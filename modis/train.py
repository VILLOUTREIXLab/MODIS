import time
import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf, DictConfig

import torch
# from torch.utils.tensorboard import SummaryWriter

from modis.training import Trainer
from modis.utils.io import load_config
from modis.utils.data import get_dataloaders, summarize_dataset
from modis.utils.display import adjust_time
from modis.utils.evaluation import evaluate_model, launch_checkpoints_evaluation
from modis.utils.plots import checkpoint_report_plots
from modis.utils.io import load_checkpoint, load_log

def parse_args():
    parser = argparse.ArgumentParser(description="MODIS Training Configuration")
    parser.add_argument('--checkpoint', type=Path, default=None, help='Checkpoint file.')
    args = parser.parse_args()
    return args

def train_loop(
    config: DictConfig,
    train_datasets: list[torch.utils.data.Dataset],
    val_datasets: list[torch.utils.data.Dataset] | None = None,
    show_dataset_summary: bool = True,
    read_args: bool = True
) -> Path | None:
    args = parse_args() if read_args else None

    # Variables
    log = []
    save_path = Path("./saved")
    # log_path = Path("./saved/log")
    device = config.device
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    init_epoch = 0
    best_epoch = 0
    best_loss = float('inf')
    val_acc = 0

    if args is not None and args.checkpoint:
        checkpoint_data = load_checkpoint(args.checkpoint)

        best_epoch = checkpoint_data['best_epoch']
        best_loss = checkpoint_data['best_loss']
        val_acc = checkpoint_data['val_acc']

        snapshot = args.checkpoint.stem.split('_')[-1]
        log_file = args.checkpoint.parent / f"checkpoint_log_{snapshot}.json"
        log = load_log(log_file)

        timestamp = checkpoint_data['timestamp']
        init_epoch = checkpoint_data['epoch']+1

        config_file = args.checkpoint.parent / 'config.yaml'
        config = OmegaConf.load(config_file)

    # Instantiate dataloaders
    train_dataloaders = get_dataloaders(train_datasets, batch_size=config.batch_size, drop_last=True, shuffle=True)

    if val_datasets is not None:
        val_dataloaders = get_dataloaders(val_datasets, batch_size=config.batch_size, drop_last=True, shuffle=True)

    if show_dataset_summary:
        print("==> Summary of train datasets")
        modality_names = [m.name for m in config.modalities]
        summarize_dataset(train_dataloaders, modality_names=modality_names)
        print()

        if val_datasets is not None:
            print("==> Summary of validation datasets")
            summarize_dataset(val_dataloaders, modality_names=modality_names)
            print()

    trainer = Trainer(config)

    # writer = SummaryWriter(log_dir=f"{log_path}/{timestamp}")

    if init_epoch > 0:
        trainer.load_model_and_optimizer_states(checkpoint_data)
        print(f"=> Resuming {config.training_mode} training on {device} device")
    else:
        print(f"==> Starting {config.training_mode} training from scratch on {device} device")

    checkpoint_file = None
    start_time = time.time()
    for epoch in range(init_epoch, init_epoch+config.num_epochs):
        metrics = []
        for i, data in enumerate(zip(*train_dataloaders)):
            x = []
            y = []
            is_labeled = []
            for i in range(len(config.modalities)):
                modal_x, modal_y = data[i][0], data[i][1]
                x.append(modal_x.to(device))
                y.append(modal_y.to(device))
                is_labeled.append(torch.tensor([True if label >= 0 else False for label in modal_y]))

                if config.training_mode == 'supervised':
                    if sum(is_labeled[i]) != modal_x.size(0):
                        raise Exception("Supervised training requires all samples to be labeled, -1 labels are invalid")

            # Train step
            batch_metrics = trainer.train_step(x, y, is_labeled)
            metrics.append(batch_metrics)

        # Process log
        epoch_metrics = {
            key: np.mean([d[key] for d in metrics if d[key] is not None]).item() if type(metrics[0][key]) != list else [np.mean(m).item() for m in zip(*[d[key] for d in metrics])]
            for key in metrics[0]
        }
        epoch_metrics['epoch_idx'] = epoch

        #
        val_acc_str = ''
        if val_datasets is not None:
            val_metrics = evaluate_model(trainer.model, val_dataloaders)
            epoch_metrics['val_acc'] = val_metrics.get('acc', 0.)
            val_acc_str = f"val_acc: {epoch_metrics['val_acc']:.4f}"

        #
        log.append(epoch_metrics)

        print(
            f"epoch: {epoch+1}/{init_epoch + config.num_epochs}, "
            f"recon_loss: {epoch_metrics['recon_loss']:.4f}, "
            f"kl_loss: {epoch_metrics['kl_loss']:.4f}, "
            f"d_train_loss: {epoch_metrics['d_train_loss']:.4f}, "
            f"d_loss: {epoch_metrics['d_loss']:.4f}, "
            f"g_loss: {epoch_metrics['g_loss']:.4f}, "
            f"d_aux_acc: {epoch_metrics['d_aux_acc']:.4f}"
            f" {val_acc_str}"
        )
        # writer.add_scalar(f"{config.model_name}/Loss/recon_loss" , epoch_metrics['recon_loss'], epoch+1)
        # writer.add_scalar(f"{config.model_name}/Loss/g_loss", epoch_metrics['g_loss'], epoch+1)
        # writer.add_scalar(f"{config.model_name}/Accuracy/d_aux_acc", epoch_metrics['d_aux_acc'], epoch+1)
        # if val_datasets is not None:
        #     writer.add_scalar(f"{config.model_name}/Accuracy/val_acc", epoch_metrics['val_acc'], epoch+1)

        # Save best checkpoint
        if val_datasets is not None:
            is_best = epoch_metrics['g_loss'] < best_loss and epoch_metrics['val_acc'] >= val_acc
        else:
            is_best = epoch_metrics['g_loss'] < best_loss
        if is_best and config.save_checkpoint_best:
            best_loss = epoch_metrics['g_loss']
            if val_datasets is not None:
                val_acc = epoch_metrics['val_acc']
            best_epoch = epoch
            checkpoint_file = trainer.save_checkpoint(
                epoch=epoch,
                best_epoch=best_epoch,
                best_loss=best_loss,
                val_acc=val_acc,
                timestamp = timestamp,
                config = config,
                log = log,
                save_path = save_path,
                is_best = True,
                verbose=False
            )

    # writer.close()

    print(f"Trained {epoch-init_epoch+1} epochs in {adjust_time(time.time() - start_time)}")
    print("==> Training finished!")
    print(f"Best model on epoch {best_epoch+1}")

    # Save latest checkpoint
    if config.save_checkpoint_latest:    
        checkpoint_file = trainer.save_checkpoint(
            epoch=epoch,
            best_epoch=best_epoch,
            best_loss=best_loss,
            val_acc=val_acc,
            timestamp=timestamp,
            config=config,
            log=log,
            save_path=save_path,
            is_best=False
        )
    
    checkpoint_dir = checkpoint_file.parent if checkpoint_file is not None else None

    del trainer
    torch.cuda.empty_cache()

    return checkpoint_dir

def train(
    config: DictConfig,
    train_datasets: list[torch.utils.data.Dataset],
    val_datasets: list[torch.utils.data.Dataset] | None = None,
    show_dataset_summary: bool = True,
    run_evaluation: bool = True,
    generate_plots: bool = True,
    read_args: bool = True
) -> Path | None:
    # Train model
    checkpoint_dir = train_loop(
        config=config,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        show_dataset_summary=show_dataset_summary,
        read_args=read_args
    )

    if checkpoint_dir is None:
        return None

    if run_evaluation:
        print()
        launch_checkpoints_evaluation(
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            checkpoint_dir=checkpoint_dir
        )

    if generate_plots:
        config = load_config(checkpoint_dir / 'config.yaml')

        if config.save_checkpoint_latest:
            checkpoint_report_plots(
                checkpoint_dir = checkpoint_dir,
                datasets = train_datasets,
                is_train = True,
                use_best = False,
                num_samples = None
            )

        if config.save_checkpoint_best:
            checkpoint_report_plots(
                checkpoint_dir = checkpoint_dir,
                datasets = train_datasets,
                is_train = True,
                use_best = True,
                num_samples = None
            )

        if val_datasets is not None:
            if config.save_checkpoint_latest:
                checkpoint_report_plots(
                    checkpoint_dir = checkpoint_dir,
                    datasets = val_datasets,
                    is_train = False,
                    use_best = False,
                    num_samples = None
                )

            if config.save_checkpoint_best:
                checkpoint_report_plots(
                    checkpoint_dir = checkpoint_dir,
                    datasets = val_datasets,
                    is_train = False,
                    use_best = True,
                    num_samples = None
                )

    return checkpoint_dir