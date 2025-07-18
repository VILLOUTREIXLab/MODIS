import json
import time
import pathlib

import numpy as np
from omegaconf import OmegaConf

import torch
from torch.autograd import grad as torch_grad
from torch.utils.tensorboard import SummaryWriter

from modis.utils.config import load_config
from modis.utils.data import get_dataloaders, summarize_dataset
from modis.utils.utils import adjust_time, accuracy, evaluate_model
from modis.utils.plots import checkpoint_report_plots
from modis.model import MODIS
from modis.losses import ClusteringLoss


def load_checkpoint(checkpoint_file: pathlib.Path) -> dict:
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file {checkpoint_file} doesn't exist.")
    checkpoint = torch.load(checkpoint_file)
    return checkpoint

def load_log(checkpoint_file: pathlib.Path) -> list:
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Log file {checkpoint_file} doesn't exist.")
    with open(checkpoint_file, 'r', encoding='utf-8') as file:
        log = json.load(file)
    return log

class Trainer:

    def __init__(self, config):
        self.model = MODIS(config)
        self.config = config

        # # Initialize weights
        # self.discriminator.apply(weights_init)
        # self.generator.apply(weights_init)

        # Optimizers
        self.optimizer = torch.optim.Adam(
            [{
                'params': [param for vae in self.model.variational_autoencoders for param in vae.parameters()],
                'lr': config.generators_lr
            },
            {
                'params': self.model.discriminator.parameters(),
                'lr': config.discriminator_lr
            }],
            betas=(config.beta1, 0.999)
        )

        # Loss functions
        self.mse_loss = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.cluster_loss = ClusteringLoss()

    def save_checkpoint(
        self,
        epoch: int, 
        timestamp: str,
        config,
        log: list,
        save_path: pathlib.Path,
        is_best: bool = False,
        verbose: bool = True
    ) -> pathlib.Path:

        checkpoint_data = {
            'epoch': epoch,
            'timestamp': timestamp,
            'config': OmegaConf.to_container(config, resolve=True),
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }

        checkpoint_dir = save_path / "checkpoints" / config.dataset_name / config.model_name / timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if is_best:
            checkpoint_file =  checkpoint_dir / f"checkpoint_best.pth"
            log_file = checkpoint_dir / f"checkpoint_log_best.json"
        else:
            checkpoint_file =  checkpoint_dir / f"checkpoint_latest.pth"
            log_file = checkpoint_dir / f"checkpoint_log_latest.json"

        torch.save(checkpoint_data, checkpoint_file)

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log, f, ensure_ascii=False, indent=4)

        if verbose:
            print(f"Saved checkpoint to {checkpoint_file}")

        return checkpoint_file
    
    def load_model_and_optimizer_states(self, checkpoint_data: dict) -> None:
        self.model.load_state_dict(checkpoint_data['model_state'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer_state'])
        print(f"Loaded state from checkpoint")

    def regularization(self, x):
        total_penalty = torch.tensor(0., device=self.model.device)
        for i in range(len(x)):
            modal_samples = x[i].detach()
            modal_samples.requires_grad_(True)
            
            # Obtain discriminator output
            latents = self.model.variational_autoencoders[i].latents(modal_samples)
            d_adv, _, _ = self.model.discriminator(latents)

            grad_output = torch.ones_like(d_adv)
            grad = torch_grad(outputs=d_adv, inputs=modal_samples, grad_outputs=grad_output, create_graph=True, retain_graph=True)[0]
            grad_norm = grad.view(grad.size(0), -1).norm(2, 1)
            total_penalty += torch.mean(grad_norm**2)
        return total_penalty

    def train_step(self, x, y, is_labeled):
        num_modalities = len(x)
        device = self.config.device
        metrics = {
            'd_train_loss': 0,
            'recon_loss': 0,
            'kl_loss': 0,
            'd_loss': 0,
            'd_cluster_loss': 0,
            'd_aux_acc': 0,
            'g_loss': 0,
            'modal_recon_loss': [],
            'modal_kl_loss': [],
            'd_adv_acc': 0
        }

        targets = []  # Modality labels
        for i in range(num_modalities):
            targets.append(torch.full((x[i].size(0),), i, dtype=torch.long).to(device).detach())

        # -------------------
        # Train discriminator
        # -------------------

        for i in range(num_modalities):
            self.model.variational_autoencoders[i].eval()
        self.model.discriminator.train()

        # Outputs

        d_adv, d_aux, d_hidden = self.model(x, discriminator_only=True)

        # Calculate mean outputs for relativistic loss
        d_adv_means = []
        for i in range(num_modalities):
            d_adv_means.append(d_adv[i].mean())

        # Losses

        # d_adv_loss = torch.tensor(0., device=device)
        # for i in range(num_modalities):
        #     d_adv_loss += self.ce_loss(d_adv[i], targets[i])

        # Relativistic GAN Loss (RpGAN)
        d_adv_loss = torch.tensor(0., device=device)
        for i in range(num_modalities):
            fake_means = [adv_mean for idx, adv_mean in enumerate(d_adv_means) if idx != i]
            fake_means_sum = torch.sum(torch.stack(fake_means), dim=0)  ## average instead?
            real_loss = torch.nn.functional.relu(1 - (d_adv[i] - fake_means_sum)).mean()

            fake_loss = torch.tensor(0., device=device)
            for j in range(num_modalities):
                if i == j: continue
                # real_means = [adv_mean for idx, adv_mean in enumerate(d_adv_means) if idx != j]  ### every not j is real or just i?
                # real_means_sum = torch.sum(torch.stack(real_means), dim=0)
                # fake_loss += torch.nn.functional.relu(1 + (d_adv[j] - real_means_sum)).mean()

                # Other approach
                fake_loss += torch.nn.functional.relu(1 + (d_adv[j] - d_adv_means[i])).mean()

            d_adv_loss += real_loss - fake_loss

        # Gradient penalties of all data (R1 and R2 regularization)
        penalty = self.regularization(x) * self.config.lambda_r
        d_adv_loss += penalty

        # Auxiliary loss
        d_aux_loss = torch.tensor(0., device=device)
        for i in range(num_modalities):
            if self.config.training_mode == 'supervised':
                d_aux_loss += self.ce_loss(d_aux[i], y[i])
            else:
                labeled_mask = is_labeled[i]
                if sum(labeled_mask) > 0:
                    d_aux_loss += self.ce_loss(d_aux[i][labeled_mask], y[i])

        # Cluster loss
        d_cluster_loss = torch.tensor(0., device=device)
        for i in range(num_modalities):
            d_cluster_loss += self.cluster_loss(d_adv[i], d_aux[i], d_hidden[i])

        d_train_loss = d_adv_loss + d_aux_loss
        d_total_loss = d_train_loss + d_cluster_loss

        # Backpropagation
        self.optimizer.zero_grad()
        d_total_loss.backward()
        # Zero out VAE gradients before step
        for vae in self.model.variational_autoencoders:
            for param in vae.parameters():
                param.grad = None
        self.optimizer.step()

        if torch.isnan(d_total_loss) == True:
            print("[!] Nan values found, aborting training")
            return

        # Logging
        metrics['d_train_loss'] = d_train_loss.item()

        # ---------------
        # Train generators (VAEs)
        # ---------------

        for i in range(num_modalities):
            self.model.variational_autoencoders[i].train()

        self.model.discriminator.eval()

        # Outputs

        recon_x, mu, logvar, d_adv, d_aux, d_hidden = self.model(x, discriminator_only=False)

        # Calculate mean outputs for relativistic loss
        d_adv_means = []
        for i in range(num_modalities):
            d_adv_means.append(d_adv[i].mean())

        # Modality pred accuracy
        d_adv_modal_acc = [accuracy(d_adv[i], targets[i]) for i in range(num_modalities)]
        d_adv_acc = sum(d_adv_modal_acc) / num_modalities

        # Classifier accuracy
        if self.config.training_mode == 'supervised':
            d_aux_acc = sum(accuracy(d_aux[i], y[i]) for i in range(num_modalities)) / num_modalities
        else:
            # Evaluate accuracy only on labeled samples
            d_aux_acc = sum(
                accuracy(d_aux[i][is_labeled[i]], y[i])
                for i in range(num_modalities)
                if sum(is_labeled[i]) > 0
            ) / num_modalities

        # Losses

        # Reconstruction
        recon_losses_modal = [self.mse_loss(recon_x[i], x[i]) for i in range(num_modalities)]
        recon_loss = sum(recon_losses_modal)  # torch.stack(recon_losses_modal).mean(dim=0)
        
        # KL
        kl_loss_modal = [-0.5 * torch.sum(1 + logvar[i] - mu[i].pow(2) - logvar[i].exp()) for i in range(num_modalities)]
        kl_loss = sum(kl_loss_modal)  # torch.stack(kl_loss_modal).mean(dim=0)

        # # Discriminator adv
        # d_adv_loss = torch.tensor(0., device=device)
        # for i in range(num_modalities):
        #     for j in range(num_modalities):
        #         if i == j: continue
        #         d_adv_loss += self.ce_loss(d_adv[i], targets[j])

        # Discriminator adv
        d_adv_loss = torch.tensor(0., device=device)
        for j in range(num_modalities):
            real_means = [adv_mean for idx, adv_mean in enumerate(d_adv_means) if idx != j]  ### every not j is real or just i?
            real_means_sum = torch.sum(torch.stack(real_means), dim=0)
            d_adv_loss += torch.nn.functional.relu(1 - (d_adv[j] - real_means_sum)).mean()

        # Discriminator aux
        d_aux_loss = torch.tensor(0., device=device)
        for i in range(num_modalities):
            if self.config.training_mode == 'supervised':
                d_aux_loss += self.ce_loss(d_aux[i], y[i])
            else:
                labeled_mask = is_labeled[i]
                if sum(labeled_mask) > 0:
                    d_aux_loss += self.ce_loss(d_aux[i][labeled_mask], y[i])

        d_cluster_loss = torch.tensor(0., device=device)
        for i in range(num_modalities):
            d_cluster_loss += self.cluster_loss(d_adv[i], d_aux[i], d_hidden[i])

        # d_loss = (1 / (num_modalities-1) * d_adv_loss) + d_aux_loss #+ d_cluster_loss  # Divide adversarial loss by the number of combinations
        # d_loss = (1 / (num_modalities-1) * d_adv_loss) + d_aux_loss
        d_loss = d_adv_loss + d_aux_loss

        g_loss = recon_loss + (self.config.beta * kl_loss) + d_loss + d_cluster_loss

        # Backpropagation
        self.optimizer.zero_grad()
        g_loss.backward()
        # Zero out discriminator gradients before step
        for param in self.model.discriminator.parameters():
            param.grad = None
        self.optimizer.step()

        # Logging
        metrics['recon_loss'] = recon_loss.item()
        metrics['kl_loss'] = kl_loss.item()
        metrics['d_loss'] = d_loss.item()
        metrics['d_cluster_loss'] = d_cluster_loss.item()
        metrics['d_adv_acc'] = d_adv_acc
        metrics['d_aux_acc'] = d_aux_acc
        metrics['g_loss'] = g_loss.item()
        metrics['modal_recon_loss'].extend([loss.item() for loss in recon_losses_modal])
        metrics['modal_kl_loss'].extend([loss.item() for loss in kl_loss_modal])

        return metrics

import argparse
def read_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument('--checkpoint', type=pathlib.Path, default=None, help='Checkpoint file.')
    args = parser.parse_args()
    return args

def train(
    config_file: str,
    train_datasets: list[torch.utils.data.DataLoader],
    val_datasets: list[torch.utils.data.DataLoader] | None = None,
    show_dataset_summary: bool = True,
    run_evaluation: bool = True,
    generate_plots: bool = True
) -> pathlib.Path:
    config = load_config(config_file)
    args = read_args()

    # Variables
    log = []
    save_path = pathlib.Path("./saved")
    log_path = pathlib.Path("./saved/log")
    device = config.device
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    init_epoch = 0

    if args.checkpoint:
        checkpoint_data = load_checkpoint(args.checkpoint)

        snapshot = args.checkpoint.stem.split('_')[-1]
        log_file = args.checkpoint.parent / f"checkpoint_log_{snapshot}.json"
        log = load_log(log_file)

        timestamp = checkpoint_data['timestamp']
        init_epoch = checkpoint_data['epoch']+1
        config = OmegaConf.create(checkpoint_data['config'])

    # Instantiate dataloaders
    train_dataloaders = get_dataloaders(train_datasets, batch_size=config.batch_size, drop_last=True, shuffle=True)

    if val_datasets is not None:
        val_dataloaders = get_dataloaders(val_datasets, batch_size=config.batch_size, drop_last=True, shuffle=True)

    if show_dataset_summary:
        print("==> Summary of train datasets")
        modality_names = [m.name for m in config.modalities]
        summarize_dataset(train_dataloaders, modality_names=modality_names)

        if val_datasets is not None:
            print("==> Summary of validation datasets")
            summarize_dataset(val_dataloaders, modality_names=modality_names)

    trainer = Trainer(config)

    writer = SummaryWriter(log_dir=f"{log_path}/{timestamp}")

    if init_epoch > 0:
        trainer.load_model_and_optimizer_states(checkpoint_data)
        print(f"=> Resuming {config.training_mode} training on {device} device")
    else:
        print(f"==> Starting {config.training_mode} training from scratch on {device} device")

    best_loss = float('inf')
    val_acc = 0
    best_epoch = 0
    start_time = time.time()
    for epoch in range(init_epoch, init_epoch+config.num_epochs):
        metrics = []
        for i, data in enumerate(zip(*train_dataloaders)):
            x = []
            y = []
            is_labeled = []
            for i in range(len(config.modalities)):
                mx, my = data[i][0], data[i][1]
                x.append(mx.to(device))
                is_labeled.append(torch.tensor([True if label != -1 else False for label in my]))
                # Remove unlabeled data when in supervised mode
                if config.training_mode == 'supervised':
                    if sum(is_labeled[i]) != mx.size(0):
                        raise Exception('Supervised training requires all samples to be labeled (avoid -1 labels)')
                    y.append(my.to(device))
                else:
                    y.append(my[is_labeled[i]].to(device))

            # Train step
            batch_metrics = trainer.train_step(x, y, is_labeled)
            metrics.append(batch_metrics)

        # Process log
        epoch_metrics = {
            key: np.mean([d[key] for d in metrics]).item() if type(metrics[0][key]) != list else [np.mean(m).item() for m in zip(*[d[key] for d in metrics])]
            for key in metrics[0]
        }
        epoch_metrics['epoch_idx'] = epoch

        #
        val_acc_str = ''
        if val_datasets is not None:
            val_metrics = evaluate_model(trainer.model, train_dataloaders)
            epoch_metrics['val_acc'] = val_metrics['acc']
            val_acc_str = f"val_acc: {val_metrics['acc']:.3f}"

        #
        log.append(epoch_metrics)

        print(
            f"epoch: {epoch+1}/{init_epoch + config.num_epochs}, "
            f"d_train_loss: {epoch_metrics['d_train_loss']:.4f}, "
            f"recon_loss: {epoch_metrics['recon_loss']:.4f}, "
            f"kl_loss: {epoch_metrics['kl_loss']:.4f}, "
            f"d_loss: {epoch_metrics['d_loss']:.4f}, "
            f"d_cluster_loss: {epoch_metrics['d_cluster_loss']:.4f}, "
            f"g_loss: {epoch_metrics['g_loss']:.4f}, "
            f"d_aux_acc: {epoch_metrics['d_aux_acc']:.4f}"
            f" {val_acc_str}"
        )
        writer.add_scalar('d_loss', epoch_metrics['d_loss'], epoch+1)
        writer.add_scalar('d_aux_acc', epoch_metrics['d_aux_acc'], epoch+1)
        writer.add_scalar('g_loss', epoch_metrics['g_loss'], epoch+1)
        if val_datasets is not None:
            writer.add_scalar('val_acc', epoch_metrics['val_acc'], epoch+1)

        # Save best checkpoint
        if val_datasets is not None:
            is_best = epoch_metrics['g_loss'] < best_loss and epoch_metrics['val_acc'] >= val_acc
        else:
            is_best = epoch_metrics['g_loss'] < best_loss
        if is_best and config.save_checkpoint:
            best_loss = epoch_metrics['g_loss']
            if val_datasets is not None:
                val_acc = epoch_metrics['val_acc']
            best_epoch = epoch
            trainer.save_checkpoint(
                epoch = epoch,
                timestamp = timestamp,
                config = config,
                log = log,
                save_path = save_path,
                is_best = True,
                verbose=False
            )

    writer.close()

    print(f"Trained {epoch-init_epoch+1} epochs in {adjust_time(time.time() - start_time)}")
    print("==> Training finished!")
    print(f"Best model on epoch {best_epoch+1}")

    # Save latest checkpoint

    if not config.save_checkpoint:
        checkpoint_dir = None
        return checkpoint_dir
    
    checkpoint_file = trainer.save_checkpoint(
        epoch = epoch,
        timestamp = timestamp,
        config = config,
        log = log,
        save_path = save_path,
        is_best = False
    )
    checkpoint_dir = checkpoint_file.parent

    # Evaluate model and save metrics

    if run_evaluation:
        metrics_data = {'latest': dict(), 'best': dict()}

        # Include last batch
        train_dataloaders = get_dataloaders(train_datasets, batch_size=config.batch_size, drop_last=False, shuffle=False)
        if val_datasets is not None:
            val_dataloaders = get_dataloaders(val_datasets, batch_size=config.batch_size, drop_last=False, shuffle=False)
    
        for checkpoint_version in ['latest', 'best']:
            if checkpoint_version == 'best':
                if checkpoint_dir is None:
                    break
                checkpoint_file_best = checkpoint_dir / f"checkpoint_best.pth"
                trainer.model.load_from_checkpoint(checkpoint_file_best, verbose=False)

            print(f"==> Evaluation metrics on train dataset for {checkpoint_version} checkpoint")
            metrics = evaluate_model(trainer.model, train_dataloaders)
            metrics_data[checkpoint_version]['train'] = metrics
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")

            if val_datasets is not None:
                print(f"==> Evaluation metrics on validation dataset for {checkpoint_version} checkpoint")
                metrics = evaluate_model(trainer.model, val_dataloaders)
                metrics_data[checkpoint_version]['validation'] = metrics
                for metric_name, metric_value in metrics.items():
                    print(f"{metric_name}: {metric_value:.4f}")

        try:
            with open(checkpoint_dir / f"checkpoints_evaluation_metrics.json", 'w', encoding='utf-8') as json_file:
                json.dump(metrics_data, json_file, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving evaluation metrics file: {e}")

    # Save report plots
    if generate_plots:
        checkpoint_report_plots(
            checkpoint_dir = checkpoint_dir,
            config_file = config_file,
            datasets = train_datasets,
            is_train = True,
            use_best = False,
            num_samples = None
        )

        checkpoint_report_plots(
            checkpoint_dir = checkpoint_dir,
            config_file = config_file,
            datasets = train_datasets,
            is_train = True,
            use_best = True,
            num_samples = None
        )

        if val_datasets is not None:
            checkpoint_report_plots(
                checkpoint_dir = checkpoint_dir,
                config_file = config_file,
                datasets = val_datasets,
                is_train = False,
                use_best = False,
                num_samples = None
            )

            checkpoint_report_plots(
                checkpoint_dir = checkpoint_dir,
                config_file = config_file,
                datasets = val_datasets,
                is_train = False,
                use_best = True,
                num_samples = None
            )

    return checkpoint_dir