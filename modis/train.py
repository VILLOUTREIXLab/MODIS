import os
import pathlib
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import grad as torch_grad

from omegaconf import DictConfig

from modis.utils.config import load_config
from modis.utils.data import summarize_dataset
from modis.utils.utils import adjust_time, accuracy
from modis.model import MODIS
from modis.losses import ClusteringLoss


class ModisTrainer:

    def __init__(self, config):
        self.model = MODIS(config)
        self.config = config

        # # Initialize weights
        # self.discriminator.apply(weights_init)
        # self.generator.apply(weights_init)

        # Optimizers
        self.optimizer = torch.optim.Adam(
            [{
                'params': [param for vae in self.model.modality_vae for param in vae.parameters()],
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

    def regularization(self, x):
        total_penalty = torch.tensor(0., device=self.model.device)
        for i in range(len(x)):
            modal_samples = x[i].detach()
            modal_samples.requires_grad_(True)
            
            # Obtain discriminator output
            latents = self.model.modality_vae[i].latents(modal_samples)
            d_adv, _, _ = self.model.discriminator(latents)

            grad_output = torch.ones_like(d_adv)
            grad = torch_grad(outputs=d_adv, inputs=modal_samples, grad_outputs=grad_output, create_graph=True, retain_graph=True)[0]
            grad_norm = grad.view(grad.size(0), -1).norm(2, 1)
            total_penalty += torch.mean(grad_norm**2)
        return total_penalty

    def train_step(self, x, y, is_labeled):
        num_modalities = len(x)
        device = self.config.device
        matrics = {
            'd_train_loss': 0,
            'd_train_aux_acc': 0,
            'recon_loss': 0,
            'kl_loss': 0,
            'd_loss': 0,
            'd_cluster_loss': 0,
            'g_loss': 0,
            'modal_recon_loss': [],
            'modal_kl_loss': []
        }

        # -------------------
        # Train discriminator
        # -------------------

        for i in range(num_modalities):
            self.model.modality_vae[i].eval()
        self.model.discriminator.train()

        # Outputs

        d_adv, d_aux, d_hidden = self.model(x, discriminator_only=True)

        # Calculate mean outputs for relativistic loss
        d_adv_means = []
        for i in range(num_modalities):
            d_adv_means.append(d_adv[i].mean())

        # Classifier accuracy
        d_aux_acc = 0
        if self.config.training_mode == 'supervised':
            for i in range(num_modalities):
                d_aux_acc += accuracy(d_aux[i], y[i])
        else:
            for i in range(num_modalities):
                labeled_mask = is_labeled[i]
                if sum(labeled_mask) > 0:
                    # Evaluate accuracy only on labeled samples
                    d_aux_acc += accuracy(d_aux[i][labeled_mask], y[i])
        d_aux_acc /= num_modalities

        ### instead of doing avg among modalities
        # d_aux_total = torch.concat([d_aux[i][is_labeled[i]] for i in range(num_modalities)], dim=0)
        # y_total = torch.concat([y[i] for i in range(num_modalities)], dim=0)
        # if y_total.size(0) == 0:
        #     print(d_aux_acc, None)
        # else:
        #     print(d_aux_acc, accuracy(d_aux_total, y_total))
        ####

        # Losses

        # Relativistic GAN Loss (RpGAN)
        d_adv_loss = torch.tensor(0., device=device)
        for i in range(num_modalities):
            fake_means = [adv_mean for idx, adv_mean in enumerate(d_adv_means) if idx != i]
            fake_means_sum = torch.sum(torch.stack(fake_means), dim=0)
            real_loss = torch.nn.functional.relu(1 - (d_adv[i] - fake_means_sum)).mean()

            fake_loss = torch.tensor(0., device=device)
            for j in range(num_modalities):
                if i == j: continue
                real_means = [adv_mean for idx, adv_mean in enumerate(d_adv_means) if idx != j]  ### every not j is real or just i?
                real_means_sum = torch.sum(torch.stack(real_means), dim=0)
                fake_loss += torch.nn.functional.relu(1 + (d_adv[j] - real_means_sum)).mean()

            d_adv_loss += real_loss - fake_loss

        # Gradient penalties of all data (R1 and R2 regularization)
        # for i in range(num_modalities):
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

        d_train_loss = d_adv_loss + d_aux_loss + d_cluster_loss

        # Backpropagation and logging

        self.optimizer.zero_grad()
        d_train_loss.backward()
        # Zero out VAE gradients before step
        for vae in self.model.modality_vae:
            for param in vae.parameters():
                param.grad = None
        self.optimizer.step()

        if torch.isnan(d_train_loss) == True:
            print("[!] Nan values found, aborting training")
            return

        # Logging
        matrics['d_train_loss'] = d_train_loss.item()
        matrics['d_train_aux_acc'] = d_aux_acc  ## do here or in the generator?

        # ---------------
        # Train generators
        # ---------------

        for i in range(num_modalities):
            self.model.modality_vae[i].train()

        self.model.discriminator.eval()

        # Outputs

        recon_x, mu, logvar, d_adv, d_aux, d_hidden = self.model(x, discriminator_only=False)

        # Calculate mean outputs for relativistic loss
        d_adv_means = []
        for i in range(num_modalities):
            d_adv_means.append(d_adv[i].mean())

        # Losses

        # Reconstruction
        recon_losses_modal = [self.mse_loss(recon_x[i], x[i]) for i in range(num_modalities)]
        recon_loss = sum(recon_losses_modal)
        
        # KL
        kl_loss_modal = [-0.5 * torch.sum(1 + logvar[i] - mu[i].pow(2) - logvar[i].exp()) for i in range(num_modalities)]
        kl_loss = sum(kl_loss_modal)

        # Discriminator
        d_adv_loss = torch.tensor(0., device=device)
        for j in range(num_modalities):
            real_means = [adv_mean for idx, adv_mean in enumerate(d_adv_means) if idx != j]  ### every not j is real or just i?
            real_means_sum = torch.sum(torch.stack(real_means), dim=0)
            d_adv_loss += torch.nn.functional.relu(1 - (d_adv[j] - real_means_sum)).mean()

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
        # d_loss = d_adv_loss + d_aux_loss + d_cluster_loss
        d_loss = d_adv_loss + d_aux_loss

        g_loss = recon_loss + (self.config.beta * kl_loss) + d_loss + d_cluster_loss

        # Backpropagation and logging

        self.optimizer.zero_grad()
        g_loss.backward()
        # Zero out discriminator gradients before step
        for param in self.model.discriminator.parameters():
            param.grad = None
        self.optimizer.step()

        # Logging
        matrics['recon_loss'] = recon_loss.item()
        matrics['kl_loss'] = kl_loss.item()
        matrics['d_loss'] = d_loss.item()
        matrics['d_cluster_loss'] = d_cluster_loss.item()
        matrics['g_loss'] = g_loss.item()
        matrics['modal_recon_loss'].extend([loss.item() for loss in recon_losses_modal])
        matrics['modal_kl_loss'].extend([loss.item() for loss in kl_loss_modal])

        return matrics

def train(
    train_datasets: list[torch.utils.data.DataLoader],
    config_file: str,
    data_summary: bool = True
):
    config = load_config(config_file)

    # Instantiate dataloaders
    train_dataloaders = [DataLoader(dataset, batch_size=config.batch_size, drop_last=True, shuffle=True)
                         for dataset in train_datasets]
    
    # Variables
    device = config.device
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    init_epoch = 0
    log = []

    # Paths
    save_path = pathlib.Path("./saved")
    checkpoint_path = save_path / "checkpoints" / config.dataset_name / config.model_name
    logs_path = save_path / "logs" / config.dataset_name / config.model_name

    if data_summary:
        print("==> Summary of train datasets")
        modality_names = [m.name for m in config.modalities]
        summarize_dataset(train_dataloaders, modality_names=modality_names)

    trainer = ModisTrainer(config)

    if init_epoch > 0:
        print(f"=> Resuming training on {device} device")
    else:
        print(f"==> Starting training from scratch on {device} device")

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
        log.append(epoch_metrics)

        print(
            f"epoch: {epoch+1}/{init_epoch + config.num_epochs}, "
            f"recon_loss: {epoch_metrics['recon_loss']:.4f}, "
            f"kl_loss: {epoch_metrics['kl_loss']:.4f}, "
            f"d_loss: {epoch_metrics['d_loss']:.4f}, "
            f"d_cluster_loss: {epoch_metrics['d_cluster_loss']:.4f}, "
            f"g_loss: {epoch_metrics['g_loss']:.4f}"
        )

        # 'd_train_loss', 'd_train_aux_acc', 'modal_recon_loss', 'modal_kl_loss'


    print(f"Trained {epoch-init_epoch+1} epochs in {adjust_time(time.time() - start_time)}")
    print("==> Training finished!")
