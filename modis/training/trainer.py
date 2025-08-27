import json
import pathlib

import torch
from omegaconf import ListConfig, OmegaConf

from modis import Model
from modis.training.losses import ClusteringLoss
from modis.utils.evaluation import accuracy


class Trainer:

    def __init__(self, config):
        self.model = Model(config)
        self.config = config
        self.use_relativistic_loss = True

        # # Initialize weights
        # self.model.apply(weights_init)

        # Optimizers
        self.optimizer = torch.optim.Adam(
            [{
                'params': [param for vae in self.model.variational_autoencoders for param in vae.parameters()],
                'lr': config.learning_rate
            },
            {
                'params': self.model.discriminator.parameters(),
                'lr': config.learning_rate
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
        best_epoch: int,
        best_loss: float,
        val_acc: float,
        timestamp: str,
        config,
        log: list,
        save_path: pathlib.Path,
        is_best: bool = False,
        verbose: bool = True
    ) -> pathlib.Path:

        checkpoint_data = {
            'epoch': epoch,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'val_acc': val_acc,
            'timestamp': timestamp,
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

        # Save config - It will be rewriten by best and latest checkpoints
        OmegaConf.save(config, checkpoint_dir / 'config.yaml')

        # Save log
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log, f, ensure_ascii=False, indent=4)

        if verbose:
            print(f"Saved checkpoint to {checkpoint_file}")

        return checkpoint_file
    
    def load_model_and_optimizer_states(self, checkpoint_data: dict) -> None:
        self.model.load_state_dict(checkpoint_data['model_state'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer_state'])
        print(f"Loaded state from checkpoint")

    def zero_centered_gradient_penalty(self, x: torch.Tensor, modality_index: int) -> torch.Tensor:
        modal_samples = x.detach().requires_grad_(True)

        # Logits from discriminator
        latents = self.model.variational_autoencoders[modality_index].latents(modal_samples)
        d_adv, _, _ = self.model.discriminator(latents)

        grad = torch.autograd.grad(outputs=d_adv.sum(), inputs=modal_samples, create_graph=True)[0]
        penalty = torch.mean(grad.norm(2, dim=1)**2)

        return penalty

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
            'r': [],
            'modal_recon_loss': [],
            'modal_kl_loss': [],
            'd_adv_acc': 0
        }

        targets = []  # Modality labels
        for i in range(num_modalities):
            targets.append(torch.full((x[i].size(0),), i, dtype=torch.long).to(device).detach())  ## remove detach?

        # -------------------
        # Train discriminator
        # -------------------

        for i in range(num_modalities):
            self.model.variational_autoencoders[i].eval()
        self.model.discriminator.train()

        # Outputs

        d_adv, d_aux, d_hidden = self.model(x, discriminator_only=True)

        # Losses

        if not self.use_relativistic_loss:
            d_adv_loss = torch.tensor(0., device=device)
            for i in range(num_modalities):
                d_adv_loss += self.ce_loss(d_adv[i], targets[i])
        else:
            # Relativistic loss
            relativistic_logits = torch.zeros_like(d_adv[0], device=device)
            for i in range(num_modalities):
                fake_digits = sum([d_adv[idx] for idx in range(num_modalities) if idx != i])
                relativistic_logits += d_adv[i] - fake_digits
            d_adv_loss = torch.nn.functional.softplus(-relativistic_logits)

            r = [self.zero_centered_gradient_penalty(x[i], modality_index=i) for i in range(num_modalities)]
            d_adv_loss += sum(r) * self.config.lambda_r / 2
            d_adv_loss = d_adv_loss.mean()

        # Auxiliary loss
        d_aux_loss = torch.tensor(0., device=device)
        for i in range(num_modalities):
            if self.config.training_mode == 'supervised':
                d_aux_loss += self.ce_loss(d_aux[i], y[i])
            else:
                if sum(is_labeled[i]) > 0:
                    d_aux_loss += self.ce_loss(d_aux[i][is_labeled[i]], y[i][is_labeled[i]])

        # Cluster loss
        d_cluster_loss = torch.tensor(0., device=device)
        for i in range(num_modalities):
            d_cluster_loss += self.cluster_loss(d_adv[i], d_aux[i], d_hidden[i])

        d_train_loss = d_adv_loss + d_aux_loss + d_cluster_loss

        if torch.isnan(d_train_loss) == True:
            print("[-] Nan values found, aborting training")
            return

        # Backpropagation
        self.optimizer.zero_grad()
        d_train_loss.backward()
        # Zero out VAE gradients before step
        for vae in self.model.variational_autoencoders:
            for param in vae.parameters():
                param.grad = None
        self.optimizer.step()

        # Logging
        metrics['d_train_loss'] = d_train_loss.item()
        if self.use_relativistic_loss:
            metrics['r'] = [penalty.item() for penalty in r]

        # ---------------
        # Train generators (VAEs)
        # ---------------

        for i in range(num_modalities):
            self.model.variational_autoencoders[i].train()

        self.model.discriminator.eval()

        # Outputs

        recon_x, mu, logvar, d_adv, d_aux, d_hidden = self.model(x, discriminator_only=False)

        # Modality pred accuracy
        d_adv_modal_acc = [accuracy(d_adv[i], targets[i]) for i in range(num_modalities)]
        d_adv_acc = sum(d_adv_modal_acc) / num_modalities

        # Classifier accuracy
        if self.config.training_mode == 'supervised':
            d_aux_acc = sum(accuracy(d_aux[i], y[i]) for i in range(num_modalities)) / num_modalities
        else:
            # Evaluate accuracy on labeled samples only
            d_aux_acc = 0
            modal_acc = [accuracy(d_aux[i][is_labeled[i]], y[i][is_labeled[i]])
                         for i in range(num_modalities)
                         if sum(is_labeled[i]) > 0]
            if len(modal_acc) > 0:
                d_aux_acc = sum(modal_acc) / len(modal_acc)
            else:
                d_aux_acc = None

        # Losses

        # Reconstruction
        recon_losses_modal = [self.mse_loss(recon_x[i], x[i]) for i in range(num_modalities)]
        recon_loss = sum(recon_losses_modal)

        # KL
        kl_loss_modal = [-0.5 * torch.sum(1 + logvar[i] - mu[i].pow(2) - logvar[i].exp()) for i in range(num_modalities)]
        if type(self.config.beta) == ListConfig:
            # Apply modality specific beta hyperparam
            kl_loss_modal = [self.config.beta[i] * kl for i, kl in enumerate(kl_loss_modal)]
            kl_loss = sum(kl_loss_modal)
        else:
            kl_loss = self.config.beta * sum(kl_loss_modal)

        # Discriminator adv
        if not self.use_relativistic_loss:
            d_adv_loss = torch.tensor(0., device=device)
            for i in range(num_modalities):
                for j in range(num_modalities):
                    if i == j: continue
                    d_adv_loss += self.ce_loss(d_adv[i], targets[j])
        else:
            relativistic_logits = torch.zeros_like(d_adv[0], device=device)
            for i in range(num_modalities):
                for j in range(num_modalities):
                    if i == j: continue
                    relativistic_logits += d_adv[j] - d_adv[i]
            d_adv_loss = torch.nn.functional.softplus(-relativistic_logits).mean()

        # Discriminator aux
        d_aux_loss = torch.tensor(0., device=device)
        for i in range(num_modalities):
            if self.config.training_mode == 'supervised':
                d_aux_loss += self.ce_loss(d_aux[i], y[i])
            else:
                if sum(is_labeled[i]) > 0:
                    d_aux_loss += self.ce_loss(d_aux[i][is_labeled[i]], y[i][is_labeled[i]])

        d_cluster_loss = torch.tensor(0., device=device)
        for i in range(num_modalities):
            d_cluster_loss += self.cluster_loss(d_adv[i], d_aux[i], d_hidden[i])

        if not self.use_relativistic_loss:
            d_loss = (1 / (num_modalities-1) * d_adv_loss) + d_aux_loss + d_cluster_loss  # Divide adversarial loss by the number of combinations
        else:
            d_loss = d_adv_loss + d_aux_loss + d_cluster_loss

        g_loss = recon_loss + kl_loss + d_loss

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