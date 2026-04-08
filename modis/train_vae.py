import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train(dataloader, config, samples):
    img_shape = dataloader.dataset[0][0].shape
    samples = samples.to(config.device)

    vae = VAE(config.latent_size, img_shape).to(config.device)
    optimizer = optim.Adam(vae.parameters(), lr=config.learning_rate)

    # Training

    logs = []
    init_epoch_idx = 0

    start_time = time.time()
    for epoch in range(init_epoch_idx, init_epoch_idx+config.n_epochs):
        recon_losses = []
        kl_losses = []
        vae.train()
        for i, (x, _) in enumerate(dataloader):
            x = x.to(config.device)
            x_hat, mu, logvar = vae(x)

            # Compute the loss

            # Reconstruction loss

            #recon_loss = F.mse_loss(x_hat, x)
            recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')        

            # Kullback-Leibler divergence
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_divergence = config.beta * kl_divergence

            #loss = 5000 * recon_loss + kl_divergence
            loss = recon_loss + kl_divergence

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_divergence.item())

        logs.append({
            'epoch_idx': epoch,
            'recon_loss': np.mean(recon_losses).item(),
            'kl_loss': np.mean(kl_losses).item()
        })    

        print(
            f'Epoch [{epoch+1}/{init_epoch_idx+config.n_epochs}] '
            f'recon_loss: {logs[-1]['recon_loss']:.4f} '
            f'kl_loss: {logs[-1]['kl_loss']:.7f}'
        )