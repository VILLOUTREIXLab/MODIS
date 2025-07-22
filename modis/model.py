import os

import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class
    """

    def __init__(
        self,
        input_size: int,
        encoder_ratios: list[float],
        latent_size: int
    ):
        super().__init__()

        def block(in_features: int, out_features: int, normalize: bool = True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def build_from_ratios(input_size: int, ratios: list, output_size: int, decoder: bool = False):
            layers = []
            current_size = input_size
            for i, ratio in enumerate(ratios):
                next_size = int(input_size * ratio) if not decoder else int(output_size * ratio)
                if i == len(ratios) - 1:  # Last layer
                    if decoder:
                        layers.extend(block(current_size, next_size))
                        layers.append(nn.Linear(next_size, output_size))
                    else:
                        layers.append(nn.Linear(current_size, next_size))
                else:
                    layers.extend(block(current_size, next_size))
                current_size = next_size
            return nn.Sequential(*layers)

        # Define scaling ratios
        decoder_ratios = encoder_ratios[::-1]  # Relative to input_size

        bottleneck_size = int(input_size * encoder_ratios[-1])
        
        self.encoder = build_from_ratios(input_size, encoder_ratios, latent_size)
        self.mu = nn.Linear(bottleneck_size, latent_size)
        self.logvar = nn.Linear(bottleneck_size, latent_size)
        self.decoder = build_from_ratios(latent_size, decoder_ratios, input_size, decoder=True)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        out = x.view(x.size(0), -1)
        out = self.encoder(out)
        return self.mu(out), self.logvar(out)

    def latents(self, x):
        return self.reparameterize(*self.encode(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

class Discriminator(nn.Module):
    """
    AC-GAN discriminator class
    """

    def __init__(self, latent_size, num_modalities, num_classes):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True)         
        )

        self.adv_layer = nn.Sequential(nn.Linear(256, num_modalities))
        self.aux_layer = nn.Sequential(nn.Linear(256, num_classes))

    def forward(self, z):
        hidden = self.fc(z)
        adv_out = self.adv_layer(hidden)
        aux_out = self.aux_layer(hidden)
        return adv_out, aux_out, hidden
    
    def predict(self, z, include_modality_pred: bool = False) -> torch.Tensor | tuple[torch.Tensor]:
        """
        Do the cluster prediction for each sample

        Args:
            include_modality_pred (boolean): Return also modality label is True
        """
        hidden = self.fc(z)
        aux_out = self.aux_layer(hidden)
        # class_pred = torch.argmax(F.softmax(aux_out, dim=1), dim=1)
        class_pred = torch.argmax(aux_out, dim=1)

        if include_modality_pred:  # useless for pred ((100/num_modalities)% acc by design)
            adv_out = self.adv_layer(hidden)
            modality_pred = torch.argmax(adv_out, dim=1)
            return class_pred, modality_pred

        return class_pred

class MODIS(nn.Module):
    """
    MODIS model class
    """

    def __init__(self, config):
        super().__init__()
        # self.model_name = config.model_name
        self.modality_names = [config.modalities[idx].name for idx in range(len(config.modalities))]
        # self.container_path = os.path.join(config.checkpoint_folder, config.model_name)
        self.num_modalities = len(config.modalities)
        self.device = config.device
        self.variational_autoencoders = nn.ModuleList([
            VAE(
                input_size = config.modalities[i].input_size,
                encoder_ratios = [1.2, 1.0, 0.75, 0.5, 0.25] if not 'encoder_ratios' in config.modalities[i] else config.modalities[i].encoder_ratios,
                latent_size = config.latent_size
            ).to(self.device) for i in range(self.num_modalities)
        ])
        self.discriminator = Discriminator(
            latent_size = config.latent_size,
            num_modalities = self.num_modalities,
            num_classes = config.num_classes
        ).to(self.device)

    def forward(self, x: list[torch.Tensor], discriminator_only: bool = True):
        """
        Feed forward the model and return the outputs required for training

        Args:
            x (list[torch.Tensor]): Each list element has the samples of an individual modality
            discriminator_only (boolean): If True, only return the outputs need to train the discriminator

        Return:
            (tuple)
        """
        if discriminator_only:
            latents = [self.get_latents(x[i], input_modality=i) for i in range(self.num_modalities)]
            d_adv, d_aux, d_hidden = zip(*(self.discriminator(latents[i].detach()) for i in range(self.num_modalities)))
            return d_adv, d_aux, d_hidden

        recon_x, mu, logvar, latents = zip(*[self.variational_autoencoders[i](x[i]) for i in range(self.num_modalities)])
        d_adv, d_aux, d_hidden = zip(*[self.discriminator(latents[i]) for i in range(self.num_modalities)])
        return recon_x, mu, logvar, d_adv, d_aux, d_hidden

    def get_latents(self, x: torch.Tensor, input_modality: int) -> torch.Tensor:
        """
        Get the latens of the input samples
        Args:
            x (torch.Tensor): Samples, shape (samples, features)
            x (torch.Tensor): Modality samples
            input_modality (int): Modality VAE index in the model to which the samples belong
        """
        self.eval()
        with torch.no_grad():
            latents = self.variational_autoencoders[input_modality].latents(x)
        return latents

    def predict(self, x: torch.Tensor, input_modality: int) -> torch.Tensor | list[torch.Tensor, torch.Tensor]:
        """
        Predict the cluster (class or label) of each sample

        Args:
            x (torch.Tensor): Samples, shape (samples, features)
            input_modality (int): Modality VAE index in the model to which the samples belong

        Return:
            (torch.Tensor): Label (cluster) prediction
        """
        self.eval()
        with torch.no_grad():
            latents = self.get_latents(x, input_modality=input_modality)
        return self.discriminator.predict(latents)
    
    def translate(self, x: torch.Tensor, input_modality: int, output_modality: int) -> torch.Tensor:
        """
        Do cross-modal translation

        Args:
            x (torch.Tensor): Samples to translate, shape (samples, features)
            input_modality (int): Modality VAE index in the model to which the samples belong
            output_modality (int): Modality VAE index in the model to which the samples will be translated

        Return:
            (torch.Tensor): Approximation of the samples in the target modality
        """
        self.eval()
        with torch.no_grad():
            latents = self.get_latents(x, input_modality=input_modality)
            recon_x = self.variational_autoencoders[output_modality].decode(latents)
        return recon_x.cpu().numpy()

    def load_from_checkpoint(self, checkpoint_file: str, verbose: bool = True) -> None:
        """Load a model from a checkpoint file"""
        import pathlib
        from modis import load_checkpoint

        checkpoint_data = load_checkpoint(pathlib.Path(checkpoint_file))
        self.load_state_dict(checkpoint_data['model_state'])
        if verbose:
            print(f"Model loaded from {checkpoint_file}")

    def load_from_state_dict(self, model_state_dict, verbose=True) -> None:
        """Load a model from a state dictionary"""
        self.load_state_dict(model_state_dict)
        if verbose:
            print(f"Model loaded from state dict")

    def print_model_params(self) -> None:
        """Detail the number of params in each module of the model"""
        for idx, vae in enumerate(self.variational_autoencoders):
            vae_params = sum([p.numel()for name,p in vae.named_parameters()])
            print(f"{self.modality_names[idx]} VAE params: {vae_params}")

        discriminator_params = sum([p.numel() for name,p in self.discriminator.named_parameters()])
        print(f"Discriminator params: {discriminator_params}")

        model_params = sum([p.numel() for name,p in self.named_parameters()])
        print(f"Total model params: {model_params}")