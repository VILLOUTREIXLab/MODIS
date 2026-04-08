"""
Neural network architectures for MODIS.

This module defines the MODIS architecture. It includes a Variational Autoencoder
(VAE) for each modality and a shared Discriminator for latent space analysis and
classification.

The key components are:

- :class:`VAE`: A Variational Autoencoder designed to learn a latent representation
  of a single data modality.
- :class:`Discriminator`: A multi-task discriminator used to classify samples in the
  latent space, distinguishing between real/fake samples and predicting their class
  labels.
- :class:`Model`: The main class that combines multiple VAEs and a single Discriminator
  to form MODIS. It handles forward passes, predictions, and cross-modal translation.
"""
import torch
import torch.nn as nn


class VAE(nn.Module):
    """Variational Autoencoder (VAE).

    Encodes input data into a latent distribution and decodes a sample from
    that distribution back into the original data space.

    Args:
        input_size (int): Number of features in the input data.
        encoder_ratios (list[float]): Ratios determining the size of each hidden
            layer in the encoder relative to ``input_size``.
        latent_size (int): Dimensionality of the latent space.
    """

    def __init__(
        self,
        input_size: int,
        encoder_ratios: list,
        latent_size: int
    ):
        super().__init__()

        def block(in_features: int, out_features: int, normalize: bool = True):
            """Create a single building block for the VAE.

            Args:
                in_features (int): Number of input features.
                out_features (int): Number of output features.
                normalize (bool): If ``True``, adds a BatchNorm1d layer.
                    Defaults to ``True``.

            Returns:
                list: A list of layer modules.
            """
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def build_from_ratios(input_size: int, ratios: list, output_size: int, decoder: bool = False):
            """Build an encoder or decoder network from a list of size ratios.

            Args:
                input_size (int): Size of the first layer's input.
                ratios (list[float]): Ratios for determining intermediate layer sizes.
                output_size (int): Size of the final layer's output.
                decoder (bool): If ``True``, builds a decoder; otherwise builds an
                    encoder. Defaults to ``False``.

            Returns:
                torch.nn.Sequential: The constructed neural network.
            """
            layers = []
            current_size = input_size
            for i, ratio in enumerate(ratios):
                next_size = int(input_size * ratio) if not decoder else int(output_size * ratio)
                if i == len(ratios) - 1:
                    if decoder:
                        layers.extend(block(current_size, next_size))
                        layers.append(nn.Linear(next_size, output_size))
                    else:
                        layers.append(nn.Linear(current_size, next_size))
                else:
                    layers.extend(block(current_size, next_size))
                current_size = next_size
            return nn.Sequential(*layers)

        decoder_ratios = encoder_ratios[::-1]
        bottleneck_size = int(input_size * encoder_ratios[-1])

        self.encoder = build_from_ratios(input_size, encoder_ratios, latent_size)
        self.mu = nn.Linear(bottleneck_size, latent_size)
        self.logvar = nn.Linear(bottleneck_size, latent_size)
        self.decoder = build_from_ratios(latent_size, decoder_ratios, input_size, decoder=True)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from the latent space using the reparameterization trick.

        Args:
            mu (torch.Tensor): Mean of the latent distribution.
            logvar (torch.Tensor): Log variance of the latent distribution.

        Returns:
            torch.Tensor: A sampled tensor from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple:
        """Encode input data into the latent distribution parameters.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            tuple: A ``(mu, logvar)`` pair representing the mean and log
            variance of the latent distribution.
        """
        out = x.view(x.size(0), -1)
        out = self.encoder(out)
        return self.mu(out), self.logvar(out)

    def latents(self, x: torch.Tensor) -> torch.Tensor:
        """Return a sample from the latent space for the given input.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: A sample drawn from the latent distribution.
        """
        return self.reparameterize(*self.encode(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent space sample back to the original data space.

        Args:
            z (torch.Tensor): A latent space sample.

        Returns:
            torch.Tensor: The reconstructed data tensor.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        """Perform a full forward pass through the VAE.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            tuple: A ``(recon_x, mu, logvar, z)`` tuple containing the
            reconstructed data, the latent mean, the latent log variance,
            and the latent sample.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z


class Discriminator(nn.Module):
    """Multi-task discriminator for the MODIS latent space.

    Operates in an Auxiliary Classifier GAN (AC-GAN) setting, performing
    two simultaneous tasks: a real/fake adversarial classification across
    modalities, and an auxiliary classification for class labels.

    Args:
        latent_size (int): Dimensionality of the input latent space.
        num_modalities (int): Number of modalities; determines the output size
            of the adversarial head.
        num_classes (int): Number of classes for the auxiliary classifier head.
    """

    def __init__(self, latent_size: int, num_modalities: int, num_classes: int):
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

    def forward(self, z: torch.Tensor) -> tuple:
        """Perform a forward pass through the discriminator.

        Args:
            z (torch.Tensor): Input latent tensor of shape
                ``(batch_size, latent_size)``.

        Returns:
            tuple: A ``(adv_out, aux_out, hidden)`` tuple containing the
            adversarial logits, the auxiliary classifier logits, and the
            penultimate hidden representation.
        """
        hidden = self.fc(z)
        adv_out = self.adv_layer(hidden)
        aux_out = self.aux_layer(hidden)
        return adv_out, aux_out, hidden

    def predict(self, z: torch.Tensor, include_modality_pred: bool = False):
        """Predict the class label for each latent sample.

        Args:
            z (torch.Tensor): Latent space samples of shape
                ``(batch_size, latent_size)``.
            include_modality_pred (bool): If ``True``, also returns the
                predicted modality label. Defaults to ``False``.

        Returns:
            torch.Tensor or tuple[torch.Tensor, torch.Tensor]:
            The predicted class label tensor of shape ``(batch_size,)``,
            or a ``(class_pred, modality_pred)`` tuple when
            ``include_modality_pred`` is ``True``.

        Note:
            The modality prediction is included only for diagnostic purposes;
            by construction the discriminator achieves roughly
            ``100 / num_modalities`` percent accuracy on this task.
        """
        hidden = self.fc(z)
        aux_out = self.aux_layer(hidden)
        class_pred = torch.argmax(aux_out, dim=1)

        if include_modality_pred:
            adv_out = self.adv_layer(hidden)
            modality_pred = torch.argmax(adv_out, dim=1)
            return class_pred, modality_pred

        return class_pred


class Model(nn.Module):
    """Main MODIS model combining multiple VAEs and a shared Discriminator.

    Each modality has its own :class:`VAE`, and a single shared
    :class:`Discriminator` operates on the latent representations produced
    by all VAEs.

    Args:
        config (omegaconf.DictConfig): Configuration object containing at least:

            - ``modalities``: list of modality configs with ``name``,
              ``input_size``, and optionally ``encoder_ratios``.
            - ``latent_size`` (int): Dimensionality of the shared latent space.
            - ``num_classes`` (int): Number of output classes.
            - ``device`` (str): Target device (``"cpu"`` or ``"cuda"``).
    """

    def __init__(self, config):
        super().__init__()
        self.modality_names = [config.modalities[idx].name for idx in range(len(config.modalities))]
        self.num_modalities = len(config.modalities)
        self.device = config.device
        self.variational_autoencoders = nn.ModuleList([
            VAE(
                input_size=config.modalities[i].input_size,
                encoder_ratios=[1.2, 1.0, 0.75, 0.5, 0.25]
                if 'encoder_ratios' not in config.modalities[i]
                else config.modalities[i].encoder_ratios,
                latent_size=config.latent_size
            ) for i in range(self.num_modalities)
        ])
        self.discriminator = Discriminator(
            latent_size=config.latent_size,
            num_modalities=self.num_modalities,
            num_classes=config.num_classes
        )
        self.to(self.device)

    def forward(self, x: list, discriminator_only: bool = True) -> tuple:
        """Perform a forward pass through the model.

        Args:
            x (list[torch.Tensor]): List of per-modality sample tensors.
            discriminator_only (bool): If ``True``, runs only the discriminator
                on detached latents (discriminator training phase). If ``False``,
                runs the full model including the VAEs. Defaults to ``True``.

        Returns:
            tuple:
            - If ``discriminator_only=True``: ``(d_adv, d_aux, d_hidden)`` —
              per-modality tuples of adversarial logits, auxiliary logits,
              and hidden representations.
            - If ``discriminator_only=False``: ``(recon_x, mu, logvar,
              d_adv, d_aux, d_hidden)`` — additionally includes per-modality
              reconstructions, latent means, and latent log variances.
        """
        if discriminator_only:
            latents = [self.get_latents(x[i], input_modality=i) for i in range(self.num_modalities)]
            d_adv, d_aux, d_hidden = zip(*(self.discriminator(latents[i].detach()) for i in range(self.num_modalities)))
            return d_adv, d_aux, d_hidden

        recon_x, mu, logvar, latents = zip(*[self.variational_autoencoders[i](x[i]) for i in range(self.num_modalities)])
        d_adv, d_aux, d_hidden = zip(*[self.discriminator(latents[i]) for i in range(self.num_modalities)])
        return recon_x, mu, logvar, d_adv, d_aux, d_hidden

    def get_latents(self, x: torch.Tensor, input_modality: int) -> torch.Tensor:
        """Return the latent representation of samples for a given modality.

        Runs the encoder of the specified VAE in evaluation mode with no
        gradient computation.

        Args:
            x (torch.Tensor): Modality samples of shape
                ``(n_samples, n_features)``.
            input_modality (int): Index of the source modality's VAE.

        Returns:
            torch.Tensor: Latent representations of shape
            ``(n_samples, latent_size)``.
        """
        self.eval()
        with torch.no_grad():
            latents = self.variational_autoencoders[input_modality].latents(x)
        return latents

    def predict(self, x: torch.Tensor, input_modality: int) -> torch.Tensor:
        """Predict the cluster or class label for each sample.

        Args:
            x (torch.Tensor): Samples of shape ``(n_samples, n_features)``.
            input_modality (int): Index of the VAE to use for encoding.

        Returns:
            torch.Tensor: Predicted class label indices of shape
            ``(n_samples,)``.
        """
        self.eval()
        with torch.no_grad():
            latents = self.get_latents(x, input_modality=input_modality)
        return self.discriminator.predict(latents)

    def translate(
        self,
        x: torch.Tensor,
        input_modality: int,
        output_modality: int
    ):
        """Perform cross-modal translation.

        Encodes samples from the source modality into the shared latent space,
        then decodes them using the target modality's VAE.

        Args:
            x (torch.Tensor): Source samples of shape
                ``(n_samples, n_features)``.
            input_modality (int): Index of the source modality's VAE (encoder).
            output_modality (int): Index of the target modality's VAE (decoder).

        Returns:
            numpy.ndarray: Translated samples in the target modality's feature
            space, shape ``(n_samples, target_features)``.
        """
        self.eval()
        with torch.no_grad():
            latents = self.get_latents(x, input_modality=input_modality)
            recon_x = self.variational_autoencoders[output_modality].decode(latents)
        return recon_x.cpu().numpy()

    def load_from_checkpoint(self, checkpoint_file: str, verbose: bool = True) -> None:
        """Load the model's state from a checkpoint file.

        Args:
            checkpoint_file (str): Path to the ``.pth`` checkpoint file.
            verbose (bool): If ``True``, prints a confirmation message.
                Defaults to ``True``.
        """
        import pathlib
        from modis import load_checkpoint

        checkpoint_data = load_checkpoint(pathlib.Path(checkpoint_file))
        self.load_state_dict(checkpoint_data['model_state'])
        if verbose:
            print(f"Model loaded from {checkpoint_file}")

    def load_from_state_dict(self, model_state_dict: dict, verbose: bool = True) -> None:
        """Load the model's state from a state dictionary.

        Args:
            model_state_dict (dict): PyTorch state dictionary mapping parameter
                names to tensors.
            verbose (bool): If ``True``, prints a confirmation message.
                Defaults to ``True``.
        """
        self.load_state_dict(model_state_dict)
        if verbose:
            print("Model loaded from state dict")

    def print_model_params(self) -> None:
        """Print the number of trainable parameters for each submodule.

        Outputs the parameter count for each modality's VAE and for the shared
        Discriminator, followed by the total parameter count.
        """
        for idx, vae in enumerate(self.variational_autoencoders):
            vae_params = sum(p.numel() for _, p in vae.named_parameters())
            print(f"{self.modality_names[idx]} VAE params: {vae_params}")

        discriminator_params = sum(p.numel() for _, p in self.discriminator.named_parameters())
        print(f"Discriminator params: {discriminator_params}")

        model_params = sum(p.numel() for _, p in self.named_parameters())
        print(f"Total model params: {model_params}")
