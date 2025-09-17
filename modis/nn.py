"""
This module defines the MODIS architecture. It includes a Variational Autoencoder 
(VAE) for each modality and a shared Discriminator for latent space analysis and
classification.

The key components are:
    - `VAE`: A Variational Autoencoder designed to learn a latent representation
      of a single data modality.
    - `Discriminator`: A multi-task discriminator used to classify samples in the
      latent space, distinguishing between real/fake samples and predicting
      their class labels.
    - `Model`: The main class that combines multiple VAEs and a single
      Discriminator to form MODIS. It handles forward passes, predictions,
      and cross-modal translation.
"""
import torch
import torch.nn as nn

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class

    This VAE is designed to encode input data into a latent distribution and
    decode a sample from that distribution back into the original data space.

    Args:
        input_size (int): The number of features in the input data.
        encoder_ratios (list[float]): A list of ratios to determine the size
                                      of each hidden layer in the encoder
                                      relative to the input size.
        latent_size (int): The dimensionality of the latent space.
    """

    def __init__(
        self,
        input_size: int,
        encoder_ratios: list[float],
        latent_size: int
    ):
        super().__init__()

        def block(in_features: int, out_features: int, normalize: bool = True):
            """
            Helper function to create a building block for the VAE.

            Args:
                in_features (int): Number of input features.
                out_features (int): Number of output features.
                normalize (bool): If True, adds a BatchNorm1d layer.

            Returns:
                list: A list of nn.Module layers.
            """
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def build_from_ratios(input_size: int, ratios: list, output_size: int, decoder: bool = False):
            """
            Builds the VAE from a list of ratios relative to input size.

            Args:
                input_size (int): The size of the first layer's input.
                ratios (list): Ratios for determining layer sizes.
                output_size (int): The size of the final layer's output.
                decoder (bool): If True, builds a decoder network; otherwise,
                                builds an encoder.

            Returns:
                nn.Sequential: The constructed neural network.
            """
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
        """
        Performs the reparameterization trick to sample from the latent space.

        Args:
            mu (torch.Tensor): The mean of the latent distribution.
            logvar (torch.Tensor): The log variance of the latent distribution.

        Returns:
            torch.Tensor: A sampled tensor from the latent space.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        Encodes the input data into the mean and log variance of the latent
        distribution.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean (mu)
                                               and log variance (logvar).
        """
        out = x.view(x.size(0), -1)
        out = self.encoder(out)
        return self.mu(out), self.logvar(out)

    def latents(self, x):
        """
        Returns a sample from the latent space for the input data.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: A sample from the latent distribution.
        """
        return self.reparameterize(*self.encode(x))

    def decode(self, z):
        """
        Decodes a latent space sample back to the original data space.

        Args:
            z (torch.Tensor): A sample from the latent space.

        Returns:
            torch.Tensor: The reconstructed data.
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Performs a full forward pass through the VAE.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple: A tuple containing the reconstructed data, mean, log variance,
                   and latent sample.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z

class Discriminator(nn.Module):
    """
    Discriminator class

    This discriminator is designed for use in an Auxiliary Classifier GAN (AC-GAN)
    setting. It performs two tasks: a binary real/fake classification and an
    auxiliary classification for a specific number of classes.

    Args:
        latent_size (int): The dimensionality of the input latent space.
        num_modalities (int): The number of modalities, used for the adversarial
                              layer's output size.
        num_classes (int): The number of classes for the auxiliary classifier.
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
        """
        Performs a forward pass through the discriminator.

        Args:
            z (torch.Tensor): The input latent tensor.

        Returns:
            tuple: A tuple containing the adversarial output, auxiliary
                   output, and the hidden layer output.
        """
        hidden = self.fc(z)
        adv_out = self.adv_layer(hidden)
        aux_out = self.aux_layer(hidden)
        return adv_out, aux_out, hidden
    
    def predict(self, z, include_modality_pred: bool = False) -> torch.Tensor | tuple[torch.Tensor]:
        """
        Predicts the class label for each sample in the latent space.

        Args:
            z (torch.Tensor): A tensor of latent space samples.
            include_modality_pred (bool): If True, also returns the modality
                                          label prediction.

        Returns:
            torch.Tensor | tuple[torch.Tensor]: The predicted class labels, or
                                                a tuple of class and modality
                                                predictions if
                                                ``include_modality_pred`` is True.
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

class Model(nn.Module):
    """
    Main model class combining multiple VAEs and a single Discriminator.

    This model is designed for multi-modal learning, where each modality has its
    own VAE, and a shared discriminator operates on the concatenated latent
    representations.

    Args:
        config (omegaconf.DictConfig): The configuration object for the model,
                                        including details about modalities,
                                        latent size, and device.
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
            ) for i in range(self.num_modalities)
        ])
        self.discriminator = Discriminator(
            latent_size = config.latent_size,
            num_modalities = self.num_modalities,
            num_classes = config.num_classes
        )
        self.to(self.device)

    def forward(self, x: list[torch.Tensor], discriminator_only: bool = True):
        """
        Performs a forward pass through the model.

        Args:
            x (list[torch.Tensor]): A list where each element contains samples
                                    for a single modality.
            discriminator_only (bool): If True, only returns the outputs needed
                                       to train the discriminator.

        Returns:
            tuple: A tuple of outputs, which varies based on
                   `discriminator_only`.
                   - If True: Returns (d_adv, d_aux, d_hidden)
                   - If False: Returns (recon_x, mu, logvar, d_adv, d_aux, d_hidden)
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
        Gets the latent representation of the input samples for a given modality.

        Args:
            x (torch.Tensor): Samples of a single modality, shape (samples, features).
            input_modality (int): The index of the VAE for the input modality.

        Returns:
            torch.Tensor: The latent representation of the samples.
        """
        self.eval()
        with torch.no_grad():
            latents = self.variational_autoencoders[input_modality].latents(x)
        return latents

    def predict(self, x: torch.Tensor, input_modality: int) -> torch.Tensor | list[torch.Tensor, torch.Tensor]:
        """
        Predicts the cluster (class or label) of each sample.

        Args:
            x (torch.Tensor): Samples, shape (samples, features).
            input_modality (int): The index of the VAE for the input modality.

        Returns:
            torch.Tensor: The predicted class labels.
        """
        self.eval()
        with torch.no_grad():
            latents = self.get_latents(x, input_modality=input_modality)
        return self.discriminator.predict(latents)
    
    def translate(self, x: torch.Tensor, input_modality: int, output_modality: int) -> torch.Tensor:
        """
        Performs cross-modal translation.

        This function takes samples from one modality, encodes them into the
        latent space, and then decodes them using the VAE of a different
        modality.

        Args:
            x (torch.Tensor): Samples to translate, shape (samples, features).
            input_modality (int): The index of the source modality's VAE.
            output_modality (int): The index of the target modality's VAE.

        Returns:
            torch.Tensor: The reconstructed samples in the target modality.
        """
        self.eval()
        with torch.no_grad():
            latents = self.get_latents(x, input_modality=input_modality)
            recon_x = self.variational_autoencoders[output_modality].decode(latents)
        return recon_x.cpu().numpy()

    def load_from_checkpoint(self, checkpoint_file: str, verbose: bool = True) -> None:
        """
        Loads the model's state from a checkpoint file.

        Args:
            checkpoint_file (str): The path to the checkpoint file.
            verbose (bool): If True, prints a confirmation message.
        """
        import pathlib
        from modis import load_checkpoint

        checkpoint_data = load_checkpoint(pathlib.Path(checkpoint_file))
        self.load_state_dict(checkpoint_data['model_state'])
        if verbose:
            print(f"Model loaded from {checkpoint_file}")

    def load_from_state_dict(self, model_state_dict, verbose=True) -> None:
        """
        Loads the model's state from a state dictionary.

        Args:
            model_state_dict (dict): The state dictionary.
            verbose (bool): If True, prints a confirmation message.
        """
        self.load_state_dict(model_state_dict)
        if verbose:
            print(f"Model loaded from state dict")

    def print_model_params(self) -> None:
        """
        Prints the number of trainable parameters for each module of the model.
        """
        for idx, vae in enumerate(self.variational_autoencoders):
            vae_params = sum([p.numel()for name,p in vae.named_parameters()])
            print(f"{self.modality_names[idx]} VAE params: {vae_params}")

        discriminator_params = sum([p.numel() for name,p in self.discriminator.named_parameters()])
        print(f"Discriminator params: {discriminator_params}")

        model_params = sum([p.numel() for name,p in self.named_parameters()])
        print(f"Total model params: {model_params}")