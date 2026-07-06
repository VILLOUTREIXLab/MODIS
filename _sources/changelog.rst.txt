.. _changelog:

Changelog
=========

All notable changes to MODIS are recorded here.

.. rubric:: Version 0.1.0

*Initial release.*

- Per-modality VAE with configurable encoder ratios.
- Shared multi-task Discriminator (adversarial + auxiliary heads).
- Supervised and semi-supervised training modes.
- Relativistic adversarial loss with zero-centred gradient penalty (R1).
- Deep Divergence-Based Clustering (DDC) loss and entropy regularisation.
- Checkpoint saving (best / latest) and resume from checkpoint.
- Post-training evaluation with accuracy, balanced accuracy, NMI, Jaccard
  Index, ARI, and F1 metrics.
- Hyperparameter grid search via :func:`~modis.utils.config.generate_grid`.
- OmegaConf-based configuration with full validation.