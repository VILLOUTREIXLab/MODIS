.. _overview:

Architecture Overview
=====================

MODIS learns a shared latent representation from data that comes in multiple
*modalities* (e.g., different measurement types, views, or feature sets).
The core idea is to project each modality into a common latent space so that
samples from different modalities representing the same underlying object
cluster together.

.. .. figure:: ../_static/modis_architecture.png
..    :alt: MODIS architecture diagram
..    :align: center
..    :width: 80%

   *MODIS architecture: each modality has its own VAE; all latents are fed
   into a shared multi-task Discriminator.*

Components
----------

Variational Autoencoders (VAEs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each modality :math:`m` is assigned its own :class:`~modis.nn.VAE`.  The VAE
encodes input data :math:`x^{(m)}` into a probabilistic latent code
:math:`z^{(m)} \sim \mathcal{N}(\mu^{(m)}, \sigma^{(m)2})` using the
reparameterisation trick, and decodes :math:`z^{(m)}` back to reconstruct
:math:`\hat{x}^{(m)}`.

The architecture of each VAE is determined by the ``encoder_ratios``
configuration field.  Layers are sized as ``int(input_size × ratio)`` for
the encoder and the reverse sequence for the decoder.

Discriminator
~~~~~~~~~~~~~

A single shared :class:`~modis.nn.Discriminator` receives latent codes from
all VAEs and is trained on two tasks simultaneously:

1. **Adversarial head** — distinguishes *which modality* a latent came from.
   The generator (VAE) is trained adversarially to fool this head, which
   forces the latent spaces of different modalities to align.

2. **Auxiliary head** — predicts the class label of a sample.  This head is
   supervised on labeled samples and drives the cluster structure in latent
   space.

Training Objectives
-------------------

The training alternates between two phases per batch.

Discriminator Phase
~~~~~~~~~~~~~~~~~~~

The VAEs are frozen and the Discriminator is updated to minimise:

.. math::

   \mathcal{L}_D = \mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{aux}} + \mathcal{L}_{\text{cluster}}

where :math:`\mathcal{L}_{\text{adv}}` is a relativistic adversarial loss
with zero-centred gradient penalty (R1 regularisation controlled by
``lambda_r``), :math:`\mathcal{L}_{\text{aux}}` is cross-entropy on labeled
samples, and :math:`\mathcal{L}_{\text{cluster}}` is the DDC clustering loss
(see :mod:`modis.training.losses`).

Generator Phase
~~~~~~~~~~~~~~~

The Discriminator is frozen and the VAEs are updated to minimise:

.. math::

   \mathcal{L}_G = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}} + \mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{aux}} + \mathcal{L}_{\text{cluster}}

The :math:`\beta` coefficient (``beta`` config field) controls the weight of
the KL divergence term and can be a scalar or a per-modality list.

Training Modes
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - ``supervised``
     - All training samples must carry a valid label (≥ 0).  An exception is
       raised if any ``-1`` label is encountered.
   * - ``semisupervised``
     - A mix of labeled (label ≥ 0) and unlabeled (label = ``-1``) samples is
       accepted.  Only labeled samples contribute to the auxiliary loss.