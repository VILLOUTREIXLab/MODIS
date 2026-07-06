.. MODIS documentation master file

MODIS Documentation
===================

**MODIS** (Multi-modal Discriminative Integration System) is a deep learning
framework for joint representation learning across multiple data modalities. It
combines a per-modality Variational Autoencoder (VAE) with a shared
multi-task Discriminator to produce a unified latent space suitable for
classification, clustering, and cross-modal translation.

.. note::

   MODIS supports both **supervised** and **semi-supervised** training modes,
   making it applicable to datasets where only a fraction of samples carry
   ground-truth labels.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   userguide/overview
   userguide/configuration
   userguide/datasets
   userguide/training
   userguide/evaluation
   userguide/model_selection
   userguide/checkpoints

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/nn
   api/training
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
