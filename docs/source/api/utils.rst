.. _api_utils:

Utilities (``modis.utils``)
============================

The ``modis.utils`` package provides I/O helpers, evaluation functions,
configuration management, display utilities, and model-selection tools.

I/O (``modis.utils.io``)
-------------------------

.. automodule:: modis.utils.io
   :no-members:

.. autofunction:: modis.utils.io.load_config

.. autofunction:: modis.utils.io.load_checkpoint

.. autofunction:: modis.utils.io.load_log

Configuration (``modis.utils.config``)
---------------------------------------

.. automodule:: modis.utils.config
   :no-members:

.. autofunction:: modis.utils.config.validate_config

.. autofunction:: modis.utils.config.config_from_dict

.. autofunction:: modis.utils.config.set_param

.. autofunction:: modis.utils.config.generate_grid

.. autofunction:: modis.utils.config.flatten_omegaconf

Evaluation (``modis.utils.evaluation``)
----------------------------------------

.. automodule:: modis.utils.evaluation
   :no-members:

.. autofunction:: modis.utils.evaluation.accuracy

.. autofunction:: modis.utils.evaluation.calc_classification_metrics

.. autofunction:: modis.utils.evaluation.evaluate_model

.. autofunction:: modis.utils.evaluation.evaluate_checkpoint

.. autofunction:: modis.utils.evaluation.launch_checkpoints_evaluation

.. autofunction:: modis.utils.evaluation.avg_mse_all_pairs

Display (``modis.utils.display``)
----------------------------------

.. automodule:: modis.utils.display
   :no-members:

.. autofunction:: modis.utils.display.adjust_time

Model Selection (``modis.utils.model_selection``)
--------------------------------------------------

.. automodule:: modis.utils.model_selection
   :no-members:

.. autofunction:: modis.utils.model_selection.generate_grid

.. autofunction:: modis.utils.model_selection.flatten_omegaconf

.. autofunction:: modis.utils.model_selection.validate_config

.. autofunction:: modis.utils.model_selection.set_param