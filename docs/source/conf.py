"""
Sphinx configuration file for the MODIS documentation.
"""

import os
import sys

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../'))

# -- Project information ------------------------------------------------------
project = "MODIS"
copyright = "2024, MODIS Contributors"
author = "MODIS Contributors"
release = "0.1.0"
version = "0.1"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    'sphinx.ext.viewcode',         # Links to source code
    "sphinx.ext.autosummary",
    'sphinx_autodoc_typehints',
    # "sphinx.ext.intersphinx",
    # "sphinx.ext.todo",
    # "sphinx.ext.coverage",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Napoleon settings (Google-style docstrings are used throughout)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_rtype = True

# -- Autodoc settings ---------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    # 'special-members': '__init__',
    "member-order": "bysource",
}
autosummary_generate = True
add_module_names = False
autoclass_content = 'both'

# -- Intersphinx mapping ------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "omegaconf": ("https://omegaconf.readthedocs.io/en/latest", None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# -- Todo extension -----------------------------------------------------------
todo_include_todos = True

# -- Options for HTML output --------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
    "collapse_navigation": False,
    "sticky_navigation": True,
}
html_title = "MODIS Documentation"
html_show_sourcelink = True
html_show_sphinx = False
