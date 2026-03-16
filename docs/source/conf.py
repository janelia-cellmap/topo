# Configuration file for the Sphinx documentation builder.

project = "topo"
copyright = "2024, Marwan Zouinkhi"
author = "Marwan Zouinkhi"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "autoapi.extension",
    "myst_nb",
    # NOTE: do NOT add myst_parser — myst_nb already includes it,
    # and having both causes "source_suffix .md is already registered"
]

# AutoAPI configuration
autoapi_type = "python"
autoapi_dirs = ["../../src"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

# MyST-NB: execute notebooks during build
nb_execution_mode = "auto"
nb_execution_timeout = 600
nb_execution_raise_on_error = True

# Theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Suppress warnings for duplicate labels from autoapi + manual docs
suppress_warnings = ["myst.domains"]
