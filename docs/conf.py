# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import datetime

# If docs/ is alongside your package root, this points to the project root
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "cleands"
author = "Jason Parker"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",      # pull in docstrings
    "sphinx.ext.autosummary",  # summary tables + stub gen
    "sphinx.ext.napoleon",     # Google/NumPy style docstrings
    "sphinx.ext.viewcode",     # links to highlighted source
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Napoleon settings (Google style on; NumPy optional)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Put __init__ doc on the function page, not duplicated on class page:
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
# KEY: Render "Attributes" as :ivar: fields (not indexed attributes)
napoleon_use_ivar = True

# Autodoc settings — respect __all__ and keep the surface clean
autodoc_default_options = {
    "members": True,             # include members listed in __all__ (plus others unless filtered)
    "undoc-members": False,      # skip objects without docstrings
    "private-members": False,    # don't include _private
    "special-members": "",       # don't include __dunder__ methods
    "inherited-members": False,  # keep pages focused on your API surface
    "member-order": "bysource",
    "show-inheritance": False,
    "ignore-module-all": False,  # <-- honor module-level __all__
    # Avoid duplicate object descriptions from wrapper Attributes
    "exclude-members": "MODEL_TYPE",
}
autodoc_typehints = "description"   # put type hints into the description instead of signature
autodoc_preserve_defaults = True    # show default values as written in code
autodoc_inherit_docstrings = True

# Autosummary — generate the *_autosummary stubs from directives
autosummary_generate = True
# Don't sweep in imported things by default; you can opt-in per module if needed
autosummary_imported_members = False

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # guard against old apidoc files at the docs root
    "cleands*.rst",
    "modules.rst",
]
