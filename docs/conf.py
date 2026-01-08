import os
import sys
import importlib.metadata

# Ensure the project root is in the path so sphinx can find the package
# if it's not already installed in the environment.
sys.path.insert(0, os.path.abspath(".."))

extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosectionlabel",
    "nbsphinx"
]

extensions.append("sphinx.ext.autodoc")
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "member-order": "groupwise",
}
autodoc_preserve_defaults = True
todo_include_todos = True
autosectionlabel_prefix_document = True

# Better API formatting
autoclass_content = "both"        # Include __init__ docstring in class description
autodoc_typehints = "none"        # Let Napoleon handle types from the docstring
add_module_names = False          # Don't show full module path (e.g. sgwt.static.Convolve -> Convolve)

extensions.append("sphinx.ext.intersphinx")
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
}

extensions.append("sphinx_copybutton")

# Use Napoleon to parse NumPy-style docstrings for a cleaner look
extensions.append("sphinx.ext.napoleon")
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # Common aliases
    "np": "numpy",
    "np.ndarray": "~numpy.ndarray",

    # Python built-ins and typing module
    "optional": "typing.Optional",
    "union": "typing.Union",
    "list": "list",
    "dict": "dict",
    "bool": "bool",
    "int": "int",
    "float": "float",
}

exclude_patterns = [
    "_build",
    "**/*.cpg",
    "**/*.dbf",
    "**/*.prj",
    "**/*.shp",
    "**/*.shx",
    "**/Shape.xml",
    "**/Shape.shp.ea.iso.xml",
    "**/PWRaw"
]

# Critical: RTD cannot run PowerWorld. Preserving local outputs.
nbsphinx_execute = 'never'

# Don't add .txt suffix to source files:
html_sourcelink_suffix = ''

master_doc = "index"

project = "ESA++"
copyright = "2026, Luke Lowery"
author = "Luke Lowery"
try:
    version = importlib.metadata.version("esapp")
except importlib.metadata.PackageNotFoundError:
    version = "unknown"
release = version

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 2,
}

autodoc_mock_imports = [
    "ctypes", 
    "win32com", 
    "win32com.client", 
    "pythoncom",
    "geopandas",
    "shapely"
]