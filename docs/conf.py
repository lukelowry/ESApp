import os
import sys
import importlib.metadata

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(".."))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "nbsphinx",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "member-order": "groupwise",
}

autodoc_preserve_defaults = True
todo_include_todos = True
autosectionlabel_prefix_document = True

autoclass_content = "both"
autodoc_typehints = "none"
add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "np": "numpy",
    "np.ndarray": "~numpy.ndarray",
    "pd": "pandas",
    "pd.DataFrame": "~pandas.DataFrame",
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
    "**/PWRaw",
]

nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
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
    "win32com", 
    "win32com.client", 
    "pythoncom",
    "geopandas",
    "shapely",
    "fiona",
    "pyproj",
]

latex_documents = [
    (master_doc, "esapp.tex", "ESA++ Documentation", author, "manual"),
]

latex_elements = {
    "preamble": r"""
\usepackage{mathrsfs}
\usepackage{breakurl}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{multirow}
\sphinxsetup{verbatimwithframe=false}
\sloppy
""",
    "figure_align": "H",
}
