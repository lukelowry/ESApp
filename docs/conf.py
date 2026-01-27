import os
import sys
import importlib.metadata

# Ensure the project root and extensions dir are in the path
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("_ext"))

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
    "grid_list",
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

autoclass_content = "init"
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
napoleon_use_param = False
napoleon_use_rtype = False
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
html_static_path = ["_static"]
html_css_files = ["custom.css"]

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
    "pointsize": "10pt",
    "fncychap": r"\usepackage[Sonny]{fncychap}",
    "fontpkg": r"""
\usepackage{lmodern}
""",
    "preamble": r"""
\usepackage{mathrsfs}
\usepackage{breakurl}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{multirow}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{xcolor}

% Sphinx code-block styling
\sphinxsetup{
    verbatimwithframe=false,
    VerbatimColor={RGB}{248,248,248},
    VerbatimBorderColor={RGB}{200,200,200},
    InnerLinkColor={RGB}{50,50,150},
    OuterLinkColor={RGB}{50,50,150}
}

% Compact lists
\setlist{nosep}
\setlength{\parskip}{0.3em}
\setlength{\parindent}{0pt}

% Modern header/footer
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\nouppercase{\leftmark}}
\fancyhead[R]{\thepage}
\fancyfoot[C]{\small ESA++ Documentation}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.2pt}
""",
    "figure_align": "H",
}
