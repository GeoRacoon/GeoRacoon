import os
import sys
from importlib.metadata import version as get_version
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GeoRacoon'
copyright = '2025, Simon Landauer, Jonas I. Liechti'
author = 'Simon Landauer, Jonas I. Liechti'



master_doc = 'index'

release = ".".join(get_version('GeoRacoon').split('.')[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx_design',
    'sphinx_togglebutton',
    "sphinx_copybutton",
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    "autoapi.extension",
    #    'ablog',
    #    'sphinx_design',
    #    'cloud_sptheme.ext.relbar_links',
    #    'cloud_sptheme.ext.index_styling'
]

# Napoleon autodoc settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

favicons = [
    "./GeoRacoon.png"
]
html_logo = "./GeoRacoon.png"

# html_theme = 'sphinx_rtd_theme'
# html_theme = 'alabaster'
html_theme = 'pydata_sphinx_theme'
# html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ['custom-pst.css']

html_context = {
    "github_user": "GeoRacoon",
    "github_repo": "GeoRacoon",
    "github_version": "main",
    "default_mode": "light",
}

# html_logo = '_static/<logo>.png'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_title = 'GeoRacoon'
html_theme_options = {
    "path_to_docs": "docs/",
    "use_edit_page_button": True,
    "use_repository_button": True,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "footer_start": ["copyright", "sphinx-version"],
    # 'repository_url': 'https://...',
    #    "home_page_in_toc": True,
    #    "show_toc_level": 2,
    #    'use_repository_button': True,
    #    "use_sidenotes": True,
}


myst_enable_extensions = [
    "dollarmath",
    "attrs_block",
    "amsmath",
    "deflist",
    # "html_admonition",
    # "html_image",
    "colon_fence",
    # "smartquotes",
    # "replacements",
    # "linkify",
    # "substitution",
]

# -- AutoApi Extension config -------------------------------------------------
autoapi_dirs = ["../src/", ]
autoapi_member_order = "groupwise"
autoapi_python_class_contnet = "both"  # use both class and __init__( docstring
autoapi_options = [
    "members",
    "undoc-members",
    "special-members",
    "imported-members",
    # "private-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_ignore = [
]
