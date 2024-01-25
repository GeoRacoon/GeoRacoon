import os
import sys
from importlib.metadata import version as get_version
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../src/landiv_blur'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LanDiv Blur'
copyright = '2024, Jonas Liechti'
author = 'Jonas I. Liechti'

master_doc = 'index'

release = ".".join(get_version('landiv_blur').split('.')[:2])

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
#    'ablog',
#    'sphinx_design',
#     'cloud_sptheme.ext.relbar_links',
#     'cloud_sptheme.ext.index_styling'
]

# Napoleon autodoc settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_theme = 'alabaster'
html_static_path = ['_static']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'pydata_sphinx_theme'
#html_theme = 'sphinx_book_theme'
# html_css_files = ["custom.css"]
# html_logo = '_static/<logo>.png'
html_title = 'LanDiv Blur'
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
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
