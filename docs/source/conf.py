import sys
import os
project = 'pydpf'
author = 'John-Joseph Brady'
version = '1.0.0'
release = '1.0.0'
extensions = ['numpydoc', 'autodoc', 'sphinx.ext.mathjax', 'sphinx_rtd_theme']
autodoc_mock_imports = ['joblib', 'torch', 'numpy', 'polars', 'pandas']
master_doc = 'index'
numpydoc_show_class_members = False
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    'logo': {
        'text': 'pydpf',
    },
    'show_toc_level': 4,
    'show_nav_level': 4,
    'navigation_depth': 2,
    'navigation_with_keys': False,
    # Left-align navigation links in the page header.
    'navbar_align': 'left',
    # Move `sphinx-version` from start to center to shrink the footer.
    'footer_start': ['sphinx-version'],
    'use_edit_page_button': False,
    'header_links_before_dropdown': 6,
    'icon_links_label': 'Quick Links',
    'icon_links': [
        {
            'name': 'Bitbucket',
            'url': 'https://github.com/John-JoB/pydpf',
            'icon': 'fa-brands fa-github',
            'type': 'fontawesome',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/pydpf/',
            'icon': 'fa-brands fa-python',
        },
    ],
}

sys.path.insert(0, os.path.abspath('../../pydpf/pydpf/'))
sys.path.append(os.path.abspath('../..'))