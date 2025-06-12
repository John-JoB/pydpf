import sys
import os
project = 'pydpf'
author = 'John-Joseph Brady'
version = '1.0.0'
release = '1.0.0'
extensions = ['numpydoc', 'autodoc', 'sphinx.ext.mathjax', 'sphinx_rtd_theme']
master_doc = 'modules'
numpydoc_show_class_members = False
html_theme = "pydata_sphinx_theme"
sys.path.insert(0, os.path.abspath('../../pydpf/'))
sys.path.append(os.path.abspath('../..'))