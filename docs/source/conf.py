import sys
import os
from unittest.mock import MagicMock
project = 'pydpf'
author = 'John-Joseph Brady'
version = '1.0.0'
release = '1.0.0'
extensions = ['numpydoc', 'autodoc']
master_doc = 'modules'
sys.modules['numpy'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['pydpf'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['polars'] = MagicMock()
sys.path.insert(0, os.path.abspath('../../pydpf/'))