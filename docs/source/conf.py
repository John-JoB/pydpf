import sys
import os
import inspect
import pkgutil
import importlib


def get_package_members(package_name):
    objects = {}

    package = importlib.import_module(package_name)

    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module = importlib.import_module(module_name)
        objects[module_name] = []


        for name, obj in inspect.getmembers(module):
            if name.startswith("_"):
                continue
            if inspect.isclass(obj) and  obj.__module__ == module_name:
                objects[module_name].append((name, "class"))
            if inspect.isfunction(obj) and  obj.__module__ == module_name and obj.__qualname__.split(".")[0] == "MyClass":
                objects[module_name].append((name, "function"))

    return objects

def generate_api_pages(app):
    package_name = "pydpf"
    objects = get_package_members(package_name)
    api_dir = os.path.join(app.srcdir, "api")
    os.makedirs(api_dir, exist_ok=True)

    index_lines = ["API Reference", "=============\n", ".. toctree::", "   :maxdepth: 2\n"]

    for module_name, members in objects.items():
        module_file = os.path.join(api_dir, f"{module_name}.rst")
        with open(module_file, "w") as f:
            f.write(f"{module_name}\n{'=' * len(module_name)}\n\n")
            f.write(".. autosummary::\n")
            f.write("   :toctree: generated\n\n")
            for name, typ in members:
                f.write(f"   {module_name}.{name}\n")

        # Add this module page to the API index
        index_lines.append(f"   {module_name}")

    # Write the API index page
    with open(os.path.join(api_dir, "index.rst"), "w") as f:
        f.write("\n".join(index_lines))


project = 'pydpf'
author = 'John-Joseph Brady'
version = '1.0'
release = '1.1.2'
extensions = ['numpydoc', "sphinx.ext.autodoc", 'sphinx.ext.mathjax', 'sphinx_rtd_theme', 'sphinx.ext.coverage', "sphinx.ext.autosummary", "sphinx.ext.napoleon"]
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
}
master_doc = 'index'
numpydoc_show_class_members = False
html_theme = "pydata_sphinx_theme"
pygments_style = 'sphinx'

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

sys.path.insert(0, os.path.abspath('../../pydpf/'))

def setup(app):
    app.connect("builder-inited", generate_api_pages)