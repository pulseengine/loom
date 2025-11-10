# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LOOM'
copyright = '2025, PulseEngine'
author = 'PulseEngine'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_needs',
    'sphinxcontrib.plantuml',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Sphinx-needs configuration ----------------------------------------------

needs_types = [
    {
        "directive": "req",
        "title": "Requirement",
        "prefix": "REQ_",
        "color": "#BFD8D2",
        "style": "node"
    },
    {
        "directive": "spec",
        "title": "Specification",
        "prefix": "SPEC_",
        "color": "#FEDCD2",
        "style": "node"
    },
    {
        "directive": "test",
        "title": "Test Case",
        "prefix": "TEST_",
        "color": "#DF744A",
        "style": "node"
    },
    {
        "directive": "verify",
        "title": "Verification",
        "prefix": "VERIFY_",
        "color": "#8FD9F4",
        "style": "node"
    },
]

needs_extra_options = [
    'category',
    'priority',
    'binaryen_pass',
    'implementations',
    'tests',
    'documentation',
    'verified',
]

needs_id_regex = '^[A-Z_0-9]+'

needs_statuses = [
    {
        "name": "planned",
        "description": "Requirement is planned but not yet started"
    },
    {
        "name": "active",
        "description": "Requirement is actively being worked on"
    },
    {
        "name": "implemented",
        "description": "Requirement has been implemented"
    },
    {
        "name": "verified",
        "description": "Requirement has been formally verified"
    },
    {
        "name": "tested",
        "description": "Requirement has test coverage"
    },
    {
        "name": "complete",
        "description": "Requirement is fully complete (implemented, verified, tested)"
    },
]

needs_table_columns = "id;title;status;priority;category"
needs_table_style = "datatables"

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# -- Todo extension configuration --------------------------------------------

todo_include_todos = True
