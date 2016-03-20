# -*- coding: utf-8 -*-

import os

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
else:
    html_theme = 'default'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'numpydoc'
]

templates_path = ['_templates']

source_suffix = '.rst'

master_doc = 'index'

project = u'commonfate'
copyright = u'2016, Antoine Liutkus'
author = u'Antoine Liutkus'

version = u'0.1.0'
release = u'0.1.0'

language = None

exclude_patterns = ['_build']

pygments_style = 'sphinx'

todo_include_todos = False

html_static_path = ['_static']

htmlhelp_basename = 'commonfatedoc'

man_pages = [
    (master_doc, 'commonfate', u'commonfate Documentation',
     [author], 1)
]

texinfo_documents = [
  (master_doc, 'commonfate', u'commonfate Documentation',
   author, 'commonfate', 'One line description of project.',
   'Miscellaneous'),
]
