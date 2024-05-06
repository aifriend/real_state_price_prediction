### SETUP DOCUMENTATION

- poetry add:
  * sphinx
  * sphinx-rtd-theme


- run:
  * make docs 
  * sphinx-quickstart 
  * sphinx-apidoc -o docs .


- add to index.rst
  * modules


- add to conf.py
  * sys.path.insert(0, os.path.abspath('..'))
  * extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc']
  * html_theme = 'sphinx_rtd_theme'


- make html