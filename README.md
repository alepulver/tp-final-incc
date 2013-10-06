tp-final-incc
=============

Learn and predict book authors from words using supervised learning.

[![Build Status](https://travis-ci.org/alepulver/tp-final-incc.png)](https://travis-ci.org/alepulver/tp-final-incc)

# Installation

For now, clone the repository and run `python setup.py install`. It requires Pyton 3.x or greater, so be sure to set up a virtual environment if needed.

# Development

Clone the repository and run the following.

```
# install packages, change accordingly if not Ubuntu
sudo apt-get install python3 python3-dev libblas-dev libatlas-dev liblapack-dev gfortran

# set up virtual environment
pyvenv-3.3 ~/my-python3-env
source ~/my-python3-env/bin/activate
wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py -O - | python
easy_install pip

# install dependencies
pip install -r requirements.txt
```

To use the notebooks:

```
# install IPython
pip install ipython pyzmq jinja2 tornado

cd notebooks
ln -s ../books_classification .

# this should open a window in the browser
ipython3 notebook --cache-size=0 --pylab inline
```

# TODO

## Must

- try keeping original author disproportions (not kfold instead of stratifiedkfold)
- try ggplot2 and ggobi

## Later
- plot accuracy in X (training number, absolute) vs Y (number of authors), with color or surface
- add more tests
- use PyTables for persistence, cache and integration with Pandas
- web and DVD interface for Project Gutenberg releases

## Maybe
- integrate extractor parameters with sklearn's grid search
- write documentation/code examples with Sphinx
- bring back support for word associations, from branch "window_optimizations", integrate and try
- try codifying 10-grams or similar by hashing
- try 1-S instead of S for features (specially for vector version, so that zeros aren't discontinuities?)
- use doulbe dispatch and inversion of control instead of decorators to deal with extraction and encoding