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

- draw authors in projected space (hierarchial features by author)
- add cache; cross validation turns into a waste of resources
- analyze classification space, draw mesh, compare with frequencies
- analyze sparsity of feature encoding

## Later

- use configurable decorators in place of tokenizers and extractors (for vocabulary, filters, cache, etc)
- add more tests

## Maybe
- finish transforming tokenizers (stemmer, lemmatizer, ...), and collapsing statistics
- integrate extractor parameters with sklearn's grid search
- add feature aggregation support and integrate with sklearn
- write documentation/code examples with Sphinx
- web and DVD interface for Project Gutenberg releases