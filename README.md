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
cd notebooks
ln -s ../books_classification .
ipython3 notebook --cache-size=0 --pylab inline
# then a window should be opened in a web browser
```

# TODO

## Must

- analyze classification space, draw mesh, compare with frequencies

## Later

- use configurable decorators in place of tokenizers and extractors (for vocabulary, filters, cache, etc)
- add more tests
- try random projections and manifold learning instead of sparse SVD
- profiling to speed up code

## Maybe
- try using word association (moving windows) to classify, and "entropified" version; not feasible without FFT?
- finish transforming tokenizers (stemmer, lemmatizer, ...), and collapsing statistics
- integrate extractor parameters with sklearn's grid search
- look at exported items in namespaces, remove some and partition if needed
- add feature aggregation support and integrate with sklearn
- write documentation/code examples with Sphinx