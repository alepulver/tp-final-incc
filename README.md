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

# TODO

## Must

- simple automatic cache with persistence (try joblib)
- add isomap graphic showing plane distances estimate and colors for each author (maybe add mesh to fill background with classiffication space)
- try finding which words (component sets) separate authors by looking at non-negative matrix factorization components, or factor analysis
- try using word association (moving windows) to classify, and "entropified" version
- use configurable decorators in place of tokenizers and extractors

## Later

- add more tests
- try random projections and manifold learning instead of sparse SVD
- profiling to speed up code

## Maybe
- finish transforming tokenizers (stemmer, lemmatizer, ...), and collapsing statistics
- integrate extractor parameters with sklearn's grid search
- look at exported items in namespaces, remove some and partition if needed
- add feature aggregation support and integrate with sklearn
- write documentation/code examples with Sphinx