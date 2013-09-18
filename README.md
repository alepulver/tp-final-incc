tp-final-incc
=============

Learn and predict book authors from words using supervised learning.

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

- update notebooks

### Plotting interface for pandas

- forget about bokeh, chaco, etc and add plot class for better GoG plotting of dataframes with an R interface (for now, later maybe JS)
- try genetic algorithm to select plots interactively for finding interesting things in a dataset, and adding information to the plots based on decisions

### Visualization

- show experiment results, copy sklearn graphs with a model
- distribution vs contributions, from the same word/author matrix
- rework filter interface (allowing to stack them and finally apply them together); and show Venn diagram relating to original sample
- partition viewer, and CV cuts viewer (very important)
- comparer for two feature sets, like entropies with different windows

## Later

- compare with tf-idf and bag of words
- compare with baseline (dummy) classifiers
- weighting windows
- using word association to classify

## Maybe
- finish transforming tokenizers (stemmer, lemmatizer, ...) and analysis helper
- integrate extractor parameters with sklearn's grid search
- simple automatic cache with persistence
- profiling to speed up code
- look at exported items in namespaces, remove some and partition if needed
- try random projections and non-negative matrix factorization, instead of sparse SVD
- add feature aggregation and integrate with sklearn