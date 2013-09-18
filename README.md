tp-final-incc
=============

Learn and predict book authors from words using supervised learning

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