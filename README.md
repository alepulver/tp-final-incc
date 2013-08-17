tp-final-incc
=============

Learn and predict book authors from words using supervised learning

# TODO

## Must

- pipeline (staged) viewer, tracker and debugger
- update notebooks

## Later

- try interesting visualizations
- distribution vs contributions, from the same word/author matrix
- compare with baseline (dummy) classifyiers
- add tests for book_collection, possibly reworking filter interface (allowing to stack them and finally apply them together)
- change matrix tests to "is permutation of", instead fo checking sum and dimensions

## Maybe
- visualize experiment results, copy sklearn graphs with objects
- comparer for two feature sets, like entropies with different windows
- weighting windows
- finish transforming tokenizers (stemmer, lemmatizer, ...) and analysis helper
- look at exported items in namespaces, remove some and partition if needed
- try random projections and non-negative matrix factorization, instead of sparse SVD
- add feature aggregation and integrate with sklearn