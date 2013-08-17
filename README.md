tp-final-incc
=============

Learn and predict book authors from words using supervised learning

# TODO

## Must

- partition viewer, and CV cuts viewer (very important)
- update notebooks
- compare with baseline (dummy) classifiers
- visualize experiment results, copy sklearn graphs with objects

## Later

- try interesting visualizations
- distribution vs contributions, from the same word/author matrix
- add tests for book_collection, possibly reworking filter interface (allowing to stack them and finally apply them together)
- weighting windows
- comparer for two feature sets, like entropies with different windows
- integrate parameters with sklearn's grid search

## Maybe
- finish transforming tokenizers (stemmer, lemmatizer, ...) and analysis helper
- pipeline (staged) viewer, tracker and debugger
- automatic cache, persistence, and change detector (automated regression testing)
- look at exported items in namespaces, remove some and partition if needed
- try random projections and non-negative matrix factorization, instead of sparse SVD
- add feature aggregation and integrate with sklearn