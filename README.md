tp-final-incc
=============

Learn and predict book authors from words using supervised learning

# TODO

## Must

- some tests for classification.py
- experiment results, aggregation and CV
- update notebooks
- try interesting visualizations

## Ideas

- distribution vs contributions, from the same word/author matrix
- comparer for two feature sets, like entropies with different windows
- use dict mixin or similar in features.py to avoid duplicated code
- weighting windows
- stemming tokenizer
- conider collapsing undesired features together (in tokenizer) rather than truncating them at the end and losing properties