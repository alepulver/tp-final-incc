import book_classification as bc
from nose.tools import *
from book_classification.tests.books import *


class DummyFeatures:
    def __init__(self, items):
        self._items = items
    def items(self):
        return self._items


def test_CollectionFeaturesExtractorWorks():
    tokenizer = bc.BasicTokenizer()
    extractor = bc.FrequenciesExtractor(tokenizer)
    collection_extractor = bc.CollectionFeaturesExtractor(extractor)

    collection_features = collection_extractor.extract_from(trainingCollection)
    for book in collection_features.collection().books():
        features = extractor.extract_from(book)
        eq_(collection_features.by_book(book), features)


def test_FeaturesEncoderCanEncodeAndDecode():
    encoder = bc.FeaturesEncoder(["one", "two", "three"])
    items = [("one", 25), ("three", 10), ("two", 50)]
    features = DummyFeatures(items)
    #eq_(list(encoder.encode(features)), [(0, 25), (2, 10), (1, 50)])
    eq_(list(encoder.decode(encoder.encode(features))), items)


def test_FeaturesEncoderIgnoresUnknownNames():
    encoder = bc.FeaturesEncoder(["one", "two", "three"])
    items = [("one", 25), ("blah", 12), ("three", 10), ("two", 50), ("hi", 1000)]
    features = DummyFeatures(items)
    #eq_(list(encoder.encode(features)), [(0, 25), (2, 10), (1, 50)])
    eq_(list(encoder.decode(encoder.encode(features))), [("one", 25), ("three", 10), ("two", 50)])


def test_CollectionFeaturesMatrixExtractor():
    tokenizer = bc.BasicTokenizer()
    extractor = bc.FrequenciesExtractor(tokenizer)
    matrix_extractor = bc.CollectionFeaturesMatrixExtractor(
        extractor, trainingCollection)

    matrixOne = matrix_extractor.extract_from(trainingCollection)
    eq_(matrixOne.shape, (2, 11))
    eq_(matrixOne.nnz, 13)
    ok_(abs(matrixOne.sum() - 2) < 10**-10)

    matrixTwo = matrix_extractor.extract_from(testingCollection)
    eq_(matrixTwo.shape, (2, 11))
    eq_(matrixTwo.nnz, 7)
    ok_(abs(matrixTwo.sum() - 0.633333333333) < 10**-10)
