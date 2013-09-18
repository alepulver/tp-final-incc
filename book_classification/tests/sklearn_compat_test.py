import book_classification as bc
from nose.tools import *
from sklearn import svm, pipeline, decomposition, cross_validation
from book_classification.tests.books import *
import numpy


def test_SklExtractorCanBeUsedAlone():
    tokenizer = bc.BasicTokenizer()
    extractor = bc.FrequenciesExtractor(tokenizer)
    transformer = bc.SklExtractor(extractor)

    books, authors = trainingCollection.as_arrays()
    transformer.fit(books, authors)
    books, authors = testingCollection.as_arrays()
    matrix = transformer.transform(books)

    result = numpy.matrix([
        [0., 0.08333333, 0.08333333, 0.,  0., 0., 0., 0.08333333, 0.08333333, 0., 0.],
        [0., 0., 0.1, 0., 0.1, 0., 0., 0., 0., 0.1, 0.]])
    ok_(numpy.allclose(matrix.todense(), result))


def test_SklExtractorCanBeUsedInPipeline():
    tokenizer = bc.BasicTokenizer()
    extractor = bc.FrequenciesExtractor(tokenizer)
    predictor = pipeline.Pipeline([
        ('extractor', bc.SklExtractor(extractor)),
        ('svd', decomposition.TruncatedSVD(10, random_state=123)),
        ('svm', svm.SVC())])

    books, authors = trainingCollection.as_arrays()
    predictor.fit(books, authors)
    #books, authors = testingCollection.as_arrays()
    eq_(list(predictor.predict(books)), list(authors))


def test_SklExtractorCanBeUsedInCrossValidation():
    tokenizer = bc.BasicTokenizer()
    extractor = bc.FrequenciesExtractor(tokenizer)
    predictor = pipeline.Pipeline([
        ('extractor', bc.SklExtractor(extractor)),
        ('svd', decomposition.TruncatedSVD(10)),
        ('svm', svm.SVC())])

    books, authors = bigCollection.as_arrays()
    scores = cross_validation.cross_val_score(predictor, books, authors,
        cv=cross_validation.StratifiedKFold(authors, n_folds=3))
    eq_(list(scores), [1, 0.5, 1])


def test_SklObserverCanLogResults():
    pass
