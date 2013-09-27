import book_classification as bc
from nose.tools import *
from sklearn import svm
from sklearn import decomposition
from book_classification.tests.books import *


def test_ClassificationModel():
    tokenizer = bc.BasicTokenizer()
    extractor = bc.FrequenciesExtractor(tokenizer)
    classification_model = bc.ClassificationModel(
        extractor, decomposition.TruncatedSVD(10), svm.SVC())
    classification_model.fit(trainingCollection)
    classification_results = classification_model.predict(testingCollection)
    eq_(classification_results._expected, classification_results._predicted)


def test_ClassificationResults():
    pass
