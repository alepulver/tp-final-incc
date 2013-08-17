import book_classification as bc
from nose.tools import *
from sklearn import svm, pipeline, decomposition, cross_validation
from book_classification.tests.books import *

def test_SklExtractorCanBeUsedAlone():
	tokenizer = bc.BasicTokenizer()
	extractor = bc.FrequenciesExtractor(tokenizer)
	transformer = bc.SklExtractor(extractor)
	
	books, authors = trainingCollection.as_arrays()
	transformer.fit(books, authors)
	books, authors = testingCollection.as_arrays()
	matrix = transformer.transform(books)

	eq_(matrix.shape, (2, 11))
	eq_(matrix.nnz, 7)
	ok_(abs(matrix.sum() - 0.633333333333) < 10**-10)

def test_SklExtractorCanBeUsedInPipeline():
	tokenizer = bc.BasicTokenizer()
	extractor = bc.FrequenciesExtractor(tokenizer)
	predictor = pipeline.Pipeline([
		('extractor', bc.SklExtractor(extractor)),
		('svd', decomposition.TruncatedSVD(10)),
		('svm', svm.SVC())])

	books, authors = trainingCollection.as_arrays()
	predictor.fit(books, authors)
	books, authors = testingCollection.as_arrays()
	eq_(list(predictor.predict(books)), list(authors))

def test_SklExtractorCanBeUsedInCrossValudation():
	# FIXME: remove random element from selection, without using this; pass random_state to sklearn when required
	#import random, numpy
	#random.seed(0)
	#numpy.random.seed(0)

	tokenizer = bc.BasicTokenizer()
	extractor = bc.FrequenciesExtractor(tokenizer)
	predictor = pipeline.Pipeline([
		('extractor', bc.SklExtractor(extractor)),
		('svd', decomposition.TruncatedSVD(10)),
		('svm', svm.SVC())])

	books, authors = bigCollection.as_arrays()
	
	print([b.title() for b in books])
	print(authors)
	
	scores = cross_validation.cross_val_score(predictor, books, authors,
		cv=cross_validation.StratifiedKFold(authors, n_folds=3))
	eq_(list(scores), [0.5, 0, 0.5])

def test_SklObserverCanLogResults():
	pass