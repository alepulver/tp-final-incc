import book_classification as bc
from nose.tools import *
from sklearn import svm

book_1 = bc.Book("Myself", "How to model classification",
	"This is a text about how to classify books.")
book_2 = bc.Book("Someone", "Animals of the mountains",
	"A book about how animals survive in extreme environments.")
book_3 = bc.Book("Myself", "Another of my books",
	"To classify a book, try processing the text and doing some math.")
book_4 = bc.Book("Someone", "Animals of the ocean",
	"The best book describing how animals adapt to survive at the bottom of the sea.")

trainingCollection = bc.BookCollection.from_books([book_1, book_2])
testingCollection = bc.BookCollection.from_books([book_4, book_3])

def test_EncodesAndDecodesAuthors():
	extractor = bc.FrequenciesExtractor(bc.BasicTokenizer())
	vocabulary = set()
	encoder = bc.FeaturesToMatrixEncoder(extractor, vocabulary, trainingCollection.authors())

	eq_(encoder.decode_authors(encoder.encode_authors(trainingCollection)),
		[book.author() for book in trainingCollection.books()])
	eq_(encoder.decode_authors(encoder.encode_authors(testingCollection)),
		[book.author() for book in testingCollection.books()])

def test_EncodesFeaturesFromTrainingCollection():
	tokenizer = bc.BasicTokenizer()
	extractor = bc.FrequenciesExtractor(tokenizer)
	vocabulary = extractor.vocabulary_for_collection(trainingCollection)
	encoder = bc.FeaturesToMatrixEncoder(extractor, vocabulary, trainingCollection.authors())
	
	matrix = encoder.encode_features(trainingCollection)
	eq_(matrix.shape, (2, 11))
	eq_(matrix.nnz, 13)
	ok_(abs(matrix.sum() - 2) < 10**-10)

def test_EncodesFeaturesFromTestingCollection():
	tokenizer = bc.BasicTokenizer()
	extractor = bc.FrequenciesExtractor(tokenizer)
	vocabulary = extractor.vocabulary_for_collection(trainingCollection)
	encoder = bc.FeaturesToMatrixEncoder(extractor, vocabulary, trainingCollection.authors())
	
	matrix = encoder.encode_features(testingCollection)
	eq_(matrix.shape, (2, 11))
	eq_(matrix.nnz, 7)
	eq_(matrix.sum(), 0.6333333333333333)

def test_EncodesOnlyFeaturesPresentInVocabulary():
	tokenizer = bc.FilteringTokenizer(bc.BasicTokenizer(), ["text", "book", "survive", "classify"])
	extractor = bc.FrequenciesExtractor(tokenizer)
	vocabulary = extractor.vocabulary_for_collection(trainingCollection)
	encoder = bc.FeaturesToMatrixEncoder(extractor, vocabulary, trainingCollection.authors())
	
	matrix = encoder.encode_features(testingCollection)
	eq_(matrix.shape, (2, 4))
	eq_(matrix.nnz, 5)
	ok_(abs(matrix.sum() - 2) < 10**-10)

def test_ClassificationModel():
	tokenizer = bc.BasicTokenizer()
	extractor = bc.FrequenciesExtractor(tokenizer)
	classification_model = bc.ClassificationModel(trainingCollection, extractor,
		bc.DummyTransformer(), svm.SVC())
	eq_(classification_model.classify(testingCollection), {book_3: "Myself", book_4: "Someone"})

def test_ClassificationResults():
	pass