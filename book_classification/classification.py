from sklearn import svm
from sklearn.decomposition import TruncatedSVD
import book_classification as bc
from scipy import sparse

class IndexerIteratorBuilder:
	def __init__(self, numeric_indexer):
		self._numeric_indexer = numeric_indexer
	
	def case_dynamic_vocabulary(self, tokenizer):
		def func(items):
			for k,v in items:
				if self._numeric_indexer.can_encode(k):
					yield k,v
		return func
	
	def case_fixed_vocabulary(self, tokenizer):
		def func(items):
			for k,v in items:
				if not self._numeric_indexer.can_encode(k):
					raise Exception('element "{}" is not in indexer vocabulary'.format(k))
				yield k,v
		return func

# TODO: separate encoder from extractor, allow each feature to produce a matrix row (list of pairs) and concatenate in encoder
class FeaturesToMatrixEncoder:
	def __init__(self, extractor, output_vocabulary, authors):
		self._extractor = extractor
		self._features_indexer = bc.NumericIndexer(output_vocabulary)
		self._authors_indexer = bc.NumericIndexer(authors)
		self._indexer_iterator = self._extractor.tokenizer().if_vocabulary(
			IndexerIteratorBuilder(self._features_indexer))

	def encode_collection(self, collection):
		matrix = self.encode_features(collection)
		authors = self.encode_authors(collection)
		return matrix, authors

	def encode_features(self, collection):
		collection_features = {}
		for book in collection.books():
			collection_features[book] = self._extractor.extract_from(book)
		
		matrix = sparse.dok_matrix((len(collection), len(self._features_indexer)))
		for i,book in enumerate(collection.books()):
			features = collection_features[book]
			for k,v in self._indexer_iterator(features.items()):
				j = self._features_indexer.encode(k)
				matrix[i, j] = v

		#return matrix.tocsc()
		return matrix

	def encode_authors(self, collection):
		return [self._authors_indexer.encode(book.author()) for book in collection.books()]
	def decode_authors(self, sequence):
		return [self._authors_indexer.decode(item) for item in sequence]

	@classmethod
	def from_collection_and_extractor(cls, collection, extractor):
		vocabulary = extractor.vocabulary_for_collection(collection)
		return self.__class__(extractor, vocabulary, training.authors())

class ClassificationModel:
	def __init__(self, training, extractor, transformer, model):
		vocabulary = extractor.vocabulary_for_collection(training)
		self._encoder = FeaturesToMatrixEncoder(extractor, vocabulary, training.authors())
		self._transformer = transformer
		self._model = model

		matrix, authors = self._encoder.encode_collection(training)
		self._transformer.fit(matrix)
		processed_matrix = self._transformer.transform(matrix)
		self._model.fit(processed_matrix, authors)

	def classify(self, collection):
		matrix, authors = self._encoder.encode_collection(collection)
		processed_matrix = self._transformer.transform(matrix)
		predicted_authors_encoded = self._model.predict(processed_matrix)
		predicted_authors = self._encoder.decode_authors(predicted_authors_encoded)

		result = {}
		for book,predicted in zip(collection.books(), predicted_authors):
			result[book] = predicted
		return result

# TODO: should be an adapter to an existing equivalent model class
class StatefulClassificationModel:
	def __init__(self, extractor, transformer, model):
		self._extractor = extractor
		self._transformer = transformer
		self._model = model

	def fit(self, books_list):
		collection = bc.BookCollection.from_books(books_list)
		# FIXME: copy transformer and model before passing?
		self._classification_model = ClassificationModel(collection, self._extractor, self._transformer, self._model)

	def predict(self, books_list):
		collection = bc.BookCollection.from_books(books_list)
		predicted_authors = self._classification_model.classify(collection)

		return [predicted_authors[book] for book in books_list]

class ClassificationResults:
	def __init__(self, classification_model, collection, expected, predicted):
		self._classification_model = classification_model
		self._collection = collection
		self._expected = expected
		self._predicted = predicted
	# allow all sklearn metrics, with proxy
	def confusion_matrix(self):
		pass

# pass CV method, etc
class Experiment:
	def __init__(self, training_col, testing_col, extractor, transformer_class, model_class):
		self._training_col = training_col
		self._testing_col = testing_col
		self._extractor = extractor
		self._matrix_builder = FeaturesToMatrixEncoder(extractor, transformer_class)
		self._model_class = model_class()

	def results(self):
		builder = MatrixBuilder(self._training_col, self._features_extractor)
		cm = ClassificationModel(builder, self._model_class)
		# IDEA: use block with implicit input/output conversion, where inside only sklearn code is used
		# (books and authors are converted to matrices and numbers respectively)
		return cm.classify(self._testing_col)

class SimpleExperiment:
	pass

class ExperimentScheme:
	pass

class ExperimentSeries:
	# take experiment constructor, dataset, partitioning/cv scheme
	# report individual and aggregated statistics
	pass