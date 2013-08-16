from sklearn import svm
from sklearn.decomposition import TruncatedSVD
import book_classification as bc
from scipy import sparse

class FeaturesEncoder:
	def __init__(self, extractor, numeric_indexer):
		self._extractor = extractor
		self._numeric_indexer = numeric_indexer
		self._encoder_iterator = self.extractor.tokenizer().if_vocabulary(self)

	def encode(self, features):
		return self._encoder_iterator(features.items())

	def decode(self, items):
		for k,v in items:
			yield self._numeric_indexer.decode(k), v

	def case_dynamic_vocabulary(self, tokenizer):
		def func(items):
			for k,v in items:
				if self._numeric_indexer.can_encode(k):
					yield self._numeric_indexer.encode(k), v
		return func
	
	def case_fixed_vocabulary(self, tokenizer):
		def func(items):
			for k,v in items:
				if not self._numeric_indexer.can_encode(k):
					raise Exception('element "{}" is not in indexer vocabulary'.format(k))
				yield self._numeric_indexer.encode(k), v
		return func

	#@classmethod
	#def from_features(self, features)

class CollectionFeatures:
	def __init__(self, collection_extractor, features_by_book):
		self._collection_extractor = collection_extractor
		self._features_by_book = features_by_book

class CollectionFeaturesExtractor:
	def __init__(self, extractor):
		self._extractor = extractor
	def extract_from(self, collection):
		result = {}
		for book in collection.books():
			result[book] = self._extractor.extract_from(book)
		return CollectionFeatures(self, result)
	def vocabulary_for(self, collection):
		pass

class FittingCollectionFeaturesEncoder:
	def __init__(self, collection_extractor, output_vocabulary=None):
		self._collection_extractor = collection_extractor
		self._output_vocabulary = output_vocabulary

	def fit(self, training):
		if self._output_vocabulary is None:
			output_vocabulary = self._collection_extractor.vocabulary_for(training)
		else:
			output_vocabulary = self._output_vocabulary
		self._features_indexer = bc.NumericIndexer(output_vocabulary)

	def transform(self, collection):
		collection_features = self._collection_extractor.extract_from(collection)
		
		matrix = sparse.dok_matrix((len(collection), len(self._features_indexer)))
		for i,book in enumerate(collection.books()):
			features = collection_features[book]
			for j,v in self._features_encoder(features):
				matrix[i, j] = v

		return matrix

class CollectionFeaturesEncoder:
	def __init__(self, collection_extractor, base_collection, output_vocabulary=None):
		self._collection_extractor = collection_extractor
		self._base_collection = base_collection
		#self._authors_indexer = bc.NumericIndexer(base_collection.authors())

		if output_vocabulary is None:
			output_vocabulary = self._collection_extractor.vocabulary_for(base_collection)
		self._features_indexer = bc.NumericIndexer(output_vocabulary)

	def encode(self, collection):
		collection_features = self._collection_extractor.extract_from(collection)
		
		matrix = sparse.dok_matrix((len(collection), len(self._features_indexer)))
		for i,book in enumerate(collection.books()):
			features = collection_features[book]
			for j,v in self._features_encoder(features):
				matrix[i, j] = v

		#authors = [self._authors_indexer.encode(book.author()) for book in collection.books()]

		#return matrix, authors
		return matrix
	
	# FIXME: should be in classification model
	def decode(self, collection, results):
		result = {}
		for book,author_code in zip(collection.books(), results):
			result[book] = self._authors_indexer.decode(author_code)
		return result

class ClassificationModel:
	def __init__(self, collection_extractor, transformer, model, output_vocabulary=None):
		self._collection_extractor = collection_extractor
		self._transformer = transformer
		self._model = model

	def fit(self, training):
		vocabulary = self._collection_extractor.vocabulary_for(training)
		self._encoder = CollectionFeaturesEncoder(
			self._collection_extractor, training, output_vocabulary)
		matrix, authors = self._encoder.encode_collection(training)
		self._transformer.fit(matrix)
		processed_matrix = self._transformer.transform(matrix)
		self._model.fit(processed_matrix, authors)

	def predict(self, collection):
		pass

	def classify(self, collection):
		matrix, authors = self._encoder.encode_collection(collection)
		processed_matrix = self._transformer.transform(matrix)
		predicted_authors_encoded = self._model.predict(processed_matrix)
		predicted_authors = self._encoder.decode_authors(predicted_authors_encoded)

		result = {}
		for book,predicted in zip(collection.books(), predicted_authors):
			result[book] = predicted
		return result

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
class SingleExperiment:
	def __init__(self, training, testing, classification_model):
		self._training = training
		self._testing = testing
		self._extractor = extractor
		self._matrix_builder = FeaturesToMatrixEncoder(extractor, transformer_class)
		self._model_class = model_class()

	def results(self):
		builder = MatrixBuilder(self._training_col, self._features_extractor)
		cm = ClassificationModel(builder, self._model_class)
		# IDEA: use block with implicit input/output conversion, where inside only sklearn code is used
		# (books and authors are converted to matrices and numbers respectively)
		return cm.classify(self._testing_col)

class MultipleExperiment:
	# take experiment constructor, dataset, partitioning/cv scheme
	# report individual and aggregated statistics
	def __init__(self, collection, classification_model, partitioning_strategy):
		pass