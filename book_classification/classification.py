from sklearn import svm
from sklearn.decomposition import TruncatedSVD
import book_classification as bc
from scipy import sparse

class CollectionFeatures:
	def __init__(self, collection, collection_extractor, features_by_book):
		self._collection = collection
		self._collection_extractor = collection_extractor
		self._features_by_book = features_by_book

	def collection(self):
		return self._collection

	def by_book(self, book):
		return self._features_by_book[book]

class CollectionFeaturesExtractor:
	def __init__(self, extractor):
		self._extractor = extractor
	
	def extract_from(self, collection):
		result = {}
		for book in collection.books():
			result[book] = self._extractor.extract_from(book)
		return CollectionFeatures(collection, self, result)
	
	def encoder_for(self, collection, vocabulary):
		if vocabulary is None:
			input_vocabulary = bc.HierarchialFeatures.from_book_collection(
				collection, bc.VocabulariesExtractor(self._extractor._tokenizer)).total().keys()
			output_vocabulary = self._extractor.features_for_vocabulary(input_vocabulary)
		else:
			output_vocabulary = self._extractor.features_for_vocabulary(vocabulary)
		encoder = FeaturesEncoder(output_vocabulary)
		return CollectionFeaturesEncoder(encoder)

class FeaturesEncoder:
	def __init__(self, vocabulary):
		self._vocabulary = vocabulary
		self._numeric_indexer = bc.NumericIndexer(self._vocabulary)

	def encode(self, features):
		for k,v in features.items():
			if self._numeric_indexer.can_encode(k):
				yield self._numeric_indexer.encode(k), v			

	def decode(self, items):
		for k,v in items:
			yield self._numeric_indexer.decode(k), v
	
	def vocabulary_size(self):
		return len(self._numeric_indexer)

class CollectionFeaturesEncoder:
	def __init__(self, features_encoder):
		self._features_encoder = features_encoder

	def encode(self, collection_features):
		num_rows = len(collection_features.collection())
		num_cols = self._features_encoder.vocabulary_size()
		matrix = sparse.dok_matrix((num_rows, num_cols))
		for i,book in enumerate(collection_features.collection().books()):
			features = collection_features.by_book(book)
			for j,v in self._features_encoder.encode(features):
				matrix[i, j] = v

		return matrix

class CollectionFeaturesMatrixExtractor:
	def __init__(self, extractor, base_collection, output_vocabulary=None):
		self._extractor = extractor
		self._training = base_collection
		self._output_vocabulary = output_vocabulary

		self._collection_features_extractor = CollectionFeaturesExtractor(self._extractor)
		self._collection_features_encoder = self._collection_features_extractor.encoder_for(
			self._training, output_vocabulary)

	def extract_from(self, collection):
		collection_features = self._collection_features_extractor.extract_from(collection)
		return self._collection_features_encoder.encode(collection_features)

class ClassificationModel:
	def __init__(self, extractor, model, output_vocabulary=None):
		self._extractor = extractor
		self._model = model
		self._output_vocabulary = output_vocabulary

	def fit(self, collection):
		self._training = collection
		self._collection_matrix_extractor = CollectionFeaturesMatrixExtractor(
			self._extractor, self._training, self._output_vocabulary)
		self._authors_indexer = bc.NumericIndexer(self._training.authors())
		
		matrix = self._collection_matrix_extractor.extract_from(self._training)
		authors = self.encode_authors(self._training)
		
		self._model.fit(matrix, authors)

	def predict(self, collection):
		matrix = self._collection_matrix_extractor.extract_from(collection)
		authors = self.encode_authors(collection)
		predicted_authors = self._model.predict(matrix)

		return ClassificationResults(self, collection,
			self.decode_authors(authors), self.decode_authors(predicted_authors))

	def encode_authors(self, collection):
		return [self._authors_indexer.encode(book.author()) for book in collection.books()]
	def decode_authors(self, sequence):
		return [self._authors_indexer.decode(author) for author in sequence]

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
	def __init__(self, training, testing, model):
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