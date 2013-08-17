from sklearn import svm
from sklearn.decomposition import TruncatedSVD
import book_classification as bc
from scipy import sparse

class Configuration:
	#def parameter_names(self):
	#	raise NotImplementedError()
	pass

class ConfigurationForVocabulary:
	# input filter
	# output filter
	# collapsing, stemming, etc
	pass

class ConfigurationForComponents:
	# window size and type
	# group size
	# transformer arguments
	pass

class ConfigurationForPipeline:
	# SVD vs NMF vs random projections
	# SVM vs random forests
	pass

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
	
	def encoder_for(self, collection, vocabulary):
		if vocabulary is None:
			input_vocabulary = bc.HierarchialFeatures.from_book_collection(
				collection, bc.VocabulariesExtractor(self._extractor.tokenizer())).total().keys()
			output_vocabulary = self._extractor.features_for_vocabulary(input_vocabulary)
		else:
			output_vocabulary = self._extractor.features_for_vocabulary(vocabulary)
		return FeaturesEncoder(output_vocabulary)

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

class CollectionFeaturesEncoder:
	def __init__(self, collection_extractor, base_collection, output_vocabulary=None):
		self._collection_extractor = collection_extractor
		self._base_collection = base_collection
		self._features_encoder = self._collection_extractor.encoder_for(
			base_collection, output_vocabulary)

	def encode(self, collection):
		collection_features = self._collection_extractor.extract_from(collection)
		
		matrix = sparse.dok_matrix((len(collection), len(self._features_indexer)))
		for i,book in enumerate(collection.books()):
			features = collection_features[book]
			for j,v in self._features_encoder(features):
				matrix[i, j] = v

		return matrix

class ClassificationModel:
	def __init__(self, collection_extractor, transformer, model, output_vocabulary=None):
		self._collection_extractor = collection_extractor
		self._transformer = transformer
		self._model = model

	def fit(self, collection):
		self._training = collection
		self._collection_encoder = self._collection_extractor.encoder_for(
			self._training, self._output_vocabulary)
		self._authors_indexer = bc.NumericIndexer(self._training.authors())
		
		matrix = self._collection_encoder.encode(self._training)
		authors = self.encode_authors(self._training)
		
		self._transformer.fit(matrix)
		processed_matrix = self._transformer.transform(matrix)
		self._model.fit(processed_matrix, authors)


	def predict(self, collection):
		matrix, authors = self._encoder.encode_collection(collection)
		processed_matrix = self._transformer.transform(matrix)
		predicted_authors_encoded = self._model.predict(processed_matrix)
		predicted_authors = self._encoder.decode_authors(predicted_authors_encoded)

		result = {}
		for book,predicted in zip(collection.books(), predicted_authors):
			result[book] = predicted
		return result

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