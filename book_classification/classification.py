from sklearn import svm
from sklearn.decomposition import TruncatedSVD
import book_classification as bc
from scipy import sparse

# TODO: aggregated feature extractors

class FeaturesToMatrixEncoder:
	def __init__(self, fixed_extractor, authors):
		self._extractor = fixed_extractor
		self._features_indexer = bc.NumericIndexer(self._extractor.vocabulary())
		self._authors_indexer = bc.NumericIndexer(authors)

	def encode_collection(self, collection):
		matrix = self.encode_features(collection)
		authors = self.encode_authors(collection.authors())
		return matrix, authors

	def encode_features(self, collection):
		collection_features = {}
		for book in collection.books():
			collection_features[book] = self._extractor.extract_from(book)
		
		matrix = sparse.dok_matrix((len(collection), len(self._features_indexer)))
		for i,book in enumerate(collection.books()):
			features = collection_features[book]
			for k,v in features.items():
				j = self._features_indexer.encode(k)
				matrix[i, j] = v

		#return matrix.tocsc()
		return matrix

	def decode_features(self, collection):
		raise NotImplementedError()

	def encode_authors(self, collection):
		return [self._authors_indexer.encode(book.author()) for book in collection.books()]
	def decode_authors(self, sequence):
		return [self._authors_indexer.decode(item) for item in sequence]

class ClassificationModel:
	def __init__(self, training, fixed_extractor, transformer, model):
		self._encoder = FeaturesToMatrixEncoder(fixed_extractor, training.authors())
		self._transformer = transformer
		self._model = model

		matrix, authors = self._encoder.encode_collection(training)
		self._transformer.fit(matrix)
		processed_matrix = self._transformer.transform(matrix)
		self._model.fit(processed_matrix, authors)

	def classify(self, collection):
		#book_indexer = bc.NumericIndexer.from_objs(collection.books())
		matrix, authors = self._encoder.encode_collection(collection)
		processed_matrix = self._transformer.transform(matrix)
		result = self._model.predict(processed_matrix)

		# decode output to book collection with possibly wrong authors?
		# or return classification results object
		return self._matrix_builder.decode_authors(result)

class ClassificationResults:
	def confusion_matrix(self):
		pass
	def sklearn_metric(self, metric):
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