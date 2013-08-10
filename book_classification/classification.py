from sklearn import svm
from sklearn.decomposition import TruncatedSVD
import book_classification as bc
from scipy import sparse

# TODO: aggregated feature extractors

# rewrite this class and rename
class MatrixBuilderCreator:
	def __init__(self, extractor, transformer_class):
		self._extractor = extractor
		self._transformer_class = transformer_class
	
	def create_from(self, collection):
		matrix_builder = MatrixBuilder(collection, self._extractor)
		return MatrixTransformer(matrix_builder, self._transformer_class())

class FeaturesToMatrixEncoder:
	def __init__(self, base_collection, fixed_extractor):
		self._base_collection = base_collection
		self._extractor = fixed_extractor
		
		authors = self._base_collection.authors()
		self._authors_indexer = bc.NumericIndexer.from_objects(authors)
		vocabulary = self._fixed_extractor.vocabulary()
		self._features_indexer = bc.NumericIndexer.from_objects(vocabulary)

	def encode_collection(self, collection):
		collection_features = {}
		for book in collection.books():
			collection_features[book] = self._features_extractor.extract_from(book)
		
		matrix = sparse.dok_matrix((len(collection), len(self._features_indexer)))
		for i,book in enumerate(collection.books()):
			features = collection_features[book]
			for k,v in features.items():
				j = self._features_indexer.encode(k)
				matrix[i, j] = v

		#return matrix.tocsc()
		return matrix

	def decode_collection(self, collection):
		raise NotImplementedError()

	def encode_authors(self, collection):
		return [self._authors_indexer.encode(book.author()) for book in collection.books()]
	def decode_authors(self, sequence):
		return [self._authors_indexer.decode(item) for item in sequence]
	def base_collection(self):
		return self._base_collection

class MatrixTransformer:
	def __init__(self, base_matrix, transformer):
		self._base_matrix = matrix_builder
		self._transformer = transformer
		self._transformer.fit(self._base_matrix)
	
	def transform(self, matrix):
		return self._transformer.transform(matrix)

class ClassificationModel:
	def __init__(self, matrix_builder, model):
		self._matrix_builder = matrix_builder
		self._model = model

		collection = self._matrix_builder.base_collection()
		matrix = self._matrix_builder.books_matrix(collection)
		authors = self._matrix_builder.encode_authors(collection)
		self._model.fit(matrix, authors)

	def classify(self, collection):
		#book_indexer = bc.NumericIndexer.from_objs(collection.books())
		data, authors = self._matrix_builder.for_collection(collection)
		result = self._model.predict(data)
		# decode output to book collection with possibly wrong authors
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
		self._matrix_builder = MatrixBuilderCreator(extractor, transformer_class)
		self._model_class = model_class()

	def results(self):
		builder = MatrixBuilder(self._training_col, self._features_extractor)
		cm = ClassificationModel(builder, self._model_class)
		# IDEA: use block with implicit input/output conversion, where inside only sklearn code is used
		# (books and authors are converted to matrices and numbers respectively)
		return cm.classify(self._testing_col)

class SimpleExperiment:
	pass