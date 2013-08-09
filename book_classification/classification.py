from sklearn import svm
from sklearn.decomposition import TruncatedSVD
import book_classification as bc
from scipy import sparse

# TODO: aggregated feature extractors

class MatrixBuilderCreator:
	def __init__(self, extractor, transformer):
		self._extractor = extractor
		self._transformer = transformer
	
	def create_from(self, collection):
		return MatrixBuilder(collection, extractor, transformer)

class MatrixBuilder:
	def __init__(self, collection, extractor, transformer):
		# always get vocabulary from extracted collection features
		# to restrict, use a restricted vocabulary inside the extractor
		# NOTE that it's required to tell the extractor to only use that vocabulary

		self._collection = collection
		self._extractor = extractor
		self._transformer = transformer

		self._data_matrix = self.build_data_matrix(collection)
		self._authors_matrix = self.build_authors_matrix(collection)
		self._svd = TruncatedSVD(50).fit(self._data_matrix)

		vocabulary = self._features_extractor.vocabulary()
		self._features_indexer = bc.NumericIndexer.from_objects(vocabulary)
		authors = self._collection.authors()
		self._authors_indexer = bc.NumericIndexer.from_objects(authors)

	def for_collection(self, collection):
		data = self._svd.transform(self.build_data_matrix(collection))
		authors = self.build_authors_matrix(collection)
		return data, authors
	def for_training_set(self):
		return self._svd.transform(self._data_matrix), self._authors_matrix

	def build_data_matrix(self, collection):
		#if collection == self._collection:
		#	return self._matrix

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

	def build_authors_matrix(self, collection):
		result = []
		for book in collection.books():
			result.append(self._authors_indexer.encode(book.author()))
		return result

	def decode_authors(self, sequence):
		return [self._authors_indexer.decode(item) for item in sequence]

class ClassificationModel:
	def __init__(self, matrix_builder, model):
		self._matrix_builder = matrix_builder
		self._model = model
		data, authors = self._matrix_builder.for_training_set()
		self._model.fit(data, authors)

	def classify(self, collection):
		#book_indexer = bc.NumericIndexer.from_objs(collection.books())
		data, authors = self._matrix_builder.for_collection(collection)
		result = self._model.predict(data)
		# decode output to book collection with possibly wrong authors
		return self._matrix_builder.decode_authors(result)

class ClassificationResults:
	pass

# pass CV method, etc
class Experiment:
	def __init__(self, training_col, testing_col, features_extractor, model_class):
		self._training_col = training_col
		self._testing_col = testing_col
		self._features_extractor = features_extractor
		self._model_class = model_class

	def results(self):
		builder = MatrixBuilder(self._training_col, self._features_extractor)
		cm = ClassificationModel(builder, self._model_class)
		# IDEA: use block with implicit input/output conversion, where inside only sklearn code is used
		# (books and authors are converted to matrices and numbers respectively)
		return cm.classify(self._testing_col)