from sklearn import svm
from sklearn.decomposition import TruncatedSVD
import book_classification as bc
from scipy import sparse


# TODO: feature aggregator for collectionfeatures, or a collectionfeatures aggregator for classificationmodel

class MatrixTransformer:
	def __init__(self, matrix_builder):
		self._matrix_builder = matrix_builder
	def for_collection(self, collection):
		pass
	def for_training_set(self):
		pass
	def decode_authors(self, sequence):
		return self._matrix_builder.decode_authors(sequence)

class MatrixBuilder:
	def __init__(self, collection, features_extractor):
		self._collection = collection
		self._features_extractor = features_extractor

		vocabulary = self._features_extractor.vocabulary()
		self._features_indexer = bc.NumericIndexer.from_objects(vocabulary)
		authors = self._collection.authors()
		self._authors_indexer = bc.NumericIndexer.from_objects(authors)

		self._data_matrix = self.build_data_matrix(collection)
		self._authors_matrix = self.build_authors_matrix(collection)
		self._svd = TruncatedSVD(50).fit(self._data_matrix)

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
	def __init__(self, matrix_builder, model_class):
		self._matrix_builder = matrix_builder
		self._model = model_class()
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

class Experiment:
	def __init__(self, model_input, testing_set, model):
		self.training_set = training_set
		self.testing_set = testing_set
		self.model = model
	def results(self):
		model_input = ModelInput(self.training_set, self.Features)

		features_train = model_input.matrix_for(self.training_set)
		features_test = model_input.matrix_for(self.testing_set)

		#print(features_train.todense())

		model = svm.SVC()
		model.fit(features_train, list(model_input.encoded_authors_for(self.training_set)))
		
		results = model_input.decode_authors(model.predict(features_test))
		results2 = {}
		for b,a in zip(self.testing_set, results):
			results2[b] = a

		#results = {}
		return results2