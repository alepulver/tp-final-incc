from sklearn import svm
import book_classification as bc
from scipy import sparse


# TODO: feature aggregator for collectionfeatures, or a collectionfeatures aggregator for classificationmodel

class ClassificationModelInput:
	def __init__(self, collection, features_extractor, features_by_book):
		self._collection = collection
		self._features_extractor = features_extractor
		self._features_by_book = features_by_book

		vocabulary = self._features_extractor.vocabulary()
		self._features_indexer = bc.NumericIndexer.from_objs(vocabulary)
		
		authors = self._collection.authors()
		self._authors_indexer = bc.NumericIndexer.from_objs(authors)

	def vocabulary(self):
		pass
	def extractor(self):
		pass
	
	def data_matrix_for(self, collection):
		another_collection_features = self._features_extractor.extract_features(collection)
		matrix = sparse.dok_matrix((len(collection), len(self._features_indexer)))

		for i,book in enumerate(collection.books()):
			features = another_collection_features.features_for(book)
			for k,v in features.items():
				j = self._features_indexer.encode(k)
				matrix[i, j] = v

		return matrix.tocsc()

	def authors_matrix_for(self, collection):
		result = []
		for book in enumerate(collection.books()):
			result.append(self._authors_indexer.encode(book.author()))
		return result

	def encode_authors(self, sequence):
		pass
	def decode_authors(self, sequence):
		pass

	def train(self, collection, modelClass):
		data = self.data_matrix_for(collection)
		authors = self.authors_matrix_for(collection)
		model = modelClass()
		model.fit(data, authors)
		return ClassificationModel(self, model)

	@classmethod
	def from_collection(cls, collection, features_extractor):
		features_by_book = {}
		for book in collection.books():
			features_by_book[book] = features_extractor.extract_from(book)
		return cls(colection, features_extractor, features_by_book)


class ClassificationModel:
	def __init__(self, model_input, model):
		self._model_input = _model_input
		self._model = model
	def classify(self, collection):
		#book_indexer = bc.NumericIndexer.from_objs(collection.books())
		result = self._model.predict(self._model_input.data_matrix_for(collection))
		# decode output to book collection with possibly wrong authors
		return result

class ClassificationResults:
	pass

class Experiment:
	def __init__(self, training_set, testing_set, Features):
		self.training_set = training_set
		self.testing_set = testing_set
		self.Features = Features
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