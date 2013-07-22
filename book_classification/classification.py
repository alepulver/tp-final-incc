from sklearn import svm
from book_classification import numeric_indexer as ni
import itertools
from scipy import sparse

from IPython import embed

def mapToDict(func, iterable):
	result = {}
	for element in iterable:
		result[element] = func(element)
	return result

def mapKeys(func, iterable):
	result = {}
	for key, value in iterable.items():
		result[func(key)] = value
	return result

class ModelInput:
	def __init__(self, books, feature_extractor):
		self._books = list(books)
		self._feature_extractor = feature_extractor
		# TODO: add lazy init
		self._features_map = mapToDict(lambda x: self._feature_extractor(x).features(), self._books)
		self._feature_indexer = ni.NumericIndexer.from_objects(itertools.chain(*(fs.keys() for fs in self._features_map.values())))
		self._authors_indexer = ni.NumericIndexer.from_objects(b.author for b in self._books)

	def matrix_for(self, books):
		matrix = sparse.dok_matrix((len(self._books), len(self._feature_indexer)))
		features_map = mapToDict(lambda x: self._feature_extractor(x).features(), books)
		for i,b in enumerate(books):
			for k,v in features_map[b].items():
				try:
					matrix[i, self._feature_indexer.encode_one(k)] = v
				except:
					pass
		return matrix.tocsc()

	def encoded_authors_for(self, books):
		return self._authors_indexer.encode_many(b.author for b in books)

	def decode_authors(self, authors):
		return self._authors_indexer.decode_many(authors)

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