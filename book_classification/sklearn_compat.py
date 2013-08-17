class SklExtractor:
	def __init__(self):
		pass
	def fit(self, books_list):
		pass
	def transform(self, books_list):
		pass

class SklAuthorDecoder:
	def __init__(self):
		pass
	def fit(self, books_list):
		pass
	def transform(self, books_list):
		pass

class SklPipelineObserver:
	def __init__(self):
		pass
	def fit(self, books_list):
		pass
	def transform(self, books_list):
		pass

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
