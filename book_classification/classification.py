import book_classification as bc


class ClassificationModel:
    def __init__(self, extractor, model, output_vocabulary=None):
        self._extractor = extractor
        self._model = model
        self._output_vocabulary = output_vocabulary

    def fit(self, collection):
        self._training = collection
        self._collection_matrix_extractor = bc.CollectionFeaturesMatrixExtractor(
            self._extractor, self._training, self._output_vocabulary)
        self._authors_indexer = bc.NumericIndexer(self._training.authors())

        matrix = self._collection_matrix_extractor.extract_from(self._training)
        authors = self.encode_authors(self._training)

        self._model.fit(matrix, authors)

    def predict(self, collection):
        matrix = self._collection_matrix_extractor.extract_from(collection)
        # XXX: if passed as strings, they will be encoded by svm
        authors = self.encode_authors(collection)
        predicted_authors = self._model.predict(matrix)

        return ClassificationResults(self, collection,
            self.decode_authors(authors), self.decode_authors(predicted_authors))

    def encode_authors(self, collection):
        return [self._authors_indexer.encode(book.author()) for book in collection.books()]

    def decode_authors(self, sequence):
        return [self._authors_indexer.decode(author) for author in sequence]


# integrate with sklearn, and produce interesting graphics; also think about results comparer
class ClassificationResults:
    def __init__(self, classification_model, collection, expected, predicted):
        self._classification_model = classification_model
        self._collection = collection
        self._expected = expected
        self._predicted = predicted

    # allow all sklearn metrics, with proxy
    def confusion_matrix(self):
        pass
