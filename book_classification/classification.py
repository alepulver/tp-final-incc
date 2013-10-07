import book_classification as bc
import random


class ClassificationModel:
    def __init__(self, extractor, dim_reducer, classifier):
        self._extractor = extractor
        self._dim_reducer = dim_reducer
        self._classifier = classifier

    def fit(self, collection):
        self._training = collection
        self._collection_matrix_extractor = bc.CollectionFeaturesMatrixExtractor(
            self._extractor, self._training)
        self._authors_indexer = bc.NumericIndexer(self._training.authors())

        matrix = self._collection_matrix_extractor.extract_from(self._training)
        authors = self.encode_authors(self._training)

        reduced_matrix = self._dim_reducer.fit_transform(matrix)
        self._classifier.fit(reduced_matrix, authors)

    def predict(self, collection):
        matrix = self._collection_matrix_extractor.extract_from(collection)
        # XXX: if passed as strings, they will be encoded by svm
        authors = self.encode_authors(collection)
        reduced_matrix = self._dim_reducer.transform(matrix)
        predicted_authors = self._classifier.predict(reduced_matrix)

        return ClassificationResults(self, collection,
            self.decode_authors(authors), self.decode_authors(predicted_authors))

    def encode_authors(self, collection):
        return [self._authors_indexer.encode(book.author()) for book in collection.books()]

    def decode_authors(self, sequence):
        return [self._authors_indexer.decode(author) for author in sequence]


class ClassificationModelFixedVoc:
    def __init__(self, extractor, dim_reducer, classifier, vocabulary):
        self._extractor = extractor
        self._collection_features_extractor = bc.CollectionFeaturesExtractor(extractor)
        self._dim_reducer = dim_reducer
        self._classifier = classifier
        self._vocabulary = vocabulary
        self._features_encoder = bc.FeaturesEncoder(self._vocabulary)
        self._collection_features_encoder = bc.CachedCollectionFeaturesEncoder(self._features_encoder)

    def fit(self, collection):
        self._training = collection
        self._authors_indexer = bc.NumericIndexer(self._training.authors())

        features = self._collection_features_extractor.extract_from(self._training)
        matrix = self._collection_features_encoder.encode(features)
        authors = self.encode_authors(self._training)

        reduced_matrix = self._dim_reducer.fit_transform(matrix)
        self._classifier.fit(reduced_matrix, authors)

    def predict(self, collection):
        features = self._collection_features_extractor.extract_from(collection)
        matrix = self._collection_features_encoder.encode(features)
        
        # XXX: if passed as strings, they will be encoded by svm
        authors = self.encode_authors(collection)
        reduced_matrix = self._dim_reducer.transform(matrix)
        predicted_authors = self._classifier.predict(reduced_matrix)

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

    def metric(self):
        return sum(1 for x,y in zip(self._expected, self._predicted) if x==y) / len(self._expected)

    def baseline_metric(self):
        return sum(len(self._collection.books_by(a)/len(self._collection))**2 for a in self._collection.authors())


class ExperimentSeries:
    pass


class ESOverAuthorsCount:
    def __init__(self, book_collection, classification_model):
        self._book_collection = book_collection
        self._classification_model = classification_model

    def run_experiment(self, config):
        for key in ['num_books', 'training_percentage', 'num_trials', 'num_authors']:
            if key not in config:
                raise Exception('missing required option %s' % key)

        num_books = config['num_books']
        collection = self._book_collection.selection().exclude_authors_below(num_books)
        total_authors = len(collection.authors())

        results = []
        for num_authors in range(2, min(total_authors+1, config['num_authors'])):
            current_results = []

            for _ in range(config['num_trials']):
                current_collection = collection.selection().sample_authors_with_books(num_authors, num_books)
                training, testing = current_collection.selection().split_per_author_percentage(config['training_percentage'])
                self._classification_model.fit(training)
                metric = self._classification_model.predict(testing).metric()
                current_results.append(metric)

            results.append(current_results)

        return results


class ESOverBiasedAuthorsCount:
    def __init__(self, book_collection, classification_model):
        self._book_collection = book_collection
        self._classification_model = classification_model

    def run_experiment(self, config):
        for key in ['min_books', 'training_percentage', 'num_trials', 'num_authors']:
            if key not in config:
                raise Exception('missing required option %s' % key)

        collection = self._book_collection.selection().exclude_authors_below(config['min_books'])
        total_authors = len(collection.authors())

        results = []
        for num_authors in range(2, min(total_authors+1, config['num_authors'])):
            current_results = []

            for _ in range(config['num_trials']):
                current_collection = collection.selection().sample_authors(num_authors)
                training, testing = current_collection.selection().split_per_author_percentage(config['training_percentage'])
                self._classification_model.fit(training)
                metric = self._classification_model.predict(testing).metric()
                current_results.append(metric)

            results.append(current_results)

        return results


class ESOverTrainingProportion:
    def __init__(self, book_collection, classification_model):
        self._book_collection = book_collection
        self._classification_model = classification_model

    def run_experiment(self, config):
        for key in ['num_books', 'num_steps', 'num_trials', 'num_authors']:
            if key not in config:
                raise Exception('missing required option %s' % key)

        results = []
        for i in range(1, config['num_steps']):
            percentage = i/config['num_steps']

            trial_results = []
            for _ in range(config['num_trials']):
                collection = self._book_collection.selection().sample_authors_with_books(
                    config['num_authors'], config['num_books'])

                training, testing = collection.selection().split_per_author_percentage(percentage)
                self._classification_model.fit(training)
                metric = self._classification_model.predict(testing).metric()
                trial_results.append(metric)

            results.append(trial_results)

        return results


class ESOverBiasedTrainingProportion:
    def __init__(self, book_collection, classification_model):
        self._book_collection = book_collection
        self._classification_model = classification_model

    def run_experiment(self, config):
        for key in ['min_books', 'num_authors', 'num_steps', 'num_trials']:
            if key not in config:
                raise Exception('missing required option %s' % key)

        results = []
        for i in range(1, config['num_steps']):
            percentage = i/config['num_steps']

            trial_results = []
            for _ in range(config['num_trials']):
                collection = self._book_collection.selection().exclude_authors_below(config['min_books'])
                collection = collection.selection().sample_authors(config['num_authors'])
                training, testing = collection.selection().split_per_author_percentage(percentage)

                self._classification_model.fit(training)
                metric = self._classification_model.predict(testing).metric()
                trial_results.append(metric)

            results.append(trial_results)

        return results
