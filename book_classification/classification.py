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
        self._config = {
            'num_books': 8,
            'training_percentage': 0.6,
            'num_trials': 3
        }

    def set_parameters(self, config):
        self._config.update(config)

    def run_experiment(self):
        num_books = self._config['num_books']
        collection = self._book_collection.selection().exclude_authors_below(num_books)
        collection = collection.selection().sample_authors(10)
        total_authors = len(collection.authors())
        results = []

        # XXX: allow passing the total number of experiments to evaluate, and
        # constants to ponderate between trials and author sets
        for num_authors in range(2, total_authors+1):
            num_sets = round(total_authors/num_authors)
            num_trials = min(num_authors, self._config['num_trials'])
            current_results = []

            for _ in range(num_sets):
                current_collection = collection.selection().sample_authors(num_authors)
                for _ in range(num_trials):
                    c = current_collection.selection().sample_books_per_author(num_books)
                    c = c.selection().split_per_author_percentage(self._config['training_percentage'])
                    training, testing = c
                    #print("%s %s" % (len(training), len(testing)))
                    self._classification_model.fit(training)
                    metric = self._classification_model.predict(testing).metric()
                    current_results.append(metric)
            results.append(current_results)

        return results


class ESOverTrainingProportion:
    def __init__(self, book_collection, classification_model):
        self._book_collection = book_collection
        self._classification_model = classification_model
        self._config = {
            'num_books': 15,
            'num_authors': 4,
            'num_steps': 10,
            'num_trials': 6,
        }

    def set_parameters(self, config):
        self._config.update(config)

    def run_experiment(self):
        results = []
        for i in range(1, self._config['num_steps']):
            percentage = i/self._config['num_steps']

            trial_results = []
            for _ in range(self._config['num_trials']):
                collection = self._book_collection.selection().sample_authors_with_books(
                    self._config['num_authors'], self._config['num_books'])

                training, testing = collection.selection().split_per_author_percentage(percentage)
                self._classification_model.fit(training)
                metric = self._classification_model.predict(testing).metric()
                trial_results.append(metric)

            results.append(trial_results)

        return results
