import multiprocessing
import book_classification as bc


class SerialCollectionFeaturesExtractor:
    def __init__(self, extractor):
        self._extractor = extractor

    def extract_from(self, collection):
        result = {}
        for book in collection.books():
            result[book] = self._extractor.extract_from(book)
        return bc.CollectionFeatures(collection, self, result)

    def encoder_for(self, collection):
        if isinstance(self._extractor, bc.PairwiseAssociationExtractor):
            return bc.DummyCollectionFeaturesEncoder()
        if isinstance(self._extractor, bc.PairwiseEntropyExtractor):
            return bc.DummyCollectionFeaturesEncoder()

        vocabulary = set()
        features = self.extract_from(collection)

        for book in features.collection().books():
            vocabulary.update(features.by_book(book).keys())

        encoder = bc.FeaturesEncoder(vocabulary)
        return bc.CollectionFeaturesEncoder(encoder)


class CollectionFeaturesFilteringExtractor:
    def __init__(self, extractor, filter_predicate):
        self._extractor = extractor
        self._filter_predicate = filter_predicate

    def extract_from(self, collection):
        features = self._extractor.extract_from(collection)
        return features.select(self._filter_predicate)


# XXX: multiprocessing module doesn't accept nested functions or lambdas
class ExtractorClosure:
    def __init__(self, extractor):
        self._extractor = extractor

    def __call__(self, book):
        return self._extractor.extract_from(book)


class ParallelCollectionFeaturesExtractor:
    def __init__(self, extractor):
        self._extractor = extractor

    def extract_from(self, collection):
        all_books = list(collection.books())
        pool = multiprocessing.Pool()

        func = ExtractorClosure(self._extractor)
        all_features = pool.map(func, all_books)

        result = {}
        for book, features in zip(all_books, all_features):
            result[book] = features
        return bc.CollectionFeatures(collection, self, result)

    def encoder_for(self, collection):
        vocabulary = set()
        features = self.extract_from(collection)

        for book in features.collection().books():
            vocabulary.update(features.by_book(book).keys())

        encoder = bc.FeaturesEncoder(vocabulary)
        return bc.CollectionFeaturesEncoder(encoder)


CollectionFeaturesExtractor = SerialCollectionFeaturesExtractor


class CollectionHierarchialFeaturesExtractor:
    def __init__(self, extractor):
        self._extractor = extractor

    def extract_from(self, collection):
        return bc.CollectionHierarchialFeatures.from_book_collection(collection, self._extractor)

    def encoder_for(self, collection):
        raise NotImplementedError()
