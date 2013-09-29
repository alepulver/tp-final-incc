import random
import hashlib
#from pyhashxx import hashxx
import book_classification as bc


class RandomContext:
    def __init__(self, seed):
        self._seed = seed

    def __enter__(self):
        self._oldstate = random.getstate()
        random.seed(self._seed)

    def __exit__(self, exc_type, exc_value, traceback):
        random.setstate(self._oldstate)


class Grouper:
    def parts_from(self, sequence):
        raise NotImplementedError()


class DummyGrouper(Grouper):
    def parts_from(self, sequence):
        return iter(sequence)


class FixedGrouper(Grouper):
    def __init__(self, parts_size):
        self._parts_size = parts_size

    def parts_from(self, sequence):
        group = []
        for token in sequence:
            if len(group) >= self._parts_size:
                yield group
                group = []
            group.append(token)
        if len(group) > 0:
            yield group

    def parts_size(self):
        return self._parts_size

    def uuid(self):
        text = "%s(%s)" % (self.__class__.__name__, self._parts_size)
        return bc.digest(text)


class SlidingGrouper(Grouper):
    def __init__(self, parts_size):
        self._parts_size = parts_size

    def parts_from(self, sequence):
        window = []
        for element in sequence:
            window.append(element)
            if len(window) >= self._parts_size:
                # need to copy because it is changed later
                yield list(window)
                window.pop(0)

    def parts_size(self):
        return self._parts_size

    def uuid(self):
        text = "%s(%s)" % (self.__class__.__name__, self._parts_size)
        return bc.digest(text)


class NumericIndexer:
    def __init__(self, objs):
        # add to list without duplicates, but avoid traversing the list
        present = set()
        self._objects = []
        for element in objs:
            if element not in present:
                present.add(element)
                self._objects.append(element)

        # XXX: same indices if same vocabulary
        self._objects.sort()
        self._indices = dict(zip(self._objects, range(len(self._objects))))

    def __len__(self):
        return len(self._objects)

    def can_encode(self, obj):
        return obj in self._indices

    def can_decode(self, index):
        return index < len(self)

    def encode(self, obj):
        return self._indices[obj]

    def decode(self, index):
        return self._objects[index]

    def vocabulary(self):
        return self._objects

    def uuid(self):
        text = "%s(%s)" % (self.__class__.__name__, bc.digest(repr(self._objects)))
        return bc.digest(text)


def digest(text):
    result = hashlib.md5()
    result.update(text.encode())
    return result.hexdigest()

#def digest2(text):
#    return hashxx(text.encode(), seed=1)
