import random


class RandomContext:
    def __init__(self, seed):
        self._seed = seed

    def __enter__(self):
        self._oldstate = random.getstate()
        random.seed(self._seed)

    def __exit__(self, exc_type, exc_value, traceback):
        random.setstate(self._oldstate)


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
