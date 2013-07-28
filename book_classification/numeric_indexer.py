class NumericIndexerBuilder:
    def __init__(self):
        self._objects = []
        self._indices = {}

    def can_encode(self, obj):
        return True

    def encode(self, obj):
        if obj not in self._indices:
            last = len(self._objects)
            self._objects.append(obj)
            self._indices[obj] = last
        return self._indices[obj]

    def build(self):
        # TODO: avoid reconstructing the dictionary
        return NumericIndexer(self._objects)

class NumericIndexer:
    def __init__(self, objs):
        self._objects = list(objs)
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

    @classmethod
    def from_objects(cls, objs):
        return cls(set(objs))