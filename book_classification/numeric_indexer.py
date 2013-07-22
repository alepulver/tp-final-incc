class NumericIndexer:
    def __init__(self, objs):
        self._objects = list(objs)
        self._indices = dict(zip(self._objects, range(len(self._objects))))

    def __len__(self):
        return len(self._indices)

    def encode_one(self, obj):
        return self._indices[obj]

    def decode_one(self, index):
        return self._objects[index]

    def encode_many(self, objs):
        return map(self.encode_one, objs)

    def decode_many(self, indices):
        return map(self.decode_one, indices)

    @classmethod
    def from_objects(cls, objs):
        return cls(set(objs))