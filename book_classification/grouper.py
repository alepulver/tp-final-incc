import book_classification as bc
import numpy


class Grouper:
    def parts_from(self, sequence):
        raise NotImplementedError()

    def parts_size(self):
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
                yield list(group)
                group = []
            group.append(token)
        if len(group) > 0:
            yield list(group)

    def parts_size(self):
        return self._parts_size


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


class NumericFixedGrouper(Grouper):
    def __init__(self, parts_size):
        self._parts_size = parts_size

    def parts_from(self, sequence):
        vector = numpy.array(list(sequence))
        assert(len(vector) >= self._parts_size)
        for i in range(len(vector) // self._parts_size):
            #window = numpy.array(vector[i:i+self._parts_size])
            start = i * self._parts_size
            end = i + self._parts_size
            window = vector[start:end]
            yield window

    def parts_size(self):
        return self._parts_size


class NumericSlidingGrouper(Grouper):
    def __init__(self, parts_size):
        self._parts_size = parts_size

    def parts_from(self, sequence):
        vector = numpy.array(list(sequence))
        assert(len(vector) >= self._parts_size)
        for i in range(len(vector) - self._parts_size):
            #window = numpy.array(vector[i:i+self._parts_size])
            window = vector[i:i+self._parts_size]
            yield window

    def parts_size(self):
        return self._parts_size
