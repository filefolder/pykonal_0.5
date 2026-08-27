"""
A module providing heap-sort functionality.

This module provides a single class (:class:`pykonal.heapq.Heap`) which
implements a binary min-heap structure whose elements are indices
(each having three components) sorted by an auxiliary value. Indices can
be pushed onto and popped from the Heap and the auxiliary sort values
can be updated on the fly, although care must be taken to resort the
Heap so as to maintain the heap invariant when updating the underlying
sort values.
"""

# Imports.
import numpy as np

from . import constants


# C Imports.
cimport numpy as np

from . cimport constants


cdef struct Index3D:
    Py_ssize_t i1, i2, i3


cdef class Heap(object):
    """
    Binary Min-Heap structure for storing indices sorted by auxiliary
    values.
    """
    def __init__(self, values):
        self.cy_values     = values
        self.cy_heap_index = np.full(values.shape, fill_value=-1)


    @property
    def heap_index(self):
        """
        [*Read only*, numpy.ndarray(shape=(N0,N1,N2), dtype=numpy.int)]
        Array of indices indicating the heap position of each node. 
        Index -1 indicates that a node is not on the heap.
        """
        return (np.asarray(self.cy_heap_index, dtype=constants.DTYPE_INT))


    @property
    def keys(self):
        """
        [*Read only*, list] Sorted list of node
        indices currently on the heap.
        """
        cdef Index3D idx
        output = []
        for i in range(self.cy_keys.size()):
            idx = self.cy_keys[i]
            output.append((idx.i1, idx.i2, idx.i3))
        return (output)


    @property
    def size(self):
        """
        [*Read only*, int] Number of node indices on the heap.
        """
        return (self.cy_keys.size())


    @property
    def values(self):
        """
        [*Read/Write*, numpy.ndarray(shape=(N0,N1,N2), dtype=numpy.float)]
        Auxiliary values to sort by.
        """
        return (np.asarray(self.cy_values))

    @values.setter
    def values(self, values):
        self.cy_values = values


    cpdef constants.BOOL_t rebind(Heap self, values):
        """
        Point the heap at *values* and restore the heap invariant against
        it. Use this when the array the heap sorts by may have been
        replaced (rather than written in place) since the heap was built.
        """
        self.cy_values = values
        return self.reheapify()


    cpdef constants.BOOL_t reheapify(Heap self):
        """
        Rebuild the heap invariant in O(n) from the current sort values
        (Floyd's algorithm).
        """
        cdef Py_ssize_t j
        if self.cy_keys.size() < 2:
            return True
        for j in range(<Py_ssize_t>(self.cy_keys.size() // 2) - 1, -1, -1):
            self.sift_up(j)
        return True


    cpdef (Py_ssize_t, Py_ssize_t, Py_ssize_t) pop(Heap self):
        """
        pop(self)

        Pop the index of the item with the smallest sort value from the
        heap, maintaining the heap invariant.

        :return: Index of node on the heap with smallest sort value.
        :rtype: tuple(int, int, int)
        :raises IndexError: If the heap is empty.
        """
        cdef Index3D last, idx_return

        if self.cy_keys.size() == 0:
            # std::vector::back()/pop_back() on an empty vector is
            # undefined behavior (no bounds check in C++). The solver's
            # FMM loop already guards calls to pop() with a `while
            # trial.cy_keys.size() > 0` check, but Heap is a general
            # purpose class -- raise a clear, catchable error here rather
            # than relying on every caller to externally guard this.
            raise IndexError("pop() called on an empty Heap.")

        last = self.cy_keys.back()
        self.cy_keys.pop_back()
        self.cy_heap_index[last.i1, last.i2, last.i3] = -1
        if self.cy_keys.size() > 0:
            idx_return = self.cy_keys[0]
            self.cy_heap_index[idx_return.i1, idx_return.i2, idx_return.i3] = -1
            self.cy_keys[0] = last
            self.cy_heap_index[last.i1, last.i2, last.i3] = 0
            self.sift_up(0)
            return ((idx_return.i1, idx_return.i2, idx_return.i3))
        return ((last.i1, last.i2, last.i3))

    cpdef constants.BOOL_t push(Heap self, Py_ssize_t i1, Py_ssize_t i2, Py_ssize_t i3):
        """
        push(self, i1, i2, i3)

        Push index onto the heap, maintaining the heap invariant.

        :param i1: First component of index.
        :type i1: int
        :param i2: Second component of index.
        :type i2: int
        :param i3: Third component of index.
        :type i3: int

        :return: True upon successful completion.
        :rtype: bool

        .. todo:: Check that index is in range.
        """
        cdef Index3D idx
        idx.i1, idx.i2, idx.i3 = i1, i2, i3
        self.cy_keys.push_back(idx)
        self.cy_heap_index[idx.i1, idx.i2, idx.i3] = self.cy_keys.size()-1
        self.sift_down(0, self.cy_keys.size()-1)
        return (True)


    cpdef constants.BOOL_t sift_down(Heap self, Py_ssize_t j_start, Py_ssize_t j):
        """
        sift_down(self, j_start, j)

        Sift the heap element at *j* down the heap (towards the root),
        without going past *j_start*, until finding a place that it
        fits.

        :param j_start: Heap index creating a barrier beyond which heap
                        element *j* cannot be sifted.
        :type j_start: int
        :param j: Heap index of element to sift towards root.
        :type j: int
        :return: Returns True upon successful execution.
        :rtype: bool
        """
        cdef Py_ssize_t j_parent
        cdef Index3D    idx_new, idx_parent

        idx_new = self.cy_keys[j]
        # Follow the path to the root, moving parents down until finding a place
        # new item fits.
        while j > j_start:
            j_parent = (j - 1) >> 1
            idx_parent = self.cy_keys[j_parent]
            if self.cy_values[idx_new.i1, idx_new.i2, idx_new.i3] < self.cy_values[idx_parent.i1, idx_parent.i2, idx_parent.i3]:
                self.cy_keys[j] = idx_parent
                self.cy_heap_index[idx_parent.i1, idx_parent.i2, idx_parent.i3] = j
                j = j_parent
                continue
            break
        self.cy_keys[j] = idx_new
        self.cy_heap_index[idx_new.i1, idx_new.i2, idx_new.i3] = j
        return (True)


    cpdef constants.BOOL_t sift_up(Heap self, Py_ssize_t j_start):
        """
        sift_up(self, j_start)

        Sift the heap element at *j_start* up the heap (away from the
        root), until finding a place that it fits.

        :param j_start: Heap index of element to sift away from root.
        :type j_start: int
        :return: Returns True upon successful execution.
        :rtype: bool
        """
        cdef Py_ssize_t j, j_child, j_end, j_right
        cdef Index3D idx_child, idx_right, idx_new
        cdef bint    has_right

        j_end = self.cy_keys.size()
        j = j_start
        idx_new = self.cy_keys[j_start]
        # Bubble up the smaller child until hitting a leaf.
        j_child = 2 * j_start + 1 # leftmost child position
        while j_child < j_end:
            # Set childpos to index of smaller child.
            j_right = j_child + 1
            has_right = j_right < j_end
            idx_child = self.cy_keys[j_child]
            # Only read cy_keys[j_right] when it's actually in bounds.
            # The original code indexed cy_keys[j_right] unconditionally
            # before checking j_right < j_end; std::vector::operator[] does
            # no bounds checking, so when j_right == j_end (the common case
            # of a parent with only a left child) that was an out-of-bounds
            # read -- undefined behavior in C++, on every pop() in the
            # solver's hot loop.
            if has_right:
                idx_right = self.cy_keys[j_right]
                if not self.cy_values[idx_child.i1, idx_child.i2, idx_child.i3] < self.cy_values[idx_right.i1, idx_right.i2, idx_right.i3]:
                    j_child = j_right
            # Move the smaller child up.
            self.cy_keys[j] = self.cy_keys[j_child]
            self.cy_heap_index[self.cy_keys[j_child].i1, self.cy_keys[j_child].i2, self.cy_keys[j_child].i3] = j
            j = j_child
            j_child = 2 * j + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        self.cy_keys[j] = idx_new
        self.cy_heap_index[idx_new.i1, idx_new.i2, idx_new.i3] = j
        self.sift_down(j_start, j)
        return (True)
