"""Numpy-related utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def to_categorical(y, num_classes=None, zero_indexed=True):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        zero_indexed: whether y has been processed so that
            used values are [0,...,np.max(y)]

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not zero_indexed:
        vals_idx = {val: i for i, val in enumerate(set(y))}
        num_classes = len(list(vals_idx.keys()))
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    if zero_indexed:
        categorical[np.arange(n), y] = 1
    else:
        categorical[np.arange(n), vals_idx[y]] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def normalize(x, axis=-1, order=2):
    """Normalizes a Numpy array.

    # Arguments
        x: Numpy array to normalize.
        axis: axis along which to normalize.
        order: Normalization order (e.g. 2 for L2 norm).

    # Returns
        A normalized copy of the array.
    """
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)
