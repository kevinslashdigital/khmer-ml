""" Neural Network base class

"""

from abc import ABC, abstractmethod

import numpy



class Base(ABC):
    """ Neural Network Base
    """

    def __init__(self, **kwargs):
        # makes the random numbers predictable
        # With the seed reset (every time), the same set of numbers will appear every time.
        # If the random seed is not reset, different numbers appear with every invocation:
        #random.seed(random_state)
        # Set float precision
        numpy.set_printoptions(precision=8)

        # Global configuration
        self.kwargs = kwargs


    def train_test_split(self, dataset, n_test_by_class=2):
        """ Split data set for training and testing
        """

        # Sort dataset following descendant label
        # dataset[:, -1].argsort # Sort the last field (column)
        # dataset = dataset[dataset[:,1].argsort(kind='mergesort')]
        sorted_dataset = dataset[dataset[:, -1].argsort()]

        # Get unique labels
        y_all = numpy.unique(sorted_dataset[:, -1])

        # Random row indices
        random_row_indices = None

        for _, label_val in enumerate(y_all):
            # Find row index where label value equals to given label_value
            search_indices = numpy.where(sorted_dataset[:, -1] == label_val)

            # Selected number of test sample equals to number of test sample
            random_idx = numpy.random.randint(search_indices[0][0], search_indices[0][-1], size=n_test_by_class)
            #mask = numpy.random.choice([False, True], lenght, p=[probability, 1-probability])
            # Merge array random row into one array
            if random_row_indices is None:
                random_row_indices = random_idx
            else:
                random_row_indices = numpy.hstack((random_row_indices, random_idx))

        # Split
        # Create mask boolean array
        # https://stackoverflow.com/questions/13092807/how-do-i-split-an-ndarray-based-on-array-of-indexes
        mask = numpy.ones(len(sorted_dataset), dtype=bool)
        mask[random_row_indices, ] = False
        # instead of a[b] you could also do a[~mask]
        X_test, X_train = sorted_dataset[random_row_indices], sorted_dataset[mask]

        # Get train and test label
        y_train = X_train[:, -1]
        y_test = X_test[:, -1]

        # Delete last column (label) from array
        X_train = numpy.delete(X_train, -1, 1)
        X_test = numpy.delete(X_test, -1, 1)

        # Return list
        return X_train, X_test, y_train, y_test


    def train_test_extract(self, dataset, n_test_by_class=2):
        """ Split data set for training and testing
        """

        # Sort dataset following descendant label
        # dataset[:, -1].argsort # Sort the last field (column)
        # dataset = dataset[dataset[:,1].argsort(kind='mergesort')]
        sorted_dataset = dataset[dataset[:, -1].argsort()]

        # Get unique labels
        y_all = numpy.unique(sorted_dataset[:, -1])

        # Random row indices
        random_row_indices = None

        for _, label_val in enumerate(y_all):
            # Find row index where label value equals to given label_value
            search_indices = numpy.where(sorted_dataset[:, -1] == label_val)

            # Selected number of test sample equals to number of test sample
            random_idx = numpy.random.randint(search_indices[0][0], search_indices[0][-1], size=n_test_by_class)
            #mask = numpy.random.choice([False, True], lenght, p=[probability, 1-probability])
            # Merge array random row into one array
            if random_row_indices is None:
                random_row_indices = random_idx
            else:
                random_row_indices = numpy.hstack((random_row_indices, random_idx))

        # Split
        # Create mask boolean array
        mask = numpy.ones(len(sorted_dataset), dtype=bool)
        mask[random_row_indices, ] = False
        # instead of a[b] you could also do a[~mask]
        #X_test, X_train = sorted_dataset[random_row_indices], sorted_dataset[mask]
        X_train = sorted_dataset
        X_test = sorted_dataset[random_row_indices].copy()

        # Get train and test label
        #y_train = X_train[:, -1]
        y_train = sorted_dataset[:, -1]
        y_test = X_test[:, -1]

        # Delete last column (label) from array
        X_train = numpy.delete(X_train, -1, 1)
        X_test = numpy.delete(X_test, -1, 1)

        # Return list
        return X_train, X_test, y_train, y_test


    def label_to_matrix(self, label_vector):
        """ Transform labels to row-based vector
        """

        #length = len(set(label_vector))#
        length = len(label_vector)
        n_label = len(set(label_vector))
        unique_labels = numpy.unique(label_vector)

        # Create matrix zeros m row x n columns
        label_matrix = numpy.zeros((length, n_label))

        # Replace value by 1 at position (i, label_vector(j))
        for i, value in enumerate(label_vector):
            #value = int(round(value))
            index = numpy.where(unique_labels == value)[0]
            label_matrix[i][index] = numpy.float(1.0)

        return label_matrix
