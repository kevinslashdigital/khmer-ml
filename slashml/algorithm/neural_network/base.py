""" Neural Network base class

"""

from abc import ABC, abstractmethod

import numpy
from numpy import exp


class Base(object):
    """ Neural Network Base
    """

    def __init__(self, random_state=0, **kwargs):
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
        # https://stackoverflow.com/questions/13092807/how-do-i-split-an-ndarray-based-on-array-of-indexes
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


    def make_weight_matrix(self, n_features, n_labels, hidden_layer_sizes):
        """ Make random weight in form of matrix for each layer
        """

        layers = []
        change_output_layouts = []
        n_inputs = 0

        for _, n_output_nodes in enumerate(hidden_layer_sizes):

            #if index  is not layers:
            if n_inputs == 0:
                n_inputs = n_features

            # Create matrix of
            # n_inputs_neuron as row and n_neurons as column
            # Random value in interal [0, 1)

            # create randomized weights
            # use scheme from Efficient Backprop by LeCun 1998 to initialize weights for hidden layer
            input_range = 1.0 / n_inputs ** (1/2)
            synaptic_weight = numpy.random.normal(loc=0, scale=input_range, size=(n_inputs, n_output_nodes))

            # create arrays of 0 for changes
            # this is essentially an array of temporary values that gets updated at each iteration
            # based on how much the weights need to change in the following iteration
            change_output = numpy.zeros((n_inputs, n_output_nodes))
            change_output_layouts.append(change_output)

            # n_output_nodes is input for next layer
            n_inputs = n_output_nodes

            # Add layer to list
            synaptic_weight = numpy.array(synaptic_weight)
            layers.append(synaptic_weight)

        # Add layer to list
        # Output layer
        synaptic_weight = numpy.random.uniform(size=(n_inputs, n_labels)) / numpy.sqrt(n_inputs)
        layers.append(synaptic_weight)


        # create arrays of 0 for changes
        # this is essentially an array of temporary values that gets updated at each iteration
        # based on how much the weights need to change in the following iteration
        change_output = numpy.zeros((n_inputs, n_labels))
        change_output_layouts.append(change_output)


        return layers, change_output_layouts


    def activation_relu(self, X):
        """Compute the rectified linear unit function inplace.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        Returns
        -------
        X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
            The transformed data.
        """

        numpy.clip(X, 0, numpy.finfo(X.dtype).max, out=X)

        return X


    def derivative_relu(self, X, delta):
        """Apply the derivative of the relu function.
        It exploits the fact that the derivative is a simple function of the output
        value from rectified linear units activation function.
        Parameters
        ----------
        Z : {array-like, sparse matrix}, shape (n_samples, n_features)
            The data which was output from the rectified linear units activation
            function during the forward pass.
        delta : {array-like}, shape (n_samples, n_features)
            The backpropagated error signal to be modified inplace.
        """
        delta[X == 0] = 0

        return delta

    def activation_tanh(self, X):
        """ tanh is a little nicer than the standard 1/(1+e^-x)
        """

        #return 1.7159 * numpy.tanh((2/3)*X)
        return numpy.tanh(X, out=X)

    def derivative_tanh(self, X, delta):
        """Apply the derivative of the hyperbolic tanh function.
        It exploits the fact that the derivative is a simple function of the output
        value from hyperbolic tangent.
        Parameters
        ----------
        Z : {array-like, sparse matrix}, shape (n_samples, n_features)
            The data which was output from the hyperbolic tangent activation
            function during the forward pass.
        delta : {array-like}, shape (n_samples, n_features)
            The backpropagated error signal to be modified inplace.
        """

        #delta *= (1 - X ** 2)
        #delta *= 1.1493*(1 - ((2/3)*X) ** 2)
        delta *= (1 - X ** 2)
        return delta


    def activation_sigmoid_exp(self, X):
        """ The Sigmoid function, which describes an S shaped curve.
        We pass the weighted sum of the inputs through this function to
        normalise them between 0 and 1.
        """

        return 1 / (1 + exp(-X))

    # derivative of our sigmoid function, in terms of the output (i.e. y)
    def derivative_sigmoid(self, X, delta):
        """Apply the derivative of the logistic sigmoid function.
        It exploits the fact that the derivative is a simple function of the output
        value from logistic function.
        Parameters
        ----------
        Z : {array-like, sparse matrix}, shape (n_samples, n_features)
            The data which was output from the logistic activation function during
            the forward pass.
        delta : {array-like}, shape (n_samples, n_features)
            The backpropagated error signal to be modified inplace.
        """
        delta *= X
        delta *= (1 - X)

        return delta


    @abstractmethod
    def load_dataset(self):
        """ load dataset extraction
        """
        pass


    @abstractmethod
    def load_model(self):
        """ load dataset extraction
        """
        pass


    @abstractmethod
    def train(self, training_input, targets):
        """ load dataset extraction
        """
        pass


    @abstractmethod
    def predict(self, test_sample):
        """ load dataset extraction
        """
        pass
