""" Back-Propagation Neural Networks Written in Python.
"""

import numpy
from numpy import dot
from slashml.algorithm.neural_network.base import Base


class NeuralNetwork(Base):
    """ Neural Network
    """

    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.5, max_iter=200, \
    momentum=0.2, random_state=1, activation='logistic', **kwargs):

        # Call super
        super(NeuralNetwork, self).__init__(random_state, **kwargs)

        # Random range of initial weight
        self.random_state = random_state

        # Number of hidden layers
        self.hidden_layer_sizes = hidden_layer_sizes

        # Learning rate: speed up the learning process
        # to reach converge point
        self.learning_rate = learning_rate

        # Maximum numbers of learning iterations (times)
        self.max_iter = max_iter

        # Momentum
        self.momentum = momentum

        # Initial synaptic weight
        self.synaptic_weights = None

        # Change output
        self.change_output_layouts = None

        # activiation function
        self.activation = activation


    def load_dataset(self):
        """ load dataset extraction
        """
        pass


    def load_model(self):
        """ load dataset extraction
        """
        pass


    def __calculate_activation_output(self, activation, output_neurons):
        """ :param activation: The activation function to be used. Can be
        logistic or tanh
        """

        if activation == 'logistic':
            return self.activation_sigmoid_exp(output_neurons)
        elif activation == 'tanh':
            return self.activation_tanh(output_neurons)
        elif activation == 'relu':
            return self.activation_relu(output_neurons)
        else:
            return None


    def __calculate_error(self, activation, output_neurons, delta_error):
        """ :param activation: The activation function to be used. Can be
        # logistic or tanh
        """

        if activation == 'logistic':
            return self.derivative_sigmoid(output_neurons, delta_error)
        elif activation == 'tanh':
            return self.derivative_tanh(output_neurons, delta_error)
        elif activation == 'relu':
            return self.derivative_relu(output_neurons, delta_error)
        else:
            return None


    def feedforward(self, input_vector):
        """ The neural network thinks.
        """
        # Output layer
        output_neurons = None
        output_by_layers = {}

        # Iterate through layers
        for layer_index, synaptic_weight in enumerate(self.synaptic_weights):

            #if layer_index == 0:
            if output_neurons is None:
                product = dot(synaptic_weight.T, input_vector)
            else:
                #product = dot(output_neurons.T, synaptic_weight)
                product = dot(synaptic_weight.T, output_neurons)

            # Calculate output based on given activation function
            output_neurons = self.__calculate_activation_output(self.activation, product)

            # Update nodes of the layer
            output_by_layers[layer_index] = output_neurons

        return output_by_layers


    def backpropagate(self, input_vector, output_by_layers, target_vector):
        """ Back propagate of learning process
        """

        # Length of the list
        layers_len = len(output_by_layers)

        delta_error = None
        error = None
        delta_error_layers = {}

        # Loop backward from length of layers
        for index in reversed(range(layers_len)):

            # Calculate the error from output layer
            # (The difference between the desired output and the predicted output).
            if delta_error is None:
                output = output_by_layers[index]
                delta_error = -(target_vector - output)
                #output_error = target_vector - output
            else:

                # Calculate the error for layer 1 (By looking at the weights in layer 1,
                # we can determine by how much layer 1 contributed to the error in layer 2).
                synaptic_weight = self.synaptic_weights[index+1]
                output = output_by_layers[index]
                delta_error = numpy.dot(synaptic_weight, error)

            # Determine error between expected output and that produces from learning process
            error = self.__calculate_error(self.activation, output, delta_error)

            # Keep tracking of delta error of each layer
            delta_error_layers[index] = error

        # Determine how much weight to be adjusted of each synaptic weight
        for indice in reversed(range(layers_len)):

            if indice - 1 >= 0:
                previous_layer_output = output_by_layers[indice-1]
            else:
                previous_layer_output = input_vector

            # Calculate how much to adjust the weights by
            #error = delta_error_layers[indice]

            previous_output = numpy.atleast_2d(previous_layer_output)
            delta = numpy.atleast_2d(delta_error_layers[indice])

            # update the weights connecting hidden to output, change == partial derivative
            change = numpy.dot(previous_output.T, delta)

            # weight adjustment
            weight_adjustment = self.learning_rate * change

            # Adjust the weights, and Momentum addition
            self.synaptic_weights[indice] -= weight_adjustment + \
                                                self.change_output_layouts[indice] * self.momentum
            self.change_output_layouts[indice] = weight_adjustment

        return self.synaptic_weights


    def start_learning(self, input_vector, target_vector):
        """ Learning process of input data and model.
        Multiplayer perceptron is adopted in this algorithm.
        """
        # One iteration of learning process composes of feedforward and backpropagation
        # Start learning feedforward following by backward propagation
        output_by_layers = self.feedforward(input_vector)
        self.synaptic_weights = self.backpropagate(input_vector, output_by_layers, target_vector)


    def train(self, training_input, targets):
        """"
        """

        # Number of layers
        n_features = len(training_input[0])

        # Number of output labels
        n_labels = len(targets[0])

        # Initialize random weight (UI,  UH,  UO), iteration = 0
        # Create matrix of n_inputs_neuron as row and n_neurons as column
        self.synaptic_weights, self.change_output_layouts = self.make_weight_matrix(n_features=n_features, \
        n_labels=n_labels, hidden_layer_sizes=self.hidden_layer_sizes)

        for idx in range(self.max_iter):
            for index, input_vector in enumerate(training_input):
                self.start_learning(input_vector, targets[index])

            # learning rate decay
            rate_decay = 0.0001
            self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * rate_decay)))


    def predict(self, test_sample):
        """ Make prediction
        """

        predictions = {}
        prediction_details = {}

        for index, vector in enumerate(test_sample):
            # Start predicting one input per time
            result_dict = self.feedforward(vector)

            # Add result to dict (key, value)
            # The expected output is at the end of list
            predictions[index] = result_dict[len(result_dict) -1]

        return predictions, prediction_details


    def accuracy(self, y_test, predictions):
        """ Get accuracy
        """

        # Prediction score
        predict_score = 0

        for index, output in predictions.items():
            if y_test[index] == numpy.argmax(output):
                predict_score += 1

        # Prediction accuracy
        accuracy = round((predict_score / len(y_test)) * 100.0, 2)

        return accuracy
