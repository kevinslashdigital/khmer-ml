""" Back-Propagation Neural Networks Written in Python.
"""

import numpy
from numpy import dot
from slashml.algorithm.neural_network.base import Base

""" import sys
sys.path.append('/Users/lion/Documents/py-workspare/slash-ml/slash-ml/slashml')

from algorithm.neural_network.base import Base
from utils.file_util import FileUtil """


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
            weight_adjustment = self.learning_rate * numpy.dot(previous_output.T, delta)

            """ weight_adjustment = self.learning_rate * numpy.dot(\
                                previous_layer_output.reshape(__len_ouput, 1), \
                                delta_error_layers[indice].reshape(1, __len_delta_error)) """

            # Adjust the weights.
            # add momentum
            #self.synaptic_weights[indice] += self.momentum * weight_adjustment
            self.synaptic_weights[indice] -= weight_adjustment + \
                                                self.change_output_layouts[indice] * self.momentum
            self.change_output_layouts[indice] = weight_adjustment

            """ weight_adjustment = numpy.dot(\
                                previous_layer_output.reshape(__len_ouput, 1), \
                                delta_error_layers[indice].reshape(1, __len_delta_error))

            # Adjust the weights.
            # add momentum
            weight_adjustment += self.momentum*self.synaptic_weights[indice]
            self.synaptic_weights[indice] += -self.learning_rate*weight_adjustment """


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
            #print('Weight {0} \n'.format(self.synaptic_weights))
            #print('Iteration no. {0} \n'.format(idx))
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

        """ import copy
        test_sample = copy.deepcopy(test_dataset) """

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
            #target_label = y_test[index]
            #max_weight = numpy.argmax(output)
            if y_test[index] == numpy.argmax(output):
                predict_score += 1
            ind_max_prob = numpy.argmax(output)

            #print('Max probab index: %d, weight: %f \n', % ind_max_prob)
            #print('Result: target label:{0} | output label: {1} | weight: {2} \n'.format(y_test[index], ind_max_prob, output[ind_max_prob]))

        # Prediction accuracy
        accuracy = round((predict_score / len(y_test)) * 100.0, 2)

        return accuracy


def demo():
    """ Sample of neural network usage
    """

    test_counter = 0
    accuracy_list = []

    from slashml.utils.file_util import FileUtil
    # Load dataset from file
    #path_to_cvs_dataset = '/Users/lion/Documents/py-workspare/slash-ml/data/dataset/matrix/iris.data.full.csv'
    path_to_cvs_dataset = '/Users/lion/Documents/py-workspare/slash-ml/data/dataset/matrix/iris.data.csv'
    #path_to_cvs_dataset = '/Users/lion/Documents/py-workspare/slash-ml/data/dataset/matrix/doc_freq_20.csv'
    dataset_matrix = FileUtil.load_csv_np(path_to_cvs_dataset)

    while test_counter < 10:
        # Array of hidden layers
        # hidden_layer_sizes = (250, 100)
        hidden_layer_sizes = (250, 100)
        learning_rate = 0.0003
        #learning_rate = 0.012 #tanh
        #learning_rate = 0.45 #logistics
        #learning_rate = 1.0
        momentum = 0.5
        #activation = 'tanh'
        activation = 'relu'
        #activation = 'logistic'

        # create a network with two input, two hidden, and one output nodes
        neural_network = NeuralNetwork(hidden_layer_sizes=hidden_layer_sizes,\
        learning_rate=learning_rate, momentum=momentum, random_state=0, max_iter=200, activation=activation)

        #X_train, X_test, y_train, y_test = neural_network.train_test_split(dataset_matrix, n_test_by_class=3)
        X_train, X_test, y_train, y_test = neural_network.train_test_extract(dataset_matrix, n_test_by_class=3)

        # Get label from dataset
        # Convert label to array of vector
        #label_vector = dataset_matrix[:, -1]
        y_label_matrix = neural_network.label_to_matrix(y_train)

        # Remove label from dataset
        #matrix_dataset = numpy.delete(dataset_matrix, numpy.s_[-1:], axis=1)

        # Start training process
        neural_network.train(X_train, y_label_matrix)

        # Perform prediction process
        predictions, _ = neural_network.predict(X_test)

        # Prediction accuracy
        accuracy = neural_network.accuracy(y_test, predictions)

        print('----------------------------\n')
        print('Accuracy is {0}.'.format(accuracy))

        # Increment counter
        test_counter = test_counter + 1

        #end = clock()
        #elapsed = end - start

        #print('Computing time %s %s' %(test_counter, elapsed))

        # Keep tracking the accuracy per operation
        accuracy_list.append(accuracy)

    mode = max(set(accuracy_list), key=accuracy_list.count)
    print("Accuracy list: ", accuracy_list)
    print("Average Accuracy", round(sum(accuracy_list)/test_counter, 2))
    print("Mode of accuracy is : ", mode)


if __name__ == '__main__':
    demo()
