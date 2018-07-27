
import numpy
import ntpath
from numpy import dot

from khmerml.utils.file_util import FileUtil
from khmerml.algorithms.base import Base
from khmerml.algorithms.neural_network.neural_network_base import NeuralNetworkBase


class NeuralNetwork(Base, NeuralNetworkBase):
  """
    Back-Propagation Neural Networks Written in Python.
  """

  def __init__(self, **kwargs):
    # Random range of initial weight
    self.random_state = kwargs['random_state']
    # Number of hidden layers
    self.hidden_layer_sizes = kwargs['hidden_layer_sizes']
    # Learning rate: speed up the learning process
    # to reach converge point
    self.learning_rate = kwargs['learning_rate']
    # Maximum numbers of learning iterations (times)
    self.max_iter = kwargs['max_iter']
    # Momentum
    self.momentum = kwargs['momentum']
    # activiation function
    self.activation = kwargs['activation']
    # Initial synaptic weight
    self.synaptic_weights = None
    # Change output
    self.change_output_layouts = None
    # Call super
    NeuralNetworkBase.__init__(self, **kwargs)

  def load_model(self):
    """
      Load train model from file
    """
    try:
      head, tail = ntpath.split(self.kwargs['train_model'])
      self.train_model = FileUtil.load_model(head+ '/neural_network_' +tail)
    except IOError as error:
      raise Exception(error)

    return self.train_model

  def save_model(self, model):
    """
      Load train model from file
    """
    try:
      head, tail = ntpath.split(self.kwargs['train_model'])
      FileUtil.save_model(head+ '/neural_network_' +tail, model)
    except IOError as error:
      print(error)

  def __calculate_activation_output(self, activation, output_neurons):
    """
      param activation: The activation function to be used. Can be
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
    """
      param activation: The activation function to be used. Can be
      (logistic or tanh)
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
    """
      The neural network thinks.
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
    """
      Back propagate of learning process
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
      # error = delta_error_layers[indice]
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
    """
      Learning process of input data and model.
      Multiplayer perceptron is adopted in this algorithm.
    """
    # One iteration of learning process composes of feedforward and backpropagation
    # Start learning feedforward following by backward propagation
    output_by_layers = self.feedforward(input_vector)
    self.synaptic_weights = self.backpropagate(input_vector, output_by_layers, target_vector)


  def train(self, train_sample):
    """
    """
    y_train = train_sample[:, -1]
    X_train = numpy.delete(train_sample, -1, 1)
    # Get label from dataset
    # Convert label to array of vector
    #label_vector = dataset_matrix[:, -1]
    y_target = self.label_to_matrix(y_train)
    # Number of layers
    n_features = len(X_train[0])
    # Number of output labels
    n_labels = len(y_target[0])
    # Initialize random weight (UI,  UH,  UO), iteration = 0
    # Create matrix of n_inputs_neuron as row and n_neurons as column
    self.synaptic_weights, self.change_output_layouts = self.make_weight_matrix(n_features=n_features, \
    n_labels=n_labels, hidden_layer_sizes=self.hidden_layer_sizes)

    for idx in range(self.max_iter):
      for index, input_vector in enumerate(X_train):
          self.start_learning(input_vector, y_target[index])
      # learning rate decay
      rate_decay = 0.0001
      self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * rate_decay)))

    # Save model
    self.save_model(self.synaptic_weights)
    # model
    return self.synaptic_weights


  def predict(self, model, test_sample):
    """
      Make prediction
    """
    # Remove class from X_test
    X_test = numpy.delete(test_sample, -1, 1)
    predictions = {}
    for index, vector in enumerate(X_test):
      # Start predicting one input per time
      result_dict = self._predict(model, vector)
      best_class = numpy.argmax(result_dict[len(result_dict) -1])
      # Add result to dict (key, value)
      # The expected output is at the end of list
      predictions[index] = best_class
      #predictions[index] = result_dict[len(result_dict) -1]
    return predictions


  def _predict(self, model, input_vector):
    """ The neural network thinks.
    """
    # Output layer
    output_neurons = None
    output_by_layers = {}
    # Iterate through layers
    for layer_index, synaptic_weight in enumerate(model):
      #if layer_index == 0:
      if output_neurons is None:
        product = dot(synaptic_weight.T, input_vector)
      else:
        product = dot(synaptic_weight.T, output_neurons)
      # Calculate output based on given activation function
      output_neurons = self.__calculate_activation_output(self.activation, product)
      # Update nodes of the layer
      output_by_layers[layer_index] = output_neurons
    return output_by_layers
