import sys, os
syspath = '/Users/lion/Documents/py-workspare/openml/khmer-ml'
sys.path.append(syspath)

from khmerml.algorithms.neural_network.neural_network_base import NeuralNetworkBase

import numpy as np
from numpy import exp


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])

    print('funcarglist', metafunc.function.__name__)

    metafunc.parametrize(argnames, [[funcargs[name] for name in argnames]
            for funcargs in funcarglist])


class TestNeuralNetworkBase:
  """
    NeuralNetwork Base class is use for calculation in
    Naive Bayes Algorithm
  """

  # def __init__(self, **kwargs):
  #   """
  #     makes the random numbers predictable
  #     With the seed reset (every time), the same set of numbers will appear every time.
  #     If the random seed is not reset, different numbers appear with every invocation:
  #     random.seed(random_state)
  #     Set float precision
  #   """
  #   np.set_printoptions(precision=8)
  #   self.kwargs = kwargs

  CONFIG = {
    'text_dir': 'data/dataset/text',
    'dataset': 'data/matrix',
    'bag_of_words': 'data/bag_of_words',
    'train_model': 'data/model/train.model',
  }

  sample_data = np.array(([5.1, 3.5, 1.4, 0.2, 1.0],
                        [4.9, 3.0, 1.4, 0.2, 1.0],
                        [4.7, 3.2, 1.3, 0.2, 1.0],
                        [5.2, 2.7, 3.9, 1.4, 2.0],
                        [5.0, 2.0, 3.5, 1.0, 2.0],
                        [5.9, 3.0, 4.2, 1.5, 2.0]))

  # Test sample
  test_vector = np.array([[4.4, 3.0, 1.3, 0.2, 1.0]])

  # Label/Class Vector
  label_vector = np.array(([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]))

  # expected_train_model = {}
  # expected_train_model[1.0] = [0.5, [0.47432024169184295, 0.3232628398791541, 0.15407854984894262, 0.04833836858006044]]
  # expected_train_model[2.0] = [0.5, [0.394919168591224, 0.20092378752886833, 0.2909930715935335, 0.11316397228637413]]

  params = {
    'test_label_to_matrix': [dict(label_vector=label_vector), ],
    'test_make_weight_matrix': [dict(n_features=4, n_labels=2, hidden_layer_sizes=(5, )), ],
    'test_activation_relu':[dict(), ],
    'test_derivative_relu':[dict(), ],
  }


  def test_label_to_matrix(self, label_vector):
    """
      Transform labels to row-based vector
    """

    nn_base = NeuralNetworkBase(**self.CONFIG)

    # Label/Class Vector
    # label_vector = np.array(([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]))
    label_vector = nn_base.label_to_matrix(label_vector)

    # Expected output
    expected_label_vector = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0],\
     [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    assert np.array_equal(expected_label_vector, label_vector)


  def test_make_weight_matrix(self, n_features, n_labels, hidden_layer_sizes):
    """
      Make random weight in form of matrix for each layer
    """
    # Instantiate Neural Network Base
    # Call for the result from Neural Network Base
    nn_base = NeuralNetworkBase(**self.CONFIG)
    output_weight_by_layers, output_weight_change_by_layers = nn_base.make_weight_matrix(n_features, n_labels, hidden_layer_sizes)

    # Label/Class Vector
    # label_vector = np.array(([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]))
    # There are 2 layers: hidden layer and output layer
    # Hidden layer is a matrix N*M
    # n_features represents the row of the matrix
    # the value in the tuple of hidden layer size represents the column of matrix.
    expected_weight_by_layers = []
    synaptic_weight_h_layer = np.array([[ 0.42156206,  0.71434969, -0.14255264,  0.49571935,  0.26126007],\
     [-0.67511967,  0.01702431,  0.41568109, -0.62916651, -0.5234192 ],\
     [-1.06500864, -0.33149057, -0.52582948,  0.43287353, -0.03835836],\
     [ 0.00622105, -0.78096484, -0.18039495, -0.0398916 ,  0.35908918]])

    synaptic_weight_o_layer = np.array([[0.31276214, 0.35999042],
       [0.12478604, 0.08451375],
       [0.37896466, 0.41309066],
       [0.11169578, 0.26012024],
       [0.31519378, 0.21803135]])

    # Expected random weights of hidden layer and output layer
    expected_weight_by_layers.append(synaptic_weight_h_layer)
    expected_weight_by_layers.append(synaptic_weight_o_layer)

    # Expected zero weigths for each iteration
    expected_weight_change_by_layers = []
    expected_weight_change_by_layers.append(np.zeros((4, 5)))
    expected_weight_change_by_layers.append(np.zeros((5, 2)))

    # To compare list of arrays use np.allclose and all

    # The result should be True
    assert all([np.allclose(x, y) for x, y in zip(expected_weight_change_by_layers, output_weight_change_by_layers)])
    # The comparision result should be False since the weights are randomly generated every time
    assert all([np.allclose(x, y) for x, y in zip(expected_weight_by_layers, output_weight_by_layers)])


  def test_activation_relu(self):
    """
    """

    # Supposed that random generated weights are as followings:
    expected_weight_by_layers = []

    synaptic_weight_h_layer = np.array([[ 0.42156206,  0.71434969, -0.14255264,  0.49571935,  0.26126007],\
     [-0.67511967,  0.01702431,  0.41568109, -0.62916651, -0.5234192 ],\
     [-1.06500864, -0.33149057, -0.52582948,  0.43287353, -0.03835836],\
     [ 0.00622105, -0.78096484, -0.18039495, -0.0398916 ,  0.35908918]])
    synaptic_weight_o_layer = np.array([[0.31276214, 0.35999042],
       [0.12478604, 0.08451375],
       [0.37896466, 0.41309066],
       [0.11169578, 0.26012024],
       [0.31519378, 0.21803135]])

    # Expected random weights of hidden layer and output layer
    expected_weight_by_layers.append(synaptic_weight_h_layer)
    expected_weight_by_layers.append(synaptic_weight_o_layer)

    # Input vector
    input_vector = np.array([5.1, 3.5, 1.4, 0.2])

    # Instantiate Neural Network Base
    # Call for the result from Neural Network Base
    nn_base = NeuralNetworkBase(**self.CONFIG)

    # Output layer
    output_neurons = None
    output_by_layers = []

    # Iterate through layers
    for layer_index, synaptic_weight in enumerate(expected_weight_by_layers):

      if output_neurons is None:
        product = np.dot(synaptic_weight.T, input_vector)
      else:
        product = np.dot(synaptic_weight.T, output_neurons)

      # Calculate output based on given activation function
      output_neurons = nn_base.activation_relu(product)

      # Update nodes of the layer
      output_by_layers.append(output_neurons)

    # Expected feedforwards learning output
    expected_output_by_ffl = []
    expected_output_by_ffl.append(np.array([0., 3.08248874, 0., 0.92413052, 0.]))
    expected_output_by_ffl.append(np.array([0.48787304, 0.50089774]))

    print(output_by_layers)
    print(expected_output_by_ffl)
    # The result should be True
    assert all([np.allclose(x, y) for x, y in zip(expected_output_by_ffl, output_by_layers)])


  def test_derivative_relu(self):
    """ Derivative relu
    """

    # Input vector
    input_vector = np.array([4.4, 3.0, 1.3, 0.2])

    # Expected feedforwards learning output
    expected_output_by_ffl = []
    expected_output_by_ffl.append(np.array([0., 3.08248874, 0., 0.92413052, 0.]))
    expected_output_by_ffl.append(np.array([0.48787304, 0.50089774]))

    # Expected output
    expected_target_vector = np.array([1.0, 0.0])

    # Supposed that random generated weights are as followings:
    expected_weight_by_layers = []

    synaptic_weight_h_layer = np.array([[ 0.42156206,  0.71434969, -0.14255264,  0.49571935,  0.26126007],\
     [-0.67511967,  0.01702431,  0.41568109, -0.62916651, -0.5234192 ],\
     [-1.06500864, -0.33149057, -0.52582948,  0.43287353, -0.03835836],\
     [ 0.00622105, -0.78096484, -0.18039495, -0.0398916 ,  0.35908918]])
    synaptic_weight_o_layer = np.array([[0.31276214, 0.35999042],
       [0.12478604, 0.08451375],
       [0.37896466, 0.41309066],
       [0.11169578, 0.26012024],
       [0.31519378, 0.21803135]])

    # Expected random weights of hidden layer and output layer
    expected_weight_by_layers.append(synaptic_weight_h_layer)
    expected_weight_by_layers.append(synaptic_weight_o_layer)

    # Length of the list
    layers_len = len(expected_output_by_ffl)
    delta_error = None
    error = None
    delta_error_layers = {}

    # Instantiate Neural Network Base
    # Call for the result from Neural Network Base
    nn_base = NeuralNetworkBase(**self.CONFIG)

     # Loop backward from length of layers
    for index in reversed(range(layers_len)):
      # Calculate the error from output layer
      # (The difference between the desired output and the predicted output).
      if delta_error is None:
        output = expected_output_by_ffl[index]
        delta_error = -(expected_target_vector - output)
        #output_error = target_vector - output
      else:
        # Calculate the error for layer 1 (By looking at the weights in layer 1,
        # we can determine by how much layer 1 contributed to the error in layer 2).
        synaptic_weight = expected_weight_by_layers[index+1]
        output = expected_output_by_ffl[index]
        delta_error = np.dot(synaptic_weight, error)

      # Determine error between expected output and that produces from learning process
      error = nn_base.derivative_relu(output, delta_error)

      # Keep tracking of delta error of each layer
      delta_error_layers[index] = error

    expected_delta_error_layers = {}
    expected_delta_error_layers[1] = np.array([-0.51212696,  0.50089774])
    expected_delta_error_layers[0] = np.array([ 0., -0.02157355, 0., 0.07309122, 0.])

    assert delta_error_layers == expected_delta_error_layers
