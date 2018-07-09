import numpy as np
from khmerml.algorithms.neural_network.neural_network_base import NeuralNetworkBase

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


CONFIG = {
    'text_dir': 'data/dataset/text',
    'dataset': 'data/matrix',
    'bag_of_words': 'data/bag_of_words',
    'train_model': 'data/model/train.model',
  }

# Instantiate Neural Network Base
# Call for the result from Neural Network Base
nn_base = NeuralNetworkBase(**CONFIG)

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
expected_delta_error_layers[0] = np.array([ 0.1, -0.02157355, 0., 0.07309122, 0.])


shared_items = {k for k in expected_delta_error_layers if k in delta_error_layers and np.array_equal(expected_delta_error_layers[k], delta_error_layers[k])}

print(shared_items)

for key, x in expected_delta_error_layers.items():
  y = delta_error_layers[key]
  #assert np.allclose(x, y)
  print(np.allclose(x, y))

#assert delta_error_layers == expected_delta_error_layers
