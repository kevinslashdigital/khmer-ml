import numpy as np

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
input_vector = np.array([4.4, 3.0, 1.3, 0.2])

# Instantiate Neural Network Base
# Call for the result from Neural Network Base
#nn_base = NeuralNetworkBase(**self.CONFIG)

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
  #output_neurons = nn_base.activation_relu(product)
  output_neurons = product

  # Update nodes of the layer
  #output_by_layers.append [layer_index] = output_neurons
  output_by_layers.append(output_neurons)

# Expected feedforwards learning output
expected_output_by_layers = []
expected_output_by_layers.append(np.array([0., 2.60708086, 0., 0.84842288, 0.]))
expected_output_by_layers.append(np.array([0.42009255, 0.44102614]))

print(output_by_layers)
print(expected_output_by_layers)
# The result should be True
assert all([np.allclose(x, y) for x, y in zip(expected_output_by_layers, output_by_layers)])
