
import numpy as np
from numpy import exp

class NeuralNetworkBase(object):
  """
    NeuralNetwork Base class is use for calculation in
    Naive Bayes Algorithm
  """

  def __init__(self, **kwargs):
    """
      makes the random numbers predictable
      With the seed reset (every time), the same set of numbers will appear every time.
      If the random seed is not reset, different numbers appear with every invocation:
      random.seed(random_state)
      Set float precision
    """
    np.set_printoptions(precision=8)
    self.kwargs = kwargs


  def label_to_matrix(self, label_vector):
    """
      Transform labels to row-based vector
    """
    length = len(label_vector)
    n_label = len(set(label_vector))
    unique_labels = np.unique(label_vector)
    # Create matrix zeros m row x n columns
    label_matrix = np.zeros((length, n_label))
    # Replace value by 1 at position (i, label_vector(j))
    for i, value in enumerate(label_vector):
      #value = int(round(value))
      index = np.where(unique_labels == value)[0]
      label_matrix[i][index] = np.float(1.0)
    return label_matrix

  def make_weight_matrix(self, n_features, n_labels, hidden_layer_sizes):
    """
      Make random weight in form of matrix for each layer
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
      synaptic_weight = np.random.normal(loc=0, scale=input_range, size=(n_inputs, n_output_nodes))

      # create arrays of 0 for changes
      # this is essentially an array of temporary values that gets updated at each iteration
      # based on how much the weights need to change in the following iteration
      change_output = np.zeros((n_inputs, n_output_nodes))
      change_output_layouts.append(change_output)

      # n_output_nodes is input for next layer
      n_inputs = n_output_nodes

      # Add layer to list
      synaptic_weight = np.array(synaptic_weight)
      layers.append(synaptic_weight)

    # Add layer to list
    # Output layer
    synaptic_weight = np.random.uniform(size=(n_inputs, n_labels)) / np.sqrt(n_inputs)
    layers.append(synaptic_weight)

    # create arrays of 0 for changes
    # this is essentially an array of temporary values that gets updated at each iteration
    # based on how much the weights need to change in the following iteration
    change_output = np.zeros((n_inputs, n_labels))
    change_output_layouts.append(change_output)
    return layers, change_output_layouts


  def activation_relu(self, X):
    """
      Compute the rectified linear unit function inplace.
      Parameters
      ----------
      X : {array-like, sparse matrix}, shape (n_samples, n_features)
          The input data.
      Returns
      -------
      X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
      The transformed data.
    """
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X


  def derivative_relu(self, X, delta):
    """
      Apply the derivative of the relu function.
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
    """
      tanh is a little nicer than the standard 1/(1+e^-x)
    """
    return np.tanh(X, out=X)

  def derivative_tanh(self, X, delta):
    """
      Apply the derivative of the hyperbolic tanh function.
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
    """
      The Sigmoid function, which describes an S shaped curve.
      We pass the weighted sum of the inputs through this function to
      normalise them between 0 and 1.
    """
    return 1 / (1 + exp(-X))

  def derivative_sigmoid(self, X, delta):
    """
      Apply the derivative of the logistic sigmoid function.
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
