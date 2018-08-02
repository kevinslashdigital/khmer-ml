import numpy as np
import math
from decimal import Decimal
from functools import reduce

class BayesBase(object):
  """
    BayesBase class is use for calculation in
    Naive Bayes Algorithm
  """

  def __init__(self, **kwargs):
    self.kwargs = kwargs
    self._train_model = {}
    self.predictions = []


  def count_classes_occurrence(self, X_train):
    """
      Count class occurences from list data set by class
    """

    _, classes_occurrence = np.unique(X_train, return_counts=True)
    return classes_occurrence


  def calculate_priori(self, X_train):
    """
      Calculate Priori Probability
    """

    # Count class occurences from X_train
    classes, classes_occurrence = np.unique(X_train[:, -1], return_counts=True)
    total_occurences = np.sum(classes_occurrence)

    # Calculate probability per class
    prioris = {}
    for index, label in enumerate(classes):
      prioris[label] = classes_occurrence[index] / total_occurences

    return prioris


  def calculate_likelihood(self, X_train):
    """
      Calculate likelihoods
    """

    # Unique class
    unique_class = np.unique(X_train[:,-1])

    likelihoods = {}
    for _, label in enumerate(unique_class):

      # Find all rows where its label equal to search label
      X_subset = X_train[X_train[:, -1] == label]
      # Delete lable from dataset
      X_subset = np.delete(X_subset, -1, 1)
      # Count total frequency of all features
      total_frequency = np.sum(X_subset)
      # Count number of features
      total_features = X_subset.shape[1]

      # Count number of frequency of each and every feature
      feature_fequency = [np.sum(feature) for feature in X_subset.T]
      # Calculate likelihood of each feature
      probabilities = [(1+ f_count)/(total_features + total_frequency) \
                  for f_count in feature_fequency]

      # Push likelihood to list
      likelihoods[label] = probabilities

    return likelihoods

  def calculate_posteriori(self, model, test_vector):
    """
      Calculate the porbability of all classes
      one class at a time.
    """
    best_posteriori, best_label = -1, None

    for label, priori_likelihood in model.items():

      # Get priori (index=0)
      priori = Decimal(priori_likelihood[0])
      # Get likelihood (index=1)
      likelihood_feature = priori_likelihood[1]

      _likelihoods = list(map(lambda x, y: math.pow(x, y), \
                      likelihood_feature, test_vector))

      likelihood = reduce(lambda x, y: Decimal(x) * Decimal(y), _likelihoods)

      # Calculate posteriori
      posteriori = priori * Decimal(likelihood)

      # Check for the best posteriori
      if best_label is None or posteriori > best_posteriori:
        best_posteriori = posteriori
        best_label = label

    return best_posteriori, best_label
