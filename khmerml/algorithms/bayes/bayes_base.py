"""
  Naive Bayes Probability
"""
import math
from decimal import Decimal
from functools import reduce

class BayesBase(object):
  """
    Naive Bayes class
  """

  def __init__(self, **kwargs):
    self.kwargs = kwargs
    self._train_model = {}
    self.predictions = []

  def count_classes_occurrence(self, dataset):
    """
      Count class occurences from list data set
      by class
    """
    classes_occurrence = {}
    for _, label in enumerate(dataset):
        classes_occurrence[label] = len(dataset[label])
    return classes_occurrence

  def calculate_class_priori(self, dataset_classes):
    """
      Calculate occurences depending on occurences
    """
    total_frequency = sum(dataset_classes.values())
    probabilities = {}
    for label, value in dataset_classes.items():
        probabilities[label] = value / total_frequency

    return probabilities


  def calculate_priori(self, dataset):
    """
      Calculate Priori Probability
    """
    classes_occurrence = self.count_classes_occurrence(dataset)
    prioris = self.calculate_class_priori(classes_occurrence)
    return prioris


  def calculate_likelihood(self, dataset):
    """
      Calculate likelihoods
    """
    # zip feature by class
    dataset_by_class = dict(dataset)
    likelihoods = {}
    for class_key, subset in dataset_by_class.items():
      zip_feature = zip(*subset)
      features = list(map(sum, zip_feature))
      del features[-1]
      total_unique_feat = len(features)
      total_freq = sum(features)
      # Calculate likelihood of each feature
      probabilities = [(1+ f_count)/(total_unique_feat + total_freq) \
                      for f_count in features]
      # Store likelihood by class
      likelihoods[class_key] = probabilities

    return likelihoods

  def calculate_posteriori(self, train_model, test_vector):
    """
      Calculate the porbability of all classes
      one class at a time.
    """
    best_posteriori, label = -1, None
    for class_index, priori_likelihood in train_model.items():
      priori = Decimal(priori_likelihood[0])
      likelihood_feature = priori_likelihood[1]
      _likelihoods = list(map(lambda x, y: math.pow(x, y), \
                      likelihood_feature, test_vector))

      likelihood = reduce(lambda x, y: Decimal(x) * Decimal(y), _likelihoods)
      posteriori = priori * likelihood
      if label is None or posteriori > best_posteriori:
        best_posteriori = posteriori
        label = class_index
    return best_posteriori, label

