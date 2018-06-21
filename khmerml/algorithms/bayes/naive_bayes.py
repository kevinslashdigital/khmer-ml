
import copy
import ntpath
import numpy as np

from khmerml.algorithms.bayes.bayes_base import BayesBase
from khmerml.utils.file_util import FileUtil
from khmerml.algorithms.base import Base

class NaiveBayes(Base, BayesBase):
  """
    Naive Bayes class
  """

  def __init__(self, **kwargs):
    # BayesBase.__init__(**kwargs)
    self.kwargs = kwargs
    self.train_model = {}
    self.predictions = []

  def load_model(self):
    """
      Load train model from file
    """
    try:
      head, tail = ntpath.split(self.kwargs['train_model'])
      self.train_model = FileUtil.load_model(head+ '/naive_bayes_' +tail)
    except IOError as error:
      raise Exception(error)

    return self.train_model

  def save_model(self, model):
    """
      Load train model from file
    """
    try:
      head, tail = ntpath.split(self.kwargs['train_model'])
      FileUtil.save_model(head+ '/naive_bayes_' +tail, model)
    except IOError as error:
      print(error)

  def train(self, X_train):
    """
      Train model
    """
    prioris = self.calculate_priori(X_train)
    likelihoods = self.calculate_likelihood(X_train)

    train_model = {}
    for label, likelihood in likelihoods.items():

      # Get priori for corresponding class
      priori = prioris[label]
      if label not in train_model:
        train_model[label] = []

      # Push priori and likelihood of each label to stack
      train_model[label].append(priori)
      train_model[label].append(likelihood)

    self.train_model = train_model
    self.save_model(train_model)
    return train_model

  def predict(self, model, X_test):
    """
      Make prediction
    """

    predictions = []
    X_test = np.delete(X_test, -1, 1)

    for index in range(X_test.shape[0]):
    #for subset in test_sample:
      # remove label from test dataset
      #test_vector = np.delete(X_test[index], -1, 1)
      test_vector = X_test[index]
      _, label = self.calculate_posteriori(model, test_vector)
      predictions.append(label)
    self.predictions = predictions

    return predictions
