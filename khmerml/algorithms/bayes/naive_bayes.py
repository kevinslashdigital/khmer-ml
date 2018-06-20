
import copy
import ntpath
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

    def train(self, dataset):
      """
        Train model
      """
      prioris = self.calculate_priori(dataset)
      likelihoods = self.calculate_likelihood(dataset)
      train_model = {}
      for class_key, likelihood in likelihoods.items():
        priori = prioris[class_key]
        if class_key not in train_model:
          train_model[class_key] = []
        train_model[class_key].append(priori)
        train_model[class_key].append(likelihood)
      self.train_model = train_model
      self.save_model(train_model)
      return train_model

    def predict(self, model, test_dataset):
      """
        Make prediction
      """
      predictions = []
      test_sample = copy.deepcopy(test_dataset)
      for subset in test_sample:
          # remove label from test dataset
          if len(test_dataset) > 1:
              del subset[-1]
          _, label = self.calculate_posteriori(model, subset)
          ''' if best_label is None or posteriori > best_prob:
              best_prob = posteriori
              best_label = label'''
          ''' if label not in predictions:
              predictions[label] = [] '''
          predictions.append(label)
      self.predictions = predictions
      return predictions



