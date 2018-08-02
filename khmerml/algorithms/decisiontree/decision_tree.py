import ntpath
import numpy as np

from khmerml.algorithms.decisiontree.tree import Tree
from khmerml.utils.file_util import FileUtil
from khmerml.algorithms.base import Base


class DecisionTree(Base, Tree):
  """
    DecisionTree class
  """

  def __init__(self, **kwargs):
    self.kwargs = kwargs
    self.train_model = {}
    self.predictions = []
    self.criterion = kwargs['criterion']
    self.prune = kwargs['prune']
    self.min_criterion = kwargs['min_criterion']
    Base.__init__(self, **kwargs)
    Tree.__init__(self, **kwargs)

  def load_model(self):
    """
      Load train model from file
    """
    try:
      head, tail = ntpath.split(self.kwargs['train_model'])
      model = FileUtil.load_model(head+ '/decision_tree_' +tail)
      self.__dict__.update(model)
    except IOError as error:
      raise Exception(error)
    return self

  def save_model(self, model):
    """
      Load train model from file
    """
    try:
      head, tail = ntpath.split(self.kwargs['train_model'])
      FileUtil.save_model(head+ '/decision_tree_' +tail, model.__dict__)
    except IOError as error:
      print(error)

  def train(self, dataset):
    """
      Train model
    """
    y_train = dataset[:, -1]
    X_train = np.delete(dataset, -1, 1)
    # Start constructing tree
    # self.build(X_train, y_train, self.criterion)
    self.build(X_train, y_train, self.criterion)
    self.save_model(self)
    # self.show_tree(10,'he')
    return self

  def predict(self, model, test_dataset):
    """
      Make prediction
    """
    # Get train and test label
    # Delete last column (label) from array
    X_test = np.delete(test_dataset, -1, 1)
    return np.array([self._predict(feature) for feature in X_test])
