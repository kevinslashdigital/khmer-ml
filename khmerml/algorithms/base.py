"""
  Machine learning abstract class
"""
from abc import ABC, abstractmethod
class Base(ABC):
  """
    Machine learning abstract class
  """

  def __init__(self, **kwargs):
    self.kwargs = kwargs

  @abstractmethod
  def load_model(self):
    """
      load dataset extraction
    """
    pass

  @abstractmethod
  def save_model(self, model):
    """
      load dataset extraction
    """
    pass

  @abstractmethod
  def train(self, dataset):
    """
      load dataset extraction
    """
    pass

  @abstractmethod
  def predict(self, model, test_dataset):
    """
      load dataset extraction
    """
    pass
