
import copy
import ntpath
import numpy as np

import sys, os
syspath = '/Users/lion/Documents/py-workspare/openml/khmer-ml'
sys.path.append(syspath)

from khmerml.utils.file_util import FileUtil
from khmerml.algorithms.bayes.naive_bayes import NaiveBayes


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])

    print('funcarglist', metafunc.function.__name__)

    metafunc.parametrize(argnames, [[funcargs[name] for name in argnames]
            for funcargs in funcarglist])

#class TestNaiveBayes(Base, BayesBase):
class TestNaiveBayes:
  """
    Naive Bayes class
  """

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

  expected_train_model = {}
  expected_train_model[1.0] = [0.5, [0.47432024169184295, 0.3232628398791541, 0.15407854984894262, 0.04833836858006044]]
  expected_train_model[2.0] = [0.5, [0.394919168591224, 0.20092378752886833, 0.2909930715935335, 0.11316397228637413]]

  params = {
    'test_train': [dict(X_train=sample_data), ],
    'test_predict': [dict(model=expected_train_model, X_test=test_vector), ],
    'test_load_model':[dict()],
  }


  def test_train(self, X_train):
    """
      Train model
    """

    # Train model and Save the model to file
    naive_bayes = NaiveBayes(**self.CONFIG)
    train_model = naive_bayes.train(X_train)

    # Expected output model
    expected_train_model = {}
    expected_train_model[1.0] = [0.5, [0.47432024169184295, 0.3232628398791541, 0.15407854984894262, 0.04833836858006044]]
    expected_train_model[2.0] = [0.5, [0.394919168591224, 0.20092378752886833, 0.2909930715935335, 0.11316397228637413]]

    assert np.array_equal(train_model, expected_train_model)


  def test_predict(self, model, X_test):
    """
      Make prediction
    """

    # Make prediction based on trained model
    naive_bayes = NaiveBayes(**self.CONFIG)
    predictions = naive_bayes.predict(model, X_test)

    # Expected predictions
    expected_predictions = [1.0]

    assert np.array_equal(predictions, expected_predictions)


  def test_load_model(self):
    """
      Load train model from file
    """
    # Save model
    naive_bayes = NaiveBayes(**self.CONFIG)
    try:
      train_model = naive_bayes.load_model()
    except RuntimeError as error:
      print(error)
    else:
      # Expected output model
      expected_train_model = {}
      expected_train_model[1.0] = [0.5, [0.47432024169184295, 0.3232628398791541, 0.15407854984894262, 0.04833836858006044]]
      expected_train_model[2.0] = [0.5, [0.394919168591224, 0.20092378752886833, 0.2909930715935335, 0.11316397228637413]]

      assert np.array_equal(train_model, expected_train_model)
