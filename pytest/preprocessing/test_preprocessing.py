import sys, os
import argparse
import pytest

syspath = 'khmer-ml'
sys.path.append(os.path.abspath(os.path.join('..', syspath)))

from khmerml.preprocessing.preprocessing_data import Preprocessing

class TestPreprocessing:
  def test_loading_data(self):
    text_dir = 'data/dataset/chatbot'
    result = Preprocessing().loading_data(text_dir, 'doc_freq','all', 1)
    fname = os.getcwd() + '/data/matrix/doc_freq_25.csv'
    assert os.path.isfile(fname)
    # pytest.xfail("not configured config")

  def test_loading_single_doc(self):
    result = Preprocessing().loading_single_doc('this is a famouse blockchain technology', 'doc_freq', 25)
    assert isinstance(result, list)
