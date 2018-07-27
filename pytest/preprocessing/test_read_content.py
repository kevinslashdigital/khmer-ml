import sys, os
import argparse
import pytest

syspath = 'khmer-ml'
sys.path.append(os.path.abspath(os.path.join('..', syspath)))
import collections
from khmerml.preprocessing.read_content import ReadContent

class TestPreprocessing:

  def test_load_content(self):
    text_dir = 'data/dataset/chatbot'
    _words_all_articles, _all_words = ReadContent().load_content(text_dir)
    assert  isinstance(_words_all_articles, dict)
    assert  isinstance(_all_words, list)

  def test_remove_stopword(self):
    result = ReadContent().remove_stopword("blockchain isn't famouse in good future")
    assert result == ['blockchain', 'famouse', 'good', 'future']

  def test_stemming_words(self):
    result = ReadContent().stemming_words( ['blockchain', 'famouse', 'good', 'future'])
    assert result == ['blockchain', 'fam', 'good', 'fut']
