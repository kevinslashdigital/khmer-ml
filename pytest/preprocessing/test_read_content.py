import sys, os
import argparse
import pytest

syspath = 'slash-ml'
sys.path.append(os.path.abspath(os.path.join('..', syspath)))
import collections
from slashml.preprocessing.read_content import ReadContent

class TestPreprocessing:

  def test_load_content(self):
    text_dir = 'data/dataset/text'
    _words_all_articles, _all_words = ReadContent().load_content(text_dir)
    assert  isinstance(_words_all_articles, list)
    assert  isinstance(_all_words, list)

  def test_remove_stopword(self):
    result = ReadContent().remove_stopword("blockchain isn't famouse in good future")
    assert result == ['blockchain', 'famouse', 'good', 'future']

  def test_stemming_words(self):
    result = ReadContent().stemming_words( ['blockchain', 'famouse', 'good', 'future'])
    assert result == ['blockchain', 'fam', 'good', 'fut']

  def test_merge_list_content(self):
    result = ReadContent().merge_list_content([[['block', 'block', 'famouse', 'famouse', 'good', 'good', 'future']]])
    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
    assert compare(result, ['block', 'famouse', 'good', 'future'])


