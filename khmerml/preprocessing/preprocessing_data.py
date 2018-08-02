"""
  This script contains a class mainly for preprocessing the data and return
"""

import os
import numpy as np
from collections import Counter
from khmerml.preprocessing.read_content import ReadContent
from khmerml.utils.file_util import FileUtil
from khmerml.utils.log import Log
from khmerml.utils.unicodesplit import UnicodeSplit

class Preprocessing(object):
  """"
    This is the main class
  """
  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def loading_data(self, folders, feature_choice, fq_type, threshold):
    """ Loading the data from txt files in folders """
    content = ReadContent(**self.kwargs)
    _words_articles, _all_words = content.load_content(folders)
    _temp_all_words = _all_words
    _temp_all_words.append(feature_choice)
    _temp_all_words.append(threshold)


    if feature_choice == 'doc_freq':
      _tfidf = self.doc_frequency(_words_articles, _all_words, feature_choice, fq_type, threshold)
    return _tfidf

  def doc_frequency(self, words_articles, all_words, feature_choice, fq_type, threshold):
    """
      This method aims at feature selection based
      on terms appearing in the articles.
      Select only terms that appear in more threshold of articles
    """
    _preq_words = dict() # words prequency with class
    _frequency = dict() # words prequency
    _label_match = dict()
    _label_count = 0
    for label in words_articles.keys(): # each class
      _label_match[label] = _label_count
      _label_count += 1
      if label not in _preq_words:
        _preq_words[label] = dict()
      for doc in words_articles[label]: # all words in each class
        for word in doc:
          if word in _frequency:
            _frequency[word] += 1
          else:
            _frequency[word] = 1
          if word in _preq_words[label]:
            _preq_words[label][word] += 1
          else:
            _preq_words[label][word] = 1
    # print('label', _label_match)

    _tfidf_mat = []
    _selected_words = []
    if fq_type == 'all':
      _selected_words = self.fq_all(_frequency, threshold)
    elif fq_type == 'class':
      _selected_words = self.fq_by_class(_preq_words, threshold)
    else:
      _selected_words = self.fq_by_top(_preq_words, threshold)

    _selected_words = list(set(_selected_words))
    _bag_of_words = self.compute_feature_matrix(_selected_words,\
                        words_articles, _label_match, feature_choice, threshold)
    # Save the dictionary for single document prediction
    path = self.kwargs['bag_of_words'] if ('bag_of_words' in self.kwargs) else 'data/bag_of_words'
    FileUtil.create_folder(path)
    FileUtil.save_pickle_dataset(path + '/'+feature_choice+'_'+str(threshold)+'.pickle', _selected_words)
    FileUtil.save_pickle_dataset(path + '/label_match.pickle', _label_match)
    return _bag_of_words


  def compute_feature_matrix(self, word_in_dic, text_in_articles, label_match, feature_choice, threshold):
    """
      Computing the feature matrix
    """
    mat = [] # feature matrix
    for label in text_in_articles.keys(): # each class
      for doc in text_in_articles[label]: # all words in each class
        row = []
        dic = Counter(doc)
        for word in word_in_dic:
          # each term or feature in article to be considered
          row.append(dic[word])# Adding to row
        row.append(label_match[label]) # adding label
        mat.append(row)

    _directory_name = FileUtil.join_path(self.kwargs['dataset'] if ('passion' in self.kwargs) else 'data/matrix' )
    FileUtil.create_folder(_directory_name)
    self.write_mat(_directory_name, feature_choice, threshold, mat)
    # returning feature matrix
    return mat

  def write_mat(self, path_to_file, feature_choice, threshold, feature_mat):
    """
      Method to write matrix into a file
    """
    FileUtil.create_folder(path_to_file)
    _mat_file = open(path_to_file + '/' + feature_choice+'_'+str(threshold)+'.csv', "w+")

    for _feature in feature_mat:
      _temp = str(_feature)
      _temp = _temp.strip(']')
      _temp = _temp.strip('[')
      _mat_file.write(_temp + '\n')

    _mat_file.close()

  def loading_single_doc(self, document, feature_choice, threshold):
    """
      Loading single document for prediction
    """
    content = ReadContent(**self.kwargs)
    if feature_choice == 'doc_freq':
      dic_load = FileUtil.load_pickle((self.kwargs['bag_of_words'] if ('bag_of_words' in self.kwargs) else 'data/bag_of_words') + \
                                      '/'+feature_choice+'_'+str(threshold)+'.pickle')
    _new_words = None
    is_unicode = self.kwargs.get('is_unicode', False)
    if is_unicode  :
      _new_words = UnicodeSplit().unicode_split(document)
    else:
      _new_words = content.remove_stopword(document)
      _new_words = content.stemming_words(_new_words)
    words = Counter(_new_words)# Count the frequency of each term
    row = []
    for word in dic_load:
      # each term or feature in article to be consider
      row.append(words[word])# Adding to row
    row.append(0)
    return row

  def fq_all(self,frequency, threshold):
    _selected_words = []
    for word in frequency:
      if frequency[word] >= threshold:
        # Consider only terms which appear in
        # documents more than a threshold
        _selected_words.append(word)
    Log('fq_all_selected_word.log').log(_selected_words)
    return _selected_words

  def fq_by_class(self,frequency_class, threshold):
    _selected_words = []
    for label,words in frequency_class.items():
      for word in frequency_class[label]:
        if words[word] >= threshold:
          _selected_words.append(word)

    Log('fq_by_class_selected_word.log').log(_selected_words)
    return _selected_words

  def fq_by_top(self,frequency_class, top):
    _selected_words = []
    for label, words in frequency_class.items():
      print('label',label)
      most_common = Counter(words).most_common(top)
      for word in most_common:
        _selected_words.append(word[0])

    Log('fq_by_class_top.log').log(_selected_words)
    return _selected_words


  def normalize_dataset(self, dataset):
    """ Transform data frequency to categorical data
    """
    feature_15 = dataset
    # % 0 : mean, 1 : mid point
    choice = 0
    # Every Attribute, determind the best information gain/gini
    for col_index in range(feature_15.shape[1] - 1):
      # Sort dataset following descendant label
      # dataset[:, -1].argsort # Sort the last field (column)
      # dataset = dataset[dataset[:,1].argsort(kind='mergesort')]
      feature = dataset[:, col_index]
      sort_feature = np.sort(feature)
      if choice == 1:
          index = np.floor(len(sort_feature)/2).astype('int')
          value = sort_feature[index]
      elif choice == 0:
          value = np.average(sort_feature)

      feature[sort_feature < value] = 0
      feature[sort_feature >= value] = 1

      dataset[:, col_index] = feature

    return dataset

