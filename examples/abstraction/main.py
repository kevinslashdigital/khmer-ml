"""
    Abstraction Layer Class
"""

import sys, os
import argparse
import time
import numpy as np

syspath = 'khmer-ml'
sys.path.append(os.path.abspath(os.path.join('..', syspath)))

from khmerml.machine_learning import MachineLearning
from khmerml.preprocessing.preprocessing_data import Preprocessing
from khmerml.utils.file_util import FileUtil
from khmerml.utils.bg_colors import Bgcolors


def perform_algo(ml, algo, dataset):
  result_acc = []
  result_acc_train = []
  result_exec_time = []
  for i in range(10):
    exec_st = time.time()
    # split dataset -> train set, test set
    training_set, test_set = ml.split_dataset(dataset, 1)
    # train
    model = algo.train(training_set)
    # make a prediction
    pred_test = algo.predict(model, test_set)
    pred_train = algo.predict(model, training_set)

    # Prediction accuracy
    acc = ml.accuracy(pred_test, test_set)
    acc_train = ml.accuracy(pred_train, training_set)
    exec_time = time.time() - exec_st
    print(acc, acc_train, exec_time)
    result_acc.append(acc)
    result_acc_train.append(acc_train)
    result_exec_time.append(exec_time)

  mean_acc = np.mean(np.array(result_acc))
  mean_acc_train = np.mean(np.array(result_acc_train))
  mean_exec_time = np.mean(np.array(result_exec_time))

  return {
    'acc': mean_acc,
    'acc_train': mean_acc_train,
    'exec_time': mean_exec_time
  }

if __name__ == "__main__":
  whole_st = time.time()

  config = {
    'text_dir': 'data/dataset/chatbot',
    'dataset': 'data/matrix',
    'bag_of_words': 'data/bag_of_words',
    'train_model': 'data/model/train.model',
    'is_unicode': 'false'
  }

  prepro = Preprocessing(**config)
  # preposessing
  dataset_matrix = prepro.loading_data(config['text_dir'], 'doc_freq', 'all', 1)
  #load dataset from file (feature data)
  filename = "doc_freq_1.csv"
  dataset_path = FileUtil.dataset_path(config, filename)
  dataset_sample = FileUtil.load_csv(dataset_path)

  prepro_time = time.time() - whole_st

  ml = MachineLearning(**config)
  # choose your algorithm
  nb_algo = ml.NiaveBayes()
  nn_algo = ml.NeuralNetwork(hidden_layer_sizes=(250, 100), learning_rate=0.012, momentum=0.5, random_state=0, max_iter=200, activation='tanh')
  dt_algo = ml.DecisionTree(criterion='gini', prune='depth', max_depth=30, min_criterion=0.05)

  nb_result = perform_algo(ml, nb_algo, dataset_sample)
  nn_result = perform_algo(ml, nn_algo, dataset_sample)
  dt_result = perform_algo(ml, dt_algo, dataset_sample)

  print(nb_result, nn_result, dt_result)

  total_execution_time = time.time() - whole_st

  result = {
    'com_time': total_execution_time,
    'text_extract_time': prepro_time,
    'figure_on_testing_data': {
      'NB': nb_result['acc'],
      'NN': nn_result['acc'],
      'DT': dt_result['acc'],
    },
    'figure_on_training_data': {
      'NB': nb_result['acc_train'],
      'NN': nn_result['acc_train'],
      'DT': dt_result['acc_train'],
    },
    'on_testing_data': {
      'NB': {'accurate': nb_result['acc'], 'time': nb_result['exec_time']},
      'NN': {'accurate': nn_result['acc'], 'time': nn_result['exec_time']},
      'DT': {'accurate': dt_result['acc'], 'time': dt_result['exec_time']},
    },
    'on_training_data': {
      'NB': {'accurate': nb_result['acc_train'], 'time': nb_result['exec_time']},
      'NN': {'accurate': nn_result['acc_train'], 'time': nn_result['exec_time']},
      'DT': {'accurate': dt_result['acc_train'], 'time': dt_result['exec_time']},
    }
  }

  print(result)

