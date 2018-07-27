"""
  Main class
"""
from khmerml.machine_learning import MachineLearning
from khmerml.preprocessing.preprocessing_data import Preprocessing
from khmerml.utils.file_util import FileUtil

#
if __name__ == "__main__":

  config = {
    'text_dir': 'data/dataset/text',
    'dataset': 'data/matrix',
    'bag_of_words': 'data/bag_of_words',
    'train_model': 'data/model/train.model',
    # 'mode': 'unicode'
  }

  """
    preposessing
  """
  prepro = Preprocessing(**config)
  # dataset_matrix = prepro.loading_data(config['text_dir'], 'doc_freq', 'all', 25)
  #load dataset from file (feature data)
  filename = "doc_freq_25.csv"
  dataset_path = FileUtil.dataset_path(config, filename)
  dataset_sample = FileUtil.load_csv(dataset_path)
  # dataset_sample = prepro.normalize_dataset(dataset_sample) # use with decision tree only
  """
    end
  """


  """
    training
  """
  ml = MachineLearning(**config)
  # split dataset -> train set, test set
  training_set, test_set = ml.split_dataset(dataset_sample, 2)
  # choose your algorithm
  algo = ml.NiaveBayes()
  # algo = ml.DecisionTree(criterion='gini', prune='depth', max_depth=30, min_criterion=0.05)
  # algo = ml.NeuralNetwork(hidden_layer_sizes=(250, 100), learning_rate=0.012, momentum=0.5, random_state=0, max_iter=200, activation='tanh')
  # train or load model
  model = algo.train(training_set)
  # model = algo.load_model()

  """
    end
  """


  """
    classify or predict
  """
  # make a prediction
  predictions = algo.predict(model, test_set)
  # Prediction accuracy
  acc = ml.accuracy(predictions, test_set)

  print('training_set', len(training_set))
  print('predictions, prediction_details', predictions, acc)
  print('label', ml.to_label(predictions, 'data/bag_of_words/label_match.pickle'))

  """
    end
  """
