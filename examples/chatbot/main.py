"""
    Chatbot class
"""

import sys, os
import argparse

syspath = 'slash-ml'
sys.path.append(os.path.abspath(os.path.join('..', syspath)))

from khmerml.machine_learning import MachineLearning
from khmerml.preprocessing.preprocessing_data import Preprocessing
from khmerml.utils.file_util import FileUtil

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode',type=str,default ='chat',help='There are two mode (chat, train, train_c, test and none), The defaul value is chat.')
  # parser.add_argument("--benchmark", help="run benchmark",action="store_true")
  # parser.add_argument('--mode',type=float,default =0.2,help='There two mode?(train and chat)')
  args = parser.parse_args()

  config = {
    'root': '/Data/Projects/ML/khmerml/slash-ml',
    'text_dir': 'data/dataset/text',
    'dataset': 'data/matrix',
    'bag_of_words': 'data/bag_of_words',
    'train_model': 'data/model/train.model',
    # 'mode': 'unicode'
  }

  ml = MachineLearning(**config)
  # choose your algorithm
  algo = ml.NiaveBayes()
  algo = ml.NeuralNetwork(hidden_layer_sizes=(250, 100), learning_rate=0.012, momentum=0.5, random_state=0, max_iter=200, activation='tanh')
  prepro = Preprocessing(**config)
  # -- mode
  if args.mode == 'train' :
    # preposessing
    dataset_matrix = prepro.loading_data(config['text_dir'], 'doc_freq', 'top', 25)
    #load dataset from file (feature data)
    filename = "doc_freq_25.csv"
    dataset_path = FileUtil.dataset_path(config, filename)
    dataset_sample = FileUtil.load_csv(dataset_path, use_numpy=True)
    dataset_sample = prepro.normalize_dataset(dataset_sample)

    ml = MachineLearning(**config)
    # split dataset -> train set, test set
    training_set, test_set = ml.split_dataset(dataset_sample, 5, use_numpy = True)
    algo = ml.NeuralNetwork(hidden_layer_sizes=(250, 100), learning_rate=0.012, momentum=0.5, random_state=0, max_iter=200, activation='tanh')
    # train
    model = algo.train(training_set)

    # make a prediction
    predictions = algo.predict(model, test_set)
    # Prediction accuracy
    acc = ml.accuracy(predictions, test_set)

    print('predictions, prediction_details', predictions, acc)
    print('label', ml.to_label(predictions,'data/bag_of_words/label_match.pickle'))
    print('Chatbot train completed!')

  elif args.mode == 'chat':
    print ("Start chatting with the bot !")
    model = algo.load_model()
    sessionid = 'Liza'
    while True:
      question 	= input('')
      # preprocess
      mat = prepro.loading_single_doc(question, 'doc_freq', 25)
      prediction = algo.predict(model, [mat])
      print('prediction', ml.to_label(prediction, 'data/bag_of_words/label_match.pickle'))
      # print(Bcolors().OKGREEN + 'You :' + Bcolors().ENDC,question)
      # print(Bcolors().OKBLUE 	+ 'Bot :' + Bcolors().ENDC,answer)
      # # save all chat to db
      # data = 'sessionid;;'+sessionid +';;question;;'+ question+ ';;response;;' + answer
      # save_conversation.save(data)

