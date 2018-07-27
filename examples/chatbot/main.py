"""
    Chatbot class
"""

import sys, os
import argparse

syspath = 'khmer-ml'
sys.path.append(os.path.abspath(os.path.join('..', syspath)))

from khmerml.machine_learning import MachineLearning
from khmerml.preprocessing.preprocessing_data import Preprocessing
from khmerml.utils.file_util import FileUtil
from khmerml.utils.bg_colors import Bgcolors

def get_answer(label):
  answer = 'sorry i don\'t understand what you are saying, tell me more!'
  if label[0] == 'greetings':
    answer = 'I am chatbot, nice to know you.'
  elif label[0] == 'goodbye':
    answer = 'thanks for chating with me.'
  elif label[0] == 'thanks':
    answer = 'thanks you so much, see you again soon.'
  return answer

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode',type=str,default ='chat',help='There are two mode (chat, train, train_c, test and none), The defaul value is chat.')
  # parser.add_argument("--benchmark", help="run benchmark",action="store_true")
  # parser.add_argument('--mode',type=float,default =0.2,help='There two mode?(train and chat)')
  args = parser.parse_args()

  config = {
    'text_dir': 'data/dataset/chatbot',
    'dataset': 'data/matrix',
    'bag_of_words': 'data/bag_of_words',
    'train_model': 'data/model/train.model',
    # 'mode': 'unicode'
  }

  ml = MachineLearning(**config)
  # choose your algorithm
  algo = ml.NiaveBayes()
  prepro = Preprocessing(**config)
  # -- mode
  if args.mode == 'train' :
    # preposessing
    dataset_matrix = prepro.loading_data(config['text_dir'], 'doc_freq', 'all', 1)
    #load dataset from file (feature data)
    filename = "doc_freq_1.csv"
    dataset_path = FileUtil.dataset_path(config, filename)
    dataset_sample = FileUtil.load_csv(dataset_path)

    ml = MachineLearning(**config)
    # split dataset -> train set, test set
    training_set, test_set = ml.split_dataset(dataset_sample, 1)
    # train
    model = algo.train(training_set)

    # make a prediction
    predictions = algo.predict(model, test_set)
    # Prediction accuracy
    acc = ml.accuracy(predictions, test_set)

    print('predictions, prediction_details', predictions, acc)
    print('label', ml.to_label(predictions,'data/bag_of_words/label_match.pickle'))
    print('==== Chatbot train completed! ====')

  elif args.mode == 'chat':
    print ("Start chatting with the bot !")
    model = algo.load_model()
    sessionid = 'Liza'
    while True:
      question 	= input('')
      # preprocess
      mat = prepro.loading_single_doc(question, 'doc_freq', 1)
      prediction = algo.predict(model, [mat])
      label = ml.to_label(prediction, 'data/bag_of_words/label_match.pickle')
      answer = get_answer(label)
      print('prediction', label)
      print(Bgcolors.OKGREEN + 'You :' + Bgcolors.ENDC,question)
      print(Bgcolors.OKBLUE 	+ 'Bot :' + Bgcolors.ENDC,answer)

