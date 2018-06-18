"""
    Factory class

"""
import random
import copy
import json
from slashml.machine_learning import MachineLearning
from slashml.preprocessing.preprocessing_data import Preprocessing
from slashml.utils.file_util import FileUtil

if __name__ == "__main__":

    config = {
        'root': '/Users/lion/Documents/py-workspare/openml/slash-ml',
        'text_dir': 'data/dataset/text',
        'dataset': 'data/dataset/matrix',
        'bag_of_words': 'data/dataset/bag_of_words',
        'train_model': 'data/naive_bayes.model',
        # 'mode': 'unicode'
    }
    # with open('config/env.sample.json') as json_data:
    #   config = json.load(json_data)
    # preposessing
    prepro = Preprocessing(**config)
    dataset_matrix = prepro.loading_data(config['text_dir'], 'doc_freq', 25)

    ml = MachineLearning(**config)

    #load dataset from file (feature data)
    filename = "doc_freq_25.csv"
    dataset_path = FileUtil.dataset_path(config, filename)
    dataset_sample = FileUtil.load_csv(dataset_path)

    # dataset -> train, test
    training_set, test_set = ml.split_dataset(dataset_sample, 3)
    # choose your algorithm
    algo = ml.NiaveBayes()
    # algo = ml.DecisionTree(criterion='gini', prune='depth', max_depth=3, min_criterion=0.05)
    # train or load model
    model = algo.train(training_set)
    # model = algo.load_model()
    # make a prediction
    predictions = algo.predict(model, test_set)
    acc = ml.accuracy(predictions,test_set)

    print('training_set', len(training_set))
    print('predictions, prediction_details', predictions, acc)
    print('label', ml.to_label(predictions))
