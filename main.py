"""
    Factory class

"""
import random
import copy
from slashml.machine_learning import MachineLearning
from slashml.preprocessing.preprocessing_data import Preprocessing
from slashml.utils.file_util import FileUtil

if __name__ == "__main__":

    config = {
        'root': '/Projects/slashml/slash-ml',
        'text_dir': 'data/dataset/text',
        'dataset': 'data/dataset/matrix',
        'train_model': 'data/naive_bayes_model.pickle',
        # 'mode': 'unicode'
    }
    # preposessing
    # prepro = Preprocessing(**config)
    # dataset_matrix = prepro.loading_data(config['text_dir'], 'doc_freq', 15)

    ml = MachineLearning(**config)

    #load dataset from file
    filename = "data.csv"
    dataset_path = FileUtil.dataset_path(config, filename)
    dataset_sample = FileUtil.load_csv(dataset_path)

    # dataset -> train, test
    training_set, test_set = ml.split_dataset(dataset_sample, 2)

    print('training_set', training_set)

    algo = ml.NiaveBayes()
    model = algo.train(training_set)
    acc = algo.predict(test_set)
