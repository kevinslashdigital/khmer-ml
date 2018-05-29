from slashml.algorithm.neural_network.main_ann import MainANN
from slashml.naive_bayes.naive_bayes import NaiveBayes

def instanceof_class(obj, my_class):
    if isinstance(obj, my_class):
        print('this is my class', obj)
    else:
        
        print('obj {0} , class {1}'.format(obj, my_class))


CONFIG = {
    'root': '/Users/lion/Documents/py-workspare/slash-ml',
    'model_dataset': 'data/dataset',
    'train_model': 'data/naive_bayes_model.pickle',
    'train_dataset': 'data/train_dataset.pickle',
    'test_dataset': 'data/test_dataset.pickle',
    'text_dir': 'data/dataset/text',
    'archive_dir': 'data/dataset/temp',
    'dataset': 'data/dataset/matrix',
    'dataset_filename': 'data.csv',
    'mode': 'unicode'
}

ANN = MainANN(hidden_layer_sizes=(100,), learning_rate=0.5, max_iter=200, momentum=0.2,\
     random_state=1, activation='logistic', **CONFIG)

instanceof_class(ANN, NaiveBayes)
