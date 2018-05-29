"""
    Factory class

"""

#from slashml.algorithm.neural_network.base import Base
from slashml.algorithm.neural_network.neural_network import NeuralNetwork

""" import sys
sys.path.append('/Users/lion/Documents/py-workspare/slash-ml/slash-ml/slashml')

from algorithm.neural_network.base import Base
from algorithm.neural_network.neural_network import NeuralNetwork
from utils.file_util import FileUtil """


class MainANN(object):
    """
        Naive Bayes class
    """


    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.5, max_iter=200, momentum=0.2,\
     random_state=1, activation='logistic', **kwargs):

        #super(MainANN, self).__init__(1, **kwargs)

        self.train_model = None
        self.predictions = None

        # Create a network with two input, two hidden, and one output nodes
        self.neural_network = NeuralNetwork(hidden_layer_sizes=hidden_layer_sizes,\
            learning_rate=learning_rate, momentum=momentum, random_state=0, \
            max_iter=max_iter, activation=activation, **kwargs)


    def load_dataset(self):
        pass


    def load_model(self):
        """ Load train model from file
        """

        """ try:
            self.train_model = FileUtil.load_model(self.kwargs)
            self.naive_bayes.train_model = self.train_model
        except IOError as error:
            raise Exception(error)
        else:
            return True """


    def train(self, training_input, targets):
        """ Train model
        """

        # Train the model with given dataset
        self.neural_network.train(training_input, targets)
        #self.train_model = self.naive_bayes.train_model

        # Save model in temporary file
        """ try:
            FileUtil.save_model(self.kwargs, self.train_model)
        except IOError as error:
            print(error)

        return self.train_model """


    def predict(self, test_sample):
        """ Make prediction
        """

        self.predictions, _ = self.neural_network.predict(test_sample)

        return self.predictions


    @staticmethod
    def demo(**kwargs):
        """ Sample of neural network usage
        """

        test_counter = 0
        accuracy_list = []

        from slashml.utils.file_util import FileUtil
        # Load dataset from file
        #path_to_cvs_dataset = '/Users/lion/Documents/py-workspare/slash-ml/data/dataset/matrix/iris.data.full.csv'
        #path_to_cvs_dataset = '/Users/lion/Documents/py-workspare/slash-ml/data/dataset/matrix/iris.data.csv'
        path_to_cvs_dataset = '/Users/lion/Documents/py-workspare/slash-ml/data/dataset/matrix/doc_freq_20.csv'
        dataset_matrix = FileUtil.load_csv_np(path_to_cvs_dataset)

        while test_counter < 1:
            # Array of hidden layers
            # hidden_layer_sizes = (250, 100)
            hidden_layer_sizes = (250, 100)
            learning_rate = 0.0003
            #learning_rate = 0.012 #tanh
            #learning_rate = 0.45 #logistics
            #learning_rate = 1.0
            momentum = 0.5
            #activation = 'tanh'
            activation = 'relu'
            #activation = 'logistic'

            max_iter = 1

            # create a network with two input, two hidden, and one output nodes
            main_ann = MainANN(hidden_layer_sizes=hidden_layer_sizes, \
            learning_rate=learning_rate, momentum=momentum, random_state=0, \
            max_iter=max_iter, activation=activation, **kwargs)

            X_train, X_test, y_train, y_test = main_ann.neural_network.train_test_split(dataset_matrix, n_test_by_class=3)

            # Get label from dataset
            # Convert label to array of vector
            #label_vector = dataset_matrix[:, -1]
            y_train_matrix = main_ann.neural_network.label_to_matrix(y_train)

            # Remove label from dataset
            #matrix_dataset = numpy.delete(dataset_matrix, numpy.s_[-1:], axis=1)

            # Start training process
            main_ann.train(X_train, y_train_matrix)

            # Perform prediction process
            predictions = main_ann.predict(X_test)

            # Prediction accuracy
            accuracy = main_ann.accuracy(y_test, predictions)

            print('----------------------------\n')
            print('Accuracy is {0}.'.format(accuracy))

            # Increment counter
            test_counter = test_counter + 1

            #end = clock()
            #elapsed = end - start

            #print('Computing time %s %s' %(test_counter, elapsed))

            # Keep tracking the accuracy per operation
            accuracy_list.append(accuracy)

        mode = max(set(accuracy_list), key=accuracy_list.count)
        print("Accuracy list: ", accuracy_list)
        print("Average Accuracy", round(sum(accuracy_list)/test_counter, 2))
        print("Mode of accuracy is : ", mode)


    def accuracy(self, y_test, predictions):
        """ Get Accuracy
        """

        return self.neural_network.accuracy(y_test, predictions)


if __name__ == "__main__":

    CONFIG = {
        #'root': '/var/www/slashml2/slash-ml',
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

    MainANN.demo(**CONFIG)
