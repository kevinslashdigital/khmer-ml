"""
    Factory class

"""
import random
import copy

from slashml.naive_bayes.naive_bayes import NaiveBayes
from slashml.utils.file_util import FileUtil
from slashml.naive_bayes.base import Base


class NaiveBayesTemplate(Base):
    """
        Naive Bayes class
    """

    def __init__(self, **kwargs):
        super(NaiveBayesTemplate, self).__init__(**kwargs)

        self.train_model = {}
        self.predictions = []
        self.naive_bayes = NaiveBayes(**kwargs)

    def load_dataset(self):
        pass

    def load_model(self):
        """ Load train model from file
        """

        try:
            self.train_model = FileUtil.load_model(self.kwargs)
            self.naive_bayes.train_model = self.train_model
        except IOError as error:
            raise Exception(error)
        else:
            return True

    def split_dataset_by_ratio(self, dataset, split_ratio):
        """ Split data set for training and testing
        """

        train_size = int(len(dataset) * split_ratio)
        train_set = []
        copy = list(dataset)
        while len(train_set) < train_size:
            index = random.randrange(len(copy))
            train_set.append(copy.pop(index))

        return [train_set, copy]


    def split_dataset(self, dataset, sample_by_class):
        """ Split data set for training and testing
        """

        dataset_by_class = self.classify_dataset_by_class(dataset)

        test_set = []
        no_train_set = 0

        for _, label in enumerate(dataset_by_class):
            no_train_set = 0
            subset = dataset_by_class[label]
            while no_train_set < sample_by_class:
                index_subset = random.randrange(len(subset))
                test_set.append(dataset_by_class[label].pop(index_subset))
                no_train_set = no_train_set + 1

        return [dataset_by_class, test_set]


    def extract_testingdata_dataset(self, dataset, sample_by_class):
        """ Extract subset from big training set
        The subset is used for testing on training model
        """

        dataset_by_class = self.classify_dataset_by_class(dataset)

        test_set = []
        no_train_set = 0

        for _, label in enumerate(dataset_by_class):
            no_train_set = 0
            subset = dataset_by_class[label]
            while no_train_set < sample_by_class:
                index_subset = random.randrange(len(subset))
                _vector = copy.deepcopy(dataset_by_class[label][index_subset])
                test_set.append(_vector)
                no_train_set = no_train_set + 1

        return [dataset_by_class, test_set]


    def classify_dataset_by_class(self, dataset):
        """ Classify dataset by class
        """

        dataset_by_class = {}
        for _, subset in enumerate(dataset):
            vector = subset
            if vector[-1] not in dataset_by_class:
                dataset_by_class[vector[-1]] = []

            dataset_by_class[vector[-1]].append(vector)

        return dataset_by_class

    def train(self, dataset):
        """ Train model
        """

        # Train the model with given dataset
        self.naive_bayes = self.naive_bayes.train(dataset)
        self.train_model = self.naive_bayes.train_model

        # Save model in temporary file
        try:
            FileUtil.save_model(self.kwargs, self.train_model)
        except IOError as error:
            print(error)

        return self.train_model

    def predict(self, test_dataset):
        """ Make prediction
        """

        self.predictions, _ = self.naive_bayes.predict(test_dataset)

        return self.predictions

    def test_naive_bayes_template(self):
        """ This function is used to test the functionality of this class
        """

        #filename = "feature22_1.csv"
        #filename = "data.csv"
        #path_to_cvs_dataset = FileUtil.path_to_file(self.kwargs, self.kwargs['dataset'], filename)
        #dataset_path = FileUtil.dataset_path(self.kwargs, filename)
        #dataset_sample = FileUtil.load_csv(path_to_cvs_dataset)

        test_counter = 0
        accuracy_list = []
        while test_counter < 20:

            # Trace computing time of this train and prediction process
            # Start time
            from time import clock
            start = clock()

            filename = "data.csv"
            path_to_cvs_dataset = FileUtil.path_to_file(self.kwargs, self.kwargs['dataset'], filename)
            #dataset_path = FileUtil.dataset_path(self.kwargs, filename)
            dataset_sample = FileUtil.load_csv(path_to_cvs_dataset)

            # Splite dataset into two subsets: traning_set and test_set
            # training_set:
                # it is used to train our model
            # test_set:
                # it is used to test our trained model
            training_set, test_set = self.split_dataset(dataset_sample, 2)

            _train_model = self.train(training_set)
            #model_existed = self.load_model()
            predicts = self.predict(test_set)

            #print("Training model ", self.naive_bayes.train_model)
            #print("Predicts ", predicts)

            accuracy = self.naive_bayes.accuracy(test_set)

            # Increment counter
            test_counter = test_counter + 1

            end = clock()
            elapsed = end - start

            print('Computing time %s %s' %(test_counter, elapsed))

            # Keep tracking the accuracy per operation
            accuracy_list.append(accuracy)

        mode = max(set(accuracy_list), key=accuracy_list.count)
        print("Accuracy list: ", accuracy_list)
        print("Average Accuracy", round(sum(accuracy_list)/test_counter, 2))
        print("Mode of accuracy is : ", mode)

    def accuracy(self, test_set):
        """ Get Accuracy
        """

        return self.naive_bayes.accuracy(test_set)

if __name__ == "__main__":

    CONFIG = {
        'root': '/Users/lion/Documents/py-workspare/slash-ml',
        'model_dataset': 'data/dataset',
        'train_model': 'data/naive_bayes_model.pickle',
        'train_dataset': 'data/train_dataset.pickle',
        'test_dataset': 'data/test_dataset.pickle',
        'text_dir': 'data/dataset/text',
        'archive_dir': 'data/dataset/temp',
        'dataset': 'data/dataset/matrix',
        'mode': 'unicode'
    }

    naive_bayes_template = NaiveBayesTemplate(**CONFIG)
    naive_bayes_template.test_naive_bayes_template()
