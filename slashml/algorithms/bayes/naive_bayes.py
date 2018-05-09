"""
    Factory class

"""
import copy

from slashml.algorithms.bayes.bayes_base import BayesBase
from slashml.utils.file_util import FileUtil
from slashml.algorithms.base import Base


class NaiveBayes(Base):
    """
        Naive Bayes class
    """

    def __init__(self, **kwargs):
        # super(NaiveBayes, self).__init__(**kwargs)
        self.kwargs = kwargs
        self.train_model = {}
        self.predictions = []
        self.naive_bayes = BayesBase(**kwargs)

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

    def save_model(self):
            """ Load train model from file
        """

        try:
            self.train_model = FileUtil.load_model(self.kwargs)
            self.naive_bayes.train_model = self.train_model
        except IOError as error:
            raise Exception(error)
        else:
            return True

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


