"""
    Factory class

"""
import copy
import numpy as np

from khmerml.algorithms.decisiontree.tree import Tree
from khmerml.utils.file_util import FileUtil
from khmerml.algorithms.base import Base


class DecisionTree(Base, Tree):
    """
        DecisionTree
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.train_model = {}
        self.predictions = []
        self.criterion = kwargs['criterion']
        self.prune = kwargs['prune']
        #self.max_depth = kwargs['max_depth']
        self.min_criterion = kwargs['min_criterion']

        # Tree to be built
        #self.root = None

        # Call super
        #super(DecisionTree, self).__init__(**kwargs)
        Base.__init__(self, **kwargs)
        Tree.__init__(self, **kwargs)


    """ def __init__(self, **kwargs):
        # BayesBase.__init__(**kwargs)
        self.kwargs = kwargs
        self.train_model = {}
        self.predictions = [] """

    def load_model(self):
        """ Load train model from file
        """

        try:
            self.train_model = FileUtil.load_model(self.kwargs)
        except IOError as error:
            raise Exception(error)
        
        return self.train_model

    def save_model(self, model):
        """ Load train model from file
        """
        
        try:
            FileUtil.save_model(self.kwargs, model)
        except IOError as error:
            print(error)

    def train(self, dataset):
        """ Train model
        """

        # Extract X_train and y_train
        #y_train = dataset[dataset[:, -1]]
        y_train = dataset[:, -1]
        X_train = np.delete(dataset, -1, 1)

        # Start constructing tree
        #self.root = Tree(self.max_depth)
        self.build(X_train, y_train, self.criterion)

        # Save final Tree to .pickle file
        self.train_model = self.build(X_train, y_train, self.criterion)
        self.save_model(self.train_model)
        return self.train_model


    def predict(self, model, test_dataset):
        """ Make prediction
        """

        # Get train and test label
        #y_test = test_dataset[:, -1]
        # Delete last column (label) from array
        X_test = np.delete(test_dataset, -1, 1)

        return np.array([self._predict(feature) for feature in X_test])