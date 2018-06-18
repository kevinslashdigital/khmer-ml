"""
    Factory class

"""
import copy

from slashml.algorithms.decisiontree.tree_base import TreeBase
from slashml.utils.file_util import FileUtil
from slashml.algorithms.base import Base


class DecisionTree(Base, TreeBase):
    """
        DecisionTree
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.train_model = {}
        self.predictions = []
        self.criterion = kwargs['criterion']
        self.prune = kwargs['prune']
        self.max_depth = kwargs['max_depth']
        self.min_criterion = kwargs['min_criterion']
        # Call super
        super(DecisionTree, self).__init__(**kwargs)


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

        X_train = dataset[dataset[:, (len(dataset[0]) -1)]];
        y_train = dataset[dataset[:, -1]];

        # self.root = Tree(self.max_depth)
        train_model = self.build(X_train, y_train, self.criterion)
        
        self.train_model = train_model
        self.save_model(train_model)
        return train_model

    def predict(self, model, test_dataset):
        """ Make prediction
        """
      
        predictions = []

        test_sample = copy.deepcopy(test_dataset)

        for subset in test_sample:
            # remove label from test dataset
            if len(test_dataset) > 1:
                del subset[-1]

            _, label = self.calculate_posteriori(model, subset)
            ''' if best_label is None or posteriori > best_prob:
                best_prob = posteriori
                best_label = label'''
            ''' if label not in predictions:
                predictions[label] = [] '''

            predictions.append(label)
        
        self.predictions = predictions
        return predictions



