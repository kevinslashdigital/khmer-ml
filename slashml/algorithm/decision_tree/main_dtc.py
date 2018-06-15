#!/bin/python
# -*- coding: utf-8 -*-

import numpy

from slashml.utils.file_util import FileUtil
from slashml.algorithm.decision_tree.base import Base
from slashml.algorithm.decision_tree.tree import Tree


class DecisionTreeClassifier(Base):

    def __init__(self, criterion='gini', prune='depth', max_depth=3, min_criterion=0.05, **kwargs):
        # Call super
        super(DecisionTreeClassifier, self).__init__(**kwargs)

        self.root = None
        self.criterion = criterion
        self.prune = prune
        self.max_depth = max_depth
        self.min_criterion = min_criterion


    def train(self, X_train, y_train):
        """ Train model
        """

        self.root = Tree(self.max_depth)
        self.root.build(X_train, y_train, self.criterion)


    def predict(self, y_test):
        """ Make prediction
        """

        return numpy.array([self.root.predict(f) for f in y_test])

    def show_tree(self):
        self.root.show_tree(0, ' ')

    def accuracy_metric(self, actual, predicted):
        """ Calculate accuracy percentage
        """

        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0
