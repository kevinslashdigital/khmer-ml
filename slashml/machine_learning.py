import random
import numpy

from slashml.algorithms.bayes.naive_bayes import NaiveBayes
from slashml.algorithms.decisiontree.decision_tree import DecisionTree
class MachineLearning(object):

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        # print('MachineLearning init',kwargs)

    # Naivae Bayes Algorithm
    def NiaveBayes(self):
        return NaiveBayes(**self.kwargs)
    
    # Decision Tree Algorithm
    def DecisionTree(self,criterion='gini', prune='depth', max_depth=3, min_criterion=0.05):
        self.kwargs['criterion'] = criterion
        self.kwargs['prune'] = prune
        self.kwargs['max_depth'] = max_depth
        self.kwargs['min_criterion'] = min_criterion
        return DecisionTree(**self.kwargs)

    # Another Algorithms

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


    def train_test_split(self, dataset, n_test_by_class=2):
        """ Split data set for training and testing
        """

        # Sort dataset following descendant label
        # dataset[:, -1].argsort # Sort the last field (column)
        # dataset = dataset[dataset[:,1].argsort(kind='mergesort')]
        sorted_dataset = dataset[dataset[:, -1].argsort()]

        # Get unique labels
        y_all = numpy.unique(sorted_dataset[:, -1])

        # Random row indices
        random_row_indices = None

        for _, label_val in enumerate(y_all):
            # Find row index where label value equals to given label_value
            search_indices = numpy.where(sorted_dataset[:, -1] == label_val)

            # Selected number of test sample equals to number of test sample
            random_idx = numpy.random.randint(search_indices[0][0], search_indices[0][-1], size=n_test_by_class)
            #mask = numpy.random.choice([False, True], lenght, p=[probability, 1-probability])
            # Merge array random row into one array
            if random_row_indices is None:
                random_row_indices = random_idx
            else:
                random_row_indices = numpy.hstack((random_row_indices, random_idx))

        # Split
        # Create mask boolean array
        mask = numpy.ones(len(sorted_dataset), dtype=bool)
        mask[random_row_indices, ] = False
        # instead of a[b] you could also do a[~mask]
        X_test, X_train = sorted_dataset[random_row_indices], sorted_dataset[mask]

        # Get train and test label
        #y_train = X_train[:, -1]
        #y_test = X_test[:, -1]

        # Delete last column (label) from array
        #X_train = numpy.delete(X_train, -1, 1)
        #X_test = numpy.delete(X_test, -1, 1)

        # Return list
        #return X_train, X_test, y_train, y_test
        return X_train, X_test


    def accuracy(self, predictions, test_set):
        """ Get Accuracy
        """

        correct = 0
        for index, _ in enumerate(test_set):
            if test_set[index][-1] == predictions[index]:
                correct += 1

        return round((correct / float(len(test_set))) * 100.0, 2)
    