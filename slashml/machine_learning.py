import random
from slashml.algorithms.bayes.naive_bayes import NaiveBayes

class MachineLearning(object):

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        # print('MachineLearning init',kwargs)

    # Naivae Bayes Algorithm
    def NiaveBayes(self):
        return NaiveBayes(**self.kwargs)

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
