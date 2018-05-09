"""
    Naive Bayes Probability
"""
import math
from decimal import Decimal

class BayesBase(object):
    """
        Naive Bayes class
    """

    def __init__(self, **kwargs):

        self.kwargs = kwargs

        self._train_model = {}
        self.predictions = []

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


    def count_classes_occurrence(self, dataset):
        """ Count class occurences from list data set
        by class
        """

        classes_occurrence = {}

        for _, label in enumerate(dataset):
            classes_occurrence[label] = len(dataset[label])

        return classes_occurrence


    def calculate_class_priori(self, dataset_classes):
        """ Calculate occurences depending on occurences
        """

        total_frequency = sum(dataset_classes.values())
        probabilities = {}

        for label, value in dataset_classes.items():
            probabilities[label] = value / total_frequency

        return probabilities


    def calculate_priori(self, dataset):
        """ Calculate Priori Probability
        """

        classes_occurrence = self.count_classes_occurrence(dataset)
        prioris = self.calculate_class_priori(classes_occurrence)

        return prioris


    def calculate_likelihood(self, dataset):
        """ Calculate likelihoods
        """

        # zip feature by class
        #dataset_by_class = cls.classify_dataset_by_class(dataset)
        dataset_by_class = dict(dataset)

        likelihoods = {}
        for class_key, subset in dataset_by_class.items():
            #if class_key not in likelihoods:
            #likelihoods[class_key] = []

            zip_feature = zip(*subset)
            features = list(map(sum, zip_feature))
            del features[-1]

            total_unique_feat = len(features)
            total_freq = sum(features)

            # Calculate likelihood of each feature
            probabilities = [(1+ f_count)/(total_unique_feat + total_freq) \
                            for f_count in features]

            # Store likelihood by class
            likelihoods[class_key] = probabilities

        return likelihoods


    def train(self, dataset):
        """ Train model
        """

        # Calculate class priori
        # Calculate likelihood of every feature per class
        prioris = self.calculate_priori(dataset)
        likelihoods = self.calculate_likelihood(dataset)

        train_model = {}
        for class_key, likelihood in likelihoods.items():
            priori = prioris[class_key]

            if class_key not in train_model:
                train_model[class_key] = []

            train_model[class_key].append(priori)
            train_model[class_key].append(likelihood)

        self.train_model = train_model

        return self


    def calculate_posteriori(self, train_model, test_vector):
        """ Calculate the porbability of all classes
            one class at a time.
        """

        best_posteriori, label = -1, None

        for class_index, priori_likelihood in train_model.items():
            priori = Decimal(priori_likelihood[0])
            likelihood_feature = priori_likelihood[1]
            _likelihoods = list(map(lambda x, y: math.pow(x, y), \
                            likelihood_feature, test_vector))

            from functools import reduce
            likelihood = reduce(lambda x, y: Decimal(x) * Decimal(y), _likelihoods)

            posteriori = priori * likelihood

            if label is None or posteriori > best_posteriori:
                best_posteriori = posteriori
                label = class_index

        return best_posteriori, label

    @classmethod
    def get_prediction_by_class(cls, predictions):
        """ get max prob per class
        """

        results = {}
        for class_key, probabilities in predictions.items():
            results[class_key] = max(probabilities)

        return results


    def predict(self, test_dataset):
        """ Make prediction
        """

        predictions = []
        prediction_details = {}

        import copy
        test_sample = copy.deepcopy(test_dataset)

        for subset in test_sample:
            # remove label from test dataset
            del subset[-1]
            _, label = self.calculate_posteriori(self.train_model, subset)
            ''' if best_label is None or posteriori > best_prob:
                best_prob = posteriori
                best_label = label'''
            ''' if label not in predictions:
                predictions[label] = [] '''

            predictions.append(label)

        self.predictions = predictions

        return predictions, prediction_details


    def accuracy(self, test_set):
        """ Get Accuracy
        """

        correct = 0
        for index, _ in enumerate(test_set):
            if test_set[index][-1] == self.predictions[index]:
                correct += 1

        return round((correct / float(len(test_set))) * 100.0, 2)

    @property
    def train_model(self):
        """
            I'm the 'train_model' property.
        """
        return self._train_model

    @train_model.setter
    def train_model(self, value):
        self._train_model = value
