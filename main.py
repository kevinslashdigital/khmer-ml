"""
    Factory class

"""
from slashml.machine_learning import MachineLearning
from slashml.preprocessing.preprocessing_data import Preprocessing
from slashml.utils.file_util import FileUtil

if __name__ == "__main__":

    config = {
        'root': '/Users/lion/Documents/py-workspare/openml/slash-ml',
        'text_dir': 'data/dataset/text',
        'dataset': 'data/dataset/matrix',
        'bag_of_words': 'data/dataset/bag_of_words',
        'train_model': 'data/naive_bayes.model',
        # 'mode': 'unicode'
    }
    # preposessing
    # prepro = Preprocessing(**config)
    # dataset_matrix = prepro.loading_data(config['text_dir'], 'doc_freq', 25)
    

    ml = MachineLearning(**config)

    #load dataset from file (feature data)
    filename = "doc_freq_25.csv"
    dataset_path = FileUtil.dataset_path(config, filename)
    dataset_sample = FileUtil.load_csv(dataset_path, use_numpy=True)

    # dataset -> train, test
    #training_set, test_set = ml.split_dataset(dataset_sample, 2)
    X_train, X_test = ml.train_test_split(dataset_sample, 2)

    # Get train and test label
    y_test = X_test

    # choose your algorithm
    # algo = ml.NiaveBayes()
    algo = ml.DecisionTree(criterion='gini', prune='depth', max_depth=50, min_criterion=0.05)
    # train or load model
    model = algo.train(X_train)
    # model = algo.load_model()
    # make a prediction
    predictions = algo.predict(model, X_test)
    acc = ml.accuracy(predictions, y_test)
 
    print('training_set', len(X_train))
    print('predictions, prediction_details', predictions, acc)
