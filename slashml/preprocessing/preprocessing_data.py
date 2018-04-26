"""
    This script contains a class mainly for preprocessing the data and return 
"""

from collections import Counter
from slashml.preprocessing.read_content import ReadContent

class Preprocessing(object):
    """"
        This is the main class
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def loading_data(self, folders, choice, threshold):
        """ Loading the data from txt files in folders"""
        content = ReadContent(**self.kwargs)
        _words_articles, _all_words = content.load_content(folders)

        if choice == 'doc_freq':
            _tfidf = self.doc_frequency(_words_articles, _all_words, threshold)
        return _tfidf

    def doc_frequency(self, words_articles, all_words, threshold):
        """
            This method aims at feature selection based
            on terms appearing in the articles.
            Select only terms that appear in more threshold of articles
        """

        _tfidf_mat = []
        _dic_in_class = []
        _selected_words = []
        for word in all_words:
            num_found = 0
            for i in range(len(words_articles)): # each class
                for words_class in words_articles[i]: # all words in each class
                    if word in words_class:
                        num_found += 1
                if num_found > threshold:
                    # Consider only terms which appear in
                    # documents more than a threshold
                    _selected_words.append(word)
        _selected_words = list(set(_selected_words))
        _words_class = self.compute_feature_matrix(_selected_words, words_articles)
        _tfidf_mat.append(_words_class)
        _tfidf_mat = self.merge_feature_vec(_tfidf_mat)
        return _tfidf_mat

    def merge_feature_vec(self, feature_mat):
        """Merging list of vectors"""
        _feature_mat = []
        for feature in feature_mat:
            for feature_vec in feature:
                _feature_mat.append(feature_vec)
        return _feature_mat

    def compute_feature_matrix(self, word_in_dic, text_in_articles):
        """ Computing the feature matrix """
        mat = []# feature matrix

        for index in range(len(text_in_articles)): # each class
            for idx_article in range(len(text_in_articles[index])):
                row = []
                dic = Counter(text_in_articles[index][idx_article])# Count the frequency of each term

                for words in word_in_dic:\
                    # each term or feature in article to be considered
                    #print(dic[words])
                    row.append(dic[words])# Adding to row
                row.append(index)
                mat.append(row)

        return mat# returning feature matrix
    
    def write_mat(self, path_to_file, feature_mat):
        """ Method to write matrix into a file
        """
        _mat_file = open(path_to_file, "w+")

        for _feature in feature_mat:
            _temp = str(_feature)
            _temp = _temp.strip(']')
            _temp = _temp.strip('[')
            _mat_file.write(_temp + '\n')

        _mat_file.close()

if __name__ == "__main__":

    config = {
        'root': '/Users/lion/Documents/py-workspare/slash-ml/slashml',
        'model_dataset': 'data/dataset',
        'dataset': 'db.khmer.json',
        'train_model': 'data/naive_bayes_model.pickle',
        'train_dataset': 'data/train_dataset.pickle',
        'test_dataset': 'data/test_dataset.pickle',
        'text_dir': 'data/dataset/text',
        'mode': 'unicode'
    }

    test = Preprocessing(**config)
    print(test.loading_data(config['text_dir'], 'doc_freq', 15))