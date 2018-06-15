"""
    This script contains a class mainly for preprocessing the data and return 
"""

from collections import Counter
from slashml.preprocessing.read_content import ReadContent
from slashml.utils.file_util import FileUtil

class Preprocessing(object):
    """"
        This is the main class
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def loading_data(self, folders, feature_choice, threshold):
        """ Loading the data from txt files in folders"""
        content = ReadContent(**self.kwargs)
        _words_articles, _all_words = content.load_content(folders)
        _temp_all_words = _all_words
        _temp_all_words.append(feature_choice)
        _temp_all_words.append(threshold)

        if feature_choice == 'doc_freq':
            _tfidf = self.doc_frequency(_words_articles, _all_words, feature_choice, threshold)
        return _tfidf

    def doc_frequency(self, words_articles, all_words, feature_choice, threshold):
        """
            This method aims at feature selection based
            on terms appearing in the articles.
            Select only terms that appear in more threshold of articles
        """

        _tfidf_mat = []
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
        _words_class = self.compute_feature_matrix(_selected_words,\
                            words_articles, feature_choice, threshold)
        _tfidf_mat.append(_words_class)
        _tfidf_mat = self.merge_feature_vec(_tfidf_mat)

        # Save the dictionary for single document prediction
        FileUtil.save_pickle_dataset(self.kwargs, self.kwargs['bag_of_words']+\
                                '/'+feature_choice+'_'+str(threshold)+'.pickle', _selected_words)

        return _tfidf_mat

    def merge_feature_vec(self, feature_mat):
        """Merging list of vectors"""
        _feature_mat = []
        for feature in feature_mat:
            for feature_vec in feature:
                _feature_mat.append(feature_vec)
        return _feature_mat

    def compute_feature_matrix(self, word_in_dic, text_in_articles, feature_choice, threshold):
        """ Computing the feature matrix """
        mat = []# feature matrix

        for index in range(len(text_in_articles)): # each class
            for idx_article in range(len(text_in_articles[index])):
                row = []
                dic = Counter(text_in_articles[index][idx_article])\
                # Count the frequency of each term

                for words in word_in_dic:\
                    # each term or feature in article to be considered
                    #print(dic[words])
                    row.append(dic[words])# Adding to row
                row.append(index)
                mat.append(row)

        # Write to .csv file
        _directory_name = FileUtil.join_path(self.kwargs, self.kwargs['dataset'])
        self.write_mat(_directory_name, feature_choice, threshold, mat)
        #self.write_mat('D:/ML_text_classification/New folder/slashml_main/data/dataset/matrix/',\
        #            feature_choice, threshold, mat)

        # returning feature matrix
        return mat

    def write_mat(self, path_to_file, feature_choice, threshold, feature_mat):
        """ Method to write matrix into a file
        """
        _mat_file = open(path_to_file + '/' + feature_choice+'_'+str(threshold)+'.csv', "w+")

        for _feature in feature_mat:
            _temp = str(_feature)
            _temp = _temp.strip(']')
            _temp = _temp.strip('[')
            _mat_file.write(_temp + '\n')

        _mat_file.close()

    def loading_single_doc(self, document, feature_choice, threshold):
        """ Loading single document for prediction
        """

        content = ReadContent(**self.kwargs)
        if feature_choice == 'doc_freq':
            dic_load = FileUtil.load_pickle(self.kwargs, self.kwargs['bag_of_words']+\
                                          '/'+feature_choice+'_'+str(threshold)+'.pickle')

        document = content.remove_stopword(document)
        article = content.stemming_words(document)
        words = Counter(article)# Count the frequency of each term
        row = []
        for word in dic_load:\
            # each term or feature in article to be considered
            row.append(words[word])# Adding to row
        #print(dic_load[len(dic_load)-1])

        return row


if __name__ == "__main__":

    CONFIG = {
        'root': 'D:/ML_text_classification/New folder/slashml_main',
        'model_dataset': 'data/dataset',
        'dataset': 'db.khmer.json',
        'train_model': 'data/naive_bayes_model.pickle',
        'train_dataset': 'data/train_dataset.pickle',
        'test_dataset': 'data/test_dataset.pickle',
        'bag_of_words': 'data/dataset/bag_of_words',
        'text_dir': 'data/dataset/text',
        'mode': 'unicode'
    }

    TEST = Preprocessing(**CONFIG)
    print(TEST.loading_data(CONFIG['text_dir'], 'doc_freq', 25))
