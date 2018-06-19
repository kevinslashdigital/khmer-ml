"""
    Chatbot class
"""

import sys, os
import argparse
sys.path.append(os.path.abspath(os.path.join('..', 'khmerml/slash-ml')))

from khmerml.machine_learning import MachineLearning
from khmerml.utils.file_util import FileUtil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,default ='chat',help='There are two mode (chat, train, train_c, test and none), The defaul value is chat.')
	# parser.add_argument("--benchmark", help="run benchmark",action="store_true")
	# parser.add_argument('--mode',type=float,default =0.2,help='There two mode?(train and chat)')
    args = parser.parse_args()

    config = {
        'root': '/Data/Projects/ML/khmerml/slash-ml',
        'text_dir': 'data/dataset/text',
        'dataset': 'data/dataset/matrix',
        'train_model': 'data/naive_bayes.model',
        # 'mode': 'unicode'
    }

    ml = MachineLearning(**config)
    # choose your algorithm
    algo = ml.NiaveBayes()
    # -- mode 
    if args.mode == 'train' :
        # preposessing
        # prepro = Preprocessing(**config)
        # dataset_matrix = prepro.loading_data(config['text_dir'], 'doc_freq', 15)

        #load dataset from file (feature data)
        filename = "data.csv"
        dataset_path = FileUtil.dataset_path(config, filename)
        dataset_sample = FileUtil.load_csv(dataset_path)

        # dataset -> train, test
        training_set, test_set = ml.split_dataset(dataset_sample, 2)
        
        # train or load model
        model = algo.train(training_set)
  
    elif args.mode == 'chat':
        print ("Start chatting with the bot !")
        algo.load_model()
        sessionid = 'Liza'
        while True:
            question 	= input('')
            algo.predict(question)
            # print(Bcolors().OKGREEN + 'You :' + Bcolors().ENDC,question)
            # print(Bcolors().OKBLUE 	+ 'Bot :' + Bcolors().ENDC,answer)
            # # save all chat to db
            # data = 'sessionid;;'+sessionid +';;question;;'+ question+ ';;response;;' + answer
            # save_conversation.save(data)
    elif args.mode == 'test':
        pass

