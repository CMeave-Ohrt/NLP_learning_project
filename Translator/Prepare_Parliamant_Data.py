import pandas as pd
import pickle
from Tokenization_for_Translation import Corpus
import numpy as np

#read in files

German_path = './Data/europarl-v7.de-en.de'
English_path = './Data/europarl-v7.de-en.en'

German_file = open(German_path, 'rb')
English_file = open(English_path, 'rb')

German_text = German_file.read()
English_text = English_file.read()

German_text = German_text.decode('utf-8', 'replace')
English_text = English_text.decode('utf-8', 'replace')

German_corpus = Corpus()
English_corpus = Corpus()

punctuation = {'.', '!', '?', ':', "'", ',', ';', '"', '-'}
German_corpus.add_punctuation(punctuation)
English_corpus.add_punctuation(punctuation)

print('Making German corpus')
German_df = German_corpus.make_vocab(German_text, 20000)
German_corpus.save_vocab('./Parameters/German_corpus.pt')

print('Making English corpus')
English_df = English_corpus.make_vocab(English_text, 20000)
English_corpus.save_vocab('./Parameters/English_corpus.pt')

print('making lists')
German_list = German_text.split('\n')
for k in range(len(German_list)):
    German_list[k] = German_corpus.START + ' ' + German_list[k] + ' ' + German_corpus.STOP
English_list = English_text.split('\n')
for k in range(len(English_list)):
    English_list[k] = English_corpus.START + ' ' + English_list[k] + ' ' + English_corpus.STOP

data = pd.DataFrame()
data['German'] = German_list
data['English'] = English_list

#randomize data
ran = np.arange(len(data))
np.random.shuffle(ran)

data_shuffled = data.iloc[ran]
data_shuffled = data_shuffled.reset_index(drop=True)

data_train = data_shuffled.iloc[:120000]
data_test = data_shuffled.iloc[120000:140000]
data_test = data_test.reset_index(drop=True)

file = open('./Data/Parliament_data_train.pt', 'wb')
pickle.dump(data_train, file)

file = open('./Data/Parliament_data_test.pt', 'wb')
pickle.dump(data_test, file)


