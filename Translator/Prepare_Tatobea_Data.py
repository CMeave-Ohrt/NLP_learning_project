import pandas as pd
import pickle
from Tokenization_for_Translation import Corpus
import numpy as np

path = './Data/deu.txt'

file = open(path, 'rb')

text = file.read().decode('utf-8', 'replace')

German_corpus = Corpus()
English_corpus = Corpus()

punctuation = {'.', '!', '?', ':', "'", ',', ';', '"', '-'}
German_corpus.add_punctuation(punctuation)
English_corpus.add_punctuation(punctuation)

lines = text.split('\n')

German_lines = []
English_lines = []

for k, line in enumerate(lines):
    English, German, _ = line.split('\t', 2)
    German_lines.append(German_corpus.START + ' ' + German + ' ' + German_corpus.STOP)
    English_lines.append(English_corpus.START + ' ' + English + ' ' + English_corpus.STOP)

data = pd.DataFrame()
data['German'] = German_lines
data['English'] = English_lines

#randomize data
ran = np.arange(len(data))
np.random.shuffle(ran)

data_shuffled = data.iloc[ran]
data_shuffled = data_shuffled.reset_index(drop=True)

data_train = data_shuffled.iloc[:200000]
data_test = data_shuffled.iloc[200000:]
data_test = data_test.reset_index(drop=True)

file = open('./Data/T2_data_train.pt', 'wb')
pickle.dump(data_train, file)

file = open('./Data/T2_data_test.pt', 'wb')
pickle.dump(data_test, file)

German_text = ' '.join(data_train['German'].values)
German_text = German_text.replace(German_corpus.START, '')
German_text = German_text.replace(German_corpus.STOP, '')

English_text = ' '.join(data_train['English'].values)
English_text = English_text.replace(English_corpus.START, '')
English_text = English_text.replace(English_corpus.STOP, '')

print('Making German corpus')
German_corpus.make_vocab(German_text, 25000)
German_corpus.save_vocab('./Parameters/German_corpus25k.pt')

print('Making English corpus')
English_corpus.make_vocab(English_text, 15000)
English_corpus.save_vocab('./Parameters/English_corpus15k.pt')

print('All done')
