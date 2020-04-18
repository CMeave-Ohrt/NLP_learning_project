from Tokenization import Corpus
import numpy as np
import os
from Vectorization import WordsToVec
from Sentiment_Analysis import SentimentAnalysis, save_model_state, load_model_state

#Run sentiment analysis using saved corpus and vectorization, save parameters

my_corpus = Corpus()

my_corpus.load_vocab('./IMDB/Parameters/movies_20k_28cut.pt')
my_corpus.add_stop_words({'.', '!', '?'})

my_converter = WordsToVec(my_corpus, embed_size=300)

path = './IMDB/Parameters/model_300d.pt'

my_converter.load_model_state(path)

my_analyzer = SentimentAnalysis(my_corpus, my_converter, train_embedding=True)

#load data

path = './IMDB/Data/train/neg'

input_data = []

for i, file_name in enumerate(os.listdir(path)):
    with open(path + '/' + file_name, 'rb') as file:
        txt = file.read()
    input_data.append(str(txt))

    if i % 1000 == 999:
        print('Done with entry {}, loading batch 1'.format(i+1))




targets_neg = np.zeros(len(input_data))

path = './IMDB/Data/train/pos'

for i, file_name in enumerate(os.listdir(path)):
    with open(path + '/' + file_name, 'rb') as file:
        txt = file.read()
    input_data.append(str(txt))

    if i % 1000 == 999:
        print('Done with entry {}, loading batch 2'.format(i+1))
        



targets_pos = np.ones(len(input_data)-len(targets_neg))

#create target data
targets = np.concatenate((targets_neg, targets_pos))

#train model and save
my_analyzer.gru_train(input_data, targets, epochs=50, batch_size=50)

print('Training error at exit:')
print(my_analyzer.gru_test(input_data, targets))

#save_model_state('./Parameters/analyzerGRU.pt', model=my_analyzer.GRU)
