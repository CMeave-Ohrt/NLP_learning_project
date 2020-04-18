from Tokenization import Corpus
import numpy as np
import os
from Vectorization import WordsToVec
from Sentiment_Analysis import SentimentAnalysis, load_model_state, save_model_state

#tests sentiment analysis

my_corpus = Corpus()

my_corpus.load_vocab('./IMDB/Parameters/movies_20k_28cut.pt')
my_corpus.add_stop_words({'.', '!', '?'})

my_converter = WordsToVec(my_corpus, embed_size=300)

path = './IMDB/Parameters/model_300d.pt'

my_converter.load_model_state(path)

my_analyzer = SentimentAnalysis(my_corpus, my_converter, train_embedding=True)

load_model_state('./IMDB/Parameters/analyzerGRU_2layer_GRU_embed.pt', my_analyzer.GRU)

path = './IMDB/Data/test/neg'

#load 1000 pos and 1000 neg test reviews
input_data = []

for i, file_name in enumerate(os.listdir(path)):
    with open(path + '/' + file_name, 'rb') as file:
        txt = file.read()
    input_data.append(str(txt))

    if i % 1000 == 999:
        print('Done with entry {}, loading batch 1'.format(i))
        break

targets_neg = np.zeros(len(input_data))

path = './IMDB/Data/test/pos'

for i, file_name in enumerate(os.listdir(path)):
    with open(path + '/' + file_name, 'rb') as file:
        txt = file.read()
    input_data.append(str(txt))

    if i % 1000 == 999:
        print('Done with entry {}, loading batch 2'.format(i))
        break

targets_pos = np.ones(len(input_data)-len(targets_neg))

targets = np.concatenate((targets_neg, targets_pos))

#print accuracy
print(my_analyzer.gru_test(input_data, targets))
