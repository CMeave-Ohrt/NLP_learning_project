from Tokenization import Corpus
import pandas as pd
import numpy as np
import os
from Vectorization import WordsToVec
import torch
import pickle
from Sentiment_Analysis import SentimentAnalysis

#tests sentiment analysis

my_corpus = Corpus()

my_corpus.load_vocab('./Parameters/movies_20k_28cut.pt')
my_corpus.add_stop_words({'.', '!', '?'})

my_converter = WordsToVec(my_corpus, embed_size=100)

path = './Parameters/model_100d.pt'

my_converter.load_model_state(path)

my_analyzer = SentimentAnalysis(my_corpus, my_converter)

my_analyzer.load_model_state('./Parameters/analyzer.pt')

path = './Data/test/neg'

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

path = './Data/test/pos'

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
print(my_analyzer.test(input_data, targets))
