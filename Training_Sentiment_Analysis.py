from Tokenization import Corpus
import pandas as pd
import numpy as np
import os
from Vectorization import WordsToVec
import torch
import pickle
from Sentiment_Analysis import SentimentAnalysis

#Run sentiment analysis using saved corpus and vectorization, save parameters

my_corpus = Corpus()

my_corpus.load_vocab('./Parameters/movies_20k_28cut.pt')
my_corpus.add_stop_words({'.', '!', '?'})

my_converter = WordsToVec(my_corpus, embed_size=100)

path = './Parameters/model_100d.pt'

my_converter.load_model_state(path)

my_analyzer = SentimentAnalysis(my_corpus, my_converter)

#load data

path = './Data/train/neg'

input_data = []

for i, file_name in enumerate(os.listdir(path)):
    with open(path + '/' + file_name, 'rb') as file:
        txt = file.read()
    input_data.append(str(txt))

    if i % 1000 == 0:
        print('Done with entry {}, loading batch 1'.format(i))

targets_neg = np.zeros(len(input_data))

path = './Data/train/pos'

for i, file_name in enumerate(os.listdir(path)):
    with open(path + '/' + file_name, 'rb') as file:
        txt = file.read()
    input_data.append(str(txt))

    if i % 1000 == 0:
        print('Done with entry {}, loading batch 2'.format(i))

targets_pos = np.ones(len(input_data)-len(targets_neg))

#create target data
targets = np.concatenate((targets_neg, targets_pos))

#train model and save
my_analyzer.train_model(input_data, targets, epochs=20, batch_size=1000)

my_analyzer.save_model_state('./Parameters/analyzer.pt')
