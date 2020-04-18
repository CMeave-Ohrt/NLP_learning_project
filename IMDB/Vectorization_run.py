from Tokenization import Corpus
import pandas as pd
import numpy as np
import os
from Vectorization import WordsToVec
import torch
import pickle

#Runs vectorization and saves parameters

my_corpus = Corpus()

#load prepared vocab
my_corpus.load_vocab('./Parameters/movies_20k_28cut.pt')
my_corpus.add_stop_words({'.', '!', '?'})

my_converter = WordsToVec(my_corpus, embed_size=300)

#load training dat
path = '../Data/train/neg'

input_data = []

for i, file_name in enumerate(os.listdir(path)):
    with open(path + '/' + file_name, 'rb') as file:
        txt = file.read()
    input_data.append(str(txt))

    if i % 1000 == 0:
        print('Done with entry {}, batch 1'.format(i))


path = '../Data/train/pos'

for i, file_name in enumerate(os.listdir(path)):
    with open(path + '/' + file_name, 'rb') as file:
        txt = file.read()
    input_data.append(str(txt))

    if i % 1000 == 0:
        print('Done with entry {}, batch 2'.format(i))


#pre-process data
data = my_converter.pre_process(input_data, 5)

#save pre-processed data to save times on later iterations
#file = open('./Parameters/training_set_preprocessed.pt', 'wb')

#pickle.dump(data, file)

print(data.shape)

#train and save model
my_converter.train_word_vects(data, epochs=4, batch_size=1000)

my_converter.save_model_state('./Parameters/model_300d.pt')

