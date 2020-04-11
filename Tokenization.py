import nltk
import pandas as pd
import numpy as np
import os
import pickle


class Corpus:
    """Create and manage language corpus. Maintains thre sets vocab, filler_words, stop_words together with dictionaries
    word_to_ind and ind_to_word translating elements of vocab to indices and vice versa"""

    def __init__(self):
        self.min_len = 3  # must be smaller than 5
        self.STOP = '#STOP'
        self.UNKNOWN = '#UNKNOWN'
        self.vocab = set()
        self.stop_words = set()
        self.filler_words = set()
        self.word_to_ind = dict()
        self.ind_to_word = dict()

    def add_stop_words(self, words):
        """input: Set of words
        These words will be added to the self.stop_words"""
        self.stop_words = self.stop_words.union(words)

    def make_vocab(self, text, top_cut, target_size):
        """Adds to self.vocab and self.filler_words:
        Text is analyzed, any word shorter than min_len is dropped, and top_cut many top entries  are added to
        self.filler_words, then set of at least target_size is added to self.vocab
        Find good numbers for top_cut using self.analyze(text)"""

        self.vocab = {self.STOP, self.UNKNOWN}

        tokens_sorted = self.analyze(text)

        #drop frequent tokens and add to filler_words
        tokens_unnecessary = tokens_sorted.iloc[:top_cut]
        self.filler_words = self.filler_words.union(set(tokens_unnecessary.Token.to_numpy().flatten()))

        #find frequency of target_size'th token and include all tokens with at least that frequency
        tokens_necessary = tokens_sorted.iloc[top_cut:].reset_index()
        count_cut = tokens_necessary['count'][target_size]
        tokens_final = tokens_necessary[tokens_necessary['count'] >= count_cut]

        #create dictionaries
        for i, token in enumerate(tokens_final['Token']):
            self.word_to_ind[token] = i
            self.ind_to_word[i] = token

        #add stop and unknown to dictionaries
        i = len(tokens_final['Token'])
        self.word_to_ind[self.STOP] = i
        self.ind_to_word[i] = self.STOP
        self.word_to_ind[self.UNKNOWN] = i+1
        self.ind_to_word[i+1] = self.UNKNOWN

        #set vocab
        self.vocab = self.vocab.union(set(tokens_final.Token.to_numpy().flatten()))

    def analyze(self, data):
        """Returns data frame listing all words of length greater than self.min_len by word count.
        data can be text or data frame with column 'Text'
        Output is dataframe with index 0, 1, ... and columns 'Token' and 'count' sorted by 'count' descending"""

        if type(data) is not str:
            data['Token'] = data.Text.map(lambda x: nltk.word_tokenize(x))
            raw_tokens = data.explode('Token').reset_index()['Token']
        else:
            raw_tokens = pd.Series(nltk.word_tokenize(data))
        #raw_tokens is series of tokens indexed 0, 1, ...

        # drop short tokens
        tokens = pd.DataFrame()
        tokens['Token'] = (raw_tokens[raw_tokens.str.len() >= self.min_len])
        tokens.reset_index(drop=True, inplace=True)

        # tally word counts
        tokens['count'] = 1
        tokens_grouped = tokens.groupby('Token').sum()

        tokens_sorted = tokens_grouped.sort_values(by='count', ascending=False)
        result = tokens_sorted.reset_index()

        return result

    def parse(self, text, use_UNKNOWN=False, use_STOP=False):
        """Takes text and converts it into numpy array of tokens. Filler words will be dropped.
        Unknown words will be transcribed as #UNKNOWN or dropped
        Stop words will be transcribed as #STOP or dropped"""

        raw_tokens = pd.DataFrame()
        raw_tokens['Token'] = pd.Series(nltk.word_tokenize(text))

        #convert stop words into STOP token if requested
        tokens_wt_stops = raw_tokens
        if use_STOP:
            tokens_wt_stops['Token'] = raw_tokens['Token'].map(lambda x: self.STOP if (x in self.stop_words) else x)



        #drop short and filler words
        tokens_necessary = tokens_wt_stops[tokens_wt_stops.Token.map(
            lambda x: (x not in self.filler_words) & (len(x) >= self.min_len))]
        tokens_necessary.reset_index(inplace=True, drop=True)

        #transcribe Unkown tokens
        tokens_final = tokens_necessary.Token.map(
            lambda x: x if (x in self.vocab) else (self.UNKNOWN if use_UNKNOWN else np.NaN))
        tokens_final = tokens_final.dropna()
        tokens_final.reset_index(inplace=True, drop=True)

        return tokens_final.to_numpy()

    def parse_to_index(self, text, use_UNKNOWN=False, use_STOP=False):
        """Takes in text and returns a list of indices, corresponding to the parsed text"""
        tokens = self.parse(text, use_UNKNOWN=use_UNKNOWN, use_STOP=use_STOP)

        result = [self.word_to_ind[token] for token in tokens]

        return result

    def save_vocab(self, path):
        """save vocab, filler words at path, directory must exist"""
        file = open(path, 'wb')

        to_save = (self.vocab, self.filler_words, self.stop_words, self.word_to_ind, self.ind_to_word)

        pickle.dump(to_save, file)

    def load_vocab(self, path):
        """load vocab, filler words from path"""
        file = open(path, 'rb')

        self.vocab, self.filler_words, self.stop_words, self.word_to_ind, self.ind_to_word = pickle.load(file)


def append_to_data(path, list):
    """auxilary function to open files below, returns list of texts"""
    for file_name in os.listdir(path):
        with open(path + '/' + file_name, 'rb') as file:
            txt = file.read()
            list.append(str(txt))
    return list




if __name__ == '__main__':

    #Read in texts, create vocab, save vocab

    my_corpus = Corpus()

    print('Loading Text samples')

    text_data = pd.DataFrame()

    text_list=[]

    text_list = append_to_data('./Data/train/neg', text_list)
    print('done 1/2')
    text_list = append_to_data('./Data/train/pos', text_list)
    print('done loading')

    text_data['Text'] = pd.Series(text_list)

    print('making vocabulary')

    #about 20k words, dropped the 28 most frequent words
    my_corpus.make_vocab(text_data, 28, 20000)

    print('saving vocabulary')

    my_corpus.save_vocab('./Parameters/movies_20k_28cut.pt')

    print('Done, filler words are')
    print(my_corpus.filler_words)


















