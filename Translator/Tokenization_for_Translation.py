
import pandas as pd
import numpy as np
import pickle


class Corpus:
    """Create and manage language corpus. Maintains three sets vocab, filler_words, stop_words together with dictionaries
    word_to_ind and ind_to_word translating elements of vocab to indices and vice versa"""

    def __init__(self):
        self.STOP = '#STOP'
        self.UNKNOWN = '#UNKNOWN'
        self.START = '#START'
        self.vocab = set()
        self.stop_words = set()
        self.punctuation = set()
        self.word_to_ind = dict()
        self.ind_to_word = dict()

    def add_stop_words(self, words):
        """input: Set of words
        These words will be added to the self.stop_words"""
        self.stop_words = self.stop_words.union(words)

    def add_punctuation(self, words):
        """input: Set of words
        These words will be added to self.punctuation"""
        self.punctuation = self.punctuation.union(words)

    def make_vocab(self, text, target_size):
        """Makes vocabulary (vocab always includes punctuation and stop and unknown token)
        Text is analyzed, set of at least target_size is added to self.vocab
        Find good numbers for top_cut using self.analyze(text)"""

        self.vocab = {self.STOP, self.UNKNOWN, self.START}.union(self.punctuation)

        tokens_sorted = self.analyze(text)

        #find frequency of target_size'th token and include all tokens with at least that frequency
        count_cut = tokens_sorted['count'][target_size]
        tokens_final = tokens_sorted[tokens_sorted['count'] >= count_cut]

        #create dictionaries

        # add stop, unknown, and start to dictionaries
        self.word_to_ind[self.START] = 2
        self.ind_to_word[2] = self.START
        self.word_to_ind[self.STOP] = 1
        self.ind_to_word[1] = self.STOP
        self.word_to_ind[self.UNKNOWN] = 0
        self.ind_to_word[0] = self.UNKNOWN

        # add punctuation to dictionaries
        for i, token in enumerate(self.punctuation):
            self.word_to_ind[token] = i + 3
            self.ind_to_word[i + 3] = token

        length = len(self.word_to_ind)

        #add all words
        for i, token in enumerate(tokens_final['Token']):
            self.word_to_ind[token] = i+length
            self.ind_to_word[i+length] = token


        #set vocab
        self.vocab = self.vocab.union(set(tokens_final.Token.to_numpy().flatten()))

    def analyze(self, data):
        """Returns data frame listing all words by count. Makes everything lower case and separates by punctuation.
        Output is dataframe with index 0, 1, ... and columns 'Token' and 'count' sorted by 'count' descending"""

        #replace all punctuation with ' '
        data_replaced = data.lower()
        for word in self.punctuation:
            data_replaced = data_replaced.replace(word, ' ')

        tokens = pd.DataFrame()

        #split text along white spaces:
        tokens['Token'] = pd.Series(data_replaced.split())

        # tally word counts
        tokens['count'] = 1
        tokens_grouped = tokens.groupby('Token').sum()

        tokens_sorted = tokens_grouped.sort_values(by='count', ascending=False)
        result = tokens_sorted.reset_index()

        return result

    def parse(self, text, use_UNKNOWN=False, use_STOP=False, use_punctuation=False, for_BLEU=False):
        """Takes text and converts it into numpy array of tokens.
        Unknown words will be transcribed as #UNKNOWN or dropped
        Stop words will be transcribed as #STOP or dropped
        Punctuation will be transcribed or dropped"""

        text_replaced = text.lower()
        #search and handle punctuation:
        for word in self.punctuation:
            if use_punctuation:
                text_replaced = text_replaced.replace(word, ' '+word+' ') #add padding around punctuation, so it will tokenize
            else:
                text_replaced = text_replaced.replace(word, ' ') #delete punctuation

        #re-capitalize tokens:
        text_replaced = text_replaced.replace(self.START.lower(), self.START)
        text_replaced = text_replaced.replace(self.STOP.lower(), self.STOP)

        split_text = text_replaced.split(' ')
        split_text = [token for token in split_text if token != '']

        raw_tokens = pd.DataFrame()
        raw_tokens['Token'] = pd.Series(split_text)

        # For BLEU score calculation, we're done here
        if for_BLEU:
            return split_text

        #convert stop words into STOP token if requested
        tokens_wt_stops = raw_tokens
        if use_STOP:
            tokens_wt_stops['Token'] = raw_tokens['Token'].map(lambda x: self.STOP if (x in self.stop_words) else x)

        if len(tokens_wt_stops) == 0:
            return np.array([])

        #transcribe Unkown tokens
        tokens_final = tokens_wt_stops.Token.map(
            lambda x: x if (x in self.vocab) else (self.UNKNOWN if use_UNKNOWN else np.NaN))
        tokens_final = tokens_final.dropna()
        tokens_final.reset_index(inplace=True, drop=True)

        return tokens_final.to_numpy()

    def parse_to_index(self, text, use_UNKNOWN=False, use_STOP=False, use_punctuation=False):
        """Takes in text and returns a list of indices, corresponding to the parsed text"""
        tokens = self.parse(text, use_UNKNOWN=use_UNKNOWN, use_STOP=use_STOP, use_punctuation=use_punctuation)

        result = [self.word_to_ind[token] for token in tokens]

        return result

    def save_vocab(self, path):
        """save vocab, filler words at path, directory must exist"""
        file = open(path, 'wb')

        to_save = (self.vocab, self.stop_words, self.punctuation, self.word_to_ind, self.ind_to_word)

        pickle.dump(to_save, file)

    def load_vocab(self, path):
        """load vocab, filler words from path"""
        file = open(path, 'rb')

        self.vocab, self.stop_words, self.punctuation, self.word_to_ind, self.ind_to_word = pickle.load(file)


