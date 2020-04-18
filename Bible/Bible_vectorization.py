
from Tokenization_for_Textgen import Corpus
from Vectorization import WordsToVec

#run vectorization

#read in file
path = './Bible/Data/Bible.txt'
file = open(path, 'r')
theText = file.read()

#load corpus
my_corpus = Corpus()
my_corpus.load_vocab('./Bible/Parameters/bible_corpus.pt')

#make vectorizer
my_vectorizer = WordsToVec(my_corpus, 150)

data = [theText] #1 element list to conform to class specs

data = my_vectorizer.pre_process(data, 7)

my_vectorizer.train_word_vects(data)

my_vectorizer.save_model_state('./Bible/Parameters/bible_vectors_150d.pt')