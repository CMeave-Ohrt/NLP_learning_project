
from Tokenization_for_Textgen import Corpus
from Vectorization import WordsToVec

#run vectorization

#read in file
path = './Bible/Data/German_Bible.txt'
file = open(path, 'rb')
theText = file.read().decode('utf-8', 'replace')

theText = theText.replace('\r\n', ' \n ')

#load corpus
my_corpus = Corpus()
my_corpus.load_vocab('./Bible/Parameters/German_bible_corpus.pt')

#make vectorizer
my_vectorizer = WordsToVec(my_corpus, 200)

data = [theText] #1 element list to conform to class specs

data = my_vectorizer.pre_process(data, 7)

my_vectorizer.train_word_vects(data)

my_vectorizer.save_model_state('./Bible/Parameters/German_bible_vectors_200d.pt')