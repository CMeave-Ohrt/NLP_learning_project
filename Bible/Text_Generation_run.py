from Tokenization_for_Textgen import Corpus
from Vectorization import WordsToVec
from Text_Generator import TextGenerator

#train model or load model and generate samples

#read in file
path = './Data/Bible.txt'
file = open(path, 'r')
theText = file.read()

#load corpus
my_corpus = Corpus()
my_corpus.load_vocab('./Parameters/bible_corpus.pt')

#load vectorizer
my_vectorizer = WordsToVec(my_corpus, 150)
my_vectorizer.load_model_state('./Parameters/bible_vectors_150d.pt')

my_generator = TextGenerator(my_corpus, my_vectorizer)

#my_generator.load_model('./Parameters/text_generator_autosave_epoch5.pt')

#print(my_generator.unparse(my_generator.generate('jesus', 100, creativity=10)))

my_generator.train_model(theText)