from Tokenization_for_Textgen import Corpus
from Vectorization import WordsToVec
from Text_Generator import TextGenerator

#train model or load model and generate samples

#read in file
path = './Data/German_Bible.txt'
file = open(path, 'rb')
theText = file.read().decode('utf-8', 'replace')

theText = theText.replace('\r\n', ' \n ')

# load corpus
my_corpus = Corpus()
my_corpus.load_vocab('./Parameters/German_bible_corpus.pt')

# load vectorizer
my_vectorizer = WordsToVec(my_corpus, 200)
my_vectorizer.load_model_state('./Parameters/German_bible_vectors_200d.pt')

my_generator = TextGenerator(my_corpus, my_vectorizer, LSTM_count=4)

my_generator.load_model('./Parameters/text_generator_autosave_epoch20.pt')

print(my_generator.unparse(my_generator.generate_with_beam('he', 1000, beam_width=100, creativity=1, window=4)))

#my_generator.train_model(theText, epochs=200, lr=0.001)
