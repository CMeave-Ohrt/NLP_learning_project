from Tokenization_for_Translation import Corpus
from Vectorization import WordsToVec
from Translator import Translator
import pickle

file = open('./Data/T2_data_train.pt', 'rb')
df = pickle.load(file)

German_corpus = Corpus()
English_corpus = Corpus()
German_corpus.load_vocab('./Parameters/German_corpus1k.pt')
English_corpus.load_vocab('./Parameters/English_corpus1k.pt')

German_vectorizer = WordsToVec(German_corpus, 150)
English_vectorizer = WordsToVec(English_corpus, 150)

German_vectorizer.load_model_state('./Parameters/German_vectors_150d')
English_vectorizer.load_model_state('./Parameters/English_vectors_150d')

my_translator = Translator(German_corpus, English_corpus, German_vectorizer, English_vectorizer,
                           dec_GRU_count=2, enc_GRU_count=2, hidden_size=128, use_feedforward=True)

my_translator.train(df, epochs=50, batch_size=32, lr=0.001, lr_reduce=0.1)
