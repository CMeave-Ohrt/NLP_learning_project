from Tokenization_for_Translation import Corpus
from Vectorization import WordsToVec
import pickle


file = open('./Data/T2_data_train.pt', 'rb')
df = pickle.load(file)

German_data = df['German']
English_data = df['English']

German_corpus = Corpus()
English_corpus = Corpus()
German_corpus.load_vocab('./Parameters/German_corpus25k.pt')
English_corpus.load_vocab('./Parameters/English_corpus15k.pt')

German_vectorizer = WordsToVec(German_corpus, 250)
English_vectorizer = WordsToVec(English_corpus, 250)

print('pre-processing')

German_data = German_vectorizer.pre_process(German_data, 7)
English_data = English_vectorizer.pre_process(English_data, 7)

print('vectorizing German data')
German_vectorizer.train_word_vects(German_data)
German_vectorizer.save_model_state('./Parameters/German_vectors_250d')

print('vectorizing English data')
English_vectorizer.train_word_vects(English_data)
English_vectorizer.save_model_state('./Parameters/English_vectors_250d')