
from Tokenization_for_Translation import Corpus
from Vectorization import WordsToVec
from Translator import Translator
import pickle

file = open('./Data/T2_data_test.pt', 'rb')
df = pickle.load(file)[:300]

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

my_translator.load_model('./Parameters/translator_feedforwardatt_epoch50.pt')

print(my_translator.test(df))


# batch, _ = my_translator.pre_processing(df[:200])
# batch = my_translator.batch_prep(batch[:10], target_as_text=True)

# my_translator.test_batch(batch)


