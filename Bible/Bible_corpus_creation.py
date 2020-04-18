
from Tokenization_for_Textgen import Corpus

#run and generate corpus

#read in file
path = './Bible/Data/Bible.txt'
file = open(path, 'r')
theText = file.read()

#generate basic corpus
my_corpus = Corpus()
punctuation = {'.', '!', '?', ':', '\n', "'", ',', ';'}
my_corpus.add_punctuation(punctuation)

df = my_corpus.analyze(theText)

my_corpus.make_vocab(theText, 6000)

print('parsing')

parsed_text = my_corpus.parse(theText, use_UNKNOWN=True, use_punctuation=True)

print(parsed_text[:100])

parsed = my_corpus.parse_to_index(theText, use_UNKNOWN=True, use_punctuation=True)

print(parsed[:100])


#saving

my_corpus.save_vocab('./Bible/Parameters/bible_corpus.pt')