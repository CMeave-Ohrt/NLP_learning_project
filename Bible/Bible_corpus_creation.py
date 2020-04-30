
from Tokenization_for_Textgen import Corpus

#run and generate corpus

#read in file
path = './Data/German_Bible.txt'
file = open(path, 'rb')
theText = file.read()

theText = theText.decode('utf-8', 'replace')

theText = theText.replace('\r\n', ' \n ')

print(theText[:1000])

#generate basic corpus
my_corpus = Corpus()
punctuation = {'.', '!', '?', ':', '\n', "'", ',', ';', '"'}
my_corpus.add_punctuation(punctuation)

df = my_corpus.analyze(theText)

print(df.describe())

my_corpus.make_vocab(theText, 10000)

print('parsing')

parsed_text = my_corpus.parse(theText, use_UNKNOWN=True, use_punctuation=True)

print(parsed_text[:100])

parsed = my_corpus.parse_to_index(theText, use_UNKNOWN=True, use_punctuation=True)

print(parsed[:100])


#saving

my_corpus.save_vocab('./Parameters/German_bible_corpus.pt')