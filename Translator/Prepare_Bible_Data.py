import pandas as pd
import pickle
from Tokenization_for_Translation import Corpus

#!!!!!!!!NOT FUNCTIONAL!!!!!!!

#read in files

German_path = './Data/German_Bible.txt'
English_path = './Data/Bible.txt'

German_file = open(German_path, 'rb')
English_file = open(English_path, 'rb')

German_text = German_file.read()
English_text = English_file.read()

German_text = German_text.decode('utf-8', 'replace')
English_text = English_text.decode('utf-8', 'replace')

English_text = English_text.replace('\r\n\r\n\r\n', '#ENDCHAP')
English_text = English_text.replace('\r\n', ' ')

German_list = German_text.split('\r\n')

for k in range(len(German_list)):
    German_list[k] = German_list[k][German_list[k].find(' ')+1:]

English_list = []
bad_list=[]

for k in range(len(German_list)):
    #Find verse number and crop it off from German text
    split_verse = German_list[k].split(' ', 1)
    verse_num = split_verse[0]
    German_list[k] = split_verse[1]

    #find verse number in English text:
    eng_ind = English_text.find(verse_num)
    eng_stop = English_text.find('#ENDCHAP')

    if eng_ind > 1500:
        bad_list.append(k)
        continue

    if eng_stop == -1:
        eng_stop = float('inf')

    #append words between last verse and this verse/chapter end
    if k > 0:
        English_list.append(English_text[ : min(eng_stop, eng_ind)])

    #crop text
    English_text = English_text[eng_ind + len(verse_num) + 1:]

#drop bad verses


#add the last verse:
English_list.append(English_text)






