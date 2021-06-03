# -*- coding: utf-8 -*-
import numpy 
import nltk
import glob
import os
import string

#read file for learning

lesFile = open('database\LesyaUkrainkaKonvalia.txt','r', encoding='utf-8')
firstText = lesFile.read()
#print(firstText)
lesFile.close()
firstText = firstText.replace("’",'').replace("‘", '').replace("’", '').replace("'", '').replace("–", ' ')

sergioFile = open('database\SergioZhadanSchoTiBudeshZhaduvati.txt','r', encoding='utf-8')
secondText = sergioFile.read()
#print(secondText)
sergioFile.close()
secondText = secondText.replace("’",'').replace("‘", '').replace("’", '').replace("'", '').replace("–", ' ')

#tokenization
nltk.download('punkt')

tokensFirst = nltk.word_tokenize(firstText)
#print(tokensFirst)

remove_punctuation = str.maketrans('', '', string.punctuation)
tokens_first = [x for x in [t.translate(remove_punctuation).lower() for t in tokensFirst] if len(x) > 0]
print(tokens_first)

tokensSecond = nltk.word_tokenize(secondText)
#print(tokensSecond)

remove_punctuation = str.maketrans('', '', string.punctuation)
tokens_second = [x for x in [t.translate(remove_punctuation).lower() for t in tokensSecond] if len(x) > 0]
print(tokens_second)