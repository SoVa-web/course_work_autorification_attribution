# -*- coding: utf-8 -*-
from nltk.util import pr
import numpy 
import nltk
import glob
import os
import string
from array import *
import csv

#array with file`s index
filesIndex = ['11', '12', '13', '14', '15', '16', '17', '18',
            '19', '110', '111', '01', '02', '03', '04', '05', '06',
            '07', '08', '09', '010']

#create csv file with dataset for analitics

open('datasetWithMetrics.csv', 'w').close()
for file in filesIndex :

    #reading start files

    nameFile = "database\\"+file+".txt" 
    print(nameFile)
    fileForClean = open(nameFile,'r', encoding='utf-8')
    textForClean = fileForClean.read()
    fileForClean.close()
    textForClean = textForClean.replace("’",'').replace("‘", '').replace("’", '').replace("'", '').replace("–", ' ').replace("«", ' ').replace("»", ' ')

    #tokenize
    nltk.download('punkt')

    tokensFile = nltk.word_tokenize(textForClean)
    #print(tokensFile)

    delPunctuation = str.maketrans('', '', string.punctuation)
    tokensWithoutPunct = [x for x in [token.translate(delPunctuation).lower() 
                                for token in tokensFile]
                                if len(x) > 0]
    print(tokensWithoutPunct)

    #defining author's style metrics

    #lexical diversity of the text

    textDev = nltk.Text(tokensWithoutPunct)
    lexicalDiversity = (len(set(textDev)) / len(textDev)) * 100 
    print("Lexical diversity : "+str(lexicalDiversity))

    #average lenght of word

    words = set(tokensWithoutPunct)
    charNum = [len(word) for word in words]
    averageNumChar = sum(charNum) / float(len(charNum))
    print("Average length words : " + str(averageNumChar))

    #average length of sentence

    sentencesList = nltk.sent_tokenize(textForClean)
    wordNum = [len(sent.split()) for sent in sentencesList]
    averageNumWord = numpy.mean(wordNum)
    print("Average length sentences : "+str(averageNumWord))

    with open('datasetWithMetrics.csv', 'a', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter='~', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([file, lexicalDiversity, averageNumChar, averageNumWord, file[0] ])