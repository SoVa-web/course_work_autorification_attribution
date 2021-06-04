# -*- coding: utf-8 -*-
import numpy 
import nltk
import string
from array import *
import csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.naive_bayes import GaussianNB

#array with file`s index
filesIndex = ['11', '12', '13', '14', '15', '16', '17', '18',
            '19', '110', '111', '01', '02', '03', '04', '05', '06',
            '07', '08', '09', '010']

#create csv file with dataset for analitics

with open('datasetWithMetrics.csv', 'w', newline='') as csvfile :
    filewriter = csv.writer(csvfile, delimiter='~', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(["nameFile","lexicalDiversity", "averageNumChar", "averageNumWord", "frequencyComma", "frequencyAnd", "author" ])
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

    #frequency "and"
    fregNum = nltk.probability.FreqDist(nltk.Text(tokensFile))
    frequencyAnd = (fregNum["і"] * 1000) / fregNum.N()

    #frequency  comma
    fregNum = nltk.probability.FreqDist(nltk.Text(tokensFile))
    frequencyComma = (fregNum[","] * 1000) / fregNum.N()


    with open('datasetWithMetrics.csv', 'a', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter='~', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([file, lexicalDiversity, averageNumChar,  averageNumWord, frequencyComma, frequencyAnd, file[0] ])

# first method defonotion author of text by using stylistic metrics using the classifier

#learning classification tree
#read dataset that we created

with open('datasetWithMetrics.csv', 'r', encoding='utf-8') as file:
    data = csv.reader(file, delimiter='~')
    data = list(map(lambda e: e[0:], data))
    headers = data.pop(0)[:-1]

    #we divide the dimensions into independent and dependent ones by column names

    x = list(map(lambda x: x[:-1], data))
    y = [x[-1] for x in data]

#we divide the dataset into training and test data

x_set_train, x_set_test, y_set_train, y_set_test = train_test_split(x, y, test_size=0.4, shuffle = True, random_state = 42 )

#create and trainig tree

decisionTree = DecisionTreeClassifier(max_depth=7)
decisionTree = decisionTree.fit(x_set_train, y_set_train)

#testing tree
#predicting the value of Y on the test data
prediction = decisionTree.predict(x_set_test)
decisionTree.predict(list(map(lambda x: x[:-1], data)))

#print result
count = 0
for i in x_set_test :
    print(i)
    print(prediction[count])
    count+=1

#accuracy
print("Accuracy of tree:", metrics.accuracy_score(y_set_test, prediction))

#graphical construction of the tree
dot_data = StringIO()
export_graphviz(decisionTree, out_file=dot_data, filled=True, special_characters=True, feature_names=headers, class_names=['0','1'])
vizTree = pydotplus.graph_from_dot_data(dot_data.getvalue())
vizTree.write_png('decisionTree.png')
Image(vizTree.create_png())




# second method Bayesian classifier


with open('datasetWithMetrics.csv', 'r', encoding='utf-8') as file:
    data = csv.reader(file, delimiter='~')
    data = list(map(lambda e: e[0:], data))
    headers = data.pop(0)[:-1]

    #we divide the dimensions into independent and dependent ones by column names

    x = list(map(lambda x: x[:-1], data))
    y = [x[-1] for x in data]

#we divide the dataset into training and test data

x_set_train, x_set_test, y_set_train, y_set_test = train_test_split(x, y, test_size=0.25, shuffle = True, random_state = 42 )

#training Gaus model

gnb = GaussianNB()
gnb.fit(x_set_train, y_set_train)
y_pred = gnb.predict(x_set_test)

count = 0
for i in x_set_test :
    print(i)
    print(y_pred[count])
    count+=1

print("Accuracy of Gaus model :", metrics.accuracy_score(y_set_test, y_pred))

