#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import string
import matplotlib.pyplot as plt; 
from collections import Counter
from collections import OrderedDict

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# In[ ]:


dictionarySize = 3000
def downloadNews():
    news1 = pd.read_csv('../input/articles1.csv')
    news2 = pd.read_csv('../input/articles2.csv')
    news3 = pd.read_csv('../input/articles3.csv')
    news1.index = range(0,                               news1.shape[0])
    news2.index = range(news1.shape[0],                  news1.shape[0]+news2.shape[0])
    news3.index = range(news1.shape[0]+news2.shape[0],   news1.shape[0]+news2.shape[0]+news3.shape[0])
    news = pd.concat([news1, news2, news3])
    return news

def getWordList(someContent, tupleLength):
    t=tupleLength
    aList = []
    for i in range(0,tupleLength):
        aList.append(someContent[i:len(someContent)-(t-i)])
    trans = list(map(list, zip(*(aList))))
    return list(map(''.join,trans))

#Gets a dictionary of Agencies to a List of their articles and a list of the Agencies
def getAgenciesContent():
    news = downloadNews()
    Agencies = news.publication.unique()
    AgencyContent = {}
    for aAgency in Agencies:
        AgenRows = news.loc[news['publication'] == aAgency]
        AgencyContent[aAgency] = AgenRows['content']
    return AgencyContent

def getPredictionTuples(someContent, tupleLength):
    prediction = someContent[tupleLength:len(someContent)-1]
    wordList = getWordList(someContent, tupleLength)
    return tuple(map(tuple, zip(*[wordList, prediction])))
#returns tuples of words with their associated output

def getContentDictionary(someContent, tupleLength):
    otherChars = string.punctuation + ' ' + ''

    totalCount = Counter()
    bigWordList = []
    for aArticle in someContent:
        aSplitList = list(re.split('(\W)', aArticle.lower()))
        aCleanSplitList = [chunk for chunk in aSplitList if chunk not in otherChars]
        wordList = getWordList(aCleanSplitList, tupleLength)
#         print('type of wordList is ' + type(wordList).__name__)
#         print('type of wordList[0] is ' + type(wordList[0]).__name__)
        bigWordList.extend(wordList)
        aCount = Counter(wordList)
        totalCount = aCount + totalCount
    return totalCount, bigWordList
#Returns a dictionary of words to their count in all content given

def getSorted(aList):
    def getCount(item):
        return -item[1]
    return sorted(aList, key=getCount)
#Sorts diction of words->Counts by the count value returns a list

def plotWordDict(tempDict):
    tempDict = dict(tempDict)
    objects = tempDict.keys()
    y_pos = np.arange(len(objects))
    performance = tempDict.values()

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('%')
    plt.title('Word Frequency')

    plt.show()
    
def getEmbedding(wordCol, aWord):
    hotEncod = np.full((1, len(wordCol)), 0)
    if aWord in wordCol:
        index = wordCol.index(aWord)
        hotEncod[0][index] = 1
    else: 
        hotEncod[0][len(wordCol)-1] = 1
    return hotEncod

def getEmbeddingIndex(wordCol, aWord):
    if aWord in wordCol:
        return wordCol.index(aWord)
    else: 
        return len(wordCol) - 1

def getWordFromEmbeding(wordCol, aWordVector):
    return getWordFromIndex(wordCol, np.nonzero(aWordVector == 1)[1][0])

def getWordFromIndex(wordCol, wordIndex):
    if wordIndex < len(wordCol)-1:
        outWord = wordCol[wordIndex]
    else: 
        outWord = './RareWord'
    return outWord
print('ran')


# In[ ]:


# word = 'dog'
# aVector = getEmbedding(top, word)
# print(aVector)

# aIndex = getEmbeddingIndex(top, word)
# print(aIndex)

# aWord = getWordFromEmbeding(top, aVector)
# print(aWord)

# aWord = getWordFromIndex(top, aIndex)
# print(aWord)


# In[ ]:


#Gets a dictionary of Agencies to their respective word counts
def getTopWordLists(wordstogether, numberOfArticlesUsed):
    # wordstogether = 1 #Length of word tuples
    # numberOfArticlesUsed = 100
    wordFreq = {}    
    wordCount = {}
    punctCount = {}
    otherChars = string.punctuation + ' ' + ''

    AgencyWordFreqDict={}
    content={}
    AgenciesWithContent = getAgenciesContent()
    for aAgency in AgenciesWithContent:
        smallContent = (AgenciesWithContent[aAgency])[0:numberOfArticlesUsed]
        AgencyWordFreqDict[aAgency], content[aAgency]  = getContentDictionary(smallContent, wordstogether)
    print('Built Dictionaries')

    for aAgency in AgenciesWithContent:
        allDict = AgencyWordFreqDict[aAgency]
        punctDict = {}
        for k in allDict:
            if k in otherChars:
                punctDict[k] = allDict[k]

        totalCount = sum(allDict.values()) #Total number of seperated items
        punctCount = sum(punctDict.values()) #Total number of seperated items
        wordCount = totalCount - punctCount
        for k in allDict:
            allDict[k] = allDict[k]/wordCount
        wordFreq[aAgency] = allDict     

    totalCount = Counter() 
    for k,v in wordFreq.items():
        totalCount += Counter(v)
    for k,v in totalCount.items():
        totalCount[k] = v/len(wordFreq)

    wordVectorSize = dictionarySize # so rare/unknown word is indexed to 5000
    top = [x[0] for x in totalCount.most_common(wordVectorSize)]
    topWords = {}
    for anAgencyName in wordFreq:
        topWords[anAgencyName] = wordFreq[anAgencyName].most_common(wordVectorSize)
        topWords[anAgencyName] = [x[0] for x in topWords[anAgencyName]]
    return top, topWords, content,totalCount
    # topWords is a dictionary containing a list of top 5000~ words for each Agency
    # top is a list of the top 5000 most common words averaged accross each of the agencies

print('ran')


# In[ ]:


top, topByAgency, content, totalCount = getTopWordLists(1,200)
print('ran')


# In[ ]:


plotWordDict(totalCount.most_common(10))


# In[ ]:


print(topByAgency.keys())
print('ran')


# In[ ]:


agency = 'Breitbart'
# X = [getEmbeddingIndex(topByAgency[agency] , x) for x in content[agency]]
X = np.concatenate([getEmbedding(topByAgency[agency] , x) for x in content[agency]])

print('done')


# In[ ]:


# print(wordFreq.keys())
# plotWordDict(Counter(wordFreq['New York Times']).most_common(10))
# plotWordDict(Counter(wordFreq['Breitbart']).most_common(10))
# plotWordDict(Counter(totalCount).most_common(10))


# In[ ]:


batchSize = X.shape[0]
print(X.shape)

def next_batch(X, batchbegin, batchend, wordcount):
    unsummedarray = X[batchbegin : batchend]
    
    summedarraysize = np.subtract(unsummedarray.shape, (wordcount-1,0))
    summedarray = np.ndarray(shape=summedarraysize, dtype=float, order='F')
    numberOfIterations = unsummedarray.shape[0]-wordcount+1
    
    for i in range(numberOfIterations):
        summedarray[i] = np.sum(unsummedarray[i:i+wordcount], 0)
    return summedarray, X[(batchbegin+wordcount):(batchend+1)]

x,y = next_batch(X, 300, 400, 1)
x2, y2 = next_batch(X, 300, 400, 2)
# for i in range(0, 100): print(x[i].shape)
# print(np.linalg.norm(x[i]-y[i]))#print(sum(abs(x[i]-y[i])))

# print(x.shape)
# print(y.shape)
# print(x2.shape)
# print(y2.shape)
print(np.linalg.norm((x[-1:]-x2)))
# print(x)
# print(x2)


# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tensorflow as tf

FLAGS = None
wordcount = 25
trainingAmount = 80000
testingAmount = 5000
learningrate = 0.1
# Create the model
x = tf.placeholder(tf.float32, [None, dictionarySize])
y_ = tf.placeholder(tf.float32, [None, dictionarySize]) #true values

W = tf.Variable(tf.zeros([dictionarySize, dictionarySize]))
b = tf.Variable(tf.zeros([dictionarySize]))
y = tf.matmul(x, W) + b #predicted values

# Define loss and optimizer

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(learningrate).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for _ in range(1):
    batch_xs, batch_ys = next_batch(X, 0, trainingAmount, wordcount)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_xs, batch_ys = next_batch(X, trainingAmount, trainingAmount + testingAmount, wordcount)
print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))


# In[ ]:


x = (1, 2, 4, 7, 12, 25)
NewYorkTimesResults = [0.2052, .160032, 0.145487, 0.145975, 0.149729, 0.149518]
BreitbartResults = [0.1762, 0.141228,0.114869,0.112735,0.112848,0.114952]

objects = x
y_pos = np.arange(len(objects))
performance = NewYorkTimesResults
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Success')
plt.title('New York Times')
plt.xlabel('Number of Preceding Words Used to Predict')

plt.show()

objects = x
y_pos = np.arange(len(objects))
performance = BreitbartResults
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Success')
plt.xlabel('Number of Preceding Words Used to Predict')
plt.title('Breitbart')

plt.show()


# In[ ]:





# In[ ]:




