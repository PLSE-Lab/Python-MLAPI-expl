# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# coding: utf-8

# In[1]:


# BIGRAM PROGRAM


# In[2]:


#Import libraries
from collections import OrderedDict
import heapq 
import pandas as pd
from nltk import word_tokenize
import string

# In[3]:


wt = word_tokenize
#memasukan data csv menggunakan pandas
data = pd.read_csv('../input/artikel.csv', encoding='latin-1')
data.head()


# In[4]:


data.tail()


# In[5]:


#tokenisasi menggunakan nltk
datatoken = data.apply(lambda row: wt(row['Berita']), axis=1)
#print(datatoken.head(10))


# In[6]:


#dari pandas ke list
datatoken = datatoken.values.tolist()
#print(datatoken)


# In[7]:


#join list of list
datatoken = sum(datatoken, [])
datatoken = [w.lower() for w in datatoken]
datatoken= [''.join(c for c in s if c not in string.punctuation) for s in datatoken]
datatoken = [s for s in datatoken if s]
#print(datatoken)


# In[8]:


listOfBigrams = []
bigramCounts = {}
unigramCounts = {}
nbyn = {}
for i in range(len(datatoken)):
	if i < len(datatoken) - 1:

			listOfBigrams.append((datatoken[i], datatoken[i + 1]))

			if (datatoken[i], datatoken[i+1]) in bigramCounts:
				bigramCounts[(datatoken[i], datatoken[i + 1])] += 1
			else:
				bigramCounts[(datatoken[i], datatoken[i + 1])] = 1

	if datatoken[i] in unigramCounts:
			unigramCounts[datatoken[i]] += 1
	else:
		unigramCounts[datatoken[i]] = 1


# In[9]:


valueprob = OrderedDict()
listOfProb = {}
for bigram in listOfBigrams:
    word1 = bigram[0]
    word2 = bigram[1]
    listOfProb[bigram] = (bigramCounts.get(bigram))/(unigramCounts.get(word1))  
    valueprob[word1+" "+word2] = listOfProb.get(bigram)
       
uniprob = {}
for unigram in datatoken:
    uniprob[unigram] = 1/ unigramCounts.get(unigram)


# In[10]:

"""  ADD ONE SMOTHING """ 
def addOneSmothing(listOfBigrams, unigramCounts, bigramCounts):
    listOfProb = {}
    cStar = {}
    for bigram in listOfBigrams:
        word1 = bigram[0]
        listOfProb[bigram] = (bigramCounts.get(bigram) + 1)/(unigramCounts.get(word1) + len(unigramCounts))
        cStar[bigram] = (bigramCounts[bigram] + 1) * unigramCounts[word1] / (unigramCounts[word1] + len(unigramCounts))
        return listOfProb, cStar
      
addOneSmothing(listOfBigrams, unigramCounts, bigramCounts)


# In[12]:
testset = []
testset = 'akan','membaik'
"""  PERPLEXITY """  
perplexity = 1
N = 0

for i in testset:
    if i in uniprob:
        N += 1
        perplexity = perplexity * (1/uniprob[i])
        
perplexity = pow(perplexity, 1/float(N))
#print("perplexity :"+str(perplexity))


# In[13]:


"""  PREDIKSI KATA YANG 	AKAN MUNCUL"""

matchedBigrams = []
checkForThisBigram = 'ada'
topDict = {}  
for bigram in listOfBigrams:
    if checkForThisBigram == bigram[0]:
            matchedBigrams.append(bigram[0]+" "+bigram[1])
         
            
#print(matchedBigrams)            
	
for singleBigram in matchedBigrams:
		topDict[singleBigram] = valueprob[singleBigram]

topBigrams = heapq.nlargest(5, topDict, key=topDict.get)
for b in topBigrams:
		print( b+" : "+str(topDict[b])+"\n")

#print(unigramCounts)    
#print(uniprob)
#print(listOfBigrams)

