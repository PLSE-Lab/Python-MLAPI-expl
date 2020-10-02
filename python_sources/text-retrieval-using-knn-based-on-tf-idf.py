#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import fnmatch, os
import re
import math

# parameters
# INPUT = "input.txt"
K = 10

def find(pattern='*', startdir=os.curdir):
    for (thisDir, subsHere, filesHere) in os.walk(startdir):
        # print(thisDir, subsHere, filesHere)
        for name in filesHere:
            if fnmatch.fnmatch(name, pattern):
                fullpath = os.path.join(thisDir,name)
                yield fullpath


def preprocess():
    '''
    preprocess dataset
    :return:
    '''
    listofFilename = [i for i in find(startdir=r'../input/20_newsgroups')]
    print(listofFilename[0])

    '''
    Text : {'fullpath of file':{'word0':tf-idf of word0,.....},.....}
    Words: {'word0':idf of word0,......}
    '''
    Text = dict()
    Words = dict()
    numberofText = len(listofFilename)
    for filename in listofFilename:
        file = open(filename, errors='ignore')
        Text[filename] = dict()
        for line in file.readlines():
            line = re.split(r'[^a-zA-Z]+', line.strip())
            for word in line:
                if word == '': continue
                word = word.lower()
                if word not in Text[filename]: Text[filename][word] = 0
                Text[filename][word] += 1
        count = Text[filename].__len__()
        for word in Text[filename]:
            Text[filename][word] /= count
            if word not in Words:
                Words[word] = 0
                Words[word] += 1
    for word in Words:
        Words[word] = math.log10(numberofText/Words[word])
    for text in Text:
        for word in Text[text]:
            Text[text][word] *= Words[word]
    dbfile = open('text-pickle', 'wb')
    pickle.dump(Text, dbfile)
    dbfile.close()
    dbfile = open('words-pickle', 'wb')
    pickle.dump(Words, dbfile)
    dbfile.close()
    print('compelete preprocess')
    
preprocess()


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../working"))
print(K)

# Any results you write to the current directory are saved as output.


# In[ ]:


def KNN(vector, Matrix, K):
    '''

    :param vector: query points
    :param Matrix: points in dataset
    :param K: K nearest neighbors to return
    :return:
    '''
    def sqr(x): return  x*x
    l = []
    for x in Matrix:
        y = vector.keys() | Matrix[x].keys()
        sum = 0
        for e in y:
            sum += sqr((vector[e] if e in vector else 0)-(Matrix[x][e] if e in Matrix[x] else 0))
        # dis = math.sqrt(sum)
        l.append([sum, x])

    l = sorted(l)
    answer = []
    for i in range(K):
        if i<len(l):
            answer.append(l[i])
    return answer


dbfile = open('../working/text-pickle', 'rb')
Text = pickle.load(dbfile)
dbfile.close()
dbfile = open('../working/words-pickle', 'rb')
Words = pickle.load(dbfile)
dbfile.close()
print(len(Text), len(Words))

v = dict()

# read input from command lines
s = input()
s = re.split(r'[^a-zA-Z]+', s.strip())
print(s)
for word in s:
    if word == '': continue
    word = word.lower()
    if word not in v: v[word] = 0
    v[word] += 1

# read input from file
# with open(INPUT,errors="ignore") as file:
#     for line in file.readlines():
#         line = re.split(r'[^a-zA-Z]+', line.strip())
#         for word in line:
#             if word == '': continue
#             word = word.lower()
#             if word not in v: v[word] = 0
#             v[word] += 1

for word in v:
    v[word] /= len(v)
    if word not in Words:
        v[word] *= math.log10(len(Text))
    else:
        v[word] *= Words[word]

answer = KNN(v,Text,K)
print(answer)
print('end')

