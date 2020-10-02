# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# This script will look for common words after eliminating some disturbing characters and often used words
# score is about 0.75
#
# The script will output a verbose file to examine processing per question pair and two distionaries to evaluate
# used words

REMOVE = '["?!;:,()/#$@“”‘’£~]'

TRAIN_NAME = '../input/train.csv'
TEST_NAME  = '../input/test.csv'
DICT_NAME  = 'dict.txt'
VERB_TRAIN_NAME  = 'verb_train.txt'
DICT_NAME_ORDER  = 'dict_ordered.txt'

EXCLUDE = ['what', 'the', 'is', 'how', 'i', 'a', 'to', 'in', 'do', 'of', 'are', 'and', 'can', 'for']

import csv
import re
import operator


def writeDict():
    f = open (DICT_NAME, 'w')
    for w in sorted(dictWords):
        f.write(w)
        f.write('   ')
        f.write(str(dictWords[w]))
        f.write('\n')
    f.close
        
def writeDictOrder():
    f = open (DICT_NAME_ORDER, 'w')

    sorted_d = sorted(dictWords.items(), key=operator.itemgetter(1))

    for w, n in sorted_d:
        f.write(str(w))
        f.write('   ')
        f.write(str(n))
        f.write('\n')
    f.close
        
def rowToDict(text):
    global dictWords
    words = text.split(' ')
    for w in words:
        if w in dictWords:
            dictWords[w] += 1
        else:
            dictWords[w] = 1
            

def textToArray(text):
    text = text.lower()
    text = re.sub(REMOVE, '', text)
    text = re.sub(' \'', ' ', text)
    text = re.sub('\' ', ' ', text)
    text = re.sub('&', ' ', text)
    rowToDict(text)
    res = text.split(' ')

    res = list (set(res) - set(EXCLUDE))

    return (res)

    
# -------------------------------------------------



def doTrain():
    y_train = []

    file = open(VERB_TRAIN_NAME, "w")    
    
    lines = 0 
    first = 1
    with open(TRAIN_NAME, 'r') as csvfile:
            
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            if first == 0:
                lines += 1
                if lines % 100000 == 0:
                    print ('line {}'.format(lines))
    
                pid = row[0]
                qid1 = row[1]
                qid2 = row[2]
                q1 = row[3]
                q2 = row[4]
                res = row[5]
                y_train.append (res)                    
                
                wa1 = textToArray(q1)
                wa2 = textToArray(q2)
        
                l = len (list(set(wa1) & set(wa2)))
                t = min (len(wa1), len(wa2))
                if t != 0:
                    inter = float(l) / t
                else:
                    inter = 1
                        
                file.write ('\n-----------\n')
                file.write ('Q1: {}\n'.format(q1))
                file.write ('S1: {}\n'.format(wa1))
                file.write ('Q2: {}\n'.format(q2))
                file.write ('S2: {}\n'.format(wa2))
                file.write ('in: {}\n'.format(inter))
                file.write ('AN: {}\n'.format(res))
        
            first = 0
    
    csvfile.close()
    file.close()
    
    return y_train

# -------------------------------------------------



def doTest():
    
    subName = 'submission.csv'
    subFile = open(subName, "w")    
    subFile.write('test_id,is_duplicate\n')                        

    lines = 0 
    first = 1
    with open(TEST_NAME, 'r') as csvfile:
            
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            if first == 0:
                lines += 1
                if lines % 100000 == 0:
                    print ('line {}'.format(lines))
    
                pid = row[0]
                q1 = row[1]
                q2 = row[2]
                    
                wa1 = textToArray(q1)
                wa2 = textToArray(q2)
        
                l = len (list(set(wa1) & set(wa2)))
                t = min (len(wa1), len(wa2))
                if t != 0:
                    inter = float(l) / t
                else:
                    inter = 1
                
                if inter == 0:
                    inter = 0.05
                if inter == 1:
                    inter = 0.95
                
                subFile.write('{},{}\n'.format(pid,inter))                        
        
            first = 0
    
    csvfile.close()

# ----------


dictWords = {}

y_train = doTrain()

writeDict()
writeDictOrder()



doTest()

exit(0)

            
            