# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

    
with open('../input/csvdata/cnnhealth.txt') as f:
  lineList = f.readlines()

for num,single in enumerate(lineList):
    single = single[single.rfind('|') + 1:]
    single = single[:single.find('http://') ]
    lineList[num] = single
print(lineList[0])

# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(lineList)
# summarize
print("vectorizer.vocabulary_")
print(vectorizer.vocabulary_)
print("vectorizer.idf_")
print(vectorizer.idf_)
# encode document

vector = vectorizer.transform(lineList)
# summarize encoded vector
print("vector.shape")
print(vector.shape)
print("vector.toarray()")
print(vector.toarray())
dict_vector=vector.toarray()
notzero = 0
for single in dict_vector[0]:
    if single != 0:
        notzero = notzero+1
print(notzero)
        

# Any results you write to the current directory are saved as output.