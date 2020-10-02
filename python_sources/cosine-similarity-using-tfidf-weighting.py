#!/usr/bin/env python
# coding: utf-8

# Hi,
# 
# I recently started working on this competition and would love comments from everyone who reads and studies NLP. I have been working on some core NLP methods (excluding deep learning) and wanted to apply those and check the performance.
# 
# If you like my implementations, please upvote and/or comment.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from collections import Counter
from nltk import word_tokenize

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# So, the tfidf basically measures the importance of a particular word in a document/paragraph/piece of text. So, in calculating tfidf, I have taken the questions as the documents and the qlist everywhere would be the list of all the questions (on both sides). The inverse document frequency is calculated over all of the questions being compared, i.e. everything given in the file.
# 
# In the next few cells I have implemented some methods to run calculate term frequency, inverse document frequency and some related functions.

# In[ ]:


def tf(question, word):
    if word not in question:
        return 0
    count = dict(Counter(question))
    q_len = len(question)
    return float(count[word]) / float(q_len)


# The two methods below are parts of implementation to the idf (inverse document frequency). The first function returns the count of sentences/documents/paragraphs where the word is present.
# 
# The second, idf method returns the value of the log.

# In[ ]:


def n_containing(qlist, word):
    return float(qlist[word])

def idf(qlist, word):
    return math.log(float(len(qlist.keys())) / (1.0 + n_containing(qlist, word)))


# The next function is the tfidf, which, as in the name, returns the tfidf value.

# In[ ]:


def tfidf(question, qlist, word):
    return tf(question, word) * idf(qlist, word)


# The tfidf function is used to return a number, of which I would be making a vector, of the two statements (I won't usually give a single word, the above is a utility function I used from an old implementation).
# 
# For the function below, v1 and v2 are two vectors (list of numbers in this case) of the same dimensions. Function returns the cosine distance between those which is the ratio of the dot product of the vectors over their RS.

# In[ ]:


def cosine(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    return np.dot(v1, v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))


# Okay, having implemented the necessary functions, let us start with running these over the dataset. We'll run it over the train.csv to check how good it performs. It is still running on my machine, and I trust this kernel very less (has been going down like after every cell I execute), so I might as well just post the output instead of running it on this again (with the code obviously).
# 
# The first is simple enough, gather the data. I am dropping the useless columns in the train data, i.e. I dropped the qids. If you are running the following cell, you need to be warned it takes some time.
# 
# There is also another quirk with this code snippet. I noticed in the questions that some of them have very less number of characters, like even 1 character questions (what were they trying to write '?'), parsing a question with less than 10 characters did not make much sense to me. The question (and its pair) containing less than 10 characters has been left out from the question list, but you can obviously use it if you would like.

# In[ ]:


train = pd.read_csv('../input/train.csv')

train_qs = train[['id', 'question1', 'question2', 'is_duplicate']]

qlist = []
count = 0
for row in train_qs.itertuples():
    try:
        if len(str(row[2])) > 10:
            q1 = word_tokenize(row[2].lower())
        if len(str(row[3])) > 10:
            q2 = word_tokenize(row[3].lower())
        qlist += q1 + q2
        count+=1
        if count%100000 == 0:
            print('At'+str(count))
#        qlist.append(q2)
    except TypeError:
        pass

# print len(qlist)
qlist = dict(Counter(qlist))
import json
with open('qlist.json', 'w') as f:
    f.write(json.dumps(qlist, indent=2))
print('All Questions added to list')


# Okay, this is the final stage. Now the questions are going to be compared. I do have some results with me (and it is taking some more time to execute on my system, so it might as well get stopped by the kernel by the time it reaches halfway).
# 
# Since I know this is gonna take a lot of time, I break the code after matching 100 pairs.

# In[ ]:


with open('submission.csv', 'a') as f:
    f.write('id,is_duplicate\n')
for row in train_qs.itertuples():
    if len(str(row[2])) > 10 and len(str(row[3])) > 10:
        wordvec1 = word_tokenize(row[2].lower())
        wordvec2 = word_tokenize(row[3].lower())
        words = wordvec1 + wordvec2
        words = list(set([word for word in words if word != '?']))

        # print words

        vec1 = []
        vec2 = []
        for word in words:
            vec1.append(tfidf(wordvec1, qlist, word))
            vec2.append(tfidf(wordvec2, qlist, word))

        with open('submission.csv', 'a') as f:
            f.write(str(row[1]) + "," + str(cosine(vec1, vec2)) + '\n')
    else:
        with open('submission.csv', 'a') as f:
            f.write(str(row[1]) + "," + '0' + '\n')
#    print str(row[1]) + "," + str(cosine(vec1, vec2))


# I agree this is not the best method to compare two questions, but this seems to set a benchmark. This is the most basic NLP implementation possible to compare two sentences using a tfidf representation of the two sentences and hence might not get a good response. This is also an unsupervised method which is basically ignoring the fact that there is a train.csv file available to "train" over.
# 
# Thank you, I appreciate anyone who reads this on spending some time on this kernel too. I would also like suggestions and methods for improvement. Also, since this is an unsupervised method I ran, could this method somehow be used in a supervised fashion, like use the cosine similarity as a feature on some other learning method.
# 
# I am mostly looking for NLP methods to implement for this and would appreciate any help/motivation to better methods around (not using deep learning, for now). All and any help is appreciated.
