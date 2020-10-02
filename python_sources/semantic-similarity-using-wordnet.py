#!/usr/bin/env python
# coding: utf-8

# Hello everyone interested in knowing more about NLP. I am looking for more methods to implement, and even though I don't seem to have a lot of results, I am enjoying what I am doing.
# 
# So, here goes another.
# 
# This implementation is trying to use very basic NLP (not deep learning) algorithms to extract semantic relations between sentences. I might combine all of my previous methods and make a feature set out of it to train with a classifier or train multiple classifiers and use bagging/boosting for accuracy.
# 
# All that is coming next is an implementation of [this paper](https://trac.research.cc.gatech.edu/ccl/export/158/SecondMindProject/SM/SM.WordNet/Paper/WordNetDotNet_Semantic_Similarity.pdf).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import nltk
from nltk.corpus import wordnet as wn
from nltk import word_tokenize

import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# The tokenize(), posTag(), and stemmer() are simple functions to perform word_tokenization, part of speech tagging and return stemmed words in two input sentences.

# In[ ]:


import nltk
from nltk import word_tokenize
import pandas as pd


def tokenize(q1, q2):
    """
        q1 and q2 are sentences/questions. Function returns a list of tokens for both.
    """
    return word_tokenize(q1), word_tokenize(q2)


def posTag(q1, q2):
    """
        q1 and q2 are lists. Function returns a list of POS tagged tokens for both.
    """
    return nltk.pos_tag(q1), nltk.pos_tag(q2)


def stemmer(tag_q1, tag_q2):
    """
        tag_q = tagged lists. Function returns a stemmed list.
    """

    stem_q1 = []
    stem_q2 = []

    for token in tag_q1:
        stem_q1.append(stem(token))

    for token in tag_q2:
        stem_q2.append(stem(token))

    return stem_q1, stem_q2


# The Micheal Lesk Algorithm is famous for word sense disambiguation (WSD).
# 
# WSD is the process of finding the best possible sense of a word from all the given senses of the word. The Micheal Lesk algorithm uses the WordNet to gather the gloss of all the senses of the word in the sentence and then calculates the maximum overlap with the senses returning whichever gives the maximum overlap.
# 
# The implementation here is a bit different from both the paper and the Micheal Lesk algorithm. My Adapted Lesk Algorithm finds the sense which is best related to the sentence (excluding the stop words). The paper defines a constant k which they use to limit the context of the sentence being matched. The overlap is calculated using the gloss of all the context words such that the most matching sense of the word will be accepted as the best sense of the word in the context.

# In[ ]:


class Lesk(object):

    def __init__(self, sentence):
        self.sentence = sentence
        self.meanings = {}
        for word in sentence:
            self.meanings[word] = ''

    def getSenses(self, word):
        # print word
        return wn.synsets(word.lower())

    def getGloss(self, senses):

        gloss = {}

        for sense in senses:
            gloss[sense.name()] = []

        for sense in senses:
            gloss[sense.name()] += word_tokenize(sense.definition())

        return gloss

    def getAll(self, word):
        senses = self.getSenses(word)

        if senses == []:
            return {word.lower(): senses}

        return self.getGloss(senses)

    def Score(self, set1, set2):
        # Base
        overlap = 0

        # Step
        for word in set1:
            if word in set2:
                overlap += 1

        return overlap

    def overlapScore(self, word1, word2):

        gloss_set1 = self.getAll(word1)
        if self.meanings[word2] == '':
            gloss_set2 = self.getAll(word2)
        else:
            # print 'here'
            gloss_set2 = self.getGloss([wn.synset(self.meanings[word2])])

        # print gloss_set2

        score = {}
        for i in gloss_set1.keys():
            score[i] = 0
            for j in gloss_set2.keys():
                score[i] += self.Score(gloss_set1[i], gloss_set2[j])

        bestSense = None
        max_score = 0
        for i in gloss_set1.keys():
            if score[i] > max_score:
                max_score = score[i]
                bestSense = i

        return bestSense, max_score

    def lesk(self, word, sentence):
        maxOverlap = 0
        context = sentence
        word_sense = []
        meaning = {}

        senses = self.getSenses(word)

        for sense in senses:
            meaning[sense.name()] = 0

        for word_context in context:
            if not word == word_context:
                score = self.overlapScore(word, word_context)
                if score[0] == None:
                    continue
                meaning[score[0]] += score[1]

        if senses == []:
            return word, None, None

        self.meanings[word] = max(meaning.keys(), key=lambda x: meaning[x])

        return word, self.meanings[word], wn.synset(self.meanings[word]).definition()


# Now, we need some similarity measures which define how related the two words are (after their best senses have been calculated). I have used two different similarity measures, one is path similarity which calculates the distance in the words in the hyponym taxonymy graph.
# 
# Wup similarity is the Wu & Palmer's similarity.

# In[ ]:


import math
import numpy as np
from scipy import spatial
from nltk.corpus import wordnet as wn
from nltk.metrics import edit_distance

def path(set1, set2):
    return wn.path_similarity(set1, set2)


def wup(set1, set2):
    return wn.wup_similarity(set1, set2)


def edit(word1, word2):
    if float(edit_distance(word1, word2)) == 0.0:
        return 0.0
    return 1.0 / float(edit_distance(word1, word2))


# In[ ]:


def computePath(q1, q2):

    R = np.zeros((len(q1), len(q2)))

    for i in range(len(q1)):
        for j in range(len(q2)):
            if q1[i][1] == None or q2[j][1] == None:
                sim = edit(q1[i][0], q2[j][0])
            else:
                sim = path(wn.synset(q1[i][1]), wn.synset(q2[j][1]))

            if sim == None:
                sim = edit(q1[i][0], q2[j][0])

            R[i, j] = sim

    # print R

    return R


# In[ ]:


def computeWup(q1, q2):

    R = np.zeros((len(q1), len(q2)))

    for i in range(len(q1)):
        for j in range(len(q2)):
            if q1[i][1] == None or q2[j][1] == None:
                sim = edit(q1[i][0], q2[j][0])
            else:
                sim = wup(wn.synset(q1[i][1]), wn.synset(q2[j][1]))

            if sim == None:
                sim = edit(q1[i][0], q2[j][0])

            R[i, j] = sim

    # print R

    return R


# This is the fifth part of the algorithm mentioned below. Combines the similarity measures calculated for the two sentences and produces a single similarity score.

# In[ ]:


def overallSim(q1, q2, R):

    sum_X = 0.0
    sum_Y = 0.0

    for i in range(len(q1)):
        max_i = 0.0
        for j in range(len(q2)):
            if R[i, j] > max_i:
                max_i = R[i, j]
        sum_X += max_i

    for i in range(len(q1)):
        max_j = 0.0
        for j in range(len(q2)):
            if R[i, j] > max_j:
                max_j = R[i, j]
        sum_Y += max_j
        
    if (float(len(q1)) + float(len(q2))) == 0.0:
        return 0.0
        
    overall = (sum_X + sum_Y) / (2 * (float(len(q1)) + float(len(q2))))

    return overall


# You can call this mostly the main method, the semantic representations here follow this algorithm:
# 
# 1.   Tokenization. 
# 
# 2.   Perform part of speech tagging.
# 
# 3.   Word sense disambiguation using adapted Lesk Algorithm.
# 
# 4.   Building a semantic similarity relations matrix R[m, n] of each pair of word senses, where R[i, j] is the semantic similarity between the most appropriate sense of word at position i of X and the most appropriate sense of word at position j of Y. Thus, R[i,j] is also the weight of edge connect from i to j. If a word does not exist in the dictionary we use the edit-distance similarity instead and output a lower associated weight.
# 
# 5.   I have not been able to compile the results from both the similarity measures. the paper mentions two methods of doing it, using match(X, Y) -- the matching measure between the word tokens X and Y -- and using the Dice Coefficient. I still figuring out some of the drawbacks of these methods and which one would be better to use. 
# **Any help on this is appreciated**

# In[ ]:


def semanticSimilarity(q1, q2):

    tokens_q1, tokens_q2 = tokenize(q1, q2)
    # stem_q1, stem_q2 = stemmer(tokens_q1, tokens_q2)
    tag_q1, tag_q2 = posTag(tokens_q1, tokens_q2)

    sentence = []
    for i, word in enumerate(tag_q1):
        if 'NN' in word[1] or 'JJ' in word[1] or 'VB' in word[1]:
            sentence.append(word[0])

    sense1 = Lesk(sentence)
    sentence1Means = []
    for word in sentence:
        sentence1Means.append(sense1.lesk(word, sentence))

    sentence = []
    for i, word in enumerate(tag_q2):
        if 'NN' in word[1] or 'JJ' in word[1] or 'VB' in word[1]:
            sentence.append(word[0])

    sense2 = Lesk(sentence)
    sentence2Means = []
    for word in sentence:
        sentence2Means.append(sense2.lesk(word, sentence))
    # for i, word in enumerate(sentence1Means):
    #     print sentence1Means[i][0], sentence2Means[i][0]

    R1 = computePath(sentence1Means, sentence2Means)
    R2 = computeWup(sentence1Means, sentence2Means)

    R = (R1 + R2) / 2

    # print R

    return overallSim(sentence1Means, sentence2Means, R)


# In[ ]:


import nltk
STOP_WORDS = nltk.corpus.stopwords.words()
def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")

    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence


# This sequential code runs all the methods above. First cleans the data, then takes two sentences and prints out the sentence path similarity output and the WUP Similarity output.

# In[ ]:


from sklearn.metrics import log_loss

X_train = pd.read_csv('../input/train.csv')
X_train = X_train.dropna(how="any")

y = X_train['is_duplicate']

print('Exported Cleaned train Data, no need for cleaning')
for col in ['question1', 'question2']:
    X_train[col] = X_train[col].apply(clean_sentence)

y_pred = []
count = 0
print('Calculating similarity for the training data, please wait.')
for row in X_train.itertuples():
    # print row
    q1 = str(row[4])
    q2 = str(row[5])

    sim = semanticSimilarity(q1, q2)
    count += 1
    if count % 10000 == 0:
        print(str(count)+", "+str(sim)+", "+str(row[6]))
    y_pred.append(sim)
    
output = pd.DataFrame(list(zip(X_train['id'].tolist(), y_pred)), columns=['id', 'similarity'])
output.to_csv('semantic_train.csv', index=False)

print("Log Loss Score:")
print(log_loss(y, np.array(y_pred)))

