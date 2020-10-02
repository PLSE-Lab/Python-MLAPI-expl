#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Importing Required Libraries

# In[ ]:


# General Libraries
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re
# specific for data preproressing and visualization
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud,STOPWORDS 
from statistics import mode
# classifiers
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from nltk.classify import ClassifierI


# In[ ]:


# Importing the traub file
TweetData =  pd.read_csv("/kaggle/input/nlp-getting-started/train.csv",index_col=0)


# In[ ]:


TweetData.head()


# In[ ]:


stop_words = set(stopwords.words('english'))


# In[ ]:


# Cleaning tweets without lammetization
def clean_tweets(x):
    clean1 = re.sub('https?://[A-Za-z0-9./]+','',x)
    clean2 = re.sub(r'[^\w\s]','',clean1).lower()
    words = word_tokenize(clean2)
    words = [w for w in words if not w in stop_words]
    return words


# In[ ]:


# # Cleaning tweets with lammetization
# from nltk.stem import WordNetLemmatizer
# lemmetizer  = WordNetLemmatizer()
# def clean_tweets(x):
#     clean1 = re.sub('https?://[A-Za-z0-9./]+','',x)
#     clean2 = re.sub(r'[^\w\s]','',clean1).lower()
#     words = word_tokenize(clean2)
#     words = [w for w in words if not w in stop_words]
#     words =[lemmetizer.lemmatize(w) for w in words]
#     return words


# In[ ]:


TweetData['words'] = TweetData['text'].apply(lambda x: clean_tweets(x))


# In[ ]:


TweetData.head()


# **We will use POS tagging ( which is used for tagging a word for its part of speech as per engilsh grammer) for filtering words which can be used as features for classification of tweets**

# In[ ]:


# POS tagging code
'''CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: "there is" ... think of it like "there exists")
FW foreign word
IN preposition/subordinating conjunction
JJ adjective 'big'
JJR adjective, comparative 'bigger'
JJS adjective, superlative 'biggest'
LS list marker 1)
MD modal could, will
NN noun, singular 'desk'
NNS noun plural 'desks'
NNP proper noun, singular 'Harrison'
NNPS proper noun, plural 'Americans'
PDT predeterminer 'all the kids'
POS possessive ending parent's
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO to go 'to' the store.
UH interjection errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when''


# In[ ]:


# All word Freqency curve with out any pos tagging filter
All_words = []
for words in TweetData['words']:
    for word in words:
            All_words.append(word)
All_words_freq = nltk.FreqDist(All_words)
Freq_word_DF = pd.DataFrame({"Data":All_words_freq.most_common(15)})
Freq_word_DF['Words'] = Freq_word_DF['Data'].apply(lambda x : x[0])
Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])
sns.set()
sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})
sns.set_context("paper", font_scale=2)
fig=plt.figure(figsize =(20,8),dpi=50)
sns.barplot('Words','freq',data = Freq_word_DF)


# **Lets see top 15 words based on its frequency after filtering on POS tagging**

# In[ ]:


# All word adjective Freqency curve 
allowed_word_type = ["JJ","JJR","JJS"]
All_words = []
for words in TweetData['words']:
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1] in allowed_word_type:
                    All_words.append(w[0])
All_words_freq = nltk.FreqDist(All_words)
Freq_word_DF = pd.DataFrame({"Data":All_words_freq.most_common(15)})
Freq_word_DF['Words'] = Freq_word_DF['Data'].apply(lambda x : x[0])
Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])
sns.set()
sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})
sns.set_context("paper", font_scale=2)
fig=plt.figure(figsize =(20,8),dpi=50)
sns.barplot('Words','freq',data = Freq_word_DF)


# In[ ]:


# All word verb Freqency curve 
allowed_word_type = ["VB","VBD","VBN","VBP","VBZ"]
All_words = []
for words in TweetData['words']:
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1] in allowed_word_type:
                    All_words.append(w[0])
All_words_freq = nltk.FreqDist(All_words)
Freq_word_DF = pd.DataFrame({"Data":All_words_freq.most_common(15)})
Freq_word_DF['Words'] = Freq_word_DF['Data'].apply(lambda x : x[0])
Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])
sns.set()
sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})
sns.set_context("paper", font_scale=2)
fig=plt.figure(figsize =(20,8),dpi=50)
sns.barplot('Words','freq',data = Freq_word_DF)


# In[ ]:


# All word Noun Freqency curve 
allowed_word_type = ["NN","NNS"]
All_words = []
for words in TweetData['words']:
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1] in allowed_word_type:
                    All_words.append(w[0])
All_words_freq = nltk.FreqDist(All_words)
Freq_word_DF = pd.DataFrame({"Data":All_words_freq.most_common(15)})
Freq_word_DF['Words'] = Freq_word_DF['Data'].apply(lambda x : x[0])
Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])
sns.set()
sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})
sns.set_context("paper", font_scale=2)
fig=plt.figure(figsize =(20,8),dpi=50)
sns.barplot('Words','freq',data = Freq_word_DF)


# **Lets make a word cloud from the data**

# In[ ]:


# wordcloud from the words
from wordcloud import WordCloud,STOPWORDS 
stopwords = set(STOPWORDS) 
comment_words = ' '
for text in TweetData['words']:
    for words in text: 
        comment_words = comment_words + words + ' '
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words)
sns.set()
plt.figure(figsize = (8, 8), facecolor = None,dpi=100) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)


# **Preprocessing the data for model building**

# In[ ]:


# # Frequency data of words 
# allowed_word_type = ["NN","NNS"]
# all_words = []
# for words in TweetData['words']:
#     pos = nltk.pos_tag(words)
#     for w in pos:
#         if w[1] in allowed_word_type:
#                     all_words.append(w[0])


# In[ ]:


allowed_word_type = ["NN","NNS","JJ"]


# In[ ]:


# Frequency Data by target
# for target 1
all_words = []
for words in TweetData[TweetData['target']==1]['words']:
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1] in allowed_word_type:
            if w[0] not in ['im','amp']:
                all_words.append(w[0])
all_words_freq_1 = nltk.FreqDist(all_words)


# In[ ]:


# for target 0
all_words = []
for words in TweetData[TweetData['target']==0]['words']:
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1] in allowed_word_type:
            if w[0] not in ['im','amp']:
                all_words.append(w[0])
all_words_freq_0 = nltk.FreqDist(all_words)


# In[ ]:


feature1 =[w[0] for w in all_words_freq_1.most_common(1000)]
feature0 = [w[0] for w in all_words_freq_0.most_common(500)]
word_feature = list(set(feature1 + feature0))


# In[ ]:


# all_words_freq = nltk.FreqDist(all_words)
# word_feature = list(all_words_freq.keys())[:2000]


# In[ ]:


TweetData['combined']  = TweetData[['words', 'target']].apply(tuple, axis=1)


# In[ ]:


def find_features(doc):
    words =  set(doc)
    features = {}
    for w in word_feature:
        features[w] = (w in words)
    return features


# In[ ]:


featuresets = [(find_features(tweet),category) for (tweet,category) in TweetData['combined']]
len(featuresets)


# In[ ]:


training_set = featuresets[:5000]
testing_set = featuresets[5000:]


# Applying Naivebase Classifier of NLTK

# In[ ]:


classifier =  nltk.NaiveBayesClassifier.train(training_set)
print("Naive Base algorith accuracy : ", nltk.classify.accuracy(classifier,testing_set)*100)
classifier.show_most_informative_features(15)


# **Applying SK Learn Classifiers**

# In[ ]:


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
# Guss_classifier = SklearnClassifier(GaussianNB())
# Guss_classifier.train(training_set)
Berr_classifier = SklearnClassifier(BernoulliNB())
Berr_classifier.train(training_set)
logistic_classifier = SklearnClassifier(LogisticRegression())
logistic_classifier.train(training_set)
SDG_classifier = SklearnClassifier(SGDClassifier())
SDG_classifier.train(training_set)
random_classifier = SklearnClassifier(RandomForestClassifier())
random_classifier.train(training_set)
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
LinerSVC_classifier =  SklearnClassifier(LinearSVC())
LinerSVC_classifier.train(training_set)
NUSVC_classifier = SklearnClassifier(NuSVC())
NUSVC_classifier.train(training_set)
print("MNB algorith accuracy : ", nltk.classify.accuracy(MNB_classifier,testing_set)*100)
print("BErr algorith accuracy : ", nltk.classify.accuracy(Berr_classifier,testing_set)*100)
print("Logistic algorith accuracy : ", nltk.classify.accuracy(logistic_classifier,testing_set)*100)
print("SDG algorith accuracy : ", nltk.classify.accuracy(SDG_classifier,testing_set)*100)
print("Random Forest algorith accuracy : ", nltk.classify.accuracy(random_classifier,testing_set)*100)
print("SVC algorith accuracy : ", nltk.classify.accuracy(SVC_classifier,testing_set)*100)
print("LinerSVC algorith accuracy : ", nltk.classify.accuracy(LinerSVC_classifier,testing_set)*100)
print("NUSVC algorith accuracy : ", nltk.classify.accuracy(NUSVC_classifier,testing_set)*100)


# **Developing a vote based combination of all classifiers**

# In[ ]:


class Voteclssifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers
        
    def classify(self,features):
        vote = []
        for c in self._classifiers:
            v = c.classify(features)
            vote.append(v)
#         if vote.count(1)== vote.count(0):
#             s=1
#         else:
#             s=mode(vote)
        return mode(vote)


# In[ ]:


votedclassifier = Voteclssifier(classifier,MNB_classifier,Berr_classifier,logistic_classifier,NUSVC_classifier)


# In[ ]:


print("VoteVlassifier algorith accuracy : ", nltk.classify.accuracy(votedclassifier,testing_set)*100)


# In[ ]:


test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv",index_col=0)


# In[ ]:


test.head(15)


# In[ ]:


test['words'] = test['text'].apply(lambda x: clean_tweets(x))
test['feature'] = test['words'].apply(lambda x : find_features(x))


# In[ ]:


votedclassifier.classify(test['feature'][0])


# In[ ]:


test['target'] = test['feature'].apply(lambda x : votedclassifier.classify(x))


# In[ ]:


Submission =  pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv",index_col=0)
Submission['target'] = test['target']
Submission.to_csv("Submission.csv")

