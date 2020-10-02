#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict


# In[ ]:


df = pd.read_csv('../input/ISIS Religious Texts v1.csv', encoding = "ISO-8859-1")
df.head()


# In[ ]:


import re
def clearstring(string):
    try:
        string = re.sub('[^A-Za-z0-9 .]+', '', string)
        string = string.split(' ')
        string = filter(None, string)
        string = [y.strip() for y in string]
        string = ' '.join(string)
    except:
        print(string)
    return string


# In[ ]:


# remove some unnecessary symbols 
for i in range(df.shape[0]):
    df['Quote'].iloc[i] = clearstring(df['Quote'].iloc[i])


# In[ ]:


# remove last row, empty row
df = df.iloc[:-1, :]
df.head()


# In[ ]:


# I just want verbs and nouns in a sentence
def get_clean_text(string):
    blob = TextBlob(string).tags
    tags = []
    # you can add more
    accept = ['NNP', 'NN', 'NNS', 'NNPS', 'VBZ', 'VBN', 'VB']
    for k in blob:
        if k[1] in accept:
            tags.append(k[0])
            
    return list(OrderedDict.fromkeys(tags))


# In[ ]:


# we need to split by ('.') to save our memory during speech tagging process to build the tree
for i in range(df.shape[0]):
    texts = df['Quote'].iloc[i].split('. ')
    tags = []
    for t in texts:
        tags += get_clean_text(t)
    df['Quote'].iloc[i] = ' '.join(list(OrderedDict.fromkeys(tags)))


# In[ ]:


df['Quote'].iloc[1]


# It cleans already
# 
# Below I want to get freq of unique for certain columns i want, you can do bar graph

# In[ ]:


df['Type'].value_counts()


# In[ ]:


df['Purpose'].value_counts()


# In[ ]:


df['Magazine'].value_counts()


# In[ ]:


df['Source'].value_counts()


# How about we predict Magazine from Source?

# In[ ]:


labels = df['Magazine'].values.copy()

# get unique label
unique_labels = np.unique(labels)
# change into int
labels = LabelEncoder().fit_transform(labels)
texts = df['Quote'].values.copy()


# I will use traditional Bag-of-Word and tf-idf for changing text to vectors space

# In[ ]:


bag_counts = CountVectorizer().fit_transform(texts)
bag_counts.shape


# That means, there are 8829 unique of word, every words in a sentence will add value by 1 if got in the vectors.
# 
# Example, sentence is 'I LOVE YOU YOU YOU', our vector got [0, 0, 0] represent 'I LOVE YOU',
# 
# that mean our vector for 'I LOVE YOU YOU YOU' is [1, 1, 3]

# In[ ]:


bag_counts_tdidf = TfidfTransformer(use_idf = True).fit_transform(bag_counts)
bag_counts_tdidf.shape


# This is the function of tf-idf
# ![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/10109d0e60cc9d50a1ea2f189bac0ac29a030a00)

# In[ ]:


# just want to show example
del bag_counts_tdidf, bag_counts


# In[ ]:


# i use pipeline to do automate processing to feed into my classifier. bag of word -> tf-idf -> SGD
magazine_clf = Pipeline([('vect', CountVectorizer()), 
                         ('tfidf', TfidfTransformer()), 
                         ('clf', SGDClassifier(loss = 'modified_huber', 
                                               penalty = 'l2', alpha = 1e-4, 
                                               n_iter = 100, random_state = 42))])


# How about some visualization for our text data?

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

current_palette = sns.color_palette()
# blue and red from seaborn
colors = [current_palette[0], current_palette[2]]

# visualize 20% of our data
_, x, _, y = train_test_split(texts, labels, test_size = 0.2)

plt.rcParams["figure.figsize"] = [10, 10]
ax = plt.subplot(111)

X = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),]).fit_transform(x).todense()
tsne = TSNE(n_components = 2).fit_transform(X)
for no, _ in enumerate(np.unique(unique_labels)):
    ax.scatter(tsne[y == no, 0], tsne[y == no, 1], c = colors[no], label = unique_labels[no])
    
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.05), fancybox = True, shadow = True, ncol = 5)
plt.show()


# We will got low accuracy for this if reduce the dimension

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size = 0.2)
magazine_clf.fit(x_train, y_train)
predicted = magazine_clf.predict(x_test)
print (np.mean(predicted == y_test))


# We got 81 accuracy for validation set! how about f1 score?

# In[ ]:


print(metrics.classification_report(y_test, predicted, target_names = unique_labels))


# Good enough! how about we do discrimination on 'Type'?

# In[ ]:


types = df['Type'].values.copy()

# get unique label
unique_types = np.unique(types)
# change into int
types = LabelEncoder().fit_transform(types)


# In[ ]:


current_palette = sns.color_palette(n_colors = unique_types.shape[0])

# visualize 20% of our data
_, x, _, y = train_test_split(texts, types, test_size = 0.2)

plt.rcParams["figure.figsize"] = [10, 10]
ax = plt.subplot(111)

X = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),]).fit_transform(x).todense()
tsne = TSNE(n_components = 2).fit_transform(X)
for no, _ in enumerate(np.unique(unique_types)):
    ax.scatter(tsne[y == no, 0], tsne[y == no, 1], c = current_palette[no], label = unique_types[no])
    
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.05), fancybox = True, shadow = True, ncol = 5)
plt.show()


# It clustered and sticked nearly each others according to population, good!

# In[ ]:


# i use pipeline to do automate processing to feed into my classifier. bag of word -> tf-idf -> SGD
type_clf = Pipeline([('vect', CountVectorizer()), 
                         ('tfidf', TfidfTransformer()), 
                         ('clf', SGDClassifier(loss = 'modified_huber', 
                                               penalty = 'l2', alpha = 1e-4, 
                                               n_iter = 100, random_state = 42))])


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(texts, types, test_size = 0.2)
type_clf.fit(x_train, y_train)
predicted = type_clf.predict(x_test)
print(np.mean(predicted == y_test))
print(metrics.classification_report(y_test, predicted, target_names = unique_types))


# ['Bible', 'Fatwa', 'Fiqh', 'Hadith Commentary'] seems overfit, anyways, it is good enough to do text classification!
