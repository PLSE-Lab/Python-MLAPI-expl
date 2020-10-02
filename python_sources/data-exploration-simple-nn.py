#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.pyplot import figure
np.random.seed(1)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/drugsComTrain_raw.csv')
test = pd.read_csv('../input/drugsComTest_raw.csv')
train.head()


# In[ ]:


train.describe()


# In[ ]:


plt.hist(train.rating)
plt.show()


# Interesting distribution, almost like the opposite of a bellcurve. People only leave reviews if the medicine was either really good or really bad it seems.

# In[ ]:


plt.figure(figsize=(10,6))
plt.hist(train.usefulCount, bins=60)
plt.show()


# As expected, extremely right skewed graph of usefulness.

# In[ ]:


plt.scatter(train.rating, train.usefulCount)
plt.show()


# Seems like the more positive reviews are found by people to be more useful.

# In[ ]:



drugcounts = train['drugName'].value_counts()
drugcounts_filtered = drugcounts[drugcounts >= 10]

plt.figure(figsize=(20,6))
plt.bar(drugcounts[:30].index, drugcounts[:30])
plt.xticks(rotation=70)
plt.show()


# Bar chart of top 30 most common drugs.

# In[ ]:


plt.figure(figsize=(20, 6))
plt.hist(drugcounts, bins=200)
plt.show()


# When the entire dataset plotted, we can see that there is a huge right skew in our data for how common certain drugs are.

# In[ ]:


filtered_train = train[train['drugName'].isin(drugcounts_filtered.index)]
print(train.shape)
print(filtered_train.shape)


# FIltering out drugs with less than 10 instances from our dataset.

# In[ ]:


rating_avgs = (filtered_train['rating'].groupby(filtered_train['drugName']).mean())
rating_avgs.sort_values()


# Series containing the effectiveness of each drug, based on average of ratings.

# In[ ]:


plt.hist(rating_avgs)


# Now our distribution looks extremely normal compared to our overall ratings histogram. This makes sense considering most reviews were either high or low.

# In[ ]:


illnesscounts = train['condition'].value_counts()
illnesscounts = illnesscounts[illnesscounts > 20]
print(illnesscounts)


# Checking to see what the most common illnesses are as well as dropping anything with under 20 cases.

# In[ ]:


filtered_train = train[train['condition'].isin(illnesscounts.index)]
curability = (filtered_train['rating'].groupby(filtered_train['condition']).mean())
curability.sort_values()


# Seeing which illnesses are most effective at being treated with medicine.

# In[ ]:


plt.hist(curability)
plt.show()


# Incredibly symmetric histogram of illness effectiveness, with its mean being centered around 7.25.

# In[ ]:


sns.boxplot(curability)


# In[ ]:


sns.boxplot(rating_avgs)


# In[ ]:


plt.scatter(rating_avgs, drugcounts_filtered)
plt.show()


# In[ ]:


b = "'@#$%^()&*;!.-"
X_train = np.array(filtered_train['review'])
X_test = np.array(test['review'])

def clean(X):
    for index, review in enumerate(X):
        for char in b:
            X[index] = X[index].replace(char, "")
    return(X)

X_train = clean(X_train)
X_test = clean(X_test)
print(X_train[:50])


# Checking to see if there's any correlation between average drug rating and the amount of times it is used. Not really seeing anything from this plot.

# Next is exploring model architectures and hyperparameters for a model predicting a specific 1-10 rating based on review.
# This is a classification NLP problem and I'll be using Keras.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from keras.utils import to_categorical
from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk

X_train = filtered_train['review']
X_test = test['review']
vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),
                             lowercase=True, max_features=5000)
test_train = pd.concat([X_train, X_test],ignore_index=True)
print(filtered_train['review'].shape)
print(test['review'].shape)
print(test_train.shape)
X_onehot = vectorizer.fit_transform(test_train)
stop_words = vectorizer.get_stop_words()
print(type(X_onehot))


# I use a CountVectorizer to vectorize each of the different reviews into 5000 feature row vectors. It's set to not add in very common words such as "and" and "the", and also ignore very rare words.

# In[ ]:


print(X_onehot.shape)
print(X_onehot.toarray())


# Sanity checking to see if the dimensions make sense, then peeking at what the actual X_onehot matrix looks like.
# We have our trimmed m number of examples, as well as 5000 columns so this looks right!

# In[ ]:


names_list = vectorizer.get_feature_names()
names = [[i] for i in names_list]
names = Word2Vec(names, min_count=1)
print(len(list(names.wv.vocab)))
print(list(names.wv.vocab)[:5])


# In[ ]:


def score_transform(X):
    y_reshaped = np.reshape(X['rating'].values, (-1, 1))
    for index, val in enumerate(y_reshaped):
        if val >= 8:
            y_reshaped[index] = 1
        elif val >= 5:
            y_reshaped[index] = 2
        else:
            y_reshaped[index] = 0
    y_result = to_categorical(y_reshaped)
    return y_result
    
    print(X_onehot)


# Creating a helper function to easily create our Y labels. We need to transform the review column to a m number of reviews by 10 columns, with a 1 in the place of the index of the label. If it were a rating of 5, the corresponding 1 would be in index 5.

# In[ ]:



y_train_test = pd.concat([filtered_train, test], ignore_index=True)
y_train = score_transform(y_train_test)
print(y_train)
print(y_train.shape)


# Checking if our helper function works. Seems good.

# In[ ]:


from numpy.random import seed
from keras.layers import Dropout, Flatten
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding

np.random.seed(1)
model = Sequential()
model.add(Dense(units=256, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary


# I experimented a lot with different kinds of simple NN architecture. It seemed like ~256 units was the sweet spot for accuracy without overfitting. Any units above would give me less accuracy as the model continued to train as well as overall performing worse, and less units also just gave me worse accuracy overall.
# I also discovered that adding any additional layers to the model resulted in a loss of accuracy, even when I kept the number extremely small as to avoid overfitting.
# I'm assuming that the data is not complicated enough for a deep network, and a single layer NN is better suited.

# In[ ]:


history = model.fit(X_onehot[:-53866], y_train[:-53866], epochs=5, batch_size=128, verbose=1, validation_data=(X_onehot[157382:157482], y_train[157382:157482]))


# I experimented with different numbers of epochs and batch sizes as well, and found roughly 7 epochs and a batch size of 128 to be optimal. Anything more than 7 epochs resulted in the model beginning to overtrain.

# In[ ]:


scores = model.evaluate(X_onehot[157482:], y_train[157482:], verbose=1)


# In[ ]:


scores[1]


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# This model predicts the if the rievew is positive, negative or neutral with a ~92% accuracy, according to this test set and output score.

# In[ ]:


all_names = [i.split() for i in X_train]
feature_vectors = names[names.wv.vocab]
print(feature_vectors[0])


# In[ ]:


np.random.seed(1)
all_names_rand = [all_names[np.random.randint(low=1, high=150000)] for i in range(5000)]
print(len(all_names_rand))


# In[ ]:


all_names_list = Word2Vec(all_names_rand, min_count=1)
all_names_vec = all_names_list[all_names_list.wv.vocab]
print(all_names[:2])


# In[ ]:


kclusterer_all = KMeansClusterer(5, distance=nltk.cluster.util.cosine_distance, repeats=10)
assigned_clusters_all = kclusterer_all.cluster(all_names_vec, assign_clusters=True)
print(len(assigned_clusters_all))


# In[ ]:


def generate_df(feature_names):
    
    all_words_dict = dict(zip(all_names_list.wv.vocab, assigned_clusters_all))
    for key in list(all_words_dict.keys()):
        if key in list(feature_names):
            pass
        else:
            del all_words_dict[key]
    sorted_names = []
    
    for cluster in range(5):
        cluster_list = []
        for key, value in all_words_dict.items():
            if value == cluster:
                
                cluster_list.append(key)
        sorted_names.append(cluster_list)
        
    for index, entry in enumerate(sorted_names):
        entry.sort()
    df_all = pd.DataFrame(sorted_names).T
    print(df_all[:50])
    return df_all, sorted_names
    

df,sorted_names_all = generate_df(names.wv.vocab)


# In[ ]:


def test_clusters(cluster_list):
    
    score_list = []
    lens = []
    
    reviewnum = 15000
    
    np.random.seed(3)
    indicies = [np.random.randint(low=1, high=150000) for i in range(reviewnum)]
    X_sample =  test_train[indicies]
    y_sample = y_train[indicies]
    
    for cluster in cluster_list:
        
        lens.append(len(cluster))

        X_onehot = vectorizer.fit_transform(X_sample)
        
        X_onehot = X_onehot.toarray()

        cluster_indexes = []
        print(len(list(names.wv.vocab)))

        for index, feature_name in enumerate(list(names.wv.vocab)):
            if feature_name in cluster:
                cluster_indexes.append(index)

        features = len(cluster_indexes)
        
        X_onehot = X_onehot[:, cluster_indexes]

        model_cluster = Sequential()
        model_cluster.add(Dense(units=256, activation='relu', input_dim=features))
        model_cluster.add(Dense(units=3, activation='softmax'))

        model_cluster.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_cluster.summary
        
        history = model_cluster.fit(X_onehot[:reviewnum-2000], y_sample[:reviewnum-2000], epochs=10, batch_size=128, verbose=2, validation_data=(X_onehot[reviewnum-2000:reviewnum-1900], y_sample[reviewnum-2000:reviewnum-1900]))
        
        y_test = score_transform(test)
        scores = model_cluster.evaluate(X_onehot[reviewnum-2000:], y_sample[reviewnum-2000:], verbose=1)
        
        score_list.append(scores[1])
    for index, entry in enumerate(score_list):
        print("cluster", index + 1, "accuracy: ", str(entry) + ". number of words for cluster: ", lens[index])


# In[ ]:


test_clusters(sorted_names_all)


# In[ ]:


test_clusters([list(names.wv.vocab)[:1000], list(names.wv.vocab)[2000:3000], list(names.wv.vocab)[3000:4000], list(names.wv.vocab)[4000:5000]])


# In[ ]:


vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, max_features=500)
X_onehot = vectorizer.fit_transform(test_train)
names_list = vectorizer.get_feature_names()
names = [[i] for i in names_list]
names = Word2Vec(names, min_count=1)

df, sorted_names_all = generate_df(names.wv.vocab)


# In[ ]:


test_clusters(sorted_names_all)

