#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df1 = pd.read_csv('../input/news-summary/news_summary.csv', encoding = 'latin-1')


# In[ ]:


df2 = pd.read_csv('../input/news-summary/news_summary_more.csv', encoding = 'latin-1')


# In[ ]:


df1.head()


# In[ ]:


df2.head()


# In[ ]:


import re  
from time import time  # To time our operations
from collections import defaultdict  # For word frequency

import spacy  

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


# In[ ]:


df1.shape, df2.shape


# In[ ]:


df1.isnull().sum(), df2.isnull().sum()


# In[ ]:


df1 = df1.dropna().reset_index(drop = True)
df1.isnull().sum()


# In[ ]:


nlp = spacy.load('en', disable=['ner', 'parser']) 

def cleaning(doc):
    txt = [token.lemma_ for token in doc if not token.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)


# In[ ]:


brief_cleaning1 = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df1['text'])
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df1['ctext'])
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df1['headlines'])
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df2['headlines'])
brief_cleaning2 = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df2['text'])


# In[ ]:


t = time()

txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning1, batch_size=5000, n_threads=-1)]

print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))


# In[ ]:


t = time()

txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning2, batch_size=5000, n_threads=-1)]

print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))


# In[ ]:


df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
df_clean.head()


# In[ ]:


from gensim.models.phrases import Phrases, Phraser


# In[ ]:


sent = [row.split() for row in df_clean['clean']]


# In[ ]:


phrases = Phrases(sent, min_count=30, progress_per=10000)


# In[ ]:


bigram = Phraser(phrases)


# In[ ]:


sentences = bigram[sent]


# In[ ]:


word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
len(word_freq)


# In[ ]:


import multiprocessing

from gensim.models import Word2Vec


# In[ ]:


cores = multiprocessing.cpu_count()
cores


# In[ ]:


w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)


# In[ ]:


t = time()

w2v_model.build_vocab(sentences, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))


# In[ ]:


t = time()

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))


# In[ ]:


w2v_model.init_sims(replace=True)


# In[ ]:


w2v_model.wv.most_similar(positive=["upgrad"])


# In[ ]:


w2v_model.wv.most_similar(positive=["bjp"])


# In[ ]:


w2v_model.wv.most_similar(positive=["iiit"])


# In[ ]:


w2v_model.wv.most_similar(positive=["amit_shah"])


# In[ ]:


w2v_model.wv.similarity("bjp", 'congress')


# In[ ]:


w2v_model.wv.most_similar(positive = ['bjp'])


# In[ ]:


w2v_model.wv.most_similar(positive = ['cristiano_ronaldo'])


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
 
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[ ]:


def tsnescatterplot(model, word, list_names):
    """ Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    its list of most similar words, and a list of words.
    """
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
        
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=19).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    
    fig, _ = plt.subplots()
    fig.set_size_inches(8, 8)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))


# In[ ]:


tsnescatterplot(w2v_model, 'football', ['dog', 'bird', 'ronaldo', 'wenger', 'iit', 'sky', 'fill', 'hey'])


# In[ ]:


tsnescatterplot(w2v_model, 'finance', [i[0] for i in w2v_model.wv.most_similar(negative=["football"])])


# In[ ]:


tsnescatterplot(w2v_model, 'finance', [i[0] for i in w2v_model.wv.most_similar(positive=["finance"])])


# In[ ]:


w2v_model.wv.syn0.shape


# In[ ]:


w2v_model.wv.most_similar("worst")


# In[ ]:


# Function to average all word vectors in a paragraph
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)
    
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    
    # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


# In[ ]:


def getAvgFeatureVecs(news, model, num_features):
    counter = 0
    newsFeatureVecs = np.zeros((len(news),num_features),dtype="float32")
    for article in news:
        # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Article %d of %d"%(counter,len(news)))
            
        newsFeatureVecs[counter] = featureVecMethod(news, model, num_features)
        counter = counter+1
        
    return newsFeatureVecs


# In[ ]:


trainDataVecs = getAvgFeatureVecs(df_clean, w2v_model, num_features = 300)


# In[ ]:


# Fitting a random forest classifier to the training data
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
    
print("Fitting random forest to training data....")    
forest = forest.fit(trainDataVecs, df_clean.iloc[:,0])


# In[ ]:


test_df = df1.iloc[:,5]
# testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)


# In[ ]:


test_df = test_df.dropna().reset_index(drop = True)
test_df.isnull().sum()


# In[ ]:


test_df.head()


# In[ ]:


test_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in test_df)


# In[ ]:


t = time()

test_txt = [cleaning(doc) for doc in nlp.pipe(test_cleaning, batch_size=5000, n_threads=-1)]

print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))


# In[ ]:


clean_test = pd.DataFrame({'clean': test_txt})
clean_test = clean_test.dropna().drop_duplicates()
clean_test.shape


# In[ ]:


clean_test.head()


# In[ ]:


testDataVecs = getAvgFeatureVecs(clean_test, w2v_model, num_features = 300)


# In[ ]:


# Predicting the sentiment values for test data and saving the results in a csv file 

result = forest.predict(testDataVecs)
output = pd.DataFrame(data={"sentiment":result})
output.to_csv( "output.csv", index=False, quoting=3 )


# In[ ]:


output.iloc[0]


# In[ ]:




