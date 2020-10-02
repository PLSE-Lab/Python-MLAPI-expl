#!/usr/bin/env python
# coding: utf-8

# ## The Last Kernel
# Time: 6:00 PM<br>
# Date: 31/12/2019<br>
# By the time you would read this kernel, it would be the end of a year, an end of a decade. I am currently moving towards learning advanced techniques, like using berts, transformer models, concepts like look-ahead and attention networks. Although all of them involves deep learning.<br>
# But before moving towards deep learning I would like to contribute a little to the community, what I have learned in NLP with Machine Learning.<br>
# I hope this kernel would also be helpful to those who want basics to learn and also those who want to move forward from basics.

# ## The problem statement
# In one word, it is simply  a binary classification problem. You are predicting whether a given tweet is about a real disaster or not. If so, predict a 1. If not, predict a 0.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import networkx as nx
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, NMF,LatentDirichletAllocation
import xgboost as xgb
from sklearn.metrics import roc_auc_score,accuracy_score,log_loss
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import seaborn as sns
from tqdm import tqdm as tqdm_base
from sklearn.model_selection import train_test_split
#from gensim.models.ldamodel import LdaModel
from sklearn import preprocessing
#from gensim.corpora import Dictionary
import umap
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


BASE_PATH = "/kaggle/input/nlp-getting-started/"


# In[ ]:


train =pd.read_csv(BASE_PATH + "train.csv")
train.head()


# In[ ]:


test =pd.read_csv(BASE_PATH + "test.csv")
test.head()


# In[ ]:


train.dtypes


# In[ ]:


train['keyword'].value_counts(dropna=False, normalize=True).head()


# In[ ]:


train['location'].value_counts(dropna=False, normalize=True).head()


# In[ ]:


plt.hist(train['target'], label='train');
plt.legend();
plt.title('Distribution of target labels');


# In[ ]:


#train['location'].fillna('NULL', inplace=True)
#train['keyword'].fillna('NULL', inplace=True)


# In[ ]:


temp = train.dropna()
temp = temp.reset_index()


# In[ ]:


temp.shape


# ## Representing the relation between location and key-Words through directed graph

# In[ ]:


g = nx.DiGraph()
for x in range(500):
    g.add_edge(temp['location'][x],temp['keyword'][x], weight=x,capacity=5,length = 100) 
pos=nx.spring_layout(g,k=0.15,)
plt.figure(figsize =(20, 20)) 
nx.draw_networkx(g,pos,alpha=0.8,node_color='red',node_size=25,font_size=9, with_label = True) 


# In[ ]:


g = nx.DiGraph()
for x in range(500,1000):
    g.add_edge(train['location'][x], train['keyword'][x]) 
    
pos=nx.spring_layout(g,k=0.15,)
plt.figure(figsize =(20, 20)) 
nx.draw_networkx(g,pos,alpha=0.8,node_color='red',node_size=25,font_size=9, with_label = True) 


# In[ ]:


g = nx.DiGraph()
for x in range(1000,1500):
    g.add_edge(train['location'][x], train['keyword'][x]) 
    
pos=nx.spring_layout(g,k=0.15,)
plt.figure(figsize =(20, 20)) 
nx.draw_networkx(g,pos,alpha=0.8,node_color='red',node_size=25,font_size=9, with_label = True) 


# In[ ]:


g = nx.DiGraph()
for x in range(1500,2000):
    g.add_edge(train['location'][x], train['keyword'][x]) 
    
pos=nx.spring_layout(g,k=0.15,)
plt.figure(figsize =(20, 20)) 
nx.draw_networkx(g,pos,alpha=0.8,node_color='red',node_size=25,font_size=9, with_label = True) 


# In[ ]:


g = nx.DiGraph()
for x in range(2000,2500):
    g.add_edge(train['location'][x], train['keyword'][x]) 
    
pos=nx.spring_layout(g,k=0.15,)
plt.figure(figsize =(20, 20)) 
nx.draw_networkx(g,pos,alpha=0.8,node_color='red',node_size=25,font_size=9, with_label = True) 


# In[ ]:


g = nx.DiGraph()
for x in range(2500,3000):
    g.add_edge(train['location'][x], train['keyword'][x]) 
    
pos=nx.spring_layout(g,k=0.15,)
plt.figure(figsize =(20, 20)) 
nx.draw_networkx(g,pos,alpha=0.8,node_color='red',node_size=25,font_size=9, with_label = True) 


# ## Lemmatizer and PorterStemmer
# Both of them are Normalization techniques, but the acutual difference is the way they normalize the word-tokens.

# In[ ]:


# Lets try out PorterStemmer first
stemmer_ = PorterStemmer()
print("The stemmed form of running is: {}".format(stemmer_.stem("running")))
print("The stemmed form of runs is: {}".format(stemmer_.stem("runs")))
print("The stemmed form of run is: {}".format(stemmer_.stem("run")))


# In[ ]:


# And then Lemmatization
lemm = WordNetLemmatizer()
print("I  case of Lemmatization, running is: {}".format(lemm.lemmatize("running")))
print("I  case of Lemmatization, runs is: {}".format(lemm.lemmatize("runs")))
print("I  case of Lemmatization, is: {}".format(lemm.lemmatize("run")))


# **Stemming** is the process of reducing inflection in words to their root forms such as mapping a group of words to the same stem even if the stem itself is not a valid word in the Language.<br>
# **Lemmatization**, unlike Stemming, reduces the inflected words properly ensuring that the root word belongs to the language. In Lemmatization root word is called Lemma. A lemma (plural lemmas or lemmata) is the canonical form, dictionary form, or citation form of a set of words.<br>
# For example, runs, running, ran are all forms of the word run, therefore run is the lemma of all these words. Because lemmatization returns an actual word of the language, it is used where it is necessary to get valid words.<br>
# 
# Source: [https://www.datacamp.com/community/tutorials/stemming-lemmatization-python](https://www.datacamp.com/community/tutorials/stemming-lemmatization-python)

# There are two aspects to show their differences:
# 
#     A stemmer will return the stem of a word, which needn't be identical to the morphological root of the word. It usually sufficient that related words map to the same stem,even if the stem is not in itself a valid root, while in lemmatisation, it will return the dictionary form of a word, which must be a valid word.
# 
#     In lemmatisation, the part of speech of a word should be first determined and the normalisation rules will be different for different part of speech, while the stemmer operates on a single word without knowledge of the context, and therefore cannot discriminate between words which have different meanings depending on part of speech.
#     
# Source: [https://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization](https://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization)
# 

# In the function below I have commented out the Lemmatizer, and worked only with PorterStemmer.<br>
# The data-cleaning part is inspired from the kernel - [Basic EDA,Cleaning and GloVe](https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove/data)

# In[ ]:


def pre_Process_data(documents):
    '''
    For preprocessing we have regularized, transformed each upper case into lower case, tokenized,
    Normalized and remove stopwords. For normalization, we have used PorterStemmer. Porter stemmer transforms 
    a sentence from this "love loving loved" to this "love love love"
    
    '''
    STOPWORDS = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    #lemm = WordNetLemmatizer()
    Tokenized_Doc=[]
    print("Pre-Processing the Data.........\n")
    for data in documents:
        review = re.sub('[^a-zA-Z]', ' ', data)
        url = re.compile(r'https?://\S+|www\.\S+')
        review = url.sub(r'',review)
        html=re.compile(r'<.*?>')
        review = html.sub(r'',review)
        emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        review = emoji_pattern.sub(r'',review)
        gen_docs = [w.lower() for w in word_tokenize(review)] 
        tokens = [stemmer.stem(token) for token in gen_docs if not token in STOPWORDS]
        #tokens = [lemm.lemmatize(token) for token in gen_docs if not token in STOPWORDS]
        final_=' '.join(tokens)
        Tokenized_Doc.append(final_)
    return Tokenized_Doc


# In[ ]:


def Vectorization(processed_data):
    '''
    Vectorization is an important step in Natural Language Processing. We have
    Used Tf_Idf vectorization in this script. The n_gram range for vectorization 
    lies between 2 and 3, that means minimum and maximum number of words in 
    the sequence that would be vectorized is two and three respectively. There
    are other different types of vectorization algorithms also, which could be added to this 
    function as required.
    
    '''
    vectorizer = TfidfVectorizer(stop_words='english', 
                                    #max_features= 200000, # keep top 200000 terms 
                                    min_df = 3, ngram_range=(2,3),
                                    smooth_idf=True)
    X = vectorizer.fit_transform(processed_data)
    print("\n Shape of the document-term matrix")
    print(X.shape) # check shape of the document-term matrix
    return X, vectorizer


# ## I gained knowledge from them
# I have no certified course in NLP, but I have gained knowledge from kernels created by the Kagglers some of them are masters while some of them are grandmasters. Especially for this kernel, I am very much thankful to [Abishek Thakur](https://www.kaggle.com/abhishek), [Parul Pandey](https://www.kaggle.com/parulpandey), [Anisotropic](https://www.kaggle.com/arthurtok).<br>
# Link to their kernels-
# * https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle
# * https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial
# * https://www.kaggle.com/parulpandey/visualizing-kannada-mnist-with-t-sne
# 
# You must visit these kernels which I have mentioned, I hope they would be very much helpful to you in your way of learning NLP.

# All the algorithms used below in the function "topic_modeling" are all unsupervised learning techniques. You could know more about them if you google them, than if I explain you in this kernel.<br>
# Or I could refer you to one of my kernels, which would also help you to understand them.<br>
# Link to my kernel - https://www.kaggle.com/basu369victor/recommender-system-using-un-supervised-learning

# In[ ]:


def topic_modeling(model,X):
    '''
    We have used three types of decomposition algorithm for unsupervised learning, anyone could 
    be selected with the help of the "model" parameter. Three of them are TruncatedSVD ,Latent
    Dirichlet Allocation and Matrix Factorization. This function is useful for comparing
    different model performances, by switching between different algorithms with the help of 
    the "model" parameter and also more algorithms could be easily added to this function.
    
    '''
    components = 900
    if model=='svd':
        print("\nTrying out Truncated SVD......")
        model_ = TruncatedSVD(n_components=components,n_iter=20)
        model_.fit(X)
    if model=='MF':
        print("\nTrying out Matrix Factorization......")
        model_ = NMF(n_components=components, random_state=1,solver='mu',
                      beta_loss='kullback-leibler', alpha=.1,max_iter=20,
                      l1_ratio=.5).fit(X)
        model_.fit(X)
    if model=='LDA':
        print("\nTrying out Latent Dirichlet Allocation......")
        #Tokenized_Doc=[doc.split() for doc in processed_data]
        #dictionary = Dictionary(Tokenized_Doc)
        #corpus = [dictionary.doc2bow(tokens) for tokens in Tokenized_Doc]
        #model_ = LdaModel(corpus, num_topics=components, id2word = dictionary)
        model_ = LatentDirichletAllocation(n_components=components,n_jobs=-1,
                                           max_iter=20,
                                           random_state=42,verbose=0
                                          )
        model_.fit(X)
    if model=='k-means':
        print("\nTrying out K-Means clustering......")
        true_k = 2
        model_ = KMeans(n_clusters=components, init='k-means++',max_iter=20, n_init=1)
        model_.fit(X)
        
    X = model_.transform(X)
    
    scl = preprocessing.StandardScaler()
    scl.fit(X)
    x_scl = scl.transform(X)

    return x_scl


# In[ ]:


def Visualize_clusters(X_topics, title):
    '''
    This function is used to visualize the clusters generated by our 
    model through unsupervised learning. We have used UMAP for better 
    visualization of clusters.
    
    '''
    #embedding = umap.UMAP(n_neighbors=30,
    #                        min_dist=0.0,
    #                        n_components=2,).fit_transform(X_topics)#20
    embedding = TSNE(n_components=2, 
                     verbose=1, random_state=0, angle=0.75).fit_transform(X_topics)
    
    plt.figure(figsize=(20,20))
    plt.gca().set_aspect('equal', 'datalim')
    #plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title(title,fontsize=16)
    plt.scatter(embedding[:, 0], embedding[:, 1], 
    c = train['target'], cmap='Spectral', alpha=0.7,
    s = 20, # size
    )
    plt.show()


# In[ ]:


processed_data = pre_Process_data(train['text'])


# In[ ]:


X, vectorizer = Vectorization(processed_data)


# In[ ]:


X_transform_1 = topic_modeling('svd',X)


# In[ ]:


Visualize_clusters(X_transform_1, "Clustering for Truncated SVD")


# In[ ]:


#X, vectorizer = Vectorization(processed_data)
X_transform_2 = topic_modeling('LDA',X)
Visualize_clusters(X_transform_2, "Clustering for LDA")


# In[ ]:


X_transform_3 = topic_modeling('MF',X)
Visualize_clusters(X_transform_3, "Clustering for MF")


# In[ ]:


X_transform_4 = topic_modeling('k-means',X)
Visualize_clusters(X_transform_4, "Clustering for K-Means")


# In[ ]:


y = train['target']
#xtrain, xvalid, ytrain, yvalid = train_test_split(X_transform_3, y, 
#                                                  stratify=y, 
#                                                  random_state=42, 
#                                                  shuffle=True)


# In[ ]:


def evaluate_performance(model):
    kf = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    i=1
    model_list=[]
    for train_index,test_index in kf.split(X_transform_3, y):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtrain,xvalid = X_transform_3[train_index],X_transform_3[test_index]
        ytrain,yvalid = y[train_index],y[test_index]
        model.fit(xtrain, ytrain)
        predictions = model.predict(xvalid)
        print("Accuracy score: "+str(accuracy_score(predictions,yvalid)))
        print("ROC score: "+str(roc_auc_score(predictions,yvalid)))
        print("Log Loss: "+ str(log_loss(predictions,yvalid)))
        i=i+1
        model_list.append(model)
    return model_list


# In[ ]:


clf = lgb.LGBMClassifier(max_depth=12,
                             learning_rate=0.5,
                             n_estimators = 1000,
                             subsample=0.25,
                           )
model_1 = evaluate_performance(clf)


# In[ ]:


clf = xgb.XGBClassifier(max_depth=12, n_estimators=600, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.5)
model_2 = evaluate_performance(clf)


# In[ ]:


clf = LogisticRegression()
model_3 = evaluate_performance(clf)


# In[ ]:


clf = KNeighborsClassifier(n_neighbors=2)
model_4 = evaluate_performance(clf)


# In[ ]:


clf = ExtraTreesClassifier(n_estimators=600,max_depth=12)
model_5 = evaluate_performance(clf)


# In[ ]:


clf = CatBoostClassifier(max_depth=12, n_estimators=600, learning_rate=0.5,verbose=0)
model_6 = evaluate_performance(clf)


# In[ ]:


processed_data = pre_Process_data(test['text'])
X, vectorizer = Vectorization(processed_data)
X_transform_1 = topic_modeling('MF',X)


# In[ ]:


prob=model_5[2].predict(X_transform_1)


# In[ ]:


my_submission = pd.DataFrame({'id': test['id'], 'target': prob})
my_submission.to_csv('SubmissionVictor.csv', index=False)
my_submission.head()


# ## The END
# If you have come till here, then I hope You have enjoyed it.<br>
# I have learned from Kagglers sharing through open-source. That's why I would also love to share my knowledge.<br>
# Feel free to say it, if you could improve this kernel any further, I would definitely listen to them.<br>
# ### Till then, do upvote if U think this kernel was informative and if you enjoyed it.
