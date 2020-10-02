#!/usr/bin/env python
# coding: utf-8

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

# Any results you write to the current directory are saved as output.


# In[ ]:


#Importing Library
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore") #for Ignoring Warnings

import sqlite3                 #To get the data from Database
import pandas as pd            #
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm
import os

from plotly import plotly
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()
from collections import Counter

#from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
#from collections import counter
#from sklearn.model_selection import cross_Validate

from sklearn.model_selection import train_test_split


# In[ ]:


#Importing the project data and resource data
project_data = pd.read_csv('../input/donorschoose-application-screening/train.csv')
resource_data = pd.read_csv('../input/donorschoose-application-screening/resources.csv')


# In[ ]:


#Shape of the project and resource data
print("project data shape" , project_data.shape)
print("resource data shape", resource_data.shape)


# In[ ]:


#display 1st five columns of project data
display(project_data.head())


# In[ ]:


#display 1st five columns of resource data
display(resource_data.head())


# In[ ]:


# Class value count in project data
project_data['project_is_approved'].value_counts()


# In[ ]:


'''#Downsampling (project_is_approved==1) class data to balance the class
#https://elitedatascience.com/imbalanced-classes

from sklearn.utils import resample
from sklearn.utils import shuffle
majority = project_data[project_data['project_is_approved']==1]
minority = project_data[project_data['project_is_approved']==0]

majority_downsample = resample(majority,replace=False,n_samples=27734,random_state=123)

project_data = pd.concat([majority_downsample,minority])

project_data['project_is_approved'].value_counts()

project_data = shuffle(project_data)  '''


# In[ ]:


#merge resource data with project data
# https://stackoverflow.com/questions/22407798/how-to-reset-a-dataframes-indexes-for-all-groups-in-one-step
price_data = resource_data.groupby('id').agg({'price':'sum','quantity':'sum'}).reset_index()


# In[ ]:


#merge all essay
project_data["essay"] = project_data["project_essay_1"].map(str) +                        project_data["project_essay_2"].map(str) +                         project_data["project_essay_3"].map(str) +                         project_data["project_essay_4"].map(str)


# In[ ]:


project_data.drop(['project_essay_1','project_essay_2','project_essay_3','project_essay_4'],axis=1,inplace=True)


# In[ ]:


#join two dataframes 
project_data =pd.merge(project_data,price_data,on='id',how='left',copy=True)


# In[ ]:


project_data.columns


# **TEXT PREPROCESSING**

# *ESSAY*

# In[ ]:


# https://stackoverflow.com/a/47091490/4084039
import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub('[^A-Za-z0-9]+', ' ', phrase)
    return phrase


# In[ ]:


stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"]


# In[ ]:


project_data = project_data[:100]
project_data['project_is_approved'].value_counts()


# In[ ]:


# Reading glove vectors in python: https://stackoverflow.com/a/38230349/4084039
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r', encoding="utf8")
    model = {}
    for line in tqdm(f):
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model
model = loadGloveModel('../input/glove2word2vec/glove_w2v.txt')


# In[ ]:


# Combining all the above statemennts 
from tqdm import tqdm
preprocessed_essays = []
# tqdm is for printing the status bar
for sentance in tqdm(project_data['essay'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    #'[^A-Za-z0-9]+'
    sent1=[]
    for e in sent.split():
        if e not in stopwords:
            
            if e.isdigit():
                sent1.append('')
            else:
                sent1.append(e)
        s= ' '.join(sent1)
    preprocessed_essays.append(s.lower())
        
   
    

    
  


# In[ ]:


sub_cat_list=[]
for i in project_data['project_subject_subcategories']:
    temp = ""
    for j in i.split(','):
        if 'The' in j.split():
            j=j.replace('The','')
        j=j.replace(' ','')
        j=j.replace('&','_')
        temp= temp+" "+j
        #print(temp)
    sub_cat_list.append(temp)
    
#print(sub_cat_list)
project_data['clean_subcategories']=sub_cat_list
#project_data.drop(['project_subject_subcategories'],axis=1,inplace=True)


    

        


# In[ ]:


cat_list=[]
for i in project_data['project_subject_categories']:
    temp = ""
    for j in i.split(','):
        if 'The' in j.split():
            j=j.replace('The','')
        j=j.replace(' ','')
        j=j.replace('&','_')
        temp= temp+" "+j
        #print(temp)
    cat_list.append(temp)
    
#print(sub_cat_list)
project_data['clean_categories']=cat_list
#project_data.drop(['project_subject_categories'],axis=1,inplace=True)


# In[ ]:


project_data.head()


# In[ ]:


project_data['essay']=preprocessed_essays


# In[ ]:


knn_data = pd.DataFrame(columns=['clean_categories', 'clean_subcategories', 'title','price','essay'])
knn_data_label = pd.DataFrame(columns=['label'])
knn_data['clean_categories']=project_data['clean_categories']
knn_data['clean_subcategories']=project_data['clean_subcategories']
knn_data['title']=project_data['project_title']
knn_data['price']=project_data['price']
knn_data['essay']=project_data['essay']
knn_data_label['label']= project_data['project_is_approved']



# In[ ]:


from sklearn.utils import resample
from sklearn.utils import shuffle
x_1,x_test,y_1,y_test = train_test_split(knn_data,knn_data_label,test_size=0.3,random_state=0)
x_tr,x_cv,y_tr,y_cv = train_test_split(x_1,y_1,test_size=0.3)



x_tr['label']= y_tr['label']
train_data_0 = x_tr[x_tr['label']==0]
train_data_1 = x_tr[x_tr['label']==1]
count_class_1, count_class_0 = x_tr.label.value_counts()
train_data_0 = train_data_0.sample(count_class_1,replace=True)
train_data = pd.concat([train_data_0,train_data_1])
train_data = shuffle(train_data)
y_tr=train_data['label']
y_tr.value_counts()
x_tr= train_data.drop(columns="label")



# In[ ]:


#y_tr.label.value_counts()


# In[ ]:


#BOW#
#essay
#TRAIN
count_vect = CountVectorizer() 
count_vect.fit(x_tr['essay'])
bow_essay_tr = count_vect.transform(x_tr['essay'])
display(bow_essay_tr.shape)

#essay
#cv
bow_essay_cv = count_vect.transform(x_cv['essay'])
display(bow_essay_cv.shape)

#essay
#test
bow_essay_test = count_vect.transform(x_test['essay'])
display(bow_essay_test.shape)



# In[ ]:


#BOW#
#title
#TRAIN
count_vect = CountVectorizer() 
count_vect.fit(x_tr['title'])
bow_title_tr = count_vect.transform(x_tr['title'])
display(bow_title_tr.shape)

#title
#cv
bow_title_cv = count_vect.transform(x_cv['title'])
display(bow_title_cv.shape)

#title
#test
bow_title_test = count_vect.transform(x_test['title'])
display(bow_title_test.shape)


# In[ ]:


#BOW#
#categories
#TRAIN
count_vect = CountVectorizer() 
count_vect.fit(x_tr['clean_categories'])
bow_categories_tr = count_vect.transform(x_tr['clean_categories'])
display(bow_categories_tr.shape)

#categories
#cv
bow_categories_cv = count_vect.transform(x_cv['clean_categories'])
display(bow_categories_cv.shape)

#categories
#test
bow_categories_test = count_vect.transform(x_test['clean_categories'])
display(bow_categories_test.shape)


# In[ ]:


#BOW#
#subcategories
#TRAIN
count_vect = CountVectorizer() 
count_vect.fit(x_tr['clean_subcategories'])
bow_subcategories_tr = count_vect.transform(x_tr['clean_subcategories'])
display(bow_subcategories_tr.shape)

#subcategories
#cv
bow_subcategories_cv = count_vect.transform(x_cv['clean_subcategories'])
display(bow_subcategories_cv.shape)

#subcategories
#test
bow_subcategories_test = count_vect.transform(x_test['clean_subcategories'])
display(bow_subcategories_test.shape)


# In[ ]:


#TFIDF
#title
#train

tf_idf_vect = TfidfVectorizer(min_df=10)
tf_idf_vect.fit(x_tr['title'])
tfidf_title_tr = tf_idf_vect.transform(x_tr['title'])
print(tfidf_title_tr.shape)

#TFIDF
#title
#cv

tfidf_title_cv = tf_idf_vect.transform(x_cv['title'])
print(tfidf_title_cv.shape)

#TFIDF
#title
#test

tfidf_title_test = tf_idf_vect.transform(x_test['title'])
print(tfidf_title_test.shape)


# In[ ]:


#TFIDF
#essay
#train

tf_idf_vect = TfidfVectorizer(min_df=10)
tf_idf_vect.fit(x_tr['essay'])
tfidf_essay_tr = tf_idf_vect.transform(x_tr['essay'])
print(tfidf_essay_tr.shape)

#TFIDF
#essay
#cv

tfidf_essay_cv = tf_idf_vect.transform(x_cv['essay'])
print(tfidf_essay_cv.shape)

#TFIDF
#essay
#test

tfidf_essay_test = tf_idf_vect.transform(x_test['essay'])
print(tfidf_essay_test.shape)


# In[ ]:


#TFIDF
#categories
#train

tf_idf_vect = TfidfVectorizer(min_df=10)
tf_idf_vect.fit(x_tr['clean_categories'])
tfidf_categories_tr = tf_idf_vect.transform(x_tr['clean_categories'])
print(tfidf_categories_tr.shape)

#TFIDF
#categories
#cv

tf_idf_vect = TfidfVectorizer(min_df=10)
tf_idf_vect.fit(x_cv['clean_categories'])
tfidf_categories_cv = tf_idf_vect.transform(x_cv['clean_categories'])
print(tfidf_categories_cv.shape)

#TFIDF
#categories
#test

tf_idf_vect = TfidfVectorizer(min_df=10)
tf_idf_vect.fit(x_test['clean_categories'])
tfidf_categories_test = tf_idf_vect.transform(x_test['clean_categories'])
print(tfidf_categories_test.shape)


# In[ ]:


#TFIDF
#subcategories
#train

tf_idf_vect = TfidfVectorizer(min_df=10)
tf_idf_vect.fit(x_tr['clean_subcategories'])
tfidf_subcategories_tr = tf_idf_vect.transform(x_tr['clean_subcategories'])
print(tfidf_subcategories_tr.shape)

#TFIDF
#subcategories
#cv


tfidf_subcategories_cv = tf_idf_vect.transform(x_cv['clean_subcategories'])
print(tfidf_subcategories_cv.shape)

#TFIDF
#subacategories
#test


tfidf_subcategories_test = tf_idf_vect.transform(x_test['clean_subcategories'])
print(tfidf_subcategories_test.shape)


# In[ ]:


# average Word2Vec
# compute average word2vec for each essay.
#Train
avg_w2v_essay_tr = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_tr['essay']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in model :
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_essay_tr.append(vector)
    
avg_w2v_essay_tr=np.asarray(avg_w2v_essay_tr)  #converting list into array
display(avg_w2v_essay_tr.shape)

#CV
avg_w2v_essay_cv = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_cv['essay']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in model :
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_essay_cv.append(vector)
    
avg_w2v_essay_cv=np.asarray(avg_w2v_essay_cv)
display(avg_w2v_essay_cv.shape)


#TEST
avg_w2v_essay_test = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_test['essay']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in model :
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_essay_test.append(vector)
avg_w2v_essay_test=np.asarray(avg_w2v_essay_test)
display(avg_w2v_essay_test.shape)

    


# In[ ]:


# average Word2Vec
# compute average word2vec for each title.
#Train
avg_w2v_title_tr = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_tr['title']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in model :
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_title_tr.append(vector)
    
avg_w2v_title_tr=np.asarray(avg_w2v_title_tr)  #converting list into array
display(avg_w2v_title_tr.shape)

#CV
avg_w2v_title_cv = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_cv['title']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in model :
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_title_cv.append(vector)
    
avg_w2v_title_cv=np.asarray(avg_w2v_title_cv)
display(avg_w2v_title_cv.shape)


#TEST
avg_w2v_title_test = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_test['title']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in model :
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_title_test.append(vector)
avg_w2v_title_test=np.asarray(avg_w2v_title_test)
display(avg_w2v_title_test.shape)


# In[ ]:


# average Word2Vec
# compute average word2vec for each subcategories.
#Train
avg_w2v_clean_subcategories_tr = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_tr['clean_subcategories']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in model :
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_clean_subcategories_tr.append(vector)
    
avg_w2v_clean_subcategories_tr=np.asarray(avg_w2v_clean_subcategories_tr)  #converting list into array
display(avg_w2v_clean_subcategories_tr.shape)

#CV
avg_w2v_clean_subcategories_cv = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_cv['clean_subcategories']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in model :
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_clean_subcategories_cv.append(vector)
    
avg_w2v_clean_subcategories_cv=np.asarray(avg_w2v_clean_subcategories_cv)
display(avg_w2v_clean_subcategories_cv.shape)


#TEST
avg_w2v_clean_subcategories_test = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_test['clean_subcategories']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in model :
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_clean_subcategories_test.append(vector)
avg_w2v_clean_subcategories_test=np.asarray(avg_w2v_clean_subcategories_test)
display(avg_w2v_clean_subcategories_test.shape)


# In[ ]:


# average Word2Vec
# compute average word2vec for each categories.
#Train
avg_w2v_clean_categories_tr = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_tr['clean_categories']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in model :
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_clean_categories_tr.append(vector)
    
avg_w2v_clean_categories_tr=np.asarray(avg_w2v_clean_categories_tr)  #converting list into array
display(avg_w2v_clean_categories_tr.shape)

#CV
avg_w2v_clean_categories_cv = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_cv['clean_categories']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in model :
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_clean_categories_cv.append(vector)
    
avg_w2v_clean_categories_cv=np.asarray(avg_w2v_clean_categories_cv)
display(avg_w2v_clean_categories_cv.shape)


#TEST
avg_w2v_clean_categories_test = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_test['clean_categories']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if word in model :
            vector += model[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_w2v_clean_categories_test.append(vector)
avg_w2v_clean_categories_test=np.asarray(avg_w2v_clean_categories_test)
display(avg_w2v_clean_categories_test.shape)


# In[ ]:


#TFIDF weighted W2v
#Essay
#train
tfidf_model = TfidfVectorizer()
tfidf_model.fit(x_tr['essay'])
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_)))
tfidf_words = set(tfidf_model.get_feature_names())


tfidf_w2v_essay_tr = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_tr['essay']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in model) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_w2v_essay_tr.append(vector)


tfidf_w2v_essay_tr=np.asarray(tfidf_w2v_essay_tr)
display(tfidf_w2v_essay_tr.shape)

#CV

tfidf_w2v_essay_cv = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_cv['essay']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in model) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_w2v_essay_cv.append(vector)


tfidf_w2v_essay_cv=np.asarray(tfidf_w2v_essay_cv)
display(tfidf_w2v_essay_cv.shape)

#Test

tfidf_w2v_essay_test = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_test['essay']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in model) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_w2v_essay_test.append(vector)


tfidf_w2v_essay_test=np.asarray(tfidf_w2v_essay_test)
display(tfidf_w2v_essay_test.shape)


# In[ ]:


#TFIDF weighted W2v
#Title
#train
tfidf_model = TfidfVectorizer()
tfidf_model.fit(x_tr['title'])
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_)))
tfidf_words = set(tfidf_model.get_feature_names())


tfidf_w2v_title_tr = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_tr['title']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in model) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_w2v_title_tr.append(vector)


tfidf_w2v_title_tr=np.asarray(tfidf_w2v_title_tr)
display(tfidf_w2v_title_tr.shape)

#CV

tfidf_w2v_title_cv = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_cv['title']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in model) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_w2v_title_cv.append(vector)


tfidf_w2v_title_cv=np.asarray(tfidf_w2v_title_cv)
display(tfidf_w2v_title_cv.shape)

#Test

tfidf_w2v_title_test = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_test['title']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in model) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_w2v_title_test.append(vector)


tfidf_w2v_title_test=np.asarray(tfidf_w2v_title_test)
display(tfidf_w2v_title_test.shape)


# In[ ]:


#TFIDF weighted W2v
#subcategories
#train
tfidf_model = TfidfVectorizer()
tfidf_model.fit(x_tr['clean_subcategories'])
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_)))
tfidf_words = set(tfidf_model.get_feature_names())


tfidf_w2v_clean_subcategories_tr = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_tr['clean_subcategories']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in model) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_w2v_clean_subcategories_tr.append(vector)


tfidf_w2v_clean_subcategories_tr=np.asarray(tfidf_w2v_clean_subcategories_tr)
display(tfidf_w2v_clean_subcategories_tr.shape)

#CV

tfidf_w2v_clean_subcategories_cv = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_cv['clean_subcategories']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in model) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_w2v_clean_subcategories_cv.append(vector)


tfidf_w2v_clean_subcategories_cv=np.asarray(tfidf_w2v_clean_subcategories_cv)
display(tfidf_w2v_clean_subcategories_cv.shape)

#Test

tfidf_w2v_clean_subcategories_test = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_test['clean_subcategories']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in model) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_w2v_clean_subcategories_test.append(vector)


tfidf_w2v_clean_subcategories_test=np.asarray(tfidf_w2v_clean_subcategories_test)
display(tfidf_w2v_clean_subcategories_test.shape)


# In[ ]:


#TFIDF weighted W2v
#subcategories
#train
tfidf_model = TfidfVectorizer()
tfidf_model.fit(x_tr['clean_categories'])
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_)))
tfidf_words = set(tfidf_model.get_feature_names())


tfidf_w2v_clean_categories_tr = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_tr['clean_categories']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in model) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_w2v_clean_categories_tr.append(vector)


tfidf_w2v_clean_categories_tr=np.asarray(tfidf_w2v_clean_categories_tr)
display(tfidf_w2v_clean_categories_tr.shape)

#CV

tfidf_w2v_clean_categories_cv = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_cv['clean_categories']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in model) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_w2v_clean_categories_cv.append(vector)


tfidf_w2v_clean_categories_cv=np.asarray(tfidf_w2v_clean_categories_cv)
display(tfidf_w2v_clean_categories_cv.shape)

#Test

tfidf_w2v_clean_categories_test = []; # the avg-w2v for each sentence/review is stored in this list
for sentence in tqdm(x_test['clean_categories']): # for each review/sentence
    vector = np.zeros(200) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in model) and (word in tfidf_words):
            vec = model[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_w2v_clean_categories_test.append(vector)


tfidf_w2v_clean_categories_test=np.asarray(tfidf_w2v_clean_categories_test)
display(tfidf_w2v_clean_categories_test.shape)


# In[ ]:


#price standardized
#train

# check this one: https://www.youtube.com/watch?v=0HOqOcln3Z4&t=530s
# standardization sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn.preprocessing import StandardScaler

# price_standardized = standardScalar.fit(project_data['price'].values)
# this will rise the error
# ValueError: Expected 2D array, got 1D array instead: array=[725.05 213.03 329.   ... 399.   287.73   5.5 ].
# Reshape your data either using array.reshape(-1, 1)

price_scalar = StandardScaler()
price_scalar.fit(x_tr['price'].values.reshape(-1,1)) # finding the mean and standard deviation of this data
print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")

# Now standardize the data with above maen and variance.
price_standardized_tr = price_scalar.transform(x_tr['price'].values.reshape(-1, 1))

#cv

price_scalar = StandardScaler()
price_scalar.fit(x_cv['price'].values.reshape(-1,1)) 
print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")

price_standardized_cv = price_scalar.transform(x_cv['price'].values.reshape(-1, 1))

#test

price_scalar = StandardScaler()
price_scalar.fit(x_test['price'].values.reshape(-1,1))
print(f"Mean : {price_scalar.mean_[0]}, Standard deviation : {np.sqrt(price_scalar.var_[0])}")

price_standardized_test = price_scalar.transform(x_test['price'].values.reshape(-1, 1))


# In[ ]:


#SET1::

# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039
from scipy.sparse import hstack
# with the same hstack function we are concatinating a sparse matrix and a dense matirx :)
X_tr =hstack((bow_categories_tr, bow_subcategories_tr,bow_essay_tr, price_standardized_tr,bow_title_tr))
X_cv=hstack((bow_categories_cv, bow_subcategories_cv,bow_essay_cv, price_standardized_cv,bow_title_cv))
X_test=hstack((bow_categories_test, bow_subcategories_test,bow_essay_test,price_standardized_test,bow_title_test))


# In[ ]:


#https://www.geeksforgeeks.org/confusion-matrix-machine-learning/

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

Accuracy_cv=[]
Accuracy_tr=[]
k = []
f1 = []
Auc_value = []
prob = []
for i in range(1,30,2):
    #instantiate learning model (k=30)
    knn= KNeighborsClassifier(n_neighbors=i)
    
    
    #fitting the model on crossvalidation train
    knn.fit(X_tr,np.ravel(y_tr))
    
    #predicting the response of the crossvalidation train 
    pred_cv = knn.predict(X_cv)
    pred_tr = knn.predict(X_tr)
    
    
    
    #evaluate the Cv accuracy
    acc_cv = accuracy_score(y_cv,pred_cv,normalize=True)*float(100)
    acc_tr = accuracy_score(y_tr,pred_tr,normalize=True)*float(100)
    Accuracy_cv.append(acc_cv)
    Accuracy_tr.append(acc_tr)
    k.append(i)
    
    Results = confusion_matrix(y_cv,pred_cv) 
    prec = Results[1][1]/(Results[1][1]+Results[0][1])
    rec =  Results[1][1]/(Results[1][1]+Results[1][0])
    mult = prec*rec
    add =  prec+rec
    result = (mult/add)
    result = 2*result
    f1.append(result)
    
    prob_score= knn.predict_proba(X_cv)
    prob_psv = prob_score[:,1]
    fpr,tpr,threshold = metrics.roc_curve(y_cv,prob_psv)
    value = auc(fpr,tpr)
    Auc_value.append(value)
       
    
Acc_table = pd.DataFrame(columns=['K-value','Accuracy_cv','Accuracy_tr','F1_score_cv','Auc_value'])
Acc_table['K-value']=k
Acc_table['Accuracy_cv']=Accuracy_cv
Acc_table['Accuracy_tr']=Accuracy_tr
Acc_table['F1_score_cv']=f1
Acc_table['Auc_value']=Auc_value
Acc_table.sort_values(["Auc_value"],axis=0,ascending=False,inplace= True)

display(Acc_table)
########################    


# In[ ]:


#Accuracy plot of Training and CV Data vs K-value
Acc_table1=Acc_table.sort_values(["K-value"],axis=0,ascending=False,inplace=False)
a=Acc_table1['K-value']
b=Acc_table1['Accuracy_tr']
c=Acc_table1['K-value']
d=Acc_table1['Accuracy_cv']
plt.title('Accuracy plot of Training and test Data vs K-value')
plt.plot(a ,b,label='training accuracy' )
plt.plot(c,d,label='CV accuracy')
plt.legend()
plt.xlabel('Number of Neighbours')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:



print(Acc_table['Accuracy_tr'].sort_values().values)


# In[ ]:


I = int(Acc_table.iloc[0][0])
print(I)

knn= KNeighborsClassifier(n_neighbors=I)

#fitting the model on crossvalidation train
knn.fit(X_tr,y_tr)

#predicting the response of the crossvalidation train
pred = knn.predict(X_test)


#evaluate the Cv accuracy
acc = accuracy_score(y_test,pred,normalize=True)*float(100)
print(acc)

confusion_mat = confusion_matrix(y_test,pred)
print(confusion_mat)
df_cm = pd.DataFrame(confusion_mat,range(2),range(2))
sns.heatmap(df_cm,annot=True,fmt='g')
plt.title('Confusion Matrix of the Test Data')
plt.xlabel('Actual Class label')
plt.ylabel('Predicted Class label')


# ROC CURVE

# In[ ]:


# calculating probability of the class
from sklearn import metrics
prob_score = []
prob_score= knn.predict_proba(X_test)
prob_psv = prob_score[:,1]

#ROC curve

fpr,tpr,threshold = metrics.roc_curve(y_test,prob_psv)
roc_auc = auc(fpr,tpr)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('knn ROC CURVE')
plt.legend()
plt.show()


# In[ ]:


prob_score
prob_psv


# In[ ]:


#SET2::
X_tr = hstack((bow_categories_tr, bow_subcategories_tr,bow_essay_tr, price_standardized_tr,tfidf_title_tr,tfidf_essay_tr))
X_cv=hstack((bow_categories_cv, bow_subcategories_cv,bow_essay_cv, price_standardized_cv,tfidf_title_cv,tfidf_essay_cv))
X_test=hstack((bow_categories_test, bow_subcategories_test,bow_essay_test,price_standardized_test,tfidf_title_test,tfidf_essay_test))


# In[ ]:


Accuracy_cv=[]
Accuracy_tr=[]
k = []
f1 = []
Auc_value = []
prob = []
for i in range(1,30,2):
    #instantiate learning model (k=30)
    knn= KNeighborsClassifier(n_neighbors=i)
    
    
    #fitting the model on crossvalidation train
    knn.fit(X_tr,np.ravel(y_tr))
    
    #predicting the response of the crossvalidation train 
    pred_cv = knn.predict(X_cv)
    pred_tr = knn.predict(X_tr)
    
    
    
    #evaluate the Cv accuracy
    acc_cv = accuracy_score(y_cv,pred_cv,normalize=True)*float(100)
    acc_tr = accuracy_score(y_tr,pred_tr,normalize=True)*float(100)
    Accuracy_cv.append(acc_cv)
    Accuracy_tr.append(acc_tr)
    k.append(i)
    
    Results = confusion_matrix(y_cv,pred_cv) 
    prec = Results[1][1]/(Results[1][1]+Results[0][1])
    rec =  Results[1][1]/(Results[1][1]+Results[1][0])
    mult = prec*rec
    add =  prec+rec
    result = (mult/add)
    result = 2*result
    f1.append(result)
    
    prob_score= knn.predict_proba(X_cv)
    prob_psv = prob_score[:,1]
    fpr,tpr,threshold = metrics.roc_curve(y_cv,prob_psv)
    value = auc(fpr,tpr)
    Auc_value.append(value)
       
    
Acc_table = pd.DataFrame(columns=['K-value','Accuracy_cv','Accuracy_tr','F1_score_cv','Auc_value'])
Acc_table['K-value']=k
Acc_table['Accuracy_cv']=Accuracy_cv
Acc_table['Accuracy_tr']=Accuracy_tr
Acc_table['F1_score_cv']=f1
Acc_table['Auc_value']=Auc_value
Acc_table.sort_values(["Auc_value"],axis=0,ascending=False,inplace= True)

display(Acc_table)
######################## 
    


# In[ ]:


#Accuracy plot of Training and CV Data vs K-value
Acc_table1=Acc_table.sort_values(["K-value"],axis=0,ascending=False,inplace=False)
a=Acc_table1['K-value']
b=Acc_table1['Accuracy_tr']
c=Acc_table1['K-value']
d=Acc_table1['Accuracy_cv']
plt.title('Accuracy plot of Training and test Data vs K-value')
plt.plot(a ,b,label='training accuracy' )
plt.plot(c,d,label='CV accuracy')
plt.legend()
plt.xlabel('Number of Neighbours')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:


I = int(Acc_table.iloc[0][0])
print(I)

knn= KNeighborsClassifier(n_neighbors=I)

#fitting the model on crossvalidation train
knn.fit(X_tr,y_tr)

#predicting the response of the crossvalidation train
pred = knn.predict(X_test)


#evaluate the Cv accuracy
acc = accuracy_score(y_test,pred,normalize=True)*float(100)
print(acc)

confusion_mat = confusion_matrix(y_test,pred)
print(confusion_mat)
df_cm = pd.DataFrame(confusion_mat,range(2),range(2))
sns.heatmap(df_cm,annot=True,fmt='g')
plt.title('Confusion Matrix of the Test Data')
plt.xlabel('Actual Class label')
plt.ylabel('Predicted Class label')


# In[ ]:


# calculating probability of the class
from sklearn import metrics
prob_score = []
prob_score= knn.predict_proba(X_test)
prob_psv = prob_score[:,1]

#ROC curve

fpr,tpr,threshold = metrics.roc_curve(y_test,prob_psv)
roc_auc = auc(fpr,tpr)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('knn ROC CURVE')
plt.legend()
plt.show()


# In[ ]:


#Taking Top 2000 feature and appling on SET2
#https://www.geeksforgeeks.org/ml-chi-square-test-for-feature-selection/
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectKBest, f_classif

chi2_features = SelectKBest(f_classif,k=2000)
X_tr = X_tr.astype(int)
X_cv = X_cv.astype(int)
X_test = X_test.astype(int)

X_tr = chi2_features.fit_transform(X_tr,y_tr)
X_cv = chi2_features.fit_transform(X_cv,y_cv)
X_test = chi2_features.fit_transform(X_test,y_test)


# In[ ]:


Accuracy_cv=[]
Accuracy_tr=[]
k = []
f1 = []
Auc_value = []
prob = []
for i in range(1,30,2):
    #instantiate learning model (k=30)
    knn= KNeighborsClassifier(n_neighbors=i)
    
    
    #fitting the model on crossvalidation train
    knn.fit(X_tr,np.ravel(y_tr))
    
    #predicting the response of the crossvalidation train 
    pred_cv = knn.predict(X_cv)
    pred_tr = knn.predict(X_tr)
    
    
    
    #evaluate the Cv accuracy
    acc_cv = accuracy_score(y_cv,pred_cv,normalize=True)*float(100)
    acc_tr = accuracy_score(y_tr,pred_tr,normalize=True)*float(100)
    Accuracy_cv.append(acc_cv)
    Accuracy_tr.append(acc_tr)
    k.append(i)
    
    Results = confusion_matrix(y_cv,pred_cv) 
    prec = Results[1][1]/(Results[1][1]+Results[0][1])
    rec =  Results[1][1]/(Results[1][1]+Results[1][0])
    mult = prec*rec
    add =  prec+rec
    result = (mult/add)
    result = 2*result
    f1.append(result)
    
    prob_score= knn.predict_proba(X_cv)
    prob_psv = prob_score[:,1]
    fpr,tpr,threshold = metrics.roc_curve(y_cv,prob_psv)
    value = auc(fpr,tpr)
    Auc_value.append(value)
       
    
Acc_table = pd.DataFrame(columns=['K-value','Accuracy_cv','Accuracy_tr','F1_score_cv','Auc_value'])
Acc_table['K-value']=k
Acc_table['Accuracy_cv']=Accuracy_cv
Acc_table['Accuracy_tr']=Accuracy_tr
Acc_table['F1_score_cv']=f1
Acc_table['Auc_value']=Auc_value
Acc_table.sort_values(["Auc_value"],axis=0,ascending=False,inplace= True)

display(Acc_table)
########################


# In[ ]:


#Accuracy plot of Training and CV Data vs K-value
Acc_table1=Acc_table.sort_values(["K-value"],axis=0,ascending=False,inplace=False)
a=Acc_table1['K-value']
b=Acc_table1['Accuracy_tr']
c=Acc_table1['K-value']
d=Acc_table1['Accuracy_cv']
plt.title('Accuracy plot of Training and test Data vs K-value')
plt.plot(a ,b,label='training accuracy' )
plt.plot(c,d,label='CV accuracy')
plt.legend()
plt.xlabel('Number of Neighbours')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:


I = int(Acc_table.iloc[0][0])
print(I)

knn= KNeighborsClassifier(n_neighbors=I)

#fitting the model on crossvalidation train
knn.fit(X_tr,y_tr)

#predicting the response of the crossvalidation train
pred = knn.predict(X_test)


#evaluate the Cv accuracy
acc = accuracy_score(y_test,pred,normalize=True)*float(100)
print(acc)

confusion_mat = confusion_matrix(y_test,pred)
print(confusion_mat)
df_cm = pd.DataFrame(confusion_mat,range(2),range(2))
sns.heatmap(df_cm,annot=True,fmt='g')
plt.title('Confusion Matrix of the Test Data')
plt.xlabel('Actual Class label')
plt.ylabel('Predicted Class label')


# In[ ]:


# calculating probability of the class
from sklearn import metrics
prob_score = []
prob_score= knn.predict_proba(X_test)
prob_psv = prob_score[:,1]

#ROC curve

fpr,tpr,threshold = metrics.roc_curve(y_test,prob_psv)
roc_auc = auc(fpr,tpr)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('knn ROC CURVE')
plt.legend()
plt.show()


# In[ ]:


#SET3::
X_tr = hstack((bow_categories_tr, bow_subcategories_tr,bow_essay_tr, price_standardized_tr,avg_w2v_title_tr,avg_w2v_essay_tr))
X_cv=hstack((bow_categories_cv, bow_subcategories_cv,bow_essay_cv, price_standardized_cv,avg_w2v_title_cv,avg_w2v_essay_cv))
X_test=hstack((bow_categories_test, bow_subcategories_test,bow_essay_test,price_standardized_test,avg_w2v_title_test,avg_w2v_essay_test))


# In[ ]:


Accuracy_cv=[]
Accuracy_tr=[]
k = []
f1 = []
Auc_value = []
prob = []
for i in range(1,30,2):
    #instantiate learning model (k=30)
    knn= KNeighborsClassifier(n_neighbors=i)
    
    
    #fitting the model on crossvalidation train
    knn.fit(X_tr,np.ravel(y_tr))
    
    #predicting the response of the crossvalidation train 
    pred_cv = knn.predict(X_cv)
    pred_tr = knn.predict(X_tr)
    
    
    
    #evaluate the Cv accuracy
    acc_cv = accuracy_score(y_cv,pred_cv,normalize=True)*float(100)
    acc_tr = accuracy_score(y_tr,pred_tr,normalize=True)*float(100)
    Accuracy_cv.append(acc_cv)
    Accuracy_tr.append(acc_tr)
    k.append(i)
    
    Results = confusion_matrix(y_cv,pred_cv) 
    prec = Results[1][1]/(Results[1][1]+Results[0][1])
    rec =  Results[1][1]/(Results[1][1]+Results[1][0])
    mult = prec*rec
    add =  prec+rec
    result = (mult/add)
    result = 2*result
    f1.append(result)
    
    prob_score= knn.predict_proba(X_cv)
    prob_psv = prob_score[:,1]
    fpr,tpr,threshold = metrics.roc_curve(y_cv,prob_psv)
    value = auc(fpr,tpr)
    Auc_value.append(value)
       
    
Acc_table = pd.DataFrame(columns=['K-value','Accuracy_cv','Accuracy_tr','F1_score_cv','Auc_value'])
Acc_table['K-value']=k
Acc_table['Accuracy_cv']=Accuracy_cv
Acc_table['Accuracy_tr']=Accuracy_tr
Acc_table['F1_score_cv']=f1
Acc_table['Auc_value']=Auc_value
Acc_table.sort_values(["Auc_value"],axis=0,ascending=False,inplace= True)

display(Acc_table)
######################## 


# In[ ]:


#Accuracy plot of Training and CV Data vs K-value
Acc_tabel1 = Acc_table.sort_values(["K-value"],axis=0,ascending=False,inplace=False)
a=Acc_table1['K-value']
b=Acc_table1['Accuracy_tr']
c=Acc_table1['K-value']
d=Acc_table1['Accuracy_cv']
plt.title('Accuracy plot of Training and test Data vs K-value')
plt.plot(a ,b,label='training accuracy' )
plt.plot(c,d,label='CV accuracy')
plt.legend()
plt.xlabel('Number of Neighbours')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:


I = int(Acc_table.iloc[0][0])
print(I)

knn= KNeighborsClassifier(n_neighbors=I)

#fitting the model on crossvalidation train
knn.fit(X_tr,y_tr)

#predicting the response of the crossvalidation train
pred = knn.predict(X_test)


#evaluate the Cv accuracy
acc = accuracy_score(y_test,pred,normalize=True)*float(100)
print(acc)

confusion_mat = confusion_matrix(y_test,pred)
print(confusion_mat)
df_cm = pd.DataFrame(confusion_mat,range(2),range(2))
sns.heatmap(df_cm,annot=True,fmt='g')
plt.title('Confusion Matrix of the Test Data')
plt.xlabel('Actual Class label')
plt.ylabel('Predicted Class label')


# In[ ]:


# calculating probability of the class
from sklearn import metrics
prob_score = []
prob_score= knn.predict_proba(X_test)
prob_psv = prob_score[:,1]

#ROC curve

fpr,tpr,threshold = metrics.roc_curve(y_test,prob_psv)
roc_auc = auc(fpr,tpr)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('knn ROC CURVE')
plt.legend()
plt.show()


# In[ ]:


X_tr = hstack((bow_categories_tr, bow_subcategories_tr,bow_essay_tr, price_standardized_tr,tfidf_w2v_title_tr,tfidf_w2v_essay_tr))
X_cv=hstack((bow_categories_cv, bow_subcategories_cv,bow_essay_cv, price_standardized_cv,tfidf_w2v_title_cv,tfidf_w2v_essay_cv))
X_test=hstack((bow_categories_test, bow_subcategories_test,bow_essay_test,price_standardized_test,tfidf_w2v_title_test,tfidf_w2v_essay_test))


# In[ ]:


Accuracy_cv=[]
Accuracy_tr=[]
k = []
f1 = []
Auc_value = []
prob = []
for i in range(1,30,2):
    #instantiate learning model (k=30)
    knn= KNeighborsClassifier(n_neighbors=i)
    
    
    #fitting the model on crossvalidation train
    knn.fit(X_tr,np.ravel(y_tr))
    
    #predicting the response of the crossvalidation train 
    pred_cv = knn.predict(X_cv)
    pred_tr = knn.predict(X_tr)
    
    
    
    #evaluate the Cv accuracy
    acc_cv = accuracy_score(y_cv,pred_cv,normalize=True)*float(100)
    acc_tr = accuracy_score(y_tr,pred_tr,normalize=True)*float(100)
    Accuracy_cv.append(acc_cv)
    Accuracy_tr.append(acc_tr)
    k.append(i)
    
    Results = confusion_matrix(y_cv,pred_cv) 
    prec = Results[1][1]/(Results[1][1]+Results[0][1])
    rec =  Results[1][1]/(Results[1][1]+Results[1][0])
    mult = prec*rec
    add =  prec+rec
    result = (mult/add)
    result = 2*result
    f1.append(result)
    
    prob_score= knn.predict_proba(X_cv)
    prob_psv = prob_score[:,1]
    fpr,tpr,threshold = metrics.roc_curve(y_cv,prob_psv)
    value = auc(fpr,tpr)
    Auc_value.append(value)
       
    
Acc_table = pd.DataFrame(columns=['K-value','Accuracy_cv','Accuracy_tr','F1_score_cv','Auc_value'])
Acc_table['K-value']=k
Acc_table['Accuracy_cv']=Accuracy_cv
Acc_table['Accuracy_tr']=Accuracy_tr
Acc_table['F1_score_cv']=f1
Acc_table['Auc_value']=Auc_value
Acc_table.sort_values(["Auc_value"],axis=0,ascending=False,inplace= True)

display(Acc_table)
######################## 


# In[ ]:


#Accuracy plot of Training and CV Data vs K-value
Acc_tabel1 = Acc_table.sort_values(["K-value"],axis=0,ascending=False,inplace=False)
a=Acc_table1['K-value']
b=Acc_table1['Accuracy_tr']
c=Acc_table1['K-value']
d=Acc_table1['Accuracy_cv']
plt.title('Accuracy plot of Training and test Data vs K-value')
plt.plot(a ,b,label='training accuracy' )
plt.plot(c,d,label='CV accuracy')
plt.legend()
plt.xlabel('Number of Neighbours')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:


I = int(Acc_table.iloc[0][0])
print(I)

knn= KNeighborsClassifier(n_neighbors=I)

#fitting the model on crossvalidation train
knn.fit(X_tr,np.ravel(y_tr))

#predicting the response of the crossvalidation train
pred = knn.predict(X_test)


#evaluate the Cv accuracy
acc = accuracy_score(y_test,pred,normalize=True)*float(100)
print(acc)

confusion_mat = confusion_matrix(y_test,pred)
print(confusion_mat)
df_cm = pd.DataFrame(confusion_mat,range(2),range(2))
sns.heatmap(df_cm,annot=True,fmt='g')
plt.title('Confusion Matrix of the Test Data')
plt.xlabel('Actual Class label')
plt.ylabel('Predicted Class label')


# In[ ]:


# calculating probability of the class
from sklearn import metrics
prob_score = []
prob_score= knn.predict_proba(X_test)
prob_psv = prob_score[:,1]

#ROC curve

fpr,tpr,threshold = metrics.roc_curve(y_test,prob_psv)
roc_auc = auc(fpr,tpr)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('knn ROC CURVE')
plt.legend()
plt.show()

