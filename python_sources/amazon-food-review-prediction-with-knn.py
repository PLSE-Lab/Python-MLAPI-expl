#!/usr/bin/env python
# coding: utf-8

# # 1 Introduction
# This python notebook for Amazon food reviews polarity prediction based on the given review data by applying K Nearest Neighbors (KNN) algorithm. To build generalized prediction model first step should be necessary cleaning of data as a part of data preprocessing. 
# 
# We will perform following data preprocessing. 
# 
# * Removing Stop-words
# * Remove any punctuations or limited set of special characters like , or . or # etc.
# * Snowball Stemming the word 
# * Convert the word to lowercase
#  
# Once we the data is cleaned to be processed we'll use below Feature generation techniques to convert text to numeric vector.
# 1. Bag Of Words (BoW)
# 1. Term Frequency - inverse document frequency (tf-idf)
# 1. Word2Vec
# 1. tf-idf weighted Word2Vec
# 
# Using KNN algorithm we will build model to predict review polarity for each technique. 
# 
# **Objective: Given a review determine whether a review is positive or negative, by appling KNN algorithm and deciding the best Feature generation technique for given problem.**
# 
# 
# 

# **1.1 Load and check data**

# In[1]:


import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
import scikitplot.metrics as skplt
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

import os
print(os.listdir("../input"))


# In[2]:


import sqlite3
con = sqlite3.connect('../input/database.sqlite')

filtered_data = pd.read_sql_query("""select * from Reviews WHERE Score != 3""",con)

filtered_data.shape
filtered_data.head(5)


# # 2 Data Preprocessing

# *  **Segregating data as positive and negative**

# In[3]:


# Here are replacing review score 1,2 as negative and 4,5 as a positive. we are skipping review score 3 considering it as a neutral.
def partition(x):
    if x<3:
        return 'negative'
    return 'positive'

actualScore = filtered_data['Score']
positiveNegative = actualScore.map(partition)
filtered_data['Score'] = positiveNegative


# * **Sorting data for time based splitting for model train and test dataset**

# In[4]:


import datetime

filtered_data["Time"] = filtered_data["Time"].map(lambda t: datetime.datetime.fromtimestamp(int(t)).strftime('%Y-%m-%d %H:%M:%S'))

sortedData = filtered_data.sort_values('ProductId',axis=0,kind="quicksort", ascending=True)
final = sortedData.drop_duplicates(subset={"UserId","ProfileName","Time","Text"},keep="first",inplace=False)

final = final[final.HelpfulnessNumerator <= final.HelpfulnessDenominator]

#As data is huge, due to computation limitation we will randomly select data. we will try to pick data in a way so that it doesn't make data imbalance problem
finalp = final[final.Score == 'positive']
finalp = finalp.sample(frac=0.035,random_state=1) #0.055

finaln = final[final.Score == 'negative']
finaln = finaln.sample(frac=0.15,random_state=1) #0.25

final = pd.concat([finalp,finaln],axis=0)

#sording data by timestamp so that it can be devided in train and test dataset for time based slicing.
final = final.sort_values('Time',axis=0,kind="quicksort", ascending=True).reset_index(drop=True)


print(final.shape)


# In[5]:


final['Score'].value_counts().plot(kind='bar')


# * ** Removing Stop-words **
# * ** Remove any punctuations or limited set of special characters like , or . or # etc. **
# * ** Snowball Stemming the word ** 
# * ** Convert the word to lowercase **

# In[6]:


import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english')) #set of stopwords
sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer

def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned

i=0
str1=' '
final_string=[]
all_positive_words=[] # store words from +ve reviews here
all_negative_words=[] # store words from -ve reviews here.
s=''

final_string=[]
all_positive_words=[] # store words from +ve reviews here
all_negative_words=[] # store words from -ve reviews here.
s=''
for sent in final['Text'].values:
    filtered_sentence=[]
    #print(sent);
    sent=cleanhtml(sent) # remove HTMl tags
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (final['Score'].values)[i] == 'positive': 
                        all_positive_words.append(s) #list of all words used to describe positive reviews
                    if(final['Score'].values)[i] == 'negative':
                        all_negative_words.append(s) #list of all words used to describe negative reviews reviews
                else:
                    continue
            else:
                continue 
    str1 = b" ".join(filtered_sentence) #final string of cleaned words    
    final_string.append(str1)
    i+=1


# # 3 Building function to find optimal K for KNN

# **To Find the optimal K we will used 10 fold cross validation method. Based on misclassifiction error for every K, we will decide the best K on  Train Data**

# In[7]:


from sklearn.cross_validation import cross_val_score

def find_optimal_k(X_train,y_train, myList):
   
    #creating odd list of K for KNN
    #myList = list(range(0,40))
    neighbors = list(filter(lambda x: x % 2 != 0, myList))

    # empty list that will hold cv scores
    cv_scores = []

    # perform 10-fold cross validation
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    # changing to misclassification error
    MSE = [1 - x for x in cv_scores]

    # determining best k
    optimal_k = neighbors[MSE.index(min(MSE))]
    print('\nThe optimal number of neighbors is %d.' % optimal_k)


    plt.figure(figsize=(10,6))
    plt.plot(list(filter(lambda x: x % 2 != 0, myList)),MSE,color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')

    print("the misclassification error for each k value is : ", np.round(MSE,3))
    
    return optimal_k


# # 4 Feature generation techniques to convert text to numeric vector.[](http://) 

# # 4.1 Appling KNN with BoW

# **Generating Bag of Wrods Vector matrix for Reviews**

# In[12]:


from sklearn.feature_extraction.text import CountVectorizer

#count_vect = CountVectorizer(ngram_range=(1,2) ) 
count_vect = CountVectorizer() 
final_bow_count = count_vect.fit_transform(final_string)#final['Text'].values)


# In[13]:


from sklearn.preprocessing import StandardScaler

final_bow_np = StandardScaler(with_mean=False).fit_transform(final_bow_count )


# **Splitting Data into Train and Test based on the timestamp of review**

# In[14]:


#We already have sorted data by timestamp so we will use first 70% of data as Train with cross validation and next 30% for test
import math
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X = final_bow_np
y = final['Score']

X_train =  final_bow_np[:math.ceil(len(final)*.7)] 
X_test = final_bow_np[math.ceil(len(final)*.7):]
y_train = y[:math.ceil(len(final)*.7)]
y_test =  y[math.ceil(len(final)*.7):]


# **Finding Optimal K by 10 fold Cross validation**

# In[ ]:


myList = list(range(0,50))

optimal_k = find_optimal_k(X_train ,y_train,myList)


# **KNN with Optimal K**

# In[76]:


knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)


# In[77]:


skplt.plot_confusion_matrix(y_test ,pred)


# In[78]:


print(classification_report(y_test ,pred))


# In[ ]:


print("Accuracy for KNN model with Bag of words is ",round(accuracy_score(y_test ,pred),3))


# # 4.2 Appling KNN with tf-idf

# **Generating tf-idf Vector matrix for Reviews**

# In[62]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vec = TfidfVectorizer()#ngram_range=(2,2))

final_tfidf_count = tf_idf_vec.fit_transform(final_string)#final['Text'].values)

#print(final_string)


# In[63]:


from sklearn.preprocessing import StandardScaler

final_tfidf_np = StandardScaler(with_mean=False).fit_transform(final_tfidf_count )


# **Splitting Data into Train and Test**

# In[29]:


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

X = final_tfidf_np
y = final['Score']

X_train =  final_tfidf_np[:math.ceil(len(final)*.7)] 
X_test = final_tfidf_np[math.ceil(len(final)*.7):]
y_train = y[:math.ceil(len(final)*.7)]
y_test =  y[math.ceil(len(final)*.7):]


# In[30]:


myList = list(range(0,40))

optimal_k = find_optimal_k(X_train ,y_train,myList)


# **KNN with Optimal K**

# In[31]:


knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)


# In[32]:


skplt.plot_confusion_matrix(y_test ,pred)


# In[33]:


print(classification_report(y_test ,pred))


# In[ ]:


print("Accuracy for KNN model with tf-idf is ",round(accuracy_score(y_test ,pred),3))


# # 4.3 Appling KNN with Avg W2V

# **Generating W2V Vector matrix for Reviews**

# In[8]:


from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle


# In[9]:


import gensim
i=0
str1=''
list_of_sent=[]
final_string_for_tfidf = []
for sent in final['Text'].values:
    filtered_sentence=[]
    sent=cleanhtml(sent)
    str1 = ''
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (cleaned_words.lower() not in stop)):    
                filtered_sentence.append(cleaned_words.lower())
                str1 += " "+cleaned_words.lower() 
            else:
                continue
    #str1 = b" ".join(filtered_sentence) #final string of cleaned words
            
    #final_string_for_tfidf.append(str1)
    list_of_sent.append(filtered_sentence)
    final_string_for_tfidf.append((str1).strip())


# In[10]:


w2v_model=gensim.models.Word2Vec(list_of_sent,min_count=5,size=50, workers=4)  


# In[11]:


sent_vectors = []; 
for sent in list_of_sent: 
    sent_vec = np.zeros(50)
    cnt_words =0; 
    for word in sent: 
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
    


# **Splitting Data into Train and Test**

# In[14]:


import math
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

X = sent_vectors #final_w2v_count
y = final['Score']

X_train =  sent_vectors[:math.ceil(len(final)*.7)]  #final_w2v_count
X_test = sent_vectors[math.ceil(len(final)*.7):] #final_w2v_count
y_train = y[:math.ceil(len(final)*.7)]
y_test =  y[math.ceil(len(final)*.7):]


# **Finding Optimal K by 10 fold Cross validation**

# In[15]:


myList = list(range(35,47))

optimal_k = find_optimal_k(X_train ,y_train,myList)


# **KNN with Optimal K**

# In[16]:


knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)


# In[17]:


skplt.plot_confusion_matrix(y_test ,pred)


# In[18]:


print(classification_report(y_test ,pred))


# In[19]:


print("Accuracy for KNN model with Word2Vec is ",round(accuracy_score(y_test ,pred),3))


# # 4.4 Appling KNN with tf-idf weighted W2V

# **Generating tf-idf W2V Vector matrix for Reviews**

# In[20]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vec_w = TfidfVectorizer()#ngram_range=(2,2))

final_tfidf_w = tf_idf_vec_w.fit_transform(final_string_for_tfidf)


# In[21]:


tfidf_feat = tf_idf_vec_w.get_feature_names()


tfidf_sent_vectors = [];
row=0;
for sent in list_of_sent:  
    sent_vec = np.zeros(50) 
    weight_sum =0;
    for word in sent:
        try:
            vec = w2v_model.wv[word]
            tf_idf = final_tfidf_w[row, tfidf_feat.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
        except Exception as e: 
            pass #print(e)
            
    try:
        sent_vec /= weight_sum
    except:
        print(e)
        
    tfidf_sent_vectors.append(sent_vec)
    row += 1


# **Finding Optimal K by 10 fold Cross validation**

# In[22]:


from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


X = tfidf_sent_vectors
y = final['Score']

X_train =  tfidf_sent_vectors[:math.ceil(len(final)*.7)] 
X_test = tfidf_sent_vectors[math.ceil(len(final)*.7):]
y_train = y[:math.ceil(len(final)*.7)]
y_test =  y[math.ceil(len(final)*.7):]


# **Finding Optimal K by 10 fold Cross validation**

# In[23]:


myList = list(range(0,40))

optimal_k = find_optimal_k(X_train ,y_train,myList)


# **KNN with Optimal K**

# In[24]:


knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)


# In[25]:


skplt.plot_confusion_matrix(y_test ,pred)


# In[26]:


print(classification_report(y_test ,pred))


# In[27]:


print("Accuracy for KNN model with tf-idf weighted Word2vec ",round(accuracy_score(y_test ,pred),3))


# # 5 Observation
# **The result of feature generation techniques and machine learning algorithms vary by application. But by comparing the accuracy of all 4 developed models, KNN model with Avg. w2v feature generation technique gives accuracy more than 75% which is the best to predict the polarity of reviews among all models **
# 
