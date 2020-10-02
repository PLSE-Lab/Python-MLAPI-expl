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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


import time
from nltk.stem.snowball import SnowballStemmer
import re
from gensim.parsing.preprocessing import remove_stopwords
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,log_loss,accuracy_score
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# # **Loading the dataset :**

# In[ ]:


fake = pd.read_csv(r'/kaggle/input/fake-and-real-news-dataset/Fake.csv')
true = pd.read_csv(r'/kaggle/input/fake-and-real-news-dataset/True.csv')
fake.head()


# In[ ]:


true.head()


# In[ ]:


print(fake.shape)
print(true.shape)


# In[ ]:


fake.info()


# **There are no null values in the dataset**

# In[ ]:


true.info()


# **There are no null values in the dataset**

# **Creating class labels :**

# In[ ]:


fake['class'] = 0
true['class'] = 1


# # **Data Visualisation :**

# In[ ]:


plt.figure(figsize=(20,12))

plt.subplot(2,2,1)
sns.countplot('subject',data=fake)
plt.title("Subjects of news on fake news dataset")
plt.grid()

plt.subplot(2,2,2)
sns.countplot('subject',data=true)
plt.title("Subjects of news on true news dataset")
plt.grid()

plt.show()


# **subject:politics in the fake news dataset is same as the subject :politicsNews in the true news dataset**
# **Therefore changing them to a common name.**

# In[ ]:


for i in range(len(true['subject'])):
    if true['subject'][i] == 'politicsNews':
        true['subject'][i] = 'politics'
        
true['subject'][10]


# **Combining the datasets:**

# In[ ]:


data = pd.concat([fake,true],axis=0)
data.head()


# **Sorting the datasets by date:**

# In[ ]:


data = pd.DataFrame.sort_values(data,by=['date'])
data.head()


# In[ ]:


#dropping unnecessary column : date
data = data.reset_index(drop=True)
data = data.drop('date',axis=1)
data.head()


# # **Text Preprocessing :**

# In[ ]:


def remove_shortforms(phrase):
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
    return phrase

def remove_special_char(text):
    text = re.sub('[^A-Za-z0-9]+'," ",text)
    return text

def remove_wordswithnum(text):
    text = re.sub("\S*\d\S*", "", text).strip()
    return text

def lowercase(text):
    text = text.lower()
    return text

def remove_stop_words(text):
    text = remove_stopwords(text)
    return text

st = SnowballStemmer(language='english')
def stemming(text):
    r= []
    for word in text :
        a = st.stem(word)
        r.append(a)
    return r

def listToString(s):  
    str1 = " "   
    return (str1.join(s))


# In[ ]:


start_time = time.time()
for i in range(len(data['text'])):
    data['text'][i] = remove_shortforms(data['text'][i])
    data['text'][i] = remove_special_char(data['text'][i])
    data['text'][i] = remove_wordswithnum(data['text'][i])
    data['text'][i] = lowercase(data['text'][i])
    data['text'][i] = remove_stop_words(data['text'][i])
    text = data['text'][i]
    text = text.split()
    data['text'][i] = stemming(text)
    s = data['text'][i]
    data['text'][i] = listToString(s)
print("Time taken to preprocess : ",time.time()-start_time," seconds")


# In[ ]:


start_time = time.time()
for i in range(len(data['title'])):
    data['title'][i] = remove_shortforms(data['title'][i])
    data['title'][i] = remove_special_char(data['title'][i])
    data['title'][i] = remove_wordswithnum(data['title'][i])
    data['title'][i] = lowercase(data['title'][i])
    data['title'][i] = remove_stop_words(data['title'][i])
    text = data['title'][i]
    text = text.split()
    data['title'][i] = stemming(text)
    s = data['title'][i]
    data['title'][i] = listToString(s)
print("Time taken to preprocess : ",time.time()-start_time," seconds")


# In[ ]:


start_time = time.time()
for i in range(len(data['subject'])):
    data['subject'][i] = remove_shortforms(data['subject'][i])
    data['subject'][i] = remove_special_char(data['subject'][i])
    data['subject'][i] = remove_wordswithnum(data['subject'][i])
    data['subject'][i] = lowercase(data['subject'][i])
    data['subject'][i] = remove_stop_words(data['subject'][i])
    text = data['subject'][i]
    text = text.split()
    data['subject'][i] = stemming(text)
    s = data['subject'][i]
    data['subject'][i] = listToString(s)
print("Time taken to preprocess : ",time.time()-start_time," seconds")


# In[ ]:


data.head()


# **Combining the texts,titles and the subjects :**

# In[ ]:


Combined_text = [None] * len(data['text'])
for i in range(len(data['title'])):
    Combined_text[i] = data['text'][i] + " " + data['title'][i] + " " + data['subject'][i]
    
data['combined_text'] = Combined_text


# In[ ]:


#dropping the unnecessary columns
data = pd.DataFrame.drop(data,columns=['title','text','subject'],axis=1)
data.head()


# # **Bag Of Words and TF-IDF encoding of the combined text:**

# In[ ]:


bow = CountVectorizer(ngram_range=(1,2))
bow_text = bow.fit_transform(data['combined_text'])


# In[ ]:


tfidf = TfidfVectorizer(ngram_range=(1,2))
tfidf_text = tfidf.fit_transform(data['combined_text'])


# In[ ]:


print(bow_text.shape)
print(tfidf_text.shape)


# # **Train-Test Split(Time based Splitting) :**

# In[ ]:


labels = data['class'].reset_index(drop=True)
labels.shape


# In[ ]:


# splitting in 70% train and 30% test
X_train_bow = bow_text[:31428]
X_test_bow = bow_text[31428:]
X_train_tfidf = tfidf_text[:31428]
X_test_tfidf = tfidf_text[31428:]
Y_train = labels[:31428]
Y_test = labels[31428:]


# **Splitted data visualisation :**

# In[ ]:


plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
sns.countplot(Y_train)
plt.grid()

plt.subplot(2,2,2)
sns.countplot(Y_test)
plt.grid()

plt.show()


# **Confusion Matrix along with Precision and Recall matrix :**

# In[ ]:


def plot_conf_matrix(Y_test,Y_pred):
    conf = confusion_matrix(Y_test,Y_pred)
    recall =(((conf.T)/(conf.sum(axis=1))).T)
    precision =(conf/conf.sum(axis=0))

    class_labels = [0,1]
    plt.figure(figsize=(10,8))
    sns.heatmap(conf,annot=True,fmt=".3f",cmap="YlOrBr",xticklabels=class_labels,yticklabels=class_labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    plt.figure(figsize=(10,8))
    sns.heatmap(precision,annot=True,fmt=".3f",cmap="YlOrBr",xticklabels=class_labels,yticklabels=class_labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    plt.figure(figsize=(10,8))
    sns.heatmap(recall,annot=True,fmt=".3f",cmap="YlOrBr",xticklabels=class_labels,yticklabels=class_labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()


# # **ML Models :**

# In[ ]:


summary = []


# # **Logistic Regression :**

# With Bag of words encoding of text :

# In[ ]:


lr = LogisticRegression(max_iter = 1200)
lr.fit(X_train_bow,Y_train)

train_pred = lr.predict(X_train_bow)
print("The train loss is : ",log_loss(Y_train,train_pred))

test_pred = lr.predict(X_test_bow)
print("The test loss is : ",log_loss(Y_test,test_pred))

plot_conf_matrix(Y_test,test_pred)
print("Accuracy :",lr.score(X_test_bow,Y_test)*100)

summary.append(("Logistic Regression with BOW",lr.score(X_test_bow,Y_test)*100))


# With TF-TDF encoding of text :

# In[ ]:


lr = LogisticRegression(max_iter=1200)
lr.fit(X_train_tfidf,Y_train)

train_pred = lr.predict(X_train_tfidf)
print("The train loss is : ",log_loss(Y_train,train_pred))

test_pred = lr.predict(X_test_tfidf)
print("The test loss is : ",log_loss(Y_test,test_pred))

plot_conf_matrix(Y_test,test_pred)
print("Accuracy :",lr.score(X_test_tfidf,Y_test)*100)

summary.append(("Logistic Regression with TF-IDF",lr.score(X_test_bow,Y_test)*100))


# # **Random Forest :**

# With Bag of words encoding of text :

# In[ ]:


rf = RandomForestClassifier()
rf.fit(X_train_bow,Y_train)

train_pred = rf.predict(X_train_bow)
print("The train loss is : ",log_loss(Y_train,train_pred))

test_pred = rf.predict(X_test_bow)
print("The test loss is : ",log_loss(Y_test,test_pred))

plot_conf_matrix(Y_test,test_pred)
print("Accuracy : ",accuracy_score(Y_test,test_pred)*100)

summary.append(("Random Forest with BOW",accuracy_score(Y_test,test_pred)*100))


# With TF-TDF encoding of text :

# In[ ]:


rf = RandomForestClassifier()
rf.fit(X_train_tfidf,Y_train)

train_pred = rf.predict(X_train_tfidf)
print("The train loss is : ",log_loss(Y_train,train_pred))

test_pred = rf.predict(X_test_tfidf)
print("The test loss is : ",log_loss(Y_test,test_pred))

print("Accuracy : ",accuracy_score(Y_test,test_pred)*100)
plot_conf_matrix(Y_test,test_pred)

summary.append(("Random Forest with TF-IDF",accuracy_score(Y_test,test_pred)*100))


# # **Linear SVM :**

# With Bag of Words encoding of text :

# In[ ]:


li_svm = SVC(kernel='linear')
li_svm.fit(X_train_bow,Y_train)

train_pred = li_svm.predict(X_train_bow)
print("The train loss is : ",log_loss(Y_train,train_pred))

test_pred = li_svm.predict(X_test_bow)
print("The test loss is : ",log_loss(Y_test,test_pred))

print("Accuracy : ",accuracy_score(Y_test,test_pred)*100)
plot_conf_matrix(Y_test,test_pred)

summary.append(("Linear SVM with BOW",accuracy_score(Y_test,test_pred)*100))


# With TF-IDF encoding of text :

# In[ ]:


li_svm = SVC(kernel='linear')
li_svm.fit(X_train_tfidf,Y_train)

train_pred = li_svm.predict(X_train_tfidf)
print("The train loss is : ",log_loss(Y_train,train_pred))

test_pred = li_svm.predict(X_test_tfidf)
print("The test loss is : ",log_loss(Y_test,test_pred))

print("Accuracy : ",accuracy_score(Y_test,test_pred)*100)
plot_conf_matrix(Y_test,test_pred)

summary.append(("Linear SVM with TF-IDF",accuracy_score(Y_test,test_pred)*100))


# # **RBF-SVM :**

# With Bag of words encoding of text :

# In[ ]:


rbf_svm = SVC(kernel='rbf')
rbf_svm.fit(X_train_bow,Y_train)

train_pred = rbf_svm.predict(X_train_bow)
print("The train loss is : ",log_loss(Y_train,train_pred))

test_pred = rbf_svm.predict(X_test_bow)
print("The test loss is : ",log_loss(Y_test,test_pred))

print("Accuracy : ",accuracy_score(Y_test,test_pred)*100)
plot_conf_matrix(Y_test,test_pred)

summary.append(("RBF-SVM with BOW",accuracy_score(Y_test,test_pred)*100))


# # **Gradient Boosing Decision Tree**

# With Bag of words encoding of text :

# In[ ]:


gbdt = GradientBoostingClassifier()
gbdt.fit(X_train_bow,Y_train)

train_pred = gbdt.predict(X_train_bow)
print("The train loss is : ",log_loss(Y_train,train_pred))

test_pred = gbdt.predict(X_test_bow)
print("The test loss is : ",log_loss(Y_test,test_pred))

print("Accuracy : ",accuracy_score(Y_test,test_pred)*100)
plot_conf_matrix(Y_test,test_pred)

summary.append(("GBDT with BOW",accuracy_score(Y_test,test_pred)*100))


# # **Decision Tree :**

# With Bag of words encoding of text :

# In[ ]:


dt = DecisionTreeClassifier()
dt.fit(X_train_bow,Y_train)

train_pred = dt.predict(X_train_bow)
print("The train loss is : ",log_loss(Y_train,train_pred))

test_pred = dt.predict(X_test_bow)
print("The test loss is : ",log_loss(Y_test,test_pred))

print("Accuracy : ",accuracy_score(Y_test,test_pred)*100)
plot_conf_matrix(Y_test,test_pred)

summary.append(("Decision Tree with BOW",accuracy_score(Y_test,test_pred)*100))


# With TF-IDF encoding of text :

# In[ ]:


dt = DecisionTreeClassifier(max_depth=100)
dt.fit(X_train_tfidf,Y_train)

train_pred = dt.predict(X_train_tfidf)
print("The train loss is : ",log_loss(Y_train,train_pred))

test_pred = dt.predict(X_test_tfidf)
print("The test loss is : ",log_loss(Y_test,test_pred))

print("Accuracy : ",accuracy_score(Y_test,test_pred)*100)
plot_conf_matrix(Y_test,test_pred)

summary.append(("Decision Tree with TF-IDF",accuracy_score(Y_test,test_pred)*100))


# # **SGD Classifier :**

# With Bag of words encoding of text :

# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train_bow,Y_train)

train_pred = sgd.predict(X_train_bow)
print("The train loss is : ",log_loss(Y_train,train_pred))

test_pred = sgd.predict(X_test_bow)
print("The test loss is : ",log_loss(Y_test,test_pred))

plot_conf_matrix(Y_test,test_pred)
print("Accuracy :",accuracy_score(Y_test,test_pred)*100)

summary.append(("SGD Classifier with BOW",lr.score(X_test_bow,Y_test)*100))


# With TF-IDF encoding of text :

# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train_tfidf,Y_train)

train_pred = sgd.predict(X_train_tfidf)
print("The train loss is : ",log_loss(Y_train,train_pred))

test_pred = sgd.predict(X_test_tfidf)
print("The test loss is : ",log_loss(Y_test,test_pred))

plot_conf_matrix(Y_test,test_pred)
print("Accuracy :",accuracy_score(Y_test,test_pred)*100)

summary.append(("SGD Classifier with TF-IDF",lr.score(X_test_bow,Y_test)*100))


# # **KNN :**

# With Bag of words encoding of text :

# In[ ]:


knn = KNeighborsClassifier()
knn.fit(X_train_bow,Y_train)

train_pred = knn.predict(X_train_bow)
print("The train loss is : ",log_loss(Y_train,train_pred))

test_pred = knn.predict(X_test_bow)
print("The test loss is : ",log_loss(Y_test,test_pred))

plot_conf_matrix(Y_test,test_pred)
print("Accuracy :",accuracy_score(Y_test,test_pred)*100)

summary.append(("KNN with BOW",lr.score(X_test_bow,Y_test)*100))


# With TF-IDF encoding of text :

# In[ ]:


knn = KNeighborsClassifier()
knn.fit(X_train_tfidf,Y_train)

train_pred = knn.predict(X_train_tfidf)
print("The train loss is : ",log_loss(Y_train,train_pred))

test_pred = knn.predict(X_test_tfidf)
print("The test loss is : ",log_loss(Y_test,test_pred))

plot_conf_matrix(Y_test,test_pred)
print("Accuracy :",accuracy_score(Y_test,test_pred)*100)

summary.append(("KNN with TF-IDF",lr.score(X_test_bow,Y_test)*100))


# # **Summary of models :**

# In[ ]:


print("Summary :")
print("        Model                         ACCURACY  ")
for i in range(len(summary)):
    print(summary[i])


# **I ran GBDT and RBF-SVM with just the BOW encoding because it already takes a lot of time to run but if you have better systems you can run them on TF-IDF encoding with hyperparameter tuning too but these accuracy are good too.**

# **Also i have put up the training losses along with the test losses for all models so that you can choose which model is overfitting and which is a better model than the rest.**

# **In my opinion GBDT with BOW encoding is the best model because it does not overfit and provides a very nice accuracy too followed by Logistic Regression with BOW encoding  which also gives a nice accuracy.**

# **Its very highly probable that GBDT with TF-IDF runs with a great accuracy too.If it will be possible I will surely update this notebook with it but if i couldn't ,be sure to try that out in case you work with this dataset.**

# **If you like my notbook please do upvote.Thanks for viewing.**
