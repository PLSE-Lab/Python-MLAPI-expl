#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Libraries
import re
import nltk # natural language tool kit
nltk.download("stopwords")
import nltk
nltk.download('punkt')
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
import nltk as nlp
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Evaluation of Classification Algoritm**
# * **Logistic Regression -> ACC: 0.9820531227566404 **
#     * tuned hyperparameters: (best parameters):  {'C': 10.0, 'penalty': 'l2'} #l2 = Ridge
#     * accuracy:  0.9820531227566404
# 
# 
# * Naive Bayes -> ACC: 0.8791866028708134 
#     * average accuracy:  0.8687312379706806
#     * average std:  0.013148098012261957
# 
# 
# * Decision Tree (CART) -> ACC: 0.9706937799043063 
#     * average accuracy:  0.9633266204315982
#     * average std:  0.01429551797233446 
# 
# 
# * K-NN -> ACC: 0.9542354630294329
#     * tuned hyperparameter K:  {'n_neighbors': 2}
#     * tuned parameter (best score):  0.9542354630294329
#     * average accuracy:  0.9492226807067798
#     * average std:  0.014787458315299876
# 
# 
# * RandomForestClassifier -> ACC: 0.9778708133971292 
#     * average accuracy:  0.9748684249344347
#     * average std:  0.009858906987874947

# **IMPORT DATA**

# In[ ]:


data = pd.read_csv("/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")
data = pd.concat([data.Category,data.Message],axis=1)
data.head()


# **CLEANING DATA**

# In[ ]:


data.dropna(axis = 0,inplace = True)
data.Category = [1 if each == "ham" else 0 for each in data.Category]


# In[ ]:


data.head()


# **DATA SAMPLE**

# In[ ]:


first_description = data.Message[2]
description = re.sub("[^a-zA-Z]"," ",first_description)
description = description.lower() 

description


# **STOPWORDS/irrelavent words**

# In[ ]:


description = nltk.word_tokenize(description)

description


# In[ ]:


description = [ word for word in description if not word in set(stopwords.words("english"))]
description


# **LEMMATAZATION**

# In[ ]:


lemma = nlp.WordNetLemmatizer()
description = [ lemma.lemmatize(word) for word in description] 

description = " ".join(description)
description


# **BAG OF WORDS**

# In[ ]:


description_list = []
for description in data.Message:
    description = re.sub("[^a-zA-Z]"," ",description)
    description = description.lower()   
    description = nltk.word_tokenize(description)
    description = [ word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [ lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)
description_list


# **CountVectorizer **

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer 
max_features = 5000
count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()  # x
print(" Commonly Used {} words: {}".format(max_features,count_vectorizer.get_feature_names()))


# **TRAIN-TEST SPLIT**

# In[ ]:


y = data.iloc[:,0].values   # Ham - Spam classes
x = sparce_matrix
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 42)


# **Evaluation of Classification Algoritm**

# **KNN**

# In[ ]:


grid = {"n_neighbors":np.arange(1,5)}
knn= KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv = 5)  # GridSearchCV
knn_cv.fit(x,y)

print("tuned hyperparameter K: ",knn_cv.best_params_)
print("tuned parameter (best score): ",knn_cv.best_score_)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=2)  # k = n_neighbors

accuracies = cross_val_score(estimator = knn, X = x_train, y= y_train, cv = 10) #Cross Validation
print("average accuracy: ",np.mean(accuracies))
print("average std: ",np.std(accuracies))


# In[ ]:


#Confusion Matrix
y_pred = knn_cv.predict(x_test)
y_true = y_test
cm = confusion_matrix(y_true,y_pred)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# **Logistic Regression**

# In[ ]:


grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}  # l1 = lasso ve l2 = ridge

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv = 5) #GridSearchCV
logreg_cv.fit(x,y)

print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)
print("accuracy: ",logreg_cv.best_score_)


# In[ ]:


#Confusion Matrix
y_pred = logreg_cv.predict(x_test)
y_true = y_test
cm = confusion_matrix(y_true,y_pred)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# **Naive Bayes & Decision Tree (CART) & RandomForestClassifier**

# In[ ]:


models = []
models.append(('Naive Bayes', GaussianNB()))
models.append(('Decision Tree (CART)',DecisionTreeClassifier())) 
models.append(('RandomForestClassifier',RandomForestClassifier(n_estimators = 100,random_state = 1)))

for name, model in models:
    model = model.fit(x_train,y_train)
    ACC = model.score(x_test,y_test)
    accuracies = cross_val_score(estimator = model, X = x_train, y= y_train, cv = 10) #CrossFold Validation
    print("{} -> ACC: {} ".format(name,ACC))
    print("average accuracy: ",np.mean(accuracies))
    print("average std: ",np.std(accuracies))
    #Confusion Matrix
    y_pred = model.predict(x_test)
    y_true = y_test
    cm = confusion_matrix(y_true,y_pred)
    f, ax = plt.subplots(figsize =(5,5))
    sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    plt.show()

