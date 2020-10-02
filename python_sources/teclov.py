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


# # TECLOV PROJECT : MEDICAL TREATMENT

# # PROBLEM STATEMENT :

# A lot has been said during the past several years about how precision medicine and, more concretely, how genetic testing is going to disrupt the way diseases like cancer are treated.
# 
# But this is only partially happening due to the huge amount of manual work still required. Once sequenced, a cancer tumor can have thousands of genetic mutations. But the challenge is distinguishing the mutations that contribute to tumor growth (drivers) from the neutral mutations (passengers). 
# 
# Currently this interpretation of genetic mutations is being done manually. This is a very time-consuming task where a clinical pathologist has to manually review and classify every single genetic mutation based on evidence from text-based clinical literature.
# 
# We need to develop a Machine Learning algorithm that, using this knowledge base as a baseline, automatically classifies genetic variations.
# 
# This problem was a competition posted on Kaggle with a award of $15,000. This was launched by  Memorial Sloan Kettering Cancer Center (MSKCC), accepted by NIPS 2017 Competition Track,  because we need your help to take personalized medicine to its full potential.
# 

# # DATA INGESTION :

# * **Training_variants** - a comma separated file containing the description of the genetic mutations used for training. Fields are ID (the id of the row used to link the mutation to the clinical evidence), Gene (the gene where this genetic mutation is located), Variation (the aminoacid change for this mutations), Class (1-9 the class this genetic mutation has been classified on)
# * **training_text** - a double pipe (||) delimited file that contains the clinical evidence (text) used to classify genetic mutations. Fields are ID (the id of the row used to link the clinical evidence to the genetic mutation), Text (the clinical evidence used to classify the genetic mutation)
# * **test_variants** - a comma separated file containing the description of the genetic mutations used for training. Fields are ID (the id of the row used to link the mutation to the clinical evidence), Gene (the gene where this genetic mutation is located), Variation (the aminoacid change for this mutations)
# * **test_text** - a double pipe (||) delimited file that contains the clinical evidence (text) used to classify genetic mutations. Fields are ID (the id of the row used to link the clinical evidence to the genetic mutation), Text (the clinical evidence used to classify the genetic mutation)

# In[ ]:


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,roc_auc_score,log_loss
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from mlxtend.classifier import StackingClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_predict,StratifiedKFold
from sklearn.pipeline import Pipeline


# ## Load training_text and training_variants

# In[ ]:


df_train_txt = pd.read_csv('/kaggle/input/msk-redefining-cancer-treatment/training_text.zip', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])
df_train_txt.head()


# In[ ]:


df_train_var = pd.read_csv('/kaggle/input/msk-redefining-cancer-treatment/training_variants.zip')
df_train_var.head()


# In[ ]:


df_test_txt = pd.read_csv('/kaggle/input/msk-redefining-cancer-treatment/test_text.zip', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])
df_test_txt.head()


# In[ ]:


df_test_var = pd.read_csv('/kaggle/input/msk-redefining-cancer-treatment/test_variants.zip')
df_test_var.head()


# <p>
#     Let's understand above data. There are 4 fields above: <br>
#     <ul>
#         <li><b>ID : </b>row id used to link the mutation to the clinical evidence</li>
#         <li><b>Gene : </b>the gene where this genetic mutation is located </li>
#         <li><b>Variation : </b>the aminoacid change for this mutations </li>
#         <li><b>Class :</b> class value 1-9, this genetic mutation has been classified on</li>
#     </ul>
#     
# Keep doing more analysis  on above data.

# **Let's join them together**

# In[ ]:


df_train = pd.merge(df_train_var, df_train_txt, how='left', on='ID')
df_train.head()


# In[ ]:


df_test = pd.merge(df_test_var, df_test_txt, how='left', on='ID')
df_test.head()


# **Lets's check for the class variable**

# In[ ]:


df_train['Class'].value_counts().plot(kind="bar")
plt.show()
print(df_train['Class'].value_counts())


# **Lets's check for the Missing values**

# In[ ]:


df_train[df_train.isnull().any(axis=1)]


# In[ ]:


df_test[df_test.isnull().any(axis=1)]


# # TEXT Preprocessing :

# **We have huge amount of text data. So, we need to pre process it. So lets write a function for the same.**

# In[ ]:


import nltk
from nltk.corpus import stopwords


# In[ ]:


# We would like to remove all stop words like a, is, an, the, ... 
# so we collecting all of them from nltk library

stop_words = set(stopwords.words('english'))


# **Defining function for training data**

# In[ ]:


def train_text_preprocess(total_text, ind, col):
    # Remove int values from text data as that might not be imp
    if type(total_text) is not int:
        string = ""
        # replacing all special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
        # replacing multiple spaces with single space
        total_text = re.sub('\s+',' ', str(total_text))
        # bring whole text to same lower-case scale.
        total_text = total_text.lower()
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from text
            if not word in stop_words:
                string += word + " "
        
        df_train[col][ind] = string


# In[ ]:


for index, row in df_train.iterrows():
    if type(row['Text']) is str:
        train_text_preprocess(row['Text'], index, 'Text')


# In[ ]:


df_train.head()


# **Defining function for test data**

# In[ ]:


def test_text_preprocess(total_text, ind, col):
    # Remove int values from text data as that might not be imp
    if type(total_text) is not int:
        string = ""
        # replacing all special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
        # replacing multiple spaces with single space
        total_text = re.sub('\s+',' ', str(total_text))
        # bring whole text to same lower-case scale.
        total_text = total_text.lower()
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from text
            if not word in stop_words:
                string += word + " "
        
        df_test[col][ind] = string


# In[ ]:


for index, row in df_test.iterrows():
    if type(row['Text']) is str:
        test_text_preprocess(row['Text'], index, 'Text')


# In[ ]:


df_test.head()


# **We want to ensure that all spaces in Gene and Variation column to be replaced by _.**

# In[ ]:


df_train.Gene      = df_train.Gene.str.replace('\s+', '_')
df_train.Variation = df_train.Variation.str.replace('\s+', '_')
df_test.Gene      = df_test.Gene.str.replace('\s+', '_')
df_test.Variation = df_test.Variation.str.replace('\s+', '_')


# **Missing values imputation**

# In[ ]:


# Merging Gene and Variation
df_train.loc[df_train['Text'].isnull(),'Text'] = df_train['Gene'] +' '+df_train['Variation']
df_test.loc[df_test['Text'].isnull(),'Text'] = df_test['Gene'] +' '+df_test['Variation']


# In[ ]:


print('Missing Values of Train data')
print(df_train.isnull().sum())
print('Missing Values of Test data')
print(df_test.isnull().sum())


# # Splitting the data :

# In[ ]:


train,test = train_test_split(df_train,test_size=0.2,random_state=42)


# In[ ]:


X_train = train['Text'].values
X_test = test['Text'].values
y_train = train['Class'].values
y_test = test['Class'].values


# # Logistic Regession 

# In[ ]:


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegression())
])
text_clf = text_clf.fit(X_train,y_train.ravel())


# In[ ]:


y_pred = text_clf.predict(X_test)
np.mean(y_pred == y_test)


# In[ ]:


print(confusion_matrix(y_test,y_pred))


# In[ ]:


X_test_final = df_test['Text'].values
predicted_class = text_clf.predict_proba(X_test_final)


# # Random Forest :

# In[ ]:


text_clf2 = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier())
])
text_clf2 = text_clf2.fit(X_train,y_train.ravel())


# In[ ]:


y_pred = text_clf2.predict(X_test)
np.mean(y_pred == y_test)

