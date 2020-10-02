#!/usr/bin/env python
# coding: utf-8

# # This notebook is a session from NUS

# In[ ]:


get_ipython().system('wget https://raw.githubusercontent.com/IRS-MR/S-MR-Workshop/master/S-MR-Workshop3/knowledge-discovery-identify-aliens/alien_train.csv')


# In[ ]:


get_ipython().system('wget https://raw.githubusercontent.com/IRS-MR/S-MR-Workshop/master/S-MR-Workshop3/knowledge-discovery-identify-aliens/alien_test.csv')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df1 = pd.read_csv('/kaggle/working/alien_train.csv')
df2 = pd.read_csv('/kaggle/working/alien_test.csv')


# In[ ]:


df = df1.append(df2, ignore_index=True)


# In[ ]:


df


# In[ ]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
df['Alien'] = lb_make.fit_transform(df['Alien'])

df.head()


# In[ ]:


for column in df.columns:
    df[column] = df[column].astype('int64')


# In[ ]:


df.to_csv('balanced_data.csv', index=False)


# In[ ]:


df.info()


# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
  
def importdata(): 
    balance_data = pd.read_csv( 
'/kaggle/working/balanced_data.csv', 
    sep= ',', header = None) 
      
    print ("Dataset Length: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
      
    print ("Dataset: ",balance_data.head()) 
    return balance_data 
   
def splitdataset(balance_data): 
  
    X = balance_data.values[:, 1:5] 
    Y = balance_data.values[:, 5] 
  
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.2, random_state = 100) 
      
    return X, Y, X_train, X_test, y_train, y_test 
      
def train_using_gini(X_train, X_test, y_train): 
  
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
      
def tarin_using_entropy(X_train, X_test, y_train): 
  
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
  
def prediction(X_test, clf_object): 
  
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 
    
def main(): 
      
    data = importdata() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train) 
      
    print("Results Using Gini Index:") 
      
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
      
    print("Results Using Entropy:") 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 
      
if __name__=="__main__": 
    main() 


# In[ ]:




