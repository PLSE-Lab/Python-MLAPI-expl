#!/usr/bin/env python
# coding: utf-8

# In[80]:


# Required Python Machine learning Packages
import pandas as pd
import numpy as np
# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
# To split the dataset into train and test datasets
from sklearn.model_selection import train_test_split
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score
import math
from matplotlib import pyplot
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, f1_score, precision_score, recall_score


# In[81]:


path="../input/mushrooms.csv"


# In[82]:


headers=['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color',
         'stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring',
         'veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat']


# In[83]:


data=pd.read_csv(path)


# In[84]:


data.isnull().values.sum()


# In[85]:


data.columns=headers


# In[86]:


data_b=data


# In[87]:


for h in headers:
    le = preprocessing.LabelEncoder()
    encoded_h = le.fit_transform(data[h])
    data_b[h+"_cat"] = encoded_h


# In[88]:


data_b.head()


# In[89]:


data_b = data_b.drop(headers, axis = 1)


# In[90]:


data_b=data_b.reset_index()


# In[91]:


for h in headers:
    if(h is "class"):
        print("skipping..")
    else:
        mean,std=data_b[h+"_cat"].mean(),data_b[h+"_cat"].std()
        if(mean != 0.0 and std != 0.0):
            std_v=(data_b[h+"_cat"] - mean)/std
            data_b.loc[:, h+"_cat"] =  std_v
        else:
            data_b.loc[:, h+"_cat"] =  0.0


# In[92]:


data_b.head(5)


# In[93]:


features = data_b.values[:,2:23]
target = data_b.values[:,1]
X_train, X_test, y_train, y_test = train_test_split(features,target, test_size = 0.33, random_state = 10)


# In[114]:


svclassifier = SVC(kernel='poly', degree=3)  
svclassifier.fit(X_train, y_train)


# In[115]:


y_pred = svclassifier.predict(X_test)


# In[116]:


print(accuracy_score(y_test,y_pred))


# In[117]:


print(confusion_matrix(y_test,y_pred))

print("F1Score: " + str(f1_score(y_test, y_pred)))
print("Precision: " + str(precision_score(y_test, y_pred)))
print("Recall: " + str(recall_score(y_test, y_pred)))




