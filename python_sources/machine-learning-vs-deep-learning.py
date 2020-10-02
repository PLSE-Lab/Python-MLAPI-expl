#!/usr/bin/env python
# coding: utf-8

# In[328]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.simplefilter('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[329]:


data = pd.read_csv('../input/Iris.csv')


# In[330]:


data.head()


# In[331]:


data.info()


# In[332]:


sns.pairplot(data=data,hue='Species')


# Iris Verginica can be easily seperated out. For the other 2, there is some overlapping which might contribute to some error in predictions.

# In[333]:


data = data.drop('Id',axis=1)


# In[334]:


data['Species'].unique()


# In[335]:


#categorising species into numerical values

def turn_numeric(iris_x):
    if iris_x == 'Iris-setosa':
        return 0
    if iris_x == 'Iris-versicolor':
        return 1
    if iris_x == 'Iris-virginica':
        return 2
    else:
        print(iris_x)
        return


# In[336]:


data['Species'] = data['Species'].apply(turn_numeric)


# In[337]:


data.head()


# In[338]:


data.isnull().values.any()


# In[339]:


X = data.drop('Species',axis=1)
y = data['Species']


# Using ML Algo - KNN to classify the species - 

# In[340]:


#scaling values so that euclidean distance can do its work
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[341]:


scaled_features = scaler.fit_transform(X)


# In[342]:


#scaled features
scaled_features


# In[343]:


df_feat = pd.DataFrame(data=scaled_features,columns=X.columns)


# In[344]:


df_feat.head()


# In[345]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_feat, y, test_size=0.3,random_state=101)


# In[346]:


from sklearn.neighbors import  KNeighborsClassifier


# In[347]:


#elbow method to calculate k value

error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,marker='o')


# K=7 seems to be the appropriate value

# In[348]:


KNN = KNeighborsClassifier(n_neighbors=7)
KNN.fit(X_train,y_train)
actual_pred = KNN.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,actual_pred))
print('\n')
print(confusion_matrix(y_test,actual_pred))


# Lets use deep learning using tensorflow - 

# In[349]:


import tensorflow as tf


# In[350]:


data.head()


# In[351]:


X_deep = data.drop('Species',axis=1)
y_deep = data['Species']


# In[352]:


X_deep_train, X_deep_test, y_deep_train, y_deep_test = train_test_split(X_deep, y_deep, test_size=0.3,random_state=101)


# In[353]:


#creating feature columns
feat_cols = []

for col in X_deep.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))


# In[354]:


feat_cols


# In[355]:


#creating an input function - 
input_func = tf.estimator.inputs.pandas_input_fn(x=X_deep_train,y=y_deep_train,batch_size=10,num_epochs=5,shuffle=True)


# In[356]:


#defining a classifier
classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3,feature_columns=feat_cols)


# In[357]:


#fitting the classifier on the input function
classifier.train(input_fn=input_func,steps=50)


# In[358]:


#evaluation
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_deep_test,batch_size=len(X_deep_test),shuffle=False)


# In[359]:


note_predictions = list(classifier.predict(input_fn=pred_fn))


# In[360]:


#each dictionary corresponds to a prediction - class_ids is the prediction
note_predictions


# In[361]:


final_preds  = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])


# In[362]:


print(classification_report(y_deep_test,final_preds))
print('\n')
print(confusion_matrix(y_deep_test,final_preds))


# Deep Learning out perform other techniques if the data size is large. But with small data size, traditional Machine Learning algorithms are preferable.
# 
# ref - https://towardsdatascience.com/why-deep-learning-is-needed-over-traditional-machine-learning-1b6a99177063

# In[ ]:




