#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from warnings import filterwarnings
filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ![image.png](attachment:image.png)

# In[ ]:


df = pd.read_csv('/kaggle/input/heart.csv')


# In[ ]:


df.dtypes


# In[ ]:


num_cols = df.select_dtypes(include = ['float','int'])


# All are in numerical data type

# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.isna().sum()


# No null values

# Your preprocessed data may contain attributes with a mixtures of scales for various quantities such as dollars, kilograms and sales volume.
# 
# Many machine learning methods expect or are more effective if the data attributes have the same scale. Two popular data scaling methods are normalization and standardization.

# In[ ]:


sns.countplot(df['NUM'])


# It is a balanced data set and follows Discrete Distribution

# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.distplot(df['AGE'])


# Normally distributed

# In[ ]:


df.columns


# In[ ]:


sns.scatterplot(df['AGE'],df['RBPS'])


# There is an correlation between Age and Blood pressure

# In[ ]:


sns.scatterplot(df['AGE'],df['CHOL'])


# In[ ]:


sns.scatterplot(df['AGE'],df['HR'])


# In[ ]:


sns.scatterplot(df['AGE'],df['OLDPACK'])


# In[ ]:


plt.figure(figsize=(15,7))
df[['AGE','RBPS','CHOL','HR']].boxplot()


# Pre processing and Modelling

# Logistic regression model

# In[ ]:


def logistic_regression(X,y):
    #X = data.drop('NUM',axis = 1)
    #y = data['NUM']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)
    #print(X_train)
    model = LogisticRegression(random_state=85,max_iter = 200, solver = 'liblinear').fit(X_train, y_train)
    print(model.score(X_train, y_train))
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


# # Normalise the data

# In[ ]:


normalized_df_train = preprocessing.normalize(df.drop('NUM',axis = 1))
normalized_df_test = df['NUM']


# In[ ]:


fig, ax=plt.subplots(1,2)
sns.distplot(df, ax=ax[0], color='y')
ax[0].set_title("Original Data")
sns.distplot(normalized_df, ax=ax[1])
ax[1].set_title("Scaled data")
plt.show()


# In[ ]:


logistic_regression(normalized_df_train,normalized_df_test)


# # Scaling

# In[ ]:


standardized_df_train = preprocessing.scale(df.drop('NUM',axis = 1))
standardized_df_test = df['NUM']


# In[ ]:


fig, ax=plt.subplots(1,2)
sns.distplot(df, ax=ax[0], color='y')
ax[0].set_title("Original Data")
sns.distplot(standardized_df, ax=ax[1])
ax[1].set_title("Scaled data")
plt.show()


# In[ ]:


logistic_regression(standardized_df_train,standardized_df_test)


# # Min Max scaling

# In[ ]:


scaled_data_train = minmax_scale(df.drop('NUM',axis = 1))
scaled_data_test = df['NUM']


# In[ ]:


fig, ax=plt.subplots(1,2)
sns.distplot(df, ax=ax[0], color='y')
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")
plt.show()


# In[ ]:


logistic_regression(scaled_data_train,scaled_data_test)


# Inference - Min max scaling produced more accuracy

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(scaled_data_train,scaled_data_test, test_size=0.25, random_state=42)


# In[ ]:


X_train


# # GridSearch

# In[ ]:


params={'solver': ['saga', 'newton-cg', 'lbfgs','liblinear','sag','saga']}

logmodel = LogisticRegression()
Classifier=GridSearchCV(model,params,scoring='accuracy',cv=10)
Classifier.fit(X_train,y_train)

print('best parameter: ', Classifier.best_params_)
print('best score: ', Classifier.best_score_)


# # Perceptron

# In[ ]:


slp = Perceptron()
accuracies=cross_val_score(slp,X_train,y_train,scoring='accuracy',cv=10)
mean_accuracy=np.mean(accuracies)
print(mean_accuracy)


# # MLP Perceptron

# In[ ]:


params={'batch_size': [32,64,128], 'hidden_layer_sizes': [50,100,150]}

mlp=MLPClassifier()
Classifier=GridSearchCV(mlp,params,scoring='accuracy',cv=10)
Classifier.fit(X_train,y_train)

print('best parameter: ', Classifier.best_params_)
print('best score: ', Classifier.best_score_)
y_pred = Classifier.predict(X_test)
accuracy_score(y_test, y_pred)


# Comparing Logistic has more CV score due to less sample size

# # Conclusion - Logistic regression works better in this case
