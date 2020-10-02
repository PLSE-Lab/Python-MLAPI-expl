#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# ## Read and Explore the data

# In[ ]:


df = pd.read_csv('../input/nasa-asteroids-classification/nasa.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df['Orbiting Body'].unique() #no significance


# In[ ]:


df['Equinox'].unique()


# All values are equal so significance,so these columns can be removed

# In[ ]:


df['Close Approach Date'].head() #no significance


# ## Data Preprocessing and Visualisation

# Let us check draw a count plot and see how many how much each categories represent

# In[ ]:


sns.countplot(x='Hazardous',data=df)


# number of Hazardous asteroids are just more than 10 percent

# In[ ]:


#Since there is a lot of data duplication will first clean a little bit then explore further
df=df.drop(['Est Dia in M(min)','Est Dia in M(max)','Est Dia in Miles(min)','Est Dia in Miles(max)','Est Dia in Feet(min)',
          'Est Dia in Feet(max)'],axis=1)
#droping multiple diameter values and keeping the value in M


# In[ ]:


df=df.drop(['Relative Velocity km per sec','Miles per hour','Miss Dist.(Astronomical)','Miss Dist.(lunar)',
            'Miss Dist.(miles)'],axis=1)


# In[ ]:


df=df.drop(['Neo Reference ID','Name','Close Approach Date','Epoch Date Close Approach','Orbit ID','Orbit Determination Date',
           'Equinox','Orbiting Body'],axis=1)


# A heatmap and see the corrulation between different parameters

# In[ ]:


plt.figure(figsize=(20,16))
sns.heatmap(data=df.corr(),cmap='coolwarm',annot=True)


# In[ ]:


# To get a more clear picture
df.corr()['Hazardous'][:-1].sort_values().plot(kind='bar')


# In[ ]:


#Let us change the Hazardous values from Boolean expression to integers
df['Hazardous']=df['Hazardous'].astype(int)


# In[ ]:


df['Hazardous']


# ## Model Creation and Evaluation

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


X=df.drop('Hazardous',axis=1)
y=df['Hazardous']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=100)


# In[ ]:


# MinMaxScaler to normalize the data
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)


# In[ ]:


X_test=scaler.transform(X_test)


# In[ ]:


X_train[0]


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix 


# In[ ]:


model_1 = RandomForestClassifier()
model_1.fit(X_train, y_train)

y_pred = model_1.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)*100
print(f'Accuracy: {round(accuracy,2)}')


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


confusion_matrix(y_test,y_pred)


# **Gradient Boosting Classifier**

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
model_2= GradientBoostingClassifier()
model_2.fit(X_train, y_train)

y_pred = model_2.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)*100
print(f'Accuracy: {round(accuracy,2)}')


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


confusion_matrix(y_test,y_pred)


# Both classifiers given same results in the dataset
