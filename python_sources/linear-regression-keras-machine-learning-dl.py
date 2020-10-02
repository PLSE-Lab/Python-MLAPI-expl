#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This notebook is built to help predict if a patient has a heart disease or not.
# We have a data which classified if patients have heart disease or not according to features in it. We will try to use this data to create a model which tries predict if a patient has this disease or not.

# ## Columns - Defined

#     age - age in years 
#     sex - (1 = male; 0 = female) 
#     cp - chest pain type 
#     trestbps - resting blood pressure (in mm Hg on admission to the hospital) 
#     chol - serum cholestoral in mg/dl 
#     fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) 
#     restecg - resting electrocardiographic results 
#     thalach - maximum heart rate achieved 
#     exang - exercise induced angina (1 = yes; 0 = no) 
#     oldpeak - ST depression induced by exercise relative to rest 
#     slope - the slope of the peak exercise ST segment 
#     ca - number of major vessels (0-3) colored by flourosopy 
#     thal - 3 = normal; 6 = fixed defect; 7 = reversable defect 
#     target - have disease or not (1=yes, 0=no)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from math import pi


# ## Load and prepare the data
# A critical step in working with neural networks is preparing the data correctly. Variables on different scales make it difficult for the network to efficiently learn the correct weights. Below, we've written the code to load and prepare the data.

# In[ ]:


data_path = '../input/heart.csv'

df = pd.read_csv(data_path)


# In[ ]:


df.head()


# In[ ]:


df.describe()


# ### Corelation plot to find important Features

# In[ ]:


fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()


# ### Inference
# It's clear that chest pain, Maximum Heart Rate and Slope are significantly important than other

# In[ ]:


sns.pairplot(data=df)


# ## Analyzing all Features

# ### 1. Men vs Women %

# In[ ]:


#print("Men vs Women Count\n", df.sex.value_counts())
men_count = len(df[df['sex']== 1])
women_count = len(df[df['sex']==0])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'Men','Women'
sizes = [men_count,women_count]
colors = ['skyblue', 'yellowgreen']
explode = (0, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.2f%%', shadow=True)
plt.show()
 


# ### 2. People who have disease vs who don't

# In[ ]:


sns.countplot(x='target', data=df)


# ### 3. Heart Disease Ratio

# In[ ]:


print("People having heart diseace vs people who doesn't: \n", df.target.value_counts())
heart_disease = len(df[df['target']==1])
no_heart_disease = len(df[df['target']==0])
labels = ["Heart Diesease", "NO Heart Disease"]
sizes = [heart_disease, no_heart_disease]
colors = ['skyblue', 'yellowgreen']
plt.figure(figsize=(8,6))

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.2f%%', shadow=True)
plt.show()
 


# ### Analyzing heart disease over age distribution

# In[ ]:


pd.crosstab(df.age, df.target).plot(kind='bar', figsize=(20, 10))
plt.title("People having heart disease vs people not having heart disease for a given age")
plt.xlabel("Age Distribution")
plt.ylabel("Heart Disease Frequency")


# ### Inference
# We could infer from the chart above that age is not a huge influening factor for heart disease

# ### Analyzing heart disease over Maximum Heart Rate

# In[ ]:


my_colors = 'yr'
pd.crosstab(df.thalach, df.target).plot(kind='bar', figsize=(20,10), color=my_colors)
plt.title("Heart diseases frequency for Maximum heart rate")
plt.xlabel("Maximum Heart Rate for a person")
plt.ylabel("Heart Disease Frequency")


# ### Inference
# We could infer from the above chart that people who have higher heart rate has a higher probability of having a heart disease

# ## Analyzing heart disease frequency over chest pain type 
# 
# Chest pain type:
#  
#    1  --> Typical angina 
#    
#    2  --> Atypical angina 
#    
#    3  --> Non-anginal pain 
#    
#    4  --> Asymptomatic

# In[ ]:


pd.crosstab(df.cp, df.target).plot(kind='bar', figsize = (20,10))
plt.title("Heart disease frequency over Chesr Pain Type")
plt.xlabel("Cheast Pain Type ")
plt.ylabel("Heart Disease Frequency")


# ### Inference
# 
# We could infer from the above chrat that people who have Atypical angina or non-anginal pain have higher probability of having heart disease

# ## Analyzing Heart disease frequency over Slope
# 
# 1 --> upsloping
# 
# 2 --> flat
# 
# 3 --> downsloping

# In[ ]:


pd.crosstab(df.slope, df.target).plot(kind='bar', figsize=(10,10))
plt.title("Heart disease frequency over Slope")
plt.xlabel("Slope Frequency")
plt.ylabel("Heart Disease Frequency")


# ## Cleaning the data

# In[ ]:



a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")
frames = [df, a, b, c]
df = pd.concat(frames, axis = 1)

to_be_dropped = ['cp', 'thal', 'slope']
df = df.drop(to_be_dropped, axis=1)
df.head()


# ## Normalizing the data

# In[ ]:


df = (df - np.min(df)) / (np.max(df) - np.min(df)).values


# ## Converting the data to features and targets

# In[ ]:


features = df.drop("target", axis=1)
targets = df.target.values


# ## Splitting the data into train and test sets

# In[ ]:


from sklearn.model_selection import train_test_split
train_features,test_features,train_targets,test_targets = train_test_split(features,targets,test_size = 0.20,random_state = 42)


# ## Finally we have prepared our data. Now it's time to train it with neural nets !!! 
# 

# In[ ]:


# Imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

# Building the model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(train_features.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(loss = 'mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
model.summary()


# ## Training the model

# In[ ]:


history = model.fit(train_features, train_targets, validation_split=0.2, epochs=100, batch_size=16, verbose=1)


# ## Evaluating the model

# In[ ]:


#print(vars(history))
plt.plot(history.history['loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# ## Prediction vs original labels

# In[ ]:


y_pred = model.predict(test_features)
plt.plot(test_targets)
plt.plot(y_pred)
plt.title('Prediction')

