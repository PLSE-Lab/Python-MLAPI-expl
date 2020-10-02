#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
#Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
#Prediction
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


spotify_data = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding = "ISO-8859-1",index_col='Unnamed: 0')


# In[ ]:


spotify_data


# Number of unique variables:

# In[ ]:


spotify_data['Genre'].nunique()


# Unique Genres:

# In[ ]:


spotify_data['Genre'].unique()


# Number of unique Artists:

# In[ ]:


spotify_data['Artist.Name'].nunique()


# Unique Artists:

# In[ ]:


spotify_data['Artist.Name'].unique()


# Genre Popularity:

# In[ ]:


plt.title("Genre Popularity in the top 50 Spotify songs - 2019")
sns.barplot(x=spotify_data['Popularity'], y=spotify_data['Genre'])


# Artist Popularity:

# In[ ]:


plt.figure(figsize=(5,10))
plt.title("Artist Popularity in the top 50 Spotify songs - 2019")
sns.barplot(x=spotify_data['Popularity'], y=spotify_data['Artist.Name'])


# Correlational Heatmap:

# In[ ]:


plt.figure(figsize=(15,17))
sns.heatmap(spotify_data.corr(),annot=True,cmap='Blues', fmt='g')


# Some correlation found:
# 1. Positive correlation
#     * Speechness and Beats per Minute
#     * Enery and Loudness
#     * Energy and Valence
# 2. Negative correlation
#     * Energy and Acousticness    
#     * Valence and Popularity     
#     * Loudness and Speechness     

# Plots showing these correlations:

# In[ ]:


g0 = sns.lmplot(x='Speechiness.',y='Beats.Per.Minute',data=spotify_data)
axes = g0.axes.flatten()
axes[0].set_title('Speechiness vs Beats Per Minute')


# In[ ]:


g1 = sns.lmplot(x='Energy',y='Loudness..dB..',data=spotify_data)
axes = g1.axes.flatten()
axes[0].set_title('Energy vs Loudness')


# In[ ]:


g2 = sns.lmplot(x='Energy',y='Valence.',data=spotify_data)
axes = g2.axes.flatten()
axes[0].set_title('Energy vs Valence')


# In[ ]:


g3 = sns.lmplot(x='Energy',y='Acousticness..',data=spotify_data)
axes = g3.axes.flatten()
axes[0].set_title('Energy vs Acousticness')


# In[ ]:


g4 = sns.lmplot(x='Valence.',y='Popularity',data=spotify_data)
axes = g4.axes.flatten()
axes[0].set_title('Valence vs Popularity')


# In[ ]:


g5 = sns.lmplot(x='Loudness..dB..',y='Speechiness.',data=spotify_data)
axes = g5.axes.flatten()
axes[0].set_title('Loudness vs Speechiness')


# Popularity Prediction

# In[ ]:


y = spotify_data['Popularity']
X = spotify_data.drop(['Popularity'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3,random_state=0)


# In[ ]:


s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)


# In[ ]:


OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
my_model = XGBRegressor(random_state=0, n_estimators=1000, learning_rate=0.05)
my_pipeline = Pipeline(steps=[('preprocessor', OH_encoder),
                             ('model', my_model)])
my_pipeline.fit(X_train,y_train)
preds= my_pipeline.predict(X_valid)

print(mean_absolute_error(y_valid,preds))

