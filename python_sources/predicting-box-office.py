#!/usr/bin/env python
# coding: utf-8

# Lets try to predict the income that move can generate based on the director, actors, genre and budget.
# Unfortunately category encoders are not available in Kaggle.
# 
# #NOT MY WORK, COPIED SO I CAN MOD AND ADD EXTRA FEATURES

# In[ ]:


from __future__ import print_function

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, pipeline
#import category_encoders as ce

from sklearn import preprocessing
from collections import defaultdict

df = pd.read_csv("../input/movie_metadata.csv")
df.head()


# Lets remove NaNs and irrelevant columns

# In[ ]:


clean_data = df[df['director_name'].notnull() & df['duration'].notnull() & df['actor_2_name'].notnull() & df['genres'].notnull()
               & df['actor_1_name'].notnull() & df['actor_3_name'].notnull() & df['plot_keywords'].notnull() & df['country'].notnull()
               & df['title_year'].notnull() & df['budget'].notnull() & df['gross'].notnull()]
x_list_pre = ['director_name','duration','actor_2_name', 'genres', 'actor_1_name', 'actor_3_name', 
          'country', 'title_year','budget','gross']
x_list_enc = ['director_name','actor_2_name', 'actor_1_name', 'actor_3_name', 
          'country']
x_list_add = ['duration','title_year','budget']
x_list_train = ['director_name','duration','actor_2_name', 'genres', 'actor_1_name', 'actor_3_name', 
          'country', 'title_year','budget']


# Lets get rid of the budget outliners

# In[ ]:


clean_data_slice = clean_data.ix[:,x_list_pre]
min_max=preprocessing.RobustScaler()

e_data_minmax=min_max.fit_transform(clean_data_slice[['budget', 'gross']])
clean_data_slice['budget'] = pd.DataFrame(e_data_minmax[:,0],index=clean_data_slice.index)
clean_data_slice['gross'] = pd.DataFrame(e_data_minmax[:,1],index=clean_data_slice.index)
center_data_slice = clean_data_slice[clean_data_slice['budget']<5]
center_data_slice.head(20)

df.loc[df['title_year'].idxmin()]


# First let's encode genres

# In[ ]:



le = defaultdict(preprocessing.LabelEncoder) 
s = center_data_slice['genres'].str.split('|').apply(pd.Series, 1)
del center_data_slice['genres']
s = s.fillna('')

genres_num = s.apply(lambda x: le[x.name].fit_transform(x))
genres_num.head()


# Then let's encode all categorical columns.

# In[ ]:


encode_slice=center_data_slice[x_list_enc]
encoded_data = encode_slice.apply(lambda x: le[x.name].fit_transform(x))
encoded_data = encoded_data.join(genres_num)
encoded_data.head()
#encoder = ce.BinaryEncoder(cols=['director_name', 'country', 'actor_2_name', 'actor_1_name', 'actor_3_name']) 
# 0,33
#encoder = ce.HashingEncoder(cols=['director_name', 'country', 'actor_2_name', 'actor_1_name', 'actor_3_name']) 
# 0,36
#encoder = ce.OneHotEncoder(cols=['director_name', 'country', 'actor_2_name', 'actor_1_name', 'actor_3_name']) 
# 0,40
#
#encoder = ce.BackwardDifferenceEncoder(cols=['director_name', 'country','actor_2_name', 'actor_1_name', 'actor_3_name'])
#0.45
#encoder = ce.HelmertEncoder(cols=['director_name', 'country','actor_2_name', 'actor_1_name', 'actor_3_name'])
# 0.38
#encoder = ce.SumEncoder(cols=['director_name', 'country','actor_2_name', 'actor_1_name', 'actor_3_name'])
# 40%


# In[ ]:


encoded_data = encoded_data.join(clean_data_slice[x_list_add])
y_data = center_data_slice['gross']
encoded_data.head()


# Split data in training and test set

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(encoded_data, y_data, 
                                                    test_size=0.25, random_state=0)


# Get average of 3-fold cross-validation score using an SVC estimator

# In[ ]:


n_folds = 3

from sklearn.model_selection import KFold


# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(n_splits=n_folds, random_state=1)
kf = kf.get_n_splits(x_train)


# Let's try RandomForest

# In[ ]:


print ('Training Random Forest...')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


clf_rf = RandomForestRegressor(n_estimators=1000,max_depth=10) 
clf_rf = clf_rf.fit( x_train, y_train )
classifier_score = clf_rf.score(x_test, y_test)
print ('The classifier accuracy score is {:.2f}'.format(classifier_score))
# Get average of 3-fold cross-validation score 
score = cross_val_score(clf_rf, x_test, y_test, cv=kf)
print ('The {}-fold cross-validation accuracy score for this classifier is {:.2f}'.format(n_folds, score.mean()))

x_1=x_test['budget']

y_1 = clf_rf.predict(x_test)

plt.figure()
plt.scatter(x_1, y_test, c="darkorange", label="data")
plt.scatter(x_1, y_1, color="cornflowerblue", label="max_depth=5")
plt.xlabel("data")
plt.ylabel("target")
plt.title("Random Forest Regression")
plt.legend()
plt.show()


# There is a major problem with the predictions. Let's see if we can find any ideas looking at the smaller dataset.

# In[ ]:


plt.figure()
plt.scatter(x_1[:50], y_test[:50], c="darkorange", label="data")
plt.scatter(x_1[:50], y_1[:50], color="cornflowerblue", label="max_depth=5")
plt.xlabel("budget")
plt.ylabel("gross")
plt.title("Random Forest Regression")
plt.legend()
plt.show()


# Still the same :-(
