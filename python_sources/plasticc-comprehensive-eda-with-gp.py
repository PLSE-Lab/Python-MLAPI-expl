#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data_df = pd.read_csv('../input/training_set.csv')
train_metadata_df = pd.read_csv('../input/training_set_metadata.csv')


# **Simple Feature Engineering**

# In[ ]:


train_data_df['flux_ratio_sq'] = np.power(train_data_df['flux'] / train_data_df['flux_err'], 2.0)
train_data_df['flux_by_flux_ratio_sq'] = train_data_df['flux'] * train_data_df['flux_ratio_sq']


# **Extracting all features from the train meta data and features like minimum, maximum, mean, median, skew etc from the time-series data (train data)**

# In[ ]:


data_features = train_data_df.columns[1:]
metadata_features = train_metadata_df.columns[1:]


# In[ ]:


groupObjects = train_data_df.groupby('object_id')[data_features]

print("Add constant object features")
features = train_metadata_df.drop(['target'], axis=1)

print("Add sum of mutable object features")
features = pd.merge(features, groupObjects.agg('sum'), how='right', on='object_id', suffixes=['', '_sum'])

print("Add mean of mutable object features")
features = pd.merge(features, groupObjects.agg('mean'), how='right', on='object_id', suffixes=['', '_mean'])

print("Add median of mutable features")
features = pd.merge(features, groupObjects.agg('median'), how='right', on='object_id', suffixes=['', '_median'])

print("Add minimum of mutable features")
features = pd.merge(features, groupObjects.agg('min'), how='right', on='object_id', suffixes=['', '_min'])

print("Add maximum of mutable features")
features = pd.merge(features, groupObjects.agg('max'), how='right', on='object_id', suffixes=['', '_max'])

print("Add range of mutable features")
features = pd.merge(features, groupObjects.agg(lambda x: max(x) - min(x)), how='right', on='object_id', suffixes=['', '_range'])

print("Add standard deviation of mutable features")
features = pd.merge(features, groupObjects.agg('std'), how='right', on='object_id', suffixes=['', '_stddev'])

print("Add skew of mutable features")
features = pd.merge(features, groupObjects.agg('skew'), how='right', on='object_id', suffixes=['', '_skew'])


# In[ ]:


features = features.fillna(features.mean())


# In[ ]:


features


# In[ ]:


features = features.drop('object_id', axis=1)


# In[ ]:


targets = train_metadata_df.target.map({6:0, 15:1, 16:2, 42:3, 52:4, 53:5, 62:6, 64:7, 65:8, 67:9, 88:10, 90:11, 92:12, 95:13})


# In[ ]:


targets


# In[ ]:


features['target'] = targets


# **Engineer new features using Genetic Programming with the gplearn library.**

# In[ ]:


import gplearn
from gplearn.genetic import SymbolicTransformer


# In[ ]:


import keras 
from keras.utils import to_categorical


# In[ ]:


function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min']

gp = SymbolicTransformer(generations=100, population_size=2000,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=0, n_jobs=3)

gp.fit(features.drop('target', axis=1).values, targets.values)


# In[ ]:


engineered_features = gp._programs

for i in range(len(engineered_features)):
    for engineered_feature in engineered_features[i]:
        if engineered_feature != None:
            print(engineered_feature)


# In[ ]:


new_features = pd.DataFrame(gp.transform(features.drop('target', axis=1).values))


# In[ ]:


features = pd.concat([features, new_features], axis=1, join_axes=[features.index])


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# **Visualize the probability distributions of each feature for different astronomical source types using seaborn**

# In[ ]:


import seaborn as sns


# In[ ]:


sns.set(style="darkgrid")


# **Original features**

# In[ ]:


columns = features.columns

for column in columns[:-10]:
    sns.pairplot(x_vars=column, y_vars=column, hue='target', diag_kind='kde', data=features)


# **Engineered features**

# In[ ]:


for column in columns[-10:]:
    sns.pairplot(x_vars=column, y_vars=column, hue='target', diag_kind='kde', data=features)


# **The features for which the data distributions for different classes are very different from each other are most likely the more "important" features. This "difference" between distribution can be measured using Kullback-Leibler Divergence (a distribution similarity metric)**

# In[ ]:


# sns_plot = sns.pairplot(data=features, hue='target', diag_kind='kde')
# sns_plot.savefig('plasticc_visualizations.png')


# In[ ]:




