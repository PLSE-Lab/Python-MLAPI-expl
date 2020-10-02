#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install pomegranate


# In[ ]:


pip install graphviz


# In[ ]:


pip install libgraphviz-dev


# In[ ]:


pip install cgraph


# In[ ]:


pip install pydot


# In[ ]:


pip install libcgraph


# In[ ]:


pip install pydotplus


# In[ ]:


conda install pygraphviz


# In[ ]:


pip install networkx


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler, Normalizer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import seaborn as sns
from IPython.core.display import Image
import matplotlib.pyplot as plt
from pomegranate import *
from tqdm import tqdm
import graphviz
import pydotplus
import pygraphviz
import networkx as nx

sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(5,5)})

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Assigning a name to the dataframe
oasis_longitudinal = pd.read_csv("../input/mri-and-alzheimers/oasis_longitudinal.csv")


# In[ ]:


# Looking at the first 5 rows of the dataframe
print("Longitudinal MRI Data in Nondemented and Demented Older Adults")
oasis_longitudinal.head()


# In[ ]:


# Dropping rows containing NaN values as the Chow-Liu algorithm does not support NaN's and checking a few features for dimentionality

oasis_longitudinal = oasis_longitudinal.dropna()

print(oasis_longitudinal['Hand'].value_counts())
print(oasis_longitudinal['M/F'].value_counts())
print(oasis_longitudinal['SES'].value_counts())


# In[ ]:


oasis_longitudinal.dtypes


# In[ ]:


# Dropping unnecessary columns
X = oasis_longitudinal.drop(['Subject ID', 'MRI ID', 'Hand'], axis=1)
y = oasis_longitudinal['Group']

# Encoding categorical columns
le_group = LabelEncoder()
le_mf = LabelEncoder()

X['Group'] = le_group.fit_transform(X['Group'])
X['M/F'] = le_mf.fit_transform(X['M/F'])

# Normalizing columns with large ranges
cols_to_norm = ["MR Delay", "eTIV", "ASF"]
X[cols_to_norm] = Normalizer().fit_transform(X[cols_to_norm])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


X


# In[ ]:


X_train


# In[ ]:


X_test


# In[ ]:


# Naming the nodes of the DAG
state_names = ["Group", "Visit", "MR Delay", "Male / Female", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]

# Generating a Chow-Liu BN from the dataset
model_chow_liu = BayesianNetwork.from_samples(X_train, algorithm='chow-liu', state_names=state_names, name="Chow-Liu Bayesian Network DAG", pseudocount=0.1)


# In[ ]:


# Plotting the DAG
plt.figure(figsize=(16,8))
model_chow_liu.plot()


# In[ ]:


# 1 = Male
# 0 = Female

# Testing what diagnosis individual 9 most likely has

# The following are the observed conditions/ stats about the individual:
observations = {"Visit": 3,
                "Male / Female": 1,
                "Age": 85,
                "EDUC": 12,
                "SES": 4.0,
                "MMSE": 30.0,
                "CDR": 0.0,
                "nWBV": 0.705}

# Using the learned BN algorithm, the probability of being diagnosed with Dimentia is given, along with other features
beliefs = map( str, model_chow_liu.predict_proba( observations ) )
print ("\n".join( "{}\t\t{}".format( state.name, belief ) for state, belief in zip( model_chow_liu.states, beliefs ) ))

# 2 = Nondemented
# 1 = Demented
# 0 = Converted : subjects were characterized as nondemented at the time of their initial visit and were subsequently characterized as demented at a later visit.


# Individual 9 has the highest chance of being nondemented at ~78%

# In[ ]:


# 1 = Male
# 0 = Female

# Testing what diagnosis individual 148 most likely has

# The following are the observed conditions/ stats about the individual:
observations = {"Visit": 2, 
                "Male / Female": 0,
                "Age": 82,
                "EDUC": 18,
                "SES": 2.0,
                "MMSE": 30.0,
                "CDR": 0.0,
                "nWBV": 0.690,
               }

# Using the learned BN algorithm, the probability of being diagnosed with Dimentia is given, along with other features
beliefs = map( str, model_chow_liu.predict_proba( observations ) )
print ("\n".join( "{}\t\t{}".format( state.name, belief ) for state, belief in zip( model_chow_liu.states, beliefs ) ))

# 2 = Nondemented
# 1 = Demented
# 0 = Converted : subjects were characterized as nondemented at the time of their initial visit and were subsequently characterized as demented at a later visit.


# Individual 148 has the highest chance of being nondemented at ~97%
