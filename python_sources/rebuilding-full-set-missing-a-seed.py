#!/usr/bin/env python
# coding: utf-8

# **Code that generates 'Instant Gratification Dataset'** *(or, more precisely, a dataset similar to the original)*

# **Disclamer: I am new to Python so there is a decent chance I've messed up. As you will see below the values do not match exactly the original train set values. I may be missing another seed value.
# 
# This script is a copy of the code released by William Cukierski who generated 'Instant Gratification' dataset. [William Cukierski's code](https://www.kaggle.com/c/instant-gratification/discussion/96519#latest-561446)

# **Import packages**

# In[ ]:


import numpy as np 
import pandas as pd 
import random
from sklearn.datasets import make_classification
from sklearn import preprocessing
get_ipython().system('pip install random_name  # install random_name package. internet must be ON (in Settings on the right)')
import random_name

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Define helper functions, set seed and other variables**

# In[ ]:


NUM_SUB_DATASETS = 512
NUM_SAMPLES = 1024
NUM_FEATURES = 255
MAX_SEED = 2**32 - 1

def funny_names(X):
    random.seed(253689)
    kaggle_words = ['golden',
                    'learn',
                    'noise',
                    'goose',
                    'master',
                    'novice',
                    'expert',
                    'grandmaster',
                    'contributor',
                    'kernel',
                    'dataset',
                    'fimbus',
                    'fepid',
                    'pembus',
                    'sumble',
                    'hint',
                    'ordinal',
                    'distraction',
                    'important',
                    'dummy',
                    'sorted',
                    'unsorted',
                    'gaussian',
                    'entropy',
                    'discard'
                   ]
    return pd.DataFrame(X, columns=[random_name.generate_name() + '-' + random.choice(kaggle_words)  for x in range(0,len(X[0]))])

def create_dataset(random_seed):
    random.seed(3 + random_seed) #setting a seed for the randint() call below
    X,y = make_classification(n_samples=NUM_SAMPLES, 
                             n_features=NUM_FEATURES, 
                             n_informative=random.randint(33,47),
                             n_redundant=0,
                             n_repeated=0,
                             n_classes=2,
                             n_clusters_per_class=3,
                             weights=None,
                             flip_y=0.05,
                             class_sep=1.0,
                             hypercube=True,
                             shift=0.0,
                             scale=1.0,
                             shuffle=True,
                             random_state=random_seed)
    df = funny_names(X)
    df['wheezy-copper-turtle-magic'] = random_seed
    df = df.sample(frac=1, axis=1, random_state=random_seed, replace=False) # Shuffle column order so magic variable isn't last
    df['target'] = y
    return df


# **Generate dataset**

# In[ ]:


seed_list = random.sample(range(1,MAX_SEED), NUM_SUB_DATASETS)
df = pd.concat([create_dataset(s) for s in seed_list], axis=0, sort = False).reset_index(drop=True)
df = df.sample(frac=1, random_state=9726).reset_index(drop=True) # Shuffle rows

# Label encode our 'magic' variable
le = preprocessing.LabelEncoder()
le.fit(df['wheezy-copper-turtle-magic'])
df['wheezy-copper-turtle-magic'] = le.transform(df['wheezy-copper-turtle-magic']) 

train = pd.read_csv('../input/train.csv') # read in train.csv for comparison and columns ordering
#test = pd.read_csv('../input/test.csv')


# **Quick spot-check and output to .csv**

# In[ ]:


train.head()


# In[ ]:


df.head()


# In[ ]:


train['muggy-smalt-axolotl-pembus'].sort_values()[0:10]  # a quick check of sorted values demonstrate that we did not get the exact values in train and df but the distributions look similar


# In[ ]:


df['muggy-smalt-axolotl-pembus'].sort_values()[0:10]


# In[ ]:


train['muggy-smalt-axolotl-pembus'].describe()


# In[ ]:


df['muggy-smalt-axolotl-pembus'].describe()


# In[ ]:


col_names = train.columns
df = df.reindex(col_names[1:], axis=1)  # sort df to match the order of columns in train

print("df shape = {}, train shape = {}".format(df.shape,train.shape))
#print(df.shape, train.shape) # id column is missing in the generate set since William likely added it later during Train / Test / Private_Test split
df.head()  # last quick look after sorting columns

df.to_csv('all.csv', index=False)  # output the generated set to .csv

