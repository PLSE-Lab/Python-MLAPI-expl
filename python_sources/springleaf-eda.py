#!/usr/bin/env python
# coding: utf-8

# # Springleaf EDA
# Notebook used in data exploration section of Coursera course on competitive data science: 
# - https://www.coursera.org/learn/competitive-data-science/lecture/nLD7Y/springleaf-competition-eda-i

# In[ ]:


import os
import numpy as np
import pandas as pd 
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

import seaborn


# In[ ]:


def autolabel(arrayA):
    ''' label each colored square with the corresponding data value. 
    If value > 20, the text is in black, else in white.
    '''
    arrayA = np.array(arrayA)
    for i in range(arrayA.shape[0]):
        for j in range(arrayA.shape[1]):
                plt.text(j,i, "%.2f"%arrayA[i,j], ha='center', va='bottom',color='w')

def hist_it(feat):
    '''Plot histogram of features with label'''
    plt.figure(figsize=(16,4))
    feat[Y==0].hist(bins=range(int(feat.min()),int(feat.max()+2)),normed=True,alpha=0.8)
    feat[Y==1].hist(bins=range(int(feat.min()),int(feat.max()+2)),normed=True,alpha=0.5)
    plt.ylim((0,1))
    
def gt_matrix(feats,sz=16):
    '''Plot heatmap of features with values greater than other features'''
    a = []
    for i,c1 in enumerate(feats):
        b = [] 
        for j,c2 in enumerate(feats):
            mask = (~train[c1].isnull()) & (~train[c2].isnull())
            if i>=j:
                b.append((train.loc[mask,c1].values>=train.loc[mask,c2].values).mean())
            else:
                b.append((train.loc[mask,c1].values>train.loc[mask,c2].values).mean())

        a.append(b)

    plt.figure(figsize = (sz,sz))
    plt.imshow(a, interpolation = 'None')
    _ = plt.xticks(range(len(feats)),feats,rotation = 90)
    _ = plt.yticks(range(len(feats)),feats,rotation = 0)
    autolabel(a)


# In[ ]:


def hist_it(feat):
    '''Plot histogram with 100 bins'''
    plt.figure(figsize=(16,4))
    feat[Y==0].hist(bins=100,range=(feat.min(),feat.max()),normed=True,alpha=0.5)
    feat[Y==1].hist(bins=100,range=(feat.min(),feat.max()),normed=True,alpha=0.5)
    plt.ylim((0,1))


# # Read the data

# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


PATH = '../input/'


# In[ ]:


train = pd.read_csv(PATH + 'train.csv.zip')

# reduce size of data to prevent kernel crashes
SAMPLE_SIZE = 1000
rand_idx = np.random.randint(0, len(train), size=SAMPLE_SIZE)
train = train.iloc[rand_idx,]
train.shape, train.head()

Y = train.target
Y.head()


# In[ ]:


# Loading too much data might cause the kerenel to crash
# Convert this to a code cell if you want to risk it
test = pd.read_csv(PATH + 'test.csv.zip')

# reduce size of data to prevent kernel crashes
rand_idx = np.random.randint(0, len(test), size=SAMPLE_SIZE)
test = test.iloc[rand_idx,]
test_ID = test.ID
test_ID.head()


# In[ ]:


import gc; gc.collect()


# # Data overview

# Probably the first thing you check is the shapes of the train and test matrices and look inside them.

# In[ ]:


train.shape, train.head()


# In[ ]:


# Loading too much data might cause the kerenel to crash
# Convert this to a code cell if you want to risk it
test.shape, test.head()


# There are almost 2000 anonymized variables! It's clear, some of them are categorical, some look like numeric. Some numeric feateures are integer typed, so probably they are event conters or dates. And others are of float type, but from the first few rows they look like integer-typed too, since fractional part is zero, but pandas treats them as `float` since there are NaN values in that features.   
# 
# From the first glance we see train has one more column `target` which we should not forget to drop before fitting a classifier. We also see `ID` column is shared between train and test, which sometimes can be succesfully used to improve the score.

# It is also useful to know if there are any NaNs in the data. You should pay attention to columns with NaNs and the number of NaNs for each row can serve as a nice feature later.

# In[ ]:


# Number of NaNs for each object
train.isnull().sum(axis=1).head(12)


# In[ ]:


# Number of NaNs for each column
train.isnull().sum(axis=0).head(12)


# Just by reviewing the head of the lists we immediately see the patterns, exactly 56 NaNs for a set of variables, and 24 NaNs for objects. 

# # Dataset cleaning

# ## Remove constant features

# All 1932 columns are anonimized which makes us to deduce the meaning of the features ourselves. We will now try to clean the dataset. 
# 
# It is usually convenient to concatenate train and test into one dataframe and do all feature engineering using it.

# In[ ]:


# combining datasets seems to crash the kernel
traintest = pd.concat([train, test], axis = 0)
traintest.shape


# First we schould look for a constant features, such features do not provide any information and only make our dataset larger. 

# In[ ]:


# `dropna = False` makes nunique treat NaNs as a distinct value
feats_counts = train.nunique(dropna = False)
feats_counts.sort_values()[:10]


# We found 5 constant features. Let's remove them.

# In[ ]:


constant_features = feats_counts.loc[feats_counts==1].index.tolist()
print (constant_features)

# Loading too much data might cause the kerenel to crash
traintest = traintest.drop(constant_features, axis = 1)
traintest.shape


# ## Remove duplicated features

# Fill NaNs with something we can find later if needed.

# In[ ]:


traintest = traintest.fillna('NaN')
traintest.head()


# Now let's encode each feature, as we discussed. 

# In[ ]:


train_enc =  pd.DataFrame(index = train.index)

for col in tqdm_notebook(traintest.columns):
    train_enc[col] = train[col].factorize()[0]

train_enc.shape, train_enc.head()


# We could also do something like this:

# In[ ]:


# train_enc[col] = train[col].map(train[col].value_counts())


# The resulting data frame is very very large, so we cannot just transpose it and use .duplicated. That is why we will use a simple loop.

# In[ ]:


dup_cols = {}

for i, c1 in enumerate(tqdm_notebook(train_enc.columns)):
    for c2 in train_enc.columns[i + 1:]:
        if c2 not in dup_cols and np.all(train_enc[c1] == train_enc[c2]):
            dup_cols[c2] = c1


# In[ ]:


dup_cols.items()


# Don't forget to save them, as it takes long time to find these.

# In[ ]:


# might have to install cPickel
#!pip install cPickle
#import cPickle as pickle
#pickle.dump(dup_cols, open('dup_cols.p', 'w'), protocol=pickle.HIGHEST_PROTOCOL)


# Drop from traintest.

# In[ ]:


traintest = traintest.drop(dup_cols.keys(), axis = 1)
traintest.shape, traintest.head()


# ## Determine data types

# Let's examine the number of unique values.

# In[ ]:


nunique = train.nunique(dropna=False)
nunique[:10]


# and build a histogram of those values

# In[ ]:


plt.figure(figsize=(14,4))
plt.hist(nunique.astype(float)/train.shape[0], bins=80, orientation='horizontal');


# Let's take a looks at the features with a huge number of unique values:

# In[ ]:


mask = (nunique.astype(float)/train.shape[0] > 0.8)
train.loc[:train.index[5], mask]


# The values are not float, they are integer, so these features are likely to be even counts. Let's look at another pack of features.

# In[ ]:


train_idx_orig = train.index
train = train.reset_index(drop=True)
Y = Y.reset_index(drop=True)
train.head(), Y.head()


# In[ ]:


mask = (nunique.astype(float)/train.shape[0] < 0.8) & (nunique.astype(float)/train.shape[0] > 0.4)
train.loc[:10, mask]


# These look like counts too. First thing to notice is the 23th line: 99999.., -99999 values look like NaNs so we should probably built a related feature. Second: the columns are sometimes placed next to each other, so the columns are probably grouped together and we can disentangle that.      

# Our conclusion: there are no floating point variables, there are some counts variables, which we will treat as numeric. 
# 
# And finally, let's pick one variable (in this case 'VAR_0015') from the third group of features.

# In[ ]:


train['VAR_0015'].value_counts()


# In[ ]:


cat_cols = list(train.select_dtypes(include=['object']).columns)
num_cols = list(train.select_dtypes(exclude=['object']).columns)


# # Go through some features

# Let's replace NaNs with something first.

# In[ ]:


train = train.replace('NaN', -999)


# Let's calculate how many times one feature is greater than the other and create cross tabel out of it. 

# In[ ]:


# select first few numeric features
feats = num_cols[:32]

# build 'mean(feat1 > feat2)' plot
gt_matrix(feats,16)


# Indeed, we see interesting patterns here. There are blocks of geatures where one is strictly greater than the other. So we can hypothesize, that each column correspondes to cumulative counts, e.g. feature number one is counts in first month, second -- total count number in first two month and so on. So we immediately understand what features we should generate to make tree-based models more efficient: the differences between consecutive values.

# ## VAR_0002, VAR_0003 

# In[ ]:


hist_it(train['VAR_0002'])
plt.ylim((0,0.05))
plt.xlim((-10,1010));


# In[ ]:


hist_it(train['VAR_0003'])
plt.ylim((0,0.03))
plt.xlim((-10,1010));


# In[ ]:


train['VAR_0002'].value_counts()[:10]


# In[ ]:


train['VAR_0003'].value_counts()[:10]


# We see there is something special about 12, 24 and so on, sowe can create another feature x mod 12. 

# ## VAR_0004

# In[ ]:


train['VAR_0004_mod50'] = train['VAR_0004'] % 50
hist_it(train['VAR_0004_mod50'])
plt.ylim((0,0.6))


# # Categorical features

# Let's take a look at categorical features we have.

# In[ ]:


train.loc[:,cat_cols].head().T


# `VAR_0200`, `VAR_0237`, `VAR_0274` look like some georgraphical data thus one could generate geography related features, we will talk later in the course.
# 
# There are some features, that are hard to identify, but look, there a date columns `VAR_0073` -- `VAR_0179`, `VAR_0204`, `VAR_0217`. It is useful to plot one date against another to find relationships. 

# In[ ]:


date_cols = [u'VAR_0073','VAR_0075',
             u'VAR_0156',u'VAR_0157',u'VAR_0158','VAR_0159',
             u'VAR_0166', u'VAR_0167',u'VAR_0168',u'VAR_0169',
             u'VAR_0176',u'VAR_0177',u'VAR_0178',u'VAR_0179',
             u'VAR_0204',
             u'VAR_0217']

for c in date_cols:
    train[c] = pd.to_datetime(train[c],format = '%d%b%y:%H:%M:%S')
    test[c] = pd.to_datetime(test[c],  format = '%d%b%y:%H:%M:%S')


# In[ ]:


c1 = 'VAR_0217'
c2 = 'VAR_0073'

# mask = (~test[c1].isnull()) & (~test[c2].isnull())
# sc2(test.ix[mask,c1].values,test.ix[mask,c2].values,alpha=0.7,c = 'black')

mask = (~train[c1].isnull()) & (~train[c2].isnull())
plt.figure(figsize=(14,4))
plt.scatter(train.loc[mask,c1].values,train.loc[mask,c2].values, c=train.loc[mask,'target'].values, 
            alpha=.5);


# We see that one date is strictly greater than the other, so the difference between them can be a good feature. Also look at horizontal line there -- it also looks like NaN, so I would rather create a new binary feature which will serve as an idicator that our time feature is NaN.
