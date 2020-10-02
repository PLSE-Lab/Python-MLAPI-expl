#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# I have experimented with some different kinds of modelling, but most of my time has been spent on everything that comes before that. I have written my own small functions, and then rewritten them or replaced them with existing ones that are standard tools in the kaggle community. This notebook is a distillation of some preliminary work I did which was not fit to be shared

# In[ ]:


import os, re, gc, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from scipy.stats import wasserstein_distance as wd
import seaborn as sns
warnings.filterwarnings("ignore")
gc.collect()


# In[ ]:


DATA_DIR = '../input/'
FILES={}
for fn in os.listdir(DATA_DIR):
    FILES[ re.search( r'[^_\.]+', fn).group() ] = DATA_DIR + fn

CAT_COL='wheezy-copper-turtle-magic'    

train = pd.read_csv(FILES['train'],index_col='id')
test = pd.read_csv(FILES['test'],index_col='id')
CATS = sorted(train[CAT_COL].unique())

# I use id as the index column to make work easier when splitting or augmenting training data
# You can waste a lot of time with poor pandas concatenation


# In[ ]:


c=train[ train[CAT_COL]==0 ].corr().abs()
high_c = c.where(np.triu(np.ones(c.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
# unpacking that statement
# get indices for upper triangular matrix not including diagonal np.triu(np.ones(c.shape), k=1).astype(bool)
# get the indexed values, and store them in a sorted list c.where().stack().sort_values(ascending=False)
high_c


# In[ ]:


high_c.hist(bins=100)
display(high_c.describe())
print(f'''# of feature pairs with correlation greater than:\n
      0.1 : {sum(high_c.gt(.1))} \n
      0.15: {sum(high_c.gt(.15))}''')


# These are the correlations between features for the training dataset where 'wheezy-copper-turtle-magic'==0 (including target as a feature).
# 
# About 1 in 40 feature pairs has a correlation greater than 0.1, and 1 in 624 have a correlation greater than 0.15
# 
# Next I'll take another look at feature importance using the Wasserstein Distance metric.
# Also known as the 'earth moving distance', it measures the separation between 2 different distributions.
# I'll be looking at the difference, for each feature, of the samples where target is 0 or 1
# 
# Note: WD of in- and out-of-target distributions for feature 'wheezy-copper-turtle-magic' == 0.30391190887413444, far above any other feature. Having accepted the wisdom of considering 512 subsets of data, I'll be trying to continue the push to provide better inputs to models

# In[ ]:


def col_sort_by_wd(df):
    distances ={}
    for c in tqdm_notebook( range(1,df.shape[1]-1) ):
        a = df.loc[ df.target==0 ].iloc[:,c]
        b = df.loc[ df.target==1 ].iloc[:,c]
        distances[train.columns[c]]= wd(a,b)
    w = pd.Series(distances)
    return w.sort_values(ascending=False)


# In[ ]:


w1 = col_sort_by_wd(train).drop(CAT_COL)
sns.distplot(w1)
w1


# In[ ]:


w2 = col_sort_by_wd(train[ train[CAT_COL]==0 ].drop(CAT_COL,axis=1))
sns.distplot(w2)
w2


# In[ ]:


w2[ w2.gt(.25) ]
# top 45 features for separating by target, when 'wheezy-copper-turtle-magic'==0


# Not shown: when varying the selection to different values of `wheezy-copper-turtle-magic` there will be about 40-50 features that are significantly more informative than the rest when considered individually. For linear models, that means you can drop all but the 50 strongest features for any given subset without significantly decreasing performance. That means you can train the same model faster, or fit a more detailed model.

# In[ ]:


sns.jointplot(w1,w2[w1.index],marker='1').set_axis_labels('Feature scores across whole dataset','Feature scores when restricted to wheezy-copper-turtle-magic==0')


# Just repeating what we've seen in tabular form, after restricting to samples for a single value of `wheezy-copper-turtle-magic`, we see an uplift of signifiance of features across the board when trying to distinguish between in- and out-of-target distribtuions.
# 
# Next, we'll see what this looks like for the feature `zippy-harlequin-otter-grandmaster` which was the strongest feature in the case where `wheezy-copper-turtle-magic` is 0.

# In[ ]:


from scipy.stats import wasserstein_distance as wd

feat='zippy-harlequin-otter-grandmaster'
sub_idx_0 = train['target']==0
sub_idx_1 = train['target']==1

sns.distplot(train[sub_idx_0][feat],bins=16,color='red')
sns.distplot(train[sub_idx_1][feat],bins=16,color='blue')

print(f'Wasserstein distance between distributions: {wd( train.loc[sub_idx_0][feat], train.loc[sub_idx_1][feat])}' )


# In[ ]:


sub_idx_0 = (train['target']==0) & (train[CAT_COL]==0)
sub_idx_1 = (train['target']==1) & (train[CAT_COL]==0)

sns.distplot(train[sub_idx_0][feat],bins=16,color='red')
sns.distplot(train[sub_idx_1][feat],bins=16,color='blue')

print(f'Wasserstein distance between distributions: {wd( train.loc[sub_idx_0][feat], train.loc[sub_idx_1][feat])}' )


# Ok, so restricting to when `wheezy-copper-turtle-magic` makes it easier to distinguish samples where target is 0 or 1.
# 
# Interestingly, neither distribution is gaussian. In particular it looks like the red distribution (target==0) is the sum of 2 or more gaussian processes.
# 
# Trying to untangle the features can drive you mad.
# * I haven't tried brute force generation of possible non-linear features that can be combined into a linear model.
# * I have tried isolating the insignificant features and reducing their dimensionality by transforming them with a non-linear manifold, and then fitting a classifier to the top features along with my synthetic ones
# * I did start to try a PyTorch implementation of a Neural Ordinary Differential Equation solver, but ran into difficulties loading the library on the Kaggle cloud to enable GPU computation -- I would be very interested to see someone pick that up

# In[ ]:


c=train[ (train[CAT_COL]==0) ].corr().abs()
high_c = c.where(np.triu(np.ones(c.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
high_c.filter(like='zippy-harlequin-otter-grandmaster')
# the correlations do shift when you isolate for target==0 or 1
# ex: replace with c=train[ (train[CAT_COL]==0) & (train['target']==0) ].corr().abs()


# In[ ]:


# looking at top 2 and bottom 2 correlations with zippy-harlequin-otter-grandmaster under CAT_COL==0
cols_of_interest=['skimpy-copper-fowl-grandmaster','flaky-chocolate-beetle-grandmaster','zippy-harlequin-otter-grandmaster','blurry-wisteria-oyster-master','crabby-carmine-flounder-sorted' ]
feats = train[sub_idx_0][cols_of_interest]
sns.pairplot(train[ train[CAT_COL]==0 ],vars=cols_of_interest,markers=['1','2'],hue='target',palette='husl')
print(f'Multivariate distribution plot where {str(CAT_COL)}==0 and target==0')


# It's hard to pick an angle which will enable greater separation for the least significant features.
# The strong features have separation by slightly different mean values
# The weak features are only separated by changes in variance, resulting in much more overlap and lower WD scores.
# 
# I've seen a lot of people trim their dataset by applying a variance threshold, and I believe that is just another way of achieving a similar or the same result.
# 
# I believe the features with low WD scores are features that have undergone a non-linear transformation.
# It's so much easier to forget them, the models train faster and so you can get a better score by searching more hyperparameters.
# 
# 
# The challenge is to invert these unknown non-linear transforms, and recover linearly useful features under within reasonable time and computing resources.
# 
# We were given a poem as a clue, and I think there is another trick to be uncovered.

# In[ ]:


names = list(train.drop(["target"],axis=1).columns.values)


# In[ ]:


first_names = []
second_names = []
third_names = []
fourth_names = []

for name in names:
    words = name.split("-")
    first_names.append(words[0])
    second_names.append(words[1])
    third_names.append(words[2])
    fourth_names.append(words[3])


# In[ ]:


fns = {x for x in first_names}
sns = {x for x in second_names}
tns = {x for x in third_names}
fons = {x for x in fourth_names}

ans = [fns, sns, tns, fons]


# In[ ]:


for i in range(len(ans)):
    for j in range(i+1,len(ans)):
        print(f'{i+1} & {j+1}: {ans[i]&ans[j]}')


# In[ ]:


display( train.filter(like='-blue-').head() )
train.filter(like='coral').head()


# `stinky-maroon-blue-kernel` and `greasy-sepia-coral-dataset` each have a word that occurs out of its expected position. I attribute this to chance, and not an intentional clue.

# In[ ]:


pd.Series(first_names).value_counts().hist()
pd.Series(first_names).value_counts()


# We've already split up training into 512 separate models.
# Doing something further starting with first names is an idea. There are 85 words in that position, with no more than 7 occurences of any given word.

# In[ ]:


pd.Series(second_names).value_counts().hist()
pd.Series(second_names).value_counts()


# In[ ]:


pd.Series(third_names).value_counts().hist()
pd.Series(third_names).value_counts()


# In[ ]:


pd.Series(fourth_names).value_counts().hist()
pd.Series(fourth_names).value_counts()


# In[ ]:


pd.Series(fourth_names).nunique()


# Just 26 words in the final position.
# Other than our useful `wheezy-copper-turtle-magic`, there are 5 to 19 features for any given word in that fourth position.
# This seems like an even better prospect for further slicing and dicing.
# Is there a clue in `26`? It is an unsual number, the number of letters in the english alphabet.

# In[ ]:


train.shape


# In[ ]:


np.log(262144)/np.log(2)


# Part of the reason why I think there are clues to be found is that the dimensions chosen feel like a puzzle constructed with obvious care to make the whole thing easier to grasp.
# 
# * samples in the training set: 2^18 or 262144
# * subcategories: 2^9 or 512
# * features: 2^8 or 256
# 
# What other ways are there to slice and dice the data before fitting a model.
# For any given subcategory, only about 50 features are useful for any of the classifiers explored in the competition.
# 
# It seems like data augmentation by boost leaderboard scores by 1 or 2% by the end, but the bulk of missclassifications will remain even after doubling training time with pseudo-labelling as one example.

# ### Thanks to everyone for reading, especially if you have contributed to our understanding by sharing your own work early and often.
# 

# ### Clues
# This is an anonymized, binary classification dataset found on a USB stick that washed ashore in a bottle. There was no data dictionary with the dataset, but this poem was handwritten on an accompanying scrap of paper:
# 
#     Silly column names abound,
#     but the test set is a mystery.
#     Careful how you pick and slice,
#     or be left behind by history.
# 

# Most significant in the poem to me are the non-rhyming lines,
# 
#     Silly column names abound,
# 
#     Careful how you pick and slice,
#     
#     
# But perhaps there is a time-series element?
# 
#     or be left behind by history.
# 

# In[ ]:


len('Instant Gratification')
# Unfortunately this does not appear to be a cipher key


# In[ ]:




