#!/usr/bin/env python
# coding: utf-8

# # How to think like a Data Scientist - DonorsChoose EDA
# 

# ## Introduction
# 
# Kaggle's got an amazing community and there are so many great kernels published. But so many of them focus on the technical aspects - how to write a model, how to use a tool. You only see the final kernel, running like a well-architected piece of software.
# 
# But we all know Data Science doesn't work like that. You get a dataset, you poke at it, try things, go back and revisit your assumptions. This kernel is an attempt to document all that - to show what I'm thinking as I run code, to preserve the failed experiments and the sidetracks. I'm not going to keep cells with syntax errors, but other than that - let's start on this journey!

# There's a few standard packages to load. Pandas, numpy, os, sys, pyplot, seaborn. For this I'm also importing some ipython packages. The data's assumed to be downloaded and available in a directory which I define here.

# In[3]:


import pandas as pd, numpy as np

import os,sys

get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display, HTML
import matplotlib.pyplot as plt, seaborn as sns

datadir = '../input/'


# After importing packages, load in all the data. I'm going to assume that it's small enough to fit in memory, and go back and try something else if it doesn't. 
# 
# Based on reading the background at https://www.kaggle.com/c/donorschoose-application-screening/data , I expect that the sample submission, train, and test datasets should have a unique key in the 'id' column, but 'resources' should have multiple rows per id, so I'll set that as the index for those dataframes. 

# In[4]:


resources = pd.read_csv(datadir + 'resources.csv')
sample_submission = pd.read_csv(datadir + 'sample_submission.csv', index_col='id')
test = pd.read_csv(datadir + 'test.csv', index_col='id')
train = pd.read_csv(datadir + 'train.csv', index_col='id')


# There's a warning, which I've seen before and just means I should be careful with the datatypes of the data I loaded. Looks like everything loaded ok. 
# 
# Next I want to go through all the files and understand what's in them. The descriptions at https://www.kaggle.com/c/donorschoose-application-screening/data are useful but it's always good to check for myself. I've got no particular order in mind. I'll start with the sample submission just to get it out of the way.
# 
# Going in, I expect to see the following - unique ID column, equal to the unique ID column of the test set, disjoint from the unique ID column of the train set. I think it should probably be a subset of the IDs in the resources column - but perhaps it's possible to have a proposal with no resources? Would be strange, but perhaps data could be missing. 
# 
# Let's see what we find. 

# # Sample Submission EDA

# In[5]:


print("Sample submission has %d rows. "
      "Validation - this should be the same as length of test set, which has %d." % (
        len(sample_submission), len(test)))
print("Sample submission has same items as test set:" , set(sample_submission.index) == set(test.index))
print("Sample submission has same order as test set:" , all(sample_submission.index == test.index))
print("Sample submission is disjoint from train indices:", len(set(sample_submission.index) & set(train.index)) == 0)
print("Sample submission is subset of resources ids:", len(set(sample_submission.index) - set(resources['id'])) == 0)

display(sample_submission.head())
display(sample_submission.describe())


# There's about 78 thousand test essays. Sample submission is pre-populated with a fixed constant - we'll be able to use the dataframe later to validate the format of our submissions. All my assumptions about the format were valid. 
# 
# Let's move on to resources.

# # Resources EDA
# 
# I really don't know what I'll find there, so I'll start by just looking at the data.

# In[6]:


display(resources.head())
display(resources.tail())


# So, we can immediately see that IDs are repeated multiple times in the resources dataframe - makes sense, proposals can ask for many resources. We'll need to be careful with index/id column of this dataframe, and will want to explore all rows with the same ID together in some form. Prices and quntities shown in numbers, but the text descriptions seem all over the place. 
# 
# A natural value to explore is the total price, both per resource and per proposal. I'll add that in. In addition, I want to look for any unusual descriptions, so I'll put in some obvious features of the description to find unusual ones.

# In[7]:


# First, just check for missing values in columns:
for col in resources.columns:
  print("%s has %d missing values." % (col, resources[col].isnull().sum()))


# No missing values in id, quantity, and price means that obvious numeric features and groups should all work. For simplifying things down the road, I'm going to fill missing descriptions with the empty string instead of a floating-point nan as is the pandas default.

# In[8]:


resources['description'].fillna('', inplace=True)
resources['description'].isnull().sum() # Should be 0 now


# OK, ready to calculate some numeric helper features. 

# In[9]:


# Total cost: quantity x price
resources['total_cost'] = resources['quantity']*resources['price'] 

# After grouping by id, we can sum the price to get the total cost of a proposal and count the items in it.
proposals = pd.DataFrame({
  'summed_cost_in_proposal':resources.groupby('id').sum()['total_cost'],
  'num_items_in_proposal':resources.groupby('id').count()['total_cost']
})

# Length of description in characters, treating it as string.
resources['desc_len'] = resources['description'].str.len()
resources['desc_len_words'] = resources['description'].map(lambda x: len( x.split(' ')))

# Stats for all of the numeric columns so far
display(resources.describe())
display(proposals.describe())


# OK, so I've got a bunch of numbers now. Here's what I can tell based on them. 
# 
# Individual items vary in price greatly - from 0 to 9999. (Those numbers look suspicious - I'll have to look at what had that price later). Number of items requested of a type varies from 1 to 800, but over 50% of the items were requested at a quantity of 1, and 75% were at 2 or less. 
# 
# Most descriptions are pretty short, 11 words is the 75th percentile, but at least one of them is crazy long (192 word description, really??? Come on...) 
# 
# Fortunately, every proposal had at least 1 item in it with nonzero cost, so I don't have to worry about zero-cost proposals. 
# 
# Everything varies over multiple orders of magnitude, so I should look at it on a log scale. I'll look at the distributions and see what crops up. 

# In[10]:


from pandas.api.types import is_numeric_dtype
for df in resources, proposals:
  for col in df.columns:
    values = df[col]
    if is_numeric_dtype(values):
      ax = sns.distplot(np.log1p(values), kde=False, axlabel='log_'+col)
      plt.show(ax)


# On a log scale, most of the distributions have a peaked shape, with some reasonable expected magnitude. The quantities don't, though - num_items_in_proposal and quantity both have the lowest value as the most common value. 
# 
# Next, I'll look at what the outliers are like. Are they real data or is there junk here? The extremes of the distributions are all pretty likely to lead us to find data that would be problematic. I'll start with looking at the resources requested in large quantities. 

# In[11]:


display(resources[ resources['quantity'] > 100])
display(resources[ resources['quantity'] > 500])


# We can immediately see some obvious patterns - books, notebooks, binders, paper, pencils, some electronic supplies are often ordered in bulk. Let's look next at the shortest/longest descriptions and lowest/highest prices.

# In[29]:


for col in ['price','total_cost','desc_len_words', 'desc_len']:
  display( resources.sort_values(by=col).head(10))
  display( resources.sort_values(by=col).tail(10))


# Surprisingly, I see some prices listed as zero, for the lowest-price items. They have poor descriptions, as well.  Perhaps they make sense in the context of the full order - some thing like "Standard Shipping" isn't really an item, more of a modifier. Some of the other zero-price things look like descriptors as well - "BLACK (PP1) - CLASSROOM SELECT" looks like the color of something else in the order.
# 
# The most expensive individual items are all the same google exploration kit. Interesting to note that for that one, the quantity isn't really accurate - it's a 30-student kit. Number of licenses could also be something reported either in the quantity or in the description. The most expensive total cost items look to include electronics, laptops, that kind of thing. 
# 
# The shortest and longest descriptions all seem to include books, which may either have a one-word title or a long detailed description. Googling, 'stargirl' and 'woolbur' definitely are; 'wonder' is a movie. Some short descriptions are basically completely missing. 
# 
# At this point, I've got a reasonable idea of what kind of information is in the resources file. I'll eventually consider finding a way to categorize the descriptions in a reasonable way, to group books together and laptops together and things like that. It may help to detect invalid descriptions and prices. 
# 
# I'll wrap up by looking at the most expensive and least expensive total packages.

# In[16]:


cheapest_total_id = proposals['summed_cost_in_proposal'].idxmin()
most_expensive_total_id = proposals['summed_cost_in_proposal'].idxmax()
biggest_proposal_id = proposals['num_items_in_proposal'].idxmax()

display(resources[resources['id']==cheapest_total_id])
display(resources[resources['id']==most_expensive_total_id])
display(resources[resources['id']==biggest_proposal_id])


# Fortunately, looks like the zero-price items don't combine to create a fake zero-price combination of resources. The cheapest proposal asked for a hundred bucks of exercise equipment, and the most expensive for chromebooks and a course license for 71. 
# 
# The biggest proposal by number of items (100) looks like a collection of books. Clearly, there's going to be some grouping of these resources that needs to be done, since these 100 books should be treated pretty similarly to that Junie Jones 25-book set that was listed as a single line item earlier. 
# 
# At this point, I've got a pretty good idea of how I want to approach this dataset. What I'll want to do later is categorize each description into groups - 'book', 'electronics', 'supplies', and perhaps extract quantities of them. Then be able to, for each group and for each proposal, get a total cost and total quantity - so with each ID, I'll have one row of data, with two columns (price, quantity) for each type of item. 

# # Train and test EDA
# 
# What do I expect to get from EDA of train/test data? 
# 
# Biggest thing I want to find out is how similar test data is to train data, in distribution. I want to see if there are outliers or invalid data in either. I know that essays 3 and 4 are going to be missing some of the time - how often does that happen in train and test? 

# In[17]:


display(train.head())
display(test.head())


# In[18]:


print("%d train examples, %d test examples" % (len(train), len(test)))


# In[19]:


print("How many examples in train have essays 3 and 4?")
display(train['project_essay_3'].isnull().value_counts())
display(train['project_essay_4'].isnull().value_counts())
print("How many examples in test have essays 3 and 4?")
display(test['project_essay_3'].isnull().value_counts())
display(test['project_essay_4'].isnull().value_counts())


# Looks like a vast majority of both train and test are missing essays 3 and 4 - this isn't a marginal few, it's most of them. Unsurprisingly, the numbers are the same for both essays 3 and 4. Good. 
# 
# From here on out, I'll do pretty standard things for each variable in the training dataset. Look at their distributions, and look at the correlation with the label variable.  Look at the distribution of the label. 

# In[20]:


train['project_is_approved'].value_counts()


# Interesting - looks like most projects are approved! Our task then is just to find markers of proposals that are so poor they fall out of that top 85%. 
# 
# Next step is EDA on the rest of the categorical variables. Want to find their distributions, and how related they are to the label. 

# In[21]:


categorical_variables = ['teacher_prefix','school_state','project_grade_category', 'project_subject_categories','project_subject_subcategories']
for var in categorical_variables:
  print("Exploring", var)
  print("There are %d categories in this column in training dataset:" % train[var].nunique() )
  print("There are %d categories in this column in test dataset:" % test[var].nunique() )
  
  counts = train[var].fillna("MISSING").value_counts()
  freqs =  counts/counts.sum()
  frac_accepted = train.fillna("MISSING").groupby(var).mean()['project_is_approved']
  result = pd.DataFrame({'count':counts, 'frequency':freqs, 'acceptance_rate':frac_accepted, 'value':counts.index},
                        index=counts.index)
  
  print("Value counts in train:" )
  display(result) # Show the values
  
  # plot the frequency
  result.plot.bar(x='value',y='frequency')
  plt.xlabel(var)
  plt.ylabel('frequency')
  plt.show()
  plt.close()
  
  # plot acceptance rate
  result.plot.bar(x='value',y='acceptance_rate')
  plt.xlabel(var)
  plt.ylabel('acceptance rate')
  plt.show()
  plt.close()
  
  print("Value counts in test:")
  counts = test[var].fillna("MISSING").value_counts()
  freqs =  counts/counts.sum()
  result = pd.DataFrame({'count':counts, 'frequency':freqs,'value':counts.index})
  display(result)
  
  result.plot.bar(x='value',y='frequency')
  plt.xlabel(var)
  plt.ylabel('frequency')
  plt.show()
  plt.close()
    


# 
# teacher_prefix seems not to be related to acceptance rate much at all. I suppose it's possible to generate a "gender" variable grouping together Mrs/Ms, or treating 'teacher' like a missing value, but given that the acceptance rate is pretty similar for all the prefixes it's probably not too useful, at least not for major gains in accuracy. Distribution in test looks similar to train. 
# 
# All the states are represented in school_state, and there's even a pretty good number of each, with 139 being the lowest. Acceptance rates seem to vary from the low 80s to the high 80s, 89 for DE to 81 for DC.  Unclear whether this is just variation due to small N or whether it's predictive - we'll have to see in cross-validation. Distribution in test seems similar to train. 
# 
# The grade categories are arranged in order, and should perhaps be treated as numeric (1-4) for modeling. All have pretty similar acceptance rates and same distribution in train/test.
# 
# Project subject categories - there's 51 of them (just like states). Some of them appear to have overlap - Literacy&Language, Math&Science are both individual and combined, in either order! There seems to be more variation across categories than across states - I can see an 80.9% acceptance rate for "special needs", and a 92% acceptance rate for "warmth, care, and hunger". Huh, that makes a bit of sense. Who would reject a Warmth, Care, and Hunger proposal? But a 78% acceptance rate for "Math & Science, Heath & Sports"  when there's an 84% acceptance rate for "Health & Sports, Math & Science"? That seems strange. One way to treat this feature is to create dummy variables for each category, but allow proposals to have multiple entries - so as to avoid treating "Health & Sports, Math & Science" as different than the reverse. 
# 
# Obviously, as I get down the frequency list to smaller and smaller numbers, the acceptance rates start fluctuating wildly. As before train and test distributions seem the same. 
# 
# Project subcategories looks the same at the top, but has far too many tiny categories with 1 entry to be useful as is. 
# 
# Last thing to look at here - teacher number of previously posted projects.

# In[22]:


display(train['teacher_number_of_previously_posted_projects'].describe())
print("Correlation:", train[['teacher_number_of_previously_posted_projects', 'project_is_approved']].corr().iloc[0]['project_is_approved'])


# # Validating cross-validation

# Are we there yet? Can I be done with EDA, import Keras and xgboost and go at it?
# 
# There's one more piece that I count under exploratory, and that's setting up the validation. I want to run a basic model and verify that the accuracy that I see in this kernel is the same as I get when I submit to the leaderboard. This is especially critical in a dataset like this, since the "resources" file has both train and test info and is a prime way that information might leak. 

# In[23]:


# First, get the folds
from sklearn.model_selection import KFold
num_folds = 5

all_indices = pd.Series(train.index)
kf = KFold(n_splits = num_folds, shuffle=True, random_state = 12345)
fold_indices = [
                [all_indices[x] for x in train_test]
                for train_test in kf.split(all_indices)
                ]
# fold_indices is a list of pairs, train and test, to be used for cross_validation


# In[24]:


# next, put making features into a function - that can be called on both train/test, and train/validate data.
def featurize_train_test(train, test, resources):
  """
  This function takes as input a training dataframe, a test dataframe, and the resources dataframe.
  It returns a train dataframe, test dataframe, and targets, ready for input into an ML model.
  
  For the EDA, I'm only going to pick out a few simple features and run logistic regression. 
  So the features I'm using are the number of previously posted projects, the grade, total cost, number of items.
  I'm deliberately using a lot of things from the resources file to ensure that I catch any information leaks early.
  """
  train_labels = train['project_is_approved'].values
  
  # Set this up to work on both train and test data, for validation and real use
  if 'project_is_approved' in test.columns:
    test_labels = test['project_is_approved']
  else:
    test_labels = pd.Series(index=test.index, data=np.nan)
  
  train_features = pd.DataFrame(index=train.index)
  test_features = pd.DataFrame(index=test.index)
  
  train_features['prev_proj'] = train['teacher_number_of_previously_posted_projects']
  test_features['prev_proj'] = test['teacher_number_of_previously_posted_projects']
  
  def grade_to_int(x):
    return {'Grades PreK-2':0, "Grades 3-5": 1, "Grades 6-8": 2, "Grades 9-12":3}[x]
  
  train_features['grade_int'] = train['project_grade_category'].map(grade_to_int)
  test_features['grade_int'] = test['project_grade_category'].map(grade_to_int)
  
  resources['total_cost'] = resources['quantity']*resources['price']
  proposals = pd.DataFrame({
    'summed_cost_in_proposal':resources.groupby('id').sum()['total_cost'],
    'num_items_in_proposal':resources.groupby('id').count()['total_cost']
  })
  
  train_features['total_cost'] = [proposals.loc[i, 'summed_cost_in_proposal'] for i in train_features.index]
  test_features['total_cost'] = [proposals.loc[i, 'summed_cost_in_proposal'] for i in test_features.index]
  train_features['num_items'] = [proposals.loc[i, 'num_items_in_proposal'] for i in train_features.index]
  test_features['num_items'] = [proposals.loc[i, 'num_items_in_proposal'] for i in test_features.index]
  
  return (train_features.values, train_labels, test_features.values, test_labels)
  
# Make a function that makes a basic model
from sklearn.linear_model import LogisticRegression
def train_test_basic_model(train_features, train_labels, test_features):
  """
  This function should train the model on the training data, 
  test it on the test data, and return the predicted probabilities.
  """
  model = LogisticRegression()
  model.fit(train_features, train_labels)
  return model.predict_proba(test_features)[:,1]

from sklearn.metrics import roc_auc_score
def evaluate(true_labels, predicted_probs):
  """
  Possibly the most important function you'll write - evaluation. 
  Takes predicted probabilities and true labels, and gets the AUC.
  See https://www.kaggle.com/c/donorschoose-application-screening#evaluation . 
  
  Here, our evaluation function is trivial and passed on to sklearn.
  """
  return roc_auc_score(true_labels, predicted_probs)


# And now that we've prepared a featurization, model training, and evaluation function, as well as folds, the evaluation is trivial. 

# In[25]:


aucs = []
for train_ind, test_ind in fold_indices:
  tr, tr_l, te, te_l = featurize_train_test(train.loc[train_ind], train.loc[test_ind], resources)
  predicted = train_test_basic_model(tr, tr_l, te)
  aucs.append(evaluate(te_l, predicted))

print("The crossvalidated AUC is", np.mean(aucs))


# Excellent. There was enough information in the features to push the AUC above 0.5 . This is of course a horrid model, but it gives us a chance to test the cross-validation - next I'll make a submission and see whether it also gets the same 0.65 AUC. 

# In[26]:


tr, tr_l, te, te_l = featurize_train_test(train, test, resources)
predicted = train_test_basic_model(tr, tr_l, te)

new_submission = sample_submission.copy()
new_submission['project_is_approved'] = predicted

new_submission.to_csv("cv_test_lr_predictions.csv")


# And we're done! Looks like like the leaderboard value is as expected, exactly 0.65. 
# 
# So now we're ready to go with the fun stuff! Get those GPUs warmed up, and on to the next kernel! 
