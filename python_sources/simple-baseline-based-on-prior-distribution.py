#!/usr/bin/env python
# coding: utf-8

# # Simple baseline without leaks 
# [updated with test stage 2 results]
# 
# This is a simple baseline using only the prior distribution of 'A', 'B' and 'Neither' in the GAP test set. Instead of assuming that the three classes are evenly distrubuted (i.e., 1/3,1/3,1/3), we can check the prior probability of the three classes in gap_test which we will us as training set. We can then use the validation set to get a realistic idea how this baseline would perform on the test set.
# 
# In addtion, we also perform an analysis of the difference in performance of female and male pronouns. We find that the data set with only male pronouns has a lower log loss than the data set wth only female pronouns, as discussed in [Mind the GAP: A Balanced Corpus of Gendered Ambiguous Pronouns](https://arxiv.org/abs/1810.05201)

# In[39]:


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


# In[40]:


test_stage_1 = pd.read_csv("../input/test_stage_1.tsv", sep="\t")
test_stage_2 = pd.read_csv("../input/test_stage_2.tsv", sep="\t")


# In[41]:


gap_test = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv", delimiter='\t')
gap_valid = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv", delimiter='\t')


# In[42]:


gap_test[0:5]


# In[43]:


def get_prior(df):
    # count how many times neither antecedent is correct for the pronoun
    Neither_count = len(df) - sum(df["A-coref"]  |  df["B-coref"])
    # count the  A coreferences
    A_count = sum(df["A-coref"])
    # count the B coreferences
    B_count = sum(df["B-coref"])
    # total number of samples
    test_total = len(df)
    # compute the prior probabilities of the three classes
    Neither_prior = Neither_count/test_total
    A_prior = A_count/test_total
    B_prior = B_count/test_total
    print("Prior probabilities:")
    print("Neither: "+str(Neither_prior),"A: "+str(A_prior),"B: "+str(B_prior))
    # sanity check whether everything adds up
    assert Neither_count + A_count + B_count == test_total
    return A_prior, B_prior, Neither_prior

A_prior,B_prior,Neither_prior = get_prior(gap_test)


# In[44]:


sample_submission = pd.read_csv("../input/sample_submission_stage_1.csv")


# In[45]:


def assign_prior(df):
    sub = pd.DataFrame()
    for index, row in df.iterrows():
        sub.loc[index, "ID"] = row["ID"]
        sub.loc[index, "A"] = A_prior
        sub.loc[index, "B"] = B_prior
        sub.loc[index, "NEITHER"] = Neither_prior
    return sub


# In[46]:


train = assign_prior(gap_test)
valid = assign_prior(gap_valid)


# # Evaluation

# In[47]:


from sklearn.metrics import log_loss

def get_gold(df):
    gold = []
    for index, row in df.iterrows():
        if (row["A-coref"]):
            gold.append("A") 
        else:
            if (row["B-coref"]):
                gold.append("B") 
            else:
                gold.append("NEITHER")
    return gold


# In[48]:


train_gold = get_gold(gap_test)
valid_gold = get_gold(gap_valid)


# In[49]:


train_pred = train[["A","B","NEITHER"]]
log_loss(train_gold,train_pred)


# This is the log loss for the training set (i.e., gap_test). Let's now check how the same distribution would work on the validation data set (i.e. gap_valid)

# In[50]:


valid_pred = valid[["A","B","NEITHER"]]
log_loss(valid_gold,valid_pred)


# Not suprinsingly, the log loss is slightly higher for the validation set. Let's now create our submission based on the test_stage_1 data set (i.e., gap_train):
# 

# In[51]:


sub1 = assign_prior(test_stage_1)


# In[52]:


sub1[0:4]


# In[53]:


sub1.to_csv("submission_1.csv", index=False)


# When you submit this file to the leaderboard you will get a score of 0.95201. Even better than on the train and validation data set.
# That's great, but note that we just got lucky. We don't know what the distribution of the three classes will be for the stage 2 test set. It is probably similar but not exactly the same one we've seen for the other three data sets.

# More importantly, the organizers have already annouced that the distribution between male and female pronouns will be different.
# Currently, those two types of pronouns are evenly distributed for all three data sets we have access to.

# In[54]:


set(gap_test["Pronoun"]).union(set(gap_valid["Pronoun"])).union(set(test_stage_1["Pronoun"]))


# In[55]:


female_pronouns = ['she','her','hers']
male_pronouns = ['he','him','his']


# In[56]:


female_gap_test = gap_test[gap_test["Pronoun"].str.lower().isin(female_pronouns)]
male_gap_test = gap_test[gap_test["Pronoun"].str.lower().isin(male_pronouns)]
female_gap_valid = gap_valid[gap_valid["Pronoun"].str.lower().isin(female_pronouns)]
male_gap_valid = gap_valid[gap_valid["Pronoun"].str.lower().isin(male_pronouns)]


# In[57]:


len(female_gap_test) == len(male_gap_test)


# In[58]:


len(female_gap_valid) == len(male_gap_valid)


# In[59]:


train_female = assign_prior(female_gap_test)
train_male = assign_prior(male_gap_test)
valid_female = assign_prior(female_gap_valid)
valid_male = assign_prior(male_gap_valid)


# In[60]:


train_gold_female = get_gold(female_gap_test)
train_gold_male = get_gold(male_gap_test)


# In[61]:


train_pred_female = train_female[["A","B","NEITHER"]]
log_loss(train_gold_female,train_pred_female)


# In[62]:


train_pred_male = train_male[["A","B","NEITHER"]]
log_loss(train_gold_male,train_pred_male)


# As the authors in [Mind the GAP: A Balanced Corpus of Gendered Ambiguous Pronouns](https://arxiv.org/abs/1810.05201) describe, the performance for resolving female pronouns is lower than for male pronouns.
# 
# Interestingly enough, this difference is reversed for the validation data set. 

# In[63]:


valid_gold_female = get_gold(female_gap_valid)
valid_gold_male = get_gold(male_gap_valid)


# In[64]:


valid_pred_female = valid_female[["A","B","NEITHER"]]
log_loss(valid_gold_female,valid_pred_female)


# In[65]:


valid_pred_male = valid_male[["A","B","NEITHER"]]
log_loss(valid_gold_male,valid_pred_male)


# # Test Stage 2 results

# In[66]:


female_test_stage_2 = test_stage_2[test_stage_2["Pronoun"].str.lower().isin(female_pronouns)]
male_test_stage_2 = test_stage_2[test_stage_2["Pronoun"].str.lower().isin(male_pronouns)]


# In[67]:


len(female_test_stage_2)


# In[68]:


len(male_test_stage_2)


# The final test stage 2 has only a slightly different ration between male and female pronouns. 

# In[69]:


sub2 = assign_prior(test_stage_2)


# In[70]:


sub2.head()


# In[71]:


sub2.to_csv("submission.csv", index=False)


# This submission will produce a loss of 0.94712 for the test stage 2 data set.

# In[ ]:




