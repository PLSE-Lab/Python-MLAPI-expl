#!/usr/bin/env python
# coding: utf-8

# # Getting 100% Accuraccy without Machine Learning Model
# 
# This is the solution kernel in which I use no machine learning model to achieve 100% accuracy.
# 
# Let's get started then!
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = [9, 12]

import warnings
warnings.simplefilter('ignore')


# ### Importing Data

# In[ ]:


train = pd.read_csv("/kaggle/input/whoisafriend/train.csv")
test = pd.read_csv("/kaggle/input/whoisafriend/test.csv")
sub = pd.read_csv("/kaggle/input/whoisafriend/sample_submission.csv")

train.shape, test.shape, sub.shape


# ### Secret Sauce : 
# 
# - Hypothesis : Does it matter how many times two persons meet in thier being friends? 
# 
# Realistically it does though, like if you meet a guy/girl for many times than others, there your friends right? Or not? 
# 
# Lets check whether this hypothesis works in our data or not.

# So basically we have to get the count of the interactions two persons have got, so let's aggregate the train and test datasets seperatly as they have totally different sets of persons.
# 
# Now to get count we've got to aggregate (similar to groupby in SQL) according to "Person A" and "Person B" on function *count* to get interaction count between two persons.

# In[ ]:


agg_train = train.groupby(['Person A', 'Person B'])['Years of Knowing'].count().reset_index()
agg_train.rename({
    "Years of Knowing": "Interaction Count"
}, axis=1, inplace=True)

agg_test = test.groupby(['Person A', 'Person B'])['Years of Knowing'].count().reset_index()
agg_test.rename({
    "Years of Knowing": "Interaction Count"
}, axis=1, inplace=True)


# In[ ]:


agg_train.head()


# ### Now merging this data into train and test sets to check whether this *interaction count* has any realtionship with being Friends.

# In[ ]:


train = pd.merge(train, agg_train, on=['Person A', 'Person B'], how='left')
test = pd.merge(test, agg_test, on=['Person A', 'Person B'], how='left')


# ### Distribution of *Interaction Count*

# In[ ]:


sns.lmplot('Interaction Count', 'Friends', data=train, fit_reg=False)


# In[ ]:


plt.figure(figsize=(12, 5))
sns.countplot('Interaction Count', data=train, hue='Friends')
plt.show()


# ### Guess that' it then. From the plot we can see that if Interaction Count >= 8 then they're Friends, else they're not.
# 
# So our hypothesis is right on point! That's why it is important to make hypothesis's always as they make good features if prove correct, and even might win you the competition :')

# ### Making the Submission

# In[ ]:


test['Friends'] = np.nan
test['Friends'] = test['Interaction Count'].apply(lambda x: 1 if x > 7 else 0)


# In[ ]:


test[['ID', 'Friends']].to_csv("1.0_sub.csv", index=False)


# ## Conclusion 
# 
# #### 1. The feature "Interaction Count" which I made from aggregated data is known as *Aggregated Features* and they're quite useful in many scenarios. Hence it's always a god decision for you to think for hypothesis with out checking the data first, i.e your first impressions of the problem statement. In our case : How do you make friends? What could be the factors, etc.
# 
# #### 2. A better Feature Engineering always put-performs a Perfectly Fine-Tuned Model (or most of the times) .
# 
# #### 3. Most of the machine learning models can not get the temporal effect in the data(RNN and LSTM can), as most of the aggregated features are used to represent the temporal effect of a set, in this case how many times did two persons meet each other? So, for our model to utilize the goodness of temporal relation we have to create features which can represent the same. 
# 
# 4. Some more aggregated features can be : 
#     
#     - avg_time_of_interaction
#     - where_did_they_meet_the_most
#     - avg_of_years_known
#     
#     As we can see most of these *aggregated features* describe the temporal relation that the model may not perceive. For example : to get an average the model has to go back to past and aggregate all the records and get the mean, and that *going back in past* is not possible for most of the models.
#     
# 
# 
# ### Please upvote if you liked the notebook!

# In[ ]:




