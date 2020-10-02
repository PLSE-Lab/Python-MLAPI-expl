#!/usr/bin/env python
# coding: utf-8

# ## kernel 1: rule-based solution
# #### Can we identify a prediction rule based on a first look at the data?
# We're going to import some useful tools and load the data. If this step is unfamiliar to you, try going back to [**kernel_0**](https://www.kaggle.com/nicknormandin/cuny-data-challenge-kernel0).

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss


d = pd.read_csv('../input/inspections_train.csv')
x_train0, x_test0 = train_test_split(d, test_size=0.25)


# Now our first step will be to create a feature that we think is important. In fact, creating and manipulating our features (widely referred to as **feature engineering** in data science) is one of the most important aspects of this profession, and one where creativity and subject matter expertise can deliver an enormous amount of value. 
# 
# In the previous kernel, we only loaded the `inspections_train.csv` file, but we saw that there was other data available. Let's load `violations.csv`.

# In[ ]:


violations = pd.read_csv('../input/violations.csv')
violations.head()


# We have a dataframe containing details of violations that occurred in each inspection, and it seems like it would make sense that the number of violations contributes to the probability of a failure. We'll create a dataframe that just counts the number of violations that occured and then merge it with our training / testing data!

# In[ ]:


violation_counts = violations.groupby(['camis', 'inspection_date']).size()
violation_counts = violation_counts.reset_index().set_index(['camis', 'inspection_date'])
violation_counts.columns = ['n_violations']

x_train1 = x_train0.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)
x_test1 = x_test0.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)


# Let's see what this `n_violations` feature looks like by creating a histogram. This is really easy in **pandas**.

# In[ ]:


x_train1.n_violations.hist()
plt.show()


# Based on our intution and an examination of the data, let's create a rule where we assign a pass probability of 0.9 if there were fewer than three violation records and 0.5 otherwise.

# In[ ]:


test_solution1 = ((x_test1.n_violations < 3).map(int) * 0.5) + 0.4
loss1 = log_loss(x_test1.passed.values, test_solution1)
print(f'log loss: {loss1:.3f}')


# With just one rule, we've made a nearly 50% reduction in our score metric (log loss)- that's awesome. This particular rule was intuitive and easy for us to understand, but what about the more subtle things? How do we create many sets of interconnected rules when the data is big and complex? Some important features of our data may not be quite so obvious to us. The next step would be to build models automatically, using **machine learning / statistical learning**.

# ## customize this solution!
# Want to try other values for our decision rule? Try defining your own violation cut-off value and probabilities to see what the log loss would be.
# 
# **Hint:** what about raising the `lower_prob` value above 0.9?

# In[ ]:


# edit these 3 variables
cut_off = 3
lower_prob = 0.9
upper_prob = 0.4


# don't change anything down here
def decision_rule(val):
    if val < cut_off: return lower_prob
    else: return upper_prob

custom_solution = x_test1.n_violations.map(decision_rule)
custom_loss = log_loss(x_test1.passed.values, custom_solution)
print(f'Custom loss: {custom_loss:.3f}')

loss_delta = loss1 - custom_loss
if loss_delta > 0: print(f'Loss improved {loss_delta*100 / loss1:.2f}% !')
elif loss_delta < 0: print('Loss did not improve')


# ### Submitting our solution
# In this kernel we've developed a new way to generate solutions. Now we need to generate solutions for each row in the test data, which we find in inspections_test.csv. The steps are:

# In[ ]:


# load the test data and add the `n_violations` feature
test_data = pd.read_csv('../input/inspections_test.csv')
test_data = test_data.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)



# take just the `id` and `n_violations` columns (since that's all we need)
submission = test_data[['id', 'n_violations']].copy()

# create a `Predicted` column
# for this example, we're using the custom decision rule defined by you above
submission['Predicted'] = submission.n_violations.map(decision_rule)

# drop the n_violations columns
submission = submission.drop('n_violations', axis=1)

# IMPORTANT: Kaggle expects you to name the columns `Id` and `Predicted`, so let's make sure here
submission.columns = ['Id', 'Predicted']

# write the submission to a csv file so that we can submit it after running the kernel
submission.to_csv('submission1.csv', index=False)

# let's take a look at our submission to make sure it's what we want
submission.head()

