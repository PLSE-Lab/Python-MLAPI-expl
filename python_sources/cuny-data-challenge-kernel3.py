#!/usr/bin/env python
# coding: utf-8

# ## kernel 3: feature engineering and the hazards of overfitting
# #### Is increasing the power of our model always a good thing?
# 
# 
# 
# As a first step, we're going to import some useful tools and load the data. If this step is unfamiliar to you, try going back to [**kernel_0**](https://www.kaggle.com/nicknormandin/cuny-data-challenge-kernel0).

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss


d = pd.read_csv('../input/inspections_train.csv', parse_dates=['inspection_date'])
x_train0, x_test0 = train_test_split(d, test_size=0.25)


# In addition to those tools, we're also going to import **the Random Forest machine learing model**, which is very popular in the data science community (view the documentation [**here**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)). 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# We'll create some additional features and train a more powerful model. First let's merge the inspections with the data about each venue, and also add the violation counts that we previously calculated.

# In[ ]:


# import the venue_stats file
venue_stats = pd.read_csv('../input/venues.csv').set_index('camis')
venue_stats.head()


# In[ ]:


# import the violations file
violations = pd.read_csv('../input/violations.csv', parse_dates=['inspection_date'])

x_train3 = x_train0.merge(venue_stats, 'left', left_on='camis', right_index=True)
x_test3 = x_test0.merge(venue_stats, 'left', left_on='camis', right_index=True)

violation_counts = violations.groupby(['camis', 'inspection_date']).size()
violation_counts = violation_counts.reset_index().set_index(['camis', 'inspection_date'])
violation_counts.columns = ['n_violations']

x_train3 = x_train3.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)
x_test3 = x_test3.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)


# Maybe it would be helpful to include a binary variable that tells us whether the inspection we're looking at is an initial or re-inspection. We can add those easily here.

# In[ ]:


x_train3['re_inspect'] = x_train3.inspection_type.str.contains('re-', regex=False, case=False).map(int)
x_train3['initial_inspect'] = x_train3.inspection_type.str.contains('initial', regex=False, case=False).map(int)

x_test3['re_inspect'] = x_test3.inspection_type.str.contains('re-', regex=False, case=False).map(int)
x_test3['initial_inspect'] = x_test3.inspection_type.str.contains('initial', regex=False, case=False).map(int)


# We might also be interested in knowing which borough the inspection takes place in. From a quick look at the table, I've lumped the six possible options into three categories that roughly correspond to the pass frequency that we see in the training data. Note that this is a way to **encode** categorical features- but not necessarily the smartest way.

# In[ ]:


boro_dict = {
    'Missing': 0,
    'STATEN ISLAND': 0,
    'BROOKLYN': 1,
    'MANHATTAN': 1,
    'BRONX': 2,
    'QUEENS': 2    
}

x_train3['boro_idx'] = [boro_dict[_] for _ in x_train3.boro]
x_test3['boro_idx'] = [boro_dict[_] for _ in x_test3.boro]


# Maybe the month of the inspection has some impact on pass frequency? Let's add a variable encoding the month of the year (again, this isn't necessarily the optimal way to encode this into the feature set).

# In[ ]:


x_train3['inspection_month'] = (x_train3.inspection_date.dt.strftime('%m').map(int) + 6) % 12
x_test3['inspection_month'] = (x_test3.inspection_date.dt.strftime('%m').map(int) + 6) % 12


# Finally, we'll encode the cuisine description as a numeric variable based on the pass frequency we see corresponding to that cuisine in the training data.

# In[ ]:


cuisine_hitrates = x_train3.groupby(['cuisine_description']).agg({'passed':'mean', 'id':'count'}).        rename(columns={'id':'ct'}).sort_values('passed')[['passed']]
cuisine_hitrates.columns = ['cuisine_hr']

    
x_train3 = x_train3.merge(cuisine_hitrates, 'left', left_on='cuisine_description', right_index=True).fillna(0.67)
x_test3 = x_test3.merge(cuisine_hitrates, 'left', left_on='cuisine_description', right_index=True).fillna(0.67)


# We'll make a list of all of the features we've created so that we can pass them to the model for training.

# In[ ]:


model_features = ['n_violations', 'inspection_month', 'cuisine_hr', 'boro_idx', 're_inspect', 'initial_inspect']


# We'll use the extremely popular and flexible Random Forest model to generate our predictions. Here we use the default settings of the model to train and generate predictions.

# In[ ]:


clf0 = RandomForestClassifier(n_estimators=50)
clf0.fit(x_train3[model_features], x_train3.passed)
test_solution3 = clf0.predict_proba(x_test3[model_features])
loss3a = log_loss(x_test3.passed.values, test_solution3)
print(f'log loss: {loss3a:.3f}')


# Surprisingly, our results are terrible. We've actually done worse than just guessing $0.67$ for every answer. This is the result of overfitting, and is an extremely common issue in machine learning and data science tasks. How do we fix it? We **regularize** the model. In this case, I'll set a maximum depth parameter and a minimum samples per leaf parameter, which prevents the model from getting too complicated. Hopefully this means that it will have predictions that **generalize** to the test data better, rather than overfitting our training data.

# In[ ]:


clf1 = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=10)
clf1.fit(x_train3[model_features], x_train3.passed)
test_solution4 = clf1.predict_proba(x_test3[model_features])
loss3b = log_loss(x_test3.passed.values, test_solution4)
print(f'log loss: {loss3b:.3f}')


# That's a significant improvement! By make our model a bit less powerful, we beat our best score. 

# Want to try your own parameters? Sometimes it's helpful to store them as a dictionary to keep things organized. Try tweaking these values and training the model again!

# In[ ]:


# change these parameters!
parameters = {
    'n_estimators': 50,
    'max_depth': 10,
    'min_samples_leaf': 10
}

# we use the ** operator to expand the parameters dictionary
clf_custom = RandomForestClassifier(**parameters)
clf_custom.fit(x_train3[model_features], x_train3.passed)
test_solution_cusotm = clf_custom.predict_proba(x_test3[model_features])
loss3_custom = log_loss(x_test3.passed.values, test_solution_cusotm)
print(f'log loss: {loss3_custom:.3f}')


# ### Submitting our solution
# In this kernel we've developed a new way to generate solutions. Now we need to generate solutions for each row in the test data, which we find in inspections_test.csv. The steps are:

# In[ ]:


# load the test data
test_data = pd.read_csv('../input/inspections_test.csv', parse_dates=['inspection_date'])

# replicate all of our feature engineering for the test data
test_data = test_data.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)
test_data = test_data.merge(venue_stats, 'left', left_on='camis', right_index=True)
test_data['re_inspect'] = test_data.inspection_type.str.contains('re-', regex=False, case=False).map(int)
test_data['initial_inspect'] = test_data.inspection_type.str.contains('initial', regex=False, case=False).map(int)
test_data['boro_idx'] = [boro_dict[_] for _ in test_data.boro]
test_data['inspection_month'] = (test_data.inspection_date.dt.strftime('%m').map(int) + 6) % 12
test_data = test_data.merge(cuisine_hitrates, 'left', left_on='cuisine_description', right_index=True).fillna(0.67)


# create a `Predicted` column
# for this example, we're using the model we previously trained
test_data['Predicted'] = [_[1] for _ in clf_custom.predict_proba(test_data[model_features])]

# take just the `id` and `n_violations` columns (since that's all we need)
submission = test_data[['id', 'Predicted']].copy()

# IMPORTANT: Kaggle expects you to name the columns `Id` and `Predicted`, so let's make sure here
submission.columns = ['Id', 'Predicted']

# write the submission to a csv file so that we can submit it after running the kernel
submission.to_csv('submission3.csv', index=False)

# let's take a look at our submission to make sure it's what we want
submission.head()


# In[ ]:




