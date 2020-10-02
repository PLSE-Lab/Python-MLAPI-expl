#!/usr/bin/env python
# coding: utf-8

# ## kernel 2: basic model
# #### Can we automate our ability to find rules?
# We had some really good luck with generating an intuitive feature from the data and creating a rule based on our observations. Let's use that same feature to create a model. 
# <br><br>
# As a first step, we're going to import some useful tools and load the data. If this step is unfamiliar to you, try going back to [**kernel_0**](https://www.kaggle.com/nicknormandin/cuny-data-challenge-kernel0).

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss


d = pd.read_csv('../input/inspections_train.csv')
x_train0, x_test0 = train_test_split(d, test_size=0.25)


# In addition to those tools, we're also going to import **our first machine learning model**, which is the `LogisticRegression` classifier (view the documentation [**here**[](http://)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). If you've ever taken a statistics or modeling course, you may recognize the name. 

# In[ ]:


from sklearn.linear_model import LogisticRegression


# Just like in [**kernel_1**](https://www.kaggle.com/nicknormandin/cuny-data-challenge-kernel1), we're going to create our `n_violations` feature.

# In[ ]:


violations = pd.read_csv('../input/violations.csv')
violation_counts = violations.groupby(['camis', 'inspection_date']).size()
violation_counts = violation_counts.reset_index().set_index(['camis', 'inspection_date'])
violation_counts.columns = ['n_violations']

x_train2 = x_train0.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)
x_test2 = x_test0.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)


# We can create and train the model in two lines of python code.

# In[ ]:


# create the model named 'classifier'
classifier = LogisticRegression(solver='lbfgs')

# train the model to predict 'passed' based on 'n_violatoins'
classifier.fit(x_train2[['n_violations']], x_train2['passed'])


# The output that you see here lets us know the settings of our model (which we've named `classifier`). We specified the type of solver (`lbfgs`) but otherwise we allowed the default settings to be picked. You should review the `LogisticRegression` documentation to learn about all of the interesting changes you can make.

# Now that we have a trained model, we can make predictions on new data. For a classification model like `LogisticRegression`, we can either use the `predict()` method to predict the *label* of `passed` (ie: a 0 or 1 value), or we can use the `predict_proba()` method to predict a *probability* of `passed`, which is a value between 0 and 1. Let's see the difference:

# In[ ]:


sample_example = [2]

# predict method just gives a 0 or 1
print(classifier.predict([sample_example])[0])

# predict_proba method gives us the probability between 0 and 1
print(classifier.predict_proba([sample_example])[0][1])


# Let's see how this solution compares to our previous best from [**kernel_1**](https://www.kaggle.com/nicknormandin/cuny-data-challenge-kernel1) (0.447).

# In[ ]:


test_solution2 = classifier.predict_proba(x_test2[['n_violations']])
loss2 = log_loss(x_test2.passed.values, test_solution2)
print(f'log loss: {loss2:.3f}')


# That's a huge reduction in loss! In **kernel_1**, we decided to predict 90% likelihood of passing when there were fewer than 3 violations and 40% chance when there were 3 or more. Our logistic regression model has 'learned' to assign a probability for every value. Let's see if we can plot it.

# In[ ]:


n_violations = np.linspace(0, 15, 16)
plt.plot(n_violations, [_[1] for _ in classifier.predict_proba(n_violations.reshape(-1, 1))])
plt.xlabel('number of violations'); plt.ylabel('predicted probability of pass')
plt.show()


# ### Submitting our solution
# In this kernel we've developed a new way to generate solutions. Now we need to generate solutions for each row in the test data, which we find in inspections_test.csv. The steps are:

# In[ ]:


# load the test data and add the `n_violations` feature
test_data = pd.read_csv('../input/inspections_test.csv')
test_data = test_data.merge(violation_counts, 'left', left_on=['camis', 'inspection_date'], right_index=True)

# take just the `id` and `n_violations` columns (since that's all we need)
submission = test_data[['id', 'n_violations']].copy()

# create a `Predicted` column
# for this example, we're using the model we previously trained
submission['Predicted'] = [_[1] for _ in classifier.predict_proba(submission.n_violations.values.reshape(-1, 1))]

# drop the n_violations columns
submission = submission.drop('n_violations', axis=1)

# IMPORTANT: Kaggle expects you to name the columns `Id` and `Predicted`, so let's make sure here
submission.columns = ['Id', 'Predicted']

# write the submission to a csv file so that we can submit it after running the kernel
submission.to_csv('submission2.csv', index=False)

# let's take a look at our submission to make sure it's what we want
submission.head()

