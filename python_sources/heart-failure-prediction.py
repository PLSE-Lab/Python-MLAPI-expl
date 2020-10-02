#!/usr/bin/env python
# coding: utf-8

# We are going to need the following packages:

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.api as sm
import seaborn as sns
sns.set()

import os


# Let's have a look at the data:

# In[ ]:


data_filepath = "../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(data_filepath)
data.head()


# For getting a better idea about the range and spread of the data, let's make a simple scatterplot with "ejection_fraction" and "serum_creatinine" as axes.
# 

# In[ ]:


x = data[["ejection_fraction", "serum_creatinine"]]
x.describe()


# In[ ]:


plt.scatter(data["ejection_fraction"], data["serum_creatinine"])
plt.xlabel("Percentage of ejection")
plt.ylabel("Serum Creatinine level")
plt.show()


# Since the objective of the task is to predict if a patient dies due to a heart attack, let's create a model using Logistic Regression which gives probability of an event.
# 
# For the first regression we will use the column "DEATH_EVENT" as dependent variable and all the
# rest columns, except "time", as independent variables.

# In[ ]:


x1 = data.copy()
x1 = x1.drop(columns=["time", "DEATH_EVENT"])
x1.head()


# Since the ranges of values in different columns differ greatly, let's standardize them

# In[ ]:


x_scaled = preprocessing.scale(x1)
x_scaled


# And now let's create a logistic regression model using scaled input

# In[ ]:


x = sm.add_constant(x_scaled)
y = data['DEATH_EVENT']

reg_log = sm.Logit(y,x)
results_log1 = reg_log.fit()


# We managed to get a model that fit our data. Let's check the table below describing the result.
# As we can see, the only significant variables, that is, those with P < 0,05, are x1, x5, x8 (let's ignore x3 with its P=0,042 as well).

# In[ ]:


results_log1.summary()


# Let's update our model and drop all variables except x1 ("age"), x3 ("ejection fraction) and x8 ("serum creatinine").

# In[ ]:


x2 = data.copy()
x2 = x2[["age", "ejection_fraction", "serum_creatinine"]]
x_scaled = preprocessing.scale(x2)
x = sm.add_constant(x_scaled)


# In[ ]:


reg_log = sm.Logit(y,x)
results_log2 = reg_log.fit()


# As we can see, we got a model that still fits our data although its optimization function value is slightly greater (0.51 vs 0.49). Let's look at the summary table:

# In[ ]:


results_log2.summary()


# The LLR p-value for this model is less than the same value for the previous model, which means the last model is slightly better that the first one. And all variables we use for the model are significant (with P equals 0).
# 
# What if we omit the age field creating a model? Will it get much worse? Let's test it!

# In[ ]:


x3 = data.copy()
x3 = x3[["ejection_fraction", "serum_creatinine"]]
x_scaled = preprocessing.scale(x3)
x = sm.add_constant(x_scaled)
reg_log = sm.Logit(y,x)
results_log3 = reg_log.fit()


# In[ ]:


results_log3.summary()


# It seems the model still works, although its LLR p-value is slightly greater while Pseudo R-squared and Log-Likelihood values are lower than those for the model with three independent variables.
# 
# Now let's check how well our model predicts the death event resulting from heart failure. It's going to be a quick test on the same dataset. 

# In[ ]:


results_log3.pred_table()


# In[ ]:


cm_df = pd.DataFrame(results_log3.pred_table())
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
cm_df


# In[ ]:


cm = np.array(cm_df)
accuracy_train = (cm[0,0]+cm[1,1])/cm.sum()
accuracy_train


# And for comparison, let's check the accuracy of prediction for the model with three independent variables:

# In[ ]:


results_log2.pred_table()
cm_df = pd.DataFrame(results_log2.pred_table())
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
cm_df


# In[ ]:


cm = np.array(cm_df)
accuracy_train = (cm[0,0]+cm[1,1])/cm.sum()
accuracy_train


# So, the logistic regression model with two independent variables (ejection fraction and serum creatinine) predict the outcome of heart failure in 75% of cases and adding the third variable (age) improves the rate for 2%. Not bad! :)
