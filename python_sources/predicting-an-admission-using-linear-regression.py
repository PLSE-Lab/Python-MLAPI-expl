#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import mean_absolute_error


# In[ ]:


dataset = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')


# In[ ]:


dataset.info()
dataset.describe()


# There are no null values in the dataset ! Let's remove the Serial No column and check the correlation amongst the variables

# In[ ]:


dataset = dataset.drop('Serial No.', axis=1)
sns.pairplot(data=dataset, kind='reg')


# It is clearly visible that GRE score, TOEFL score, CGPA have heavy influence on the chances of admission. Research doesn't seem to have high infuence on the chances of admission. Let's try to check the correlation values

# In[ ]:


dataset.corr(method='pearson')


# Above data confirms the understanding. Also there exist multicollinearity in the data. GRE, CGPA, and TOEFL scores are heavily correlated. 
# Let's try to build the model with all independent variables first.

# In[ ]:


y=dataset['Chance of Admit ']
dataset=dataset.drop('Chance of Admit ',axis=1)

def create_stats_fit_model(x):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
    regressor_OLS = sm.OLS(endog=y_train,exog=x_train).fit()
    print(regressor_OLS.summary())
    predictions =  regressor_OLS.predict(x_test)
    print("Mean absolute error is ",mean_absolute_error(y_test,predictions))


# Let's now call the method with full set of independent variables

# In[ ]:


dataset_temp = np.append(arr = np.ones((500,1)).astype(int), values = dataset ,axis=1)
dataset_temp1 = dataset_temp[:, [0,1,2,3,4,5,6,7]]
create_stats_fit_model(dataset_temp1)


# Let's remove x4 which has the highest pvalue and rerun

# In[ ]:


dataset_temp1 = dataset_temp[:, [0,1,2,3,5,6,7]]
create_stats_fit_model(dataset_temp1)


# Let's remove x3 and rerun

# In[ ]:


dataset_temp1 = dataset_temp[:, [0,1,2,5,6,7]]
create_stats_fit_model(dataset_temp1)


# There is no observable improvement further to this model. It seems that as per my analysis, I would 
# consider below variables -
# GRE Score, TOEFL Score, LOR, CGPA, Research
# 
# 
# I still think that this could be improved as have not considered the impact of multicollinearity. Please comment on what do you think.
