#!/usr/bin/env python
# coding: utf-8

# ## Context
# This dataset is created for prediction of Graduate Admissions from an Indian perspective.
# 
# ## Content
# The dataset contains several parameters which are considered important during the application for Masters Programs. The parameters included are : 
# 1. GRE Scores ( out of 340 ) 
# 2. TOEFL Scores ( out of 120 ) 
# 3. University Rating ( out of 5 ) 
# 4. Statement of Purpose and Letter of Recommendation Strength ( out of 5 ) 
# 5. Undergraduate GPA ( out of 10 ) 
# 6. Research Experience ( either 0 or 1 ) 
# 7. Chance of Admit ( ranging from 0 to 1 )
# 
# ## Acknowledgements
# This dataset is inspired by the UCLA Graduate Dataset. The test scores and GPA are in the older format. The dataset is owned by Mohan S Acharya.
# 
# ## Inspiration
# This dataset was built with the purpose of helping students in shortlisting universities with their profiles. The predicted output gives them a fair idea about their chances for a particular university.
# 
# ## Citation
# Please cite the following if you are interested in using the dataset : Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019

# In[56]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[57]:


df = pd.read_csv('../input/Admission_Predict.csv',sep = ",")
print(df.head())


# In[58]:


df.describe()


# In[59]:


df.info()


# In[ ]:





# ## Plots To Help Explore And Get A Feel For Data

# In[60]:


sns.distplot(df['GRE Score'])


# In[61]:


sns.distplot(df['TOEFL Score'])


# In[62]:


sns.distplot(df['CGPA'])


# In[63]:


sns.jointplot(x='TOEFL Score',y='GRE Score',data=df,kind='scatter')


# In[64]:


sns.jointplot(x='CGPA',y='GRE Score',data=df,kind='scatter')


# In[65]:


sns.jointplot(x='Chance of Admit ',y='CGPA',data=df,kind='scatter')


# ## Curious How Much Research Relates To Other Variables

# In[66]:


sns.countplot(df['Research'])


# In[67]:


sns.lmplot(x='Chance of Admit ',y='LOR ',data=df,hue='Research')


# In[68]:


sns.lmplot(x='CGPA',y='LOR ',data=df,hue='Research')


# In[69]:


sns.jointplot(x='University Rating',y='CGPA',data=df,kind='scatter')


# ## Split Dependent and Independent Variables

# In[70]:


from sklearn.model_selection import train_test_split

df=df.drop(['Serial No.'], axis=1)

X = df.drop('Chance of Admit ', axis=1)
y = df['Chance of Admit ']


# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# ## Instantiate DecisionTreeRegressor and Fit to Data

# In[72]:


from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)


# ## Make Prediction

# In[73]:


predict = dtr.predict(X_test)


# ## Evaluate Model Performance

# In[74]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error

dtr_mse= mean_squared_error(y_test, predict)
dtr_rmse = np.sqrt(metrics.mean_squared_error(y_test, predict))

print('Decision Tree Regression RMSE: ', dtr_rmse)
print('Decision Tree Regression MSE: ', dtr_mse)


# In[75]:


plt.scatter(predict,y_test)


# 

# ## Use Recursive Feature Elimination to Find Top Three Features and Repeat Prediction Using Only Those Features

# In[76]:


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

  
rfe = RFE(RandomForestRegressor(n_estimators=50, random_state=42), n_features_to_select=3)

rfe.fit(X_train, y_train)

#create dataframe of feature and ranking. Top 3 have '1' in rfe.ranking_ array
rfe_features_rank = pd.DataFrame({'feature':X_train.columns, 'score':rfe.ranking_})
#compose list of highest ranked features
top_three_features = rfe_features_rank[rfe_features_rank['score'] == 1]['feature'].values
print('Top three features: ', top_three_features)


# In[77]:


top3_df = df[[top_three_features[0],top_three_features[1], top_three_features[2], 'Chance of Admit ']]


# In[78]:


X = top3_df.drop('Chance of Admit ', axis=1)
y = top3_df['Chance of Admit ']


# In[79]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
dtr.fit(X_train, y_train)
predict = dtr.predict(X_test)


# In[80]:


dtr_mse= mean_squared_error(y_test, predict)
dtr_rmse = np.sqrt(metrics.mean_squared_error(y_test, predict))

print('Decision Tree Regression with only top three features RMSE: ', dtr_rmse)
print('Decision Tree Regression with only top three features MSE: ', dtr_mse)


# In[81]:


plt.scatter(predict,y_test)

