#!/usr/bin/env python
# coding: utf-8

# # Admissions Prediction
# ## Marco Gancitano

# Import needed packages for the notebook

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random


# ### Data Import

# Read in data from csv and see interesting characteristics

# In[ ]:


admissions = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
admissions = admissions.drop('Serial No.',axis = 1)


# These are the given variables:
# 1. GRE Scores ( out of 340 ) 
#     - Graduate Record Examinations
# 2. TOEFL Scores ( out of 120 )
#     - Test of English as a Foreign Language 
# 3. University Rating ( out of 5 ) 
#     - Quality Rating of Undergraduate
# 4. Statement of Purpose and Letter of Recommendation Strength ( out of 5 ) 
# 5. Undergraduate GPA ( out of 10 ) 
# 6. Research Experience ( either 0 or 1 ) 
# 7. Chance of Admit ( ranging from 0 to 1 )

# Quick look at how the data is formatted

# In[ ]:


admissions.head()


# Look at an overview of the data

# In[ ]:


admissions.describe()


# In[ ]:


# Basic correlogram
sns.pairplot(admissions)


# Encouraging to see that test scores (GRE, TOEFL, CGPA) have the best relations with chance of being accepted. I guess studying does pay off.

# Next, we look at numeric correlation between variables

# In[ ]:


corr = admissions.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# Again we see that test scores have the highest correlation with supplemental works being next (SOP, LOR) and research experience being last

# ### Modeling
# 
# 

# #### Creating data sets

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import mean_absolute_error


# In[ ]:


X = admissions.drop('Chance of Admit ',axis = 1)
y = admissions['Chance of Admit ']

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = .25,random_state = 123)


# #### Linear Regression

# In[ ]:


lin_model = LinearRegression()


# In[ ]:


lin_model.fit(X_train,y_train)


# In[ ]:


print('Mean absolute error for linear model: %0.4f' %mean_absolute_error(y_val,lin_model.predict(X_val)))


# #### Random Forest Regression

# In[ ]:


rf_model = RandomForestRegressor(n_estimators = 1000,random_state = 123)
rf_model.fit(X_train,y_train)


# In[ ]:


print('Mean absolute error for linear model: %0.4f' %mean_absolute_error(y_val,rf_model.predict(X_val)))


# With random forest regression being more accurate we'll work with that

# #### Feature Importance

# Looking at the importance of features when estimating the chance of admissions. This is another way to see the explanatory power of the given features just like the pairplot and correlation matrix.

# In[ ]:


feature_importance = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X.columns)), columns=['Value','Feature'])
plt.figure(figsize=(10, 6))
sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))
plt.title('Random Forest Feature Importance')
plt.tight_layout()


# #### Model Fine Tuning

# The problem with this feature importance is the model is still quite naive so the feature importance may be inaccurate. Lets fine tune the model and then see if it changes.

# In[ ]:


rf_model = RandomForestRegressor(n_jobs = -1,random_state = 123)
param_grid = {'n_estimators': [500, 700, 1000, 1200], 
                    'max_depth': [4, 5, 6, 7, 8], 
                    'min_samples_split': [2, 3, 4, 5, 6],
                    'max_features': [1,2,3,4,5,6,7]}
rf_grid = GridSearchCV(estimator = rf_model,param_grid = param_grid,
                       cv = 3,n_jobs = -1)
rf_grid.fit(X_train,y_train)


# In[ ]:


rf_grid.best_params_


# We see that this combination creates the best cross validation accuracy.
# - Max depth: 7
# - Max features: 2
# - Min sample split: 5
# - Trees : 500

# In[ ]:


rf_model.set_params(**rf_grid.best_params_)


# In[ ]:


rf_model.fit(X_train,y_train)


# In[ ]:


print('Mean absolute error for linear model: %0.4f' %mean_absolute_error(y_val,rf_model.predict(X_val)))


# In[ ]:


feature_importance = pd.DataFrame(sorted(zip(rf_model.feature_importances_, X.columns)), columns=['Value','Feature'])
plt.figure(figsize=(10, 6))
sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))
plt.title('Random Forest Feature Importance')
plt.tight_layout()


# Model is slightly better and we see that it isn't mostly dependent on one variable which helps with robustness

# ### Conclusion

# In conclusion, we see that test scores are the most important with getting into grad school with supplemental material behind and research experience as least important. 

# ### Future Works

# 1. Expand models tested to see which one is best for this case
# 2. Begin feature engineering
# 3. Expand on tuning of hyper-parameters

# Thank you if you've read over this notebook and feel free to ask any questions or point out any errors. I always enjoy both teaching and learning.
