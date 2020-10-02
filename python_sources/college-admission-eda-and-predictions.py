#!/usr/bin/env python
# coding: utf-8

# # College Admission: Project Overview
# 
# * Created a tool that estimates the chance of admission in college for a student.
# * Performed a simple EDA analysis of the data provided, verifying how each variable affects the admission chance.
# * Compared the prediction result for different regression models, in which Linear Regression was recommended as it scored similar to Lasso and Random Forest Regressor.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('seaborn-darkgrid')


# Importing the csv file

# In[ ]:


df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')


# In[ ]:


df.head()


# All the columns are numeric. We won't need to use a One-Hot-Encoding approach (used for categorical data)

# 1. **check for missing data**

# In[ ]:


max_nan = df.isnull().sum().max() # no data is missing
print('NaN values in our dataset: ' + str(max_nan))
print('\n')
sns.heatmap(abs(df.isnull()), cmap='viridis')


# No data is missing
# 

# In[ ]:


# Creating a new column just for EDA

# Assuming -> Chance of Admit >= 75% (Probably admitted)
#             Chance of Admit < 75% (Probably recused) 

df['probably_admitted'] = df['Chance of Admit '].apply(lambda x: 1 if x >= 0.75 else 0)


# In[ ]:


#Dataframes for continuous and discrete data
df_cont = df[['GRE Score', 'TOEFL Score', 'CGPA','Chance of Admit ']]
df_disc = df[['University Rating','SOP','LOR ', 'probably_admitted']]


# In[ ]:


for i in np.arange(0, len(df_cont.columns), 2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,7))
    
    ax1.set_title('Distribution for %s' % (df_cont.columns[i]))    
    ax1 = sns.distplot(df_cont[df_cont.columns[i]],bins=30, kde=False, ax=ax1)
    
    ax2.set_title('Distribution for %s' % (df_cont.columns[i+1]))    
    ax2 = sns.distplot(df_cont[df_cont.columns[i+1]],bins=30, kde=False, ax=ax2)


# - **GRE Score**: Normal Distribution
# - **TOEFL**: Normal Distribution
# - **CGPA**: Normal Distribution
# - **Chance of Admit**: The majority of students have between 70-80% of chance of being admite 

# In[ ]:


for i in np.arange(0, len(df_disc.columns), 2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,7))
   
    ax1.set_title('Distribution for %s' % (df_disc.columns[i]))    
    ax1 = sns.countplot(x=df_disc.columns[i], data=df_disc, ax=ax1)
    
    ax2.set_title('Distribution for %s' % (df_disc.columns[i+1]))    
    ax2 = sns.countplot(x=df_disc.columns[i+1],data=df_disc, ax=ax2)


# Correlation between variables

# In[ ]:


fig, (ax1) = plt.subplots(figsize=(9,5))
ax1 = sns.heatmap(df.drop(['Serial No.'], axis=1).corr(), linewidths=0.5, square=True, cmap='viridis')


# In[ ]:


pd.DataFrame(df.drop(['Serial No.', 'probably_admitted'], axis=1).corr()['Chance of Admit '].sort_values(ascending=False)[1:])


# Doing research don't seem to affect much on the admission chance. 
# 
# The other variables are directly linked, as it seems logic that students that score better grades on tests have a better chance of being admitted.

# # Model Building

# In[ ]:


from sklearn.model_selection import train_test_split


# ### Train-Test splits

# As all columns are numerical, we can train-test-split the DataFrame directly.

# In[ ]:


X = df.drop(['Serial No.', 'probably_admitted','Chance of Admit '], axis=1)
y = df['Chance of Admit '].values
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# ## Linear regression

# In[ ]:


from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score # module import for model validation and comparison


# In[ ]:


lm = LinearRegression()
lm.fit(X_train, y_train)

print('Negative Mean Absolute Error for Linear Regression:')
np.mean(cross_val_score(lm, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))


# ## Lasso regression

# In[ ]:


alpha = []
error = []

for i in range(1, 100):
    alpha.append(i/2000)
    lml = Lasso(alpha=(i/2000))
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))
    
plt.grid(True)
plt.title('Lasso Regression Prediction Score')
plt.ylabel('Error')
plt.xlabel('Alpha')
plt.plot(alpha, error, label='Neg Mean Absolute Error')
plt.legend()


# Choosing the alpha that gives the smaller error

# In[ ]:


index = error.index(max(error))
best_alpha = alpha[index]

lml = Lasso(alpha=best_alpha)
lml.fit(X_train, y_train)


# ## Random Forest 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf = RandomForestRegressor()
rf.fit(X_train, y_train)
print('Negative Mean Absolute Error for Random Forest Regression:')
np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))


# # Prediction Comparison

# In[ ]:


tpred_lm = lm.predict(X_test)
tpred_lml = lml.predict(X_test)
tpred_rf = rf.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_absolute_error


# In[ ]:


mae_lm = '{:2.2f}'.format(mean_absolute_error(y_test, tpred_lm)*100) + '%'
mae_lml = '{:2.2f}'.format(mean_absolute_error(y_test, tpred_lml)*100) + '%'
mae_rf = '{:2.2f}'.format(mean_absolute_error(y_test, tpred_rf)*100) + '%'

print('Mean Absolute Error for Each Regression Type:')
print('\n')
pd.DataFrame({'Linear':[mae_lm], 'Lasso':[mae_lml], 'RdnForest':[mae_rf] })


# As stated from the table above, all three methods scored similar results in a MAE comparison.
# 
# In this situation, Linear Regression is recommended as it requires less computational power.  
