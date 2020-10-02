#!/usr/bin/env python
# coding: utf-8

# # Houses Pricing - Random Forest Regression

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


df.shape


# # **Searching for null/missing values**

# In[ ]:


nulls = pd.DataFrame(df.isnull().sum())

nulls = nulls[nulls[0] >0]

print(nulls)


# # Using a Heatmap to see the null/missing values

# In[ ]:


plt.figure(figsize=(20,6))
sns.heatmap(df.isnull(), cbar=False,yticklabels=False)


# # **Dropping columns with high number of null values**

# We will drop the followed columns where we have a high number of null values:
# 
# *     * Alley
# *     * PoolQC
# *     * Fence
# *     * MiscFeature
# *     * FireplaceQu
# *     * LotFrontage
#     
# We can drop also the column *Id* because it will be irrelevant to the study

# In[ ]:


columns = ['Alley','PoolQC','Fence','MiscFeature','FireplaceQu','LotFrontage']
df.drop(columns, axis=1, inplace = True)


# # **Filling missing/null values**
# 
# 

# ## Column MasVnrType

# In[ ]:


df['MasVnrType'].unique()


# In[ ]:


df['MasVnrType'].fillna(value = 'None', inplace = True)


# In[ ]:


df['MasVnrType'].unique()


# ## Column MasVnrArea

# In[ ]:


df.update(df['MasVnrArea'].fillna(value = 0, inplace = True))


# ## Column BSMTQUAL

# In[ ]:


df.update(df['BsmtQual'].fillna(value = 'NA', inplace = True))
df['BsmtQual'].unique()


# ## Column BSMTCOND

# In[ ]:


df.update(df['BsmtCond'].fillna(value = 'NA', inplace = True))
df['BsmtCond'].unique()


# # Removing the basement data that is not crucial to the model

# In[ ]:


columns = ['BsmtExposure','BsmtFinType1','BsmtFinType2','GarageYrBlt','GarageType','GarageFinish','GarageQual','GarageCond']
df.drop(columns, axis=1, inplace = True)


# # Removing the null values in ELECTRICAL category

# In[ ]:


df['Electrical'].isnull().sum()


# In[ ]:


df.update(df.Electrical.dropna(inplace= True))
df['Electrical'].isnull().sum()


# In[ ]:


plt.figure(figsize=(20,6))
sns.heatmap(df.isnull(), cbar=False,yticklabels=False)


# # Correlations with the target column - SalePrice

# In[ ]:


CorrTarget = df.corr()
CorrTarget['SalePrice']


# # Stracting the best correlations to the model - above 0.4

# In[ ]:


dicCorrelations = dict(CorrTarget['SalePrice'])
BestCorrelations = []

for k,v in dicCorrelations.items():
    if v >0.4:
        BestCorrelations.append(k)
    else:
        continue


# In[ ]:


print(BestCorrelations)


# # Using a heatmap to visualize the correlations

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(df[BestCorrelations].corr(),cmap = 'magma',annot = True, linecolor = 'black',lw = 1)


# # Putting the best categories in another dataframe

# In[ ]:


df_bestcorr = df[BestCorrelations]
df_bestcorr.head(3)


# # Starting the Regression model
# ## First, We separate the data in test and train

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df_bestcorr.drop('SalePrice',axis=1), df_bestcorr['SalePrice'], test_size = 0.3)


# ### Reshaping the data

# In[ ]:


y_train= y_train.values.reshape(-1,1)
y_test= y_test.values.reshape(-1,1)


# # Importing the Random Forrest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train,y_train.ravel())


# In[ ]:


predictions = model.predict(X_test)


# In[ ]:


plt.figure(figsize=(12,8))
plt.scatter(y_test,predictions, marker = ('v'))
plt.xlabel('Y Test')
plt.ylabel('Predict')


# In[ ]:


error = y_test-predictions
error = error.reshape(-1,1)
sns.distplot(error,bins=50, color = 'red')


# ## Using the GridSearchCV to find the best parameters to the model

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


parameters = {'min_samples_leaf':[1,20], 'min_samples_split':[2,200],'n_estimators':[100,250,500,750,1000]}


# In[ ]:


grid = GridSearchCV(model,parameters)

grid.fit(X_train,y_train.ravel())


# In[ ]:


grid.best_params_


# In[ ]:


best_model = grid.best_estimator_


# In[ ]:


predictions = best_model.predict(X_test)

plt.figure(figsize=(12,8))
plt.scatter(y_test,predictions, marker = ('v'))
plt.xlabel('Y Test')
plt.ylabel('Predict')


# In[ ]:


error = y_test-predictions
print(error.sum())
error = error.reshape(-1,1)

sns.distplot(error,bins=50, color = 'red')


# # Let's put the new data on our dataset

# In[ ]:


predictPrice = best_model.predict(df_bestcorr.drop('SalePrice',axis=1))
x = pd.DataFrame(predictPrice,columns=['SalePrice_Predicted'])

result_comparision = pd.concat([df,x], axis = 1)
result_comparision.head()


# # Comparing the real data against the predictions

# In[ ]:


result_comparision['Model_Error'] = result_comparision.SalePrice - result_comparision.SalePrice_Predicted
result_comparision[['SalePrice','SalePrice_Predicted','Model_Error']].head()


# # Working with the test dataset

# In[ ]:


df_test = pd.read_csv('../input/test.csv', usecols = 
                      [  'OverallQual',
                         'YearBuilt',
                         'YearRemodAdd',
                         'MasVnrArea',
                         'TotalBsmtSF',
                         '1stFlrSF',
                         'GrLivArea',
                         'FullBath',
                         'TotRmsAbvGrd',
                         'Fireplaces',
                         'GarageCars',
                         'GarageArea'])


# In[ ]:


df_test.isnull().sum()


# ## Clearing the data

# In[ ]:


df_test.update(df_test['MasVnrArea'].fillna(value = 0, inplace = True))
df_test['TotalBsmtSF'].fillna(value = df_test.TotalBsmtSF.mean(), inplace = True)
df_test['GarageCars'].fillna(value = df_test.GarageCars.mean(), inplace = True)
df_test['GarageArea'].fillna(value = df_test.GarageArea.mean(), inplace = True)
df_test.isnull().sum()


# # Predicting results

# In[ ]:


predict_result = best_model.predict(df_test)
predict_result = pd.DataFrame(predict_result,columns=['SalePrice'])


# ## Creating a new dataset with the predicted results

# In[ ]:


index = pd.read_csv('../input/test.csv')
Id = index['Id']
Id = pd.DataFrame(Id)

result = pd.concat([Id,predict_result.round(2)], axis =1)
result.head()


# # Saving results

# In[ ]:


result.to_csv('submission.csv',index=False)

