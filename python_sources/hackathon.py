#!/usr/bin/env python
# coding: utf-8

# In[39]:


# importing Libraries
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from pylab import plot, show, subplot, specgram, imshow, savefig
from sklearn import preprocessing
from sklearn import cross_validation, metrics
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer

import matplotlib.pyplot as plote

get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('ggplot')


# In[40]:


train = pd.read_csv('../input/GDSChackathon.csv')


# In[41]:


train.head()


# In[42]:


train.describe()


# Finding types of datatypes

# In[43]:


list(set(train.dtypes.tolist()))


# ## function for inding missing values in dataset

# In[44]:


# To check how many columns have missing values - this can be repeated to see the progress made
def show_missing():
    missing = train.columns[train.isnull().any()].tolist()
    return missing


# from this we can find the total missing data in each columns

# In[45]:


train[show_missing()].isnull().sum()


# ## Checking is there any unespected value in score

# In[46]:


train['imdb_score'].describe()


# # Finding all categorical data

# In[47]:


cats = []
for col in train.columns.values:
    if train[col].dtype == 'object':
        cats.append(col)


# # Creating separte datasets for Continuous vs Categorical
# 

# In[48]:


df_cont = train.drop(cats, axis=1)
df_cat = train[cats]


# # Checking correlation
# correlation te;

# In[49]:


corr=train.corr()["imdb_score"]
corr[np.argsort(corr, axis=0)[::-1]]


# # Handle Missing Data for continuous data
# ## If any column contains  entries of missing data, replace those missing values with the median
# 

# In[50]:


for col in df_cont.columns.values:
      if np.sum(df_cont[col].isnull()) > 0:
        median = df_cont[col].median()
        idx = np.where(df_cont[col].isnull())[0]
         
        df_cont[col].iloc[idx] = median


# ## Normalization of contineous data

# In[51]:


df_cont[col] = Normalizer().fit_transform(df_cont[col].reshape(1,-1))[0]


# # plotting most correlated features(correlation>0.1)

# In[52]:



import seaborn as sns
corrmat = df_cont.corr()
plt.figure(figsize = (10,7))


corr_features = corrmat.index[abs(corrmat["imdb_score"])>0.1]
g = sns.heatmap(train[corr_features].corr(),annot=True,cmap="RdYlGn")


# here i tried to plot the most correlated feature which are affectin the target value

# # for categorical data

# ## Categorical Variable Genre
# splitting it in multiple parts and create column genere_0 to genere_7

# In[53]:


genresdf = pd.DataFrame(df_cat['genres'])
genresdf = pd.DataFrame(genresdf.genres.str.split('|').tolist(),columns = ["Genre_"+str(i) for i in  range(0,8)] )

genresdf=genresdf.reindex(df_cat.index)


df_cat.drop('genres',inplace = True, axis = 1)
df_cat = df_cat.merge(genresdf,left_index = True,right_index = True)


# # Handle Missing Data for Categorical Data
# 
# If any column contains entries of missing data, replace those values with the 'anystr'(we can replace here any string)
# ## Applying the sklearn.LabelEncoder
# 

# In[54]:


for col in df_cat.columns.values:
    if np.sum(df_cat[col].isnull()) > 0:
        df_cat[col] = df_cat[col].fillna('anystr')
        
    df_cat[col] = LabelEncoder().fit_transform(df_cat[col])
    


# # merging categorical and contineous data(both are normalised)

# In[55]:


df_new = df_cont.join(df_cat)
targetfet = df_new.imdb_score

features = df_new.drop(['imdb_score'], axis = 1)


# # Detailed Visualization

# In[56]:


vis_dataset = pd.concat([features, targetfet], axis = 1)


# ## No of Movies vs imdb score

# In[57]:


plt.figure(figsize=(12,8))
sns.distplot(df_cont.imdb_score.values, bins=50, kde=False)
plt.xlabel('imdb_score', fontsize=12)
plt.show()


# conclusion: maximum movies are rated b/w 6 to 7.5

# ## Distplot
# The following plot shows the distribution of imdb score (target) in the dataset.
# 
# 

# In[58]:


sns.distplot(targetfet)


# 

# ## budget of movie changing by year
# 

# In[59]:


data_groupby_year = train.groupby(train["title_year"])
data_groupby_year_mean = data_groupby_year.mean()
Budget = plt.scatter(data_groupby_year_mean.index, data_groupby_year_mean["budget"],color = "b" ,s = data_groupby_year["budget"].count())
plt.xlabel("Year")
plt.show()


# Scatter plot

# ## Boxoffice hit (Gross) of movie changing by year

# In[60]:


Gross = plt.scatter(data_groupby_year_mean.index, data_groupby_year_mean["gross"],color='r', s = data_groupby_year["gross"].count())
plt.xlabel("Year")
plt.show()


# ## language of movie
# 

# used BoxPlot

# In[61]:


train.boxplot(column="gross", by="language", rot= 90, grid=False)


# Gross of english movies are quit high

# # splitting train and test data

# In[62]:


X_train, X_test, y_train, y_test = train_test_split(features, targetfet ,test_size=0.3, random_state=7)


# # plotting feature importance

# In[63]:


import xgboost as xgb
xgb = xgb.XGBRegressor()
xgb.fit(X_train,y_train)
predictions_xgb = xgb.predict(X_test)
error_xgb = metrics.mean_squared_error(y_test, predictions_xgb)
print(error_xgb)


# In[64]:


importances = xgb.feature_importances_
feature_names = features.columns.values
data = pd.DataFrame({'features': feature_names,'importances':importances})
new_index = (data['importances'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)
plt.figure(figsize=(15,10))
ax= sns.barplot(x=sorted_data['features'], y=sorted_data['importances'])
plt.xticks(rotation= 90)
plt.xlabel('Features')
plt.ylabel('Importances')
plt.title('feature importances')


# voted users has the hieghest feature importance

# # linear regression

# In[65]:


from sklearn.linear_model import LinearRegression
clf1 = LinearRegression()
clf1.fit(X_train, y_train)
predictions_lr = clf1.predict(X_test)
error_lr = metrics.mean_squared_error(y_test, predictions_lr)
print(error_lr)


# # Linear Regression fit

# In[81]:


y_pred = clf1.predict(X_test)
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, s=20)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted imdb score')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
plt.tight_layout()


# here getting too much noise so trying next algorithm

# # Random forest
# Random Forest, the process es of finding the root node and splitting the feature so that the nodes will run randomly

# In[66]:


clf = RandomForestRegressor()

clf.fit(X_train, y_train)
predictions_lr = clf.predict(X_test)
error_lr = metrics.mean_squared_error(y_test, predictions_lr)
print(error_lr)


# # xgboost
# XGBoost stands for eXtreme Gradient Boosting.
# 
# Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction.

# In[67]:


import xgboost as xgb
xgb = xgb.XGBRegressor()
xgb.fit(X_train,y_train)
predictions_xgb = xgb.predict(X_test)
error_xgb = metrics.mean_squared_error(y_test, predictions_xgb)
print(error_xgb)


# In[83]:


y_pred


# # XGBOOST graph fit
# 

# In[75]:


y_pred = xgb.predict(X_test)

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, s=20)
plt.title('Predicted vs. Actual')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted imdb score')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)])
plt.tight_layout()


# # xgboost for finding training and testing score 

# In[68]:



import xgboost as xgb
xgb = xgb.XGBRegressor()
# train the model on the training set
xgb.fit(X_train,y_train)


# In[72]:


xgb_score_train = xgb.score(X_test, y_test)
xgb_score_test = xgb.score(X_train, y_train)
print("Training score: ",xgb_score_train)
print("Testing score: ",xgb_score_test)


# #  random forest for finding training and testing score 

# In[70]:


from sklearn.ensemble import RandomForestRegressor
dt = RandomForestRegressor(n_estimators = 1000,n_jobs=-1,random_state = 0)
dt.fit(X_train, y_train)
dt_score_train = dt.score(X_train, y_train)
print("Training score: ",dt_score_train)
dt_score_test = dt.score(X_test, y_test)
print("Testing score: ",dt_score_test)


# # so here most sutable algorithm is XGBoost
