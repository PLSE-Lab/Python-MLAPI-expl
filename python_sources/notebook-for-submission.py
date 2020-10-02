#!/usr/bin/env python
# coding: utf-8

# # Flatiron School: Use your regression skills to save the Galaxy
# 
# ### InClass prediction Competition

# ### Table of Contents
# <a id='Table of contents'></a>
# 
# ### <a href='#1. Obtaining and Viewing the Data'> 1. Obtaining and Viewing the Data </a>
# * <a href='#1.1. Test Data'> 1.1. Test Data </a>
# * <a href='#1.2. Train Data'> 1.2. Train Data </a>
# 
# ### <a href='#2. Preprocessing the Data'> 2. Preprocessing the Data </a>
# * <a href='#2.1. Handling Genre column'> 2.1. Handling Genre column </a>
# * <a href='#2.2. Handling Rating column'> 2.2. Handling Rating column</a>
# * <a href='#2.3. Handling Year_of_Release column'> 2.3. Handling Year_of_Release column </a>
# * <a href='#2.4. Handling User Count column'> 2.4. Handling User Count column </a>
# * <a href='#2.5. Handling Critic Score column'> 2.5. Handling Critic Score column </a>
# * <a href='#2.6. Handling Platform column'> 2.6. Handling Platform column </a>
# 
# ### <a href='#3. Modeling the Train Data'> 3. Modeling the Train Data </a>
# 
# ### <a href='#4. Predicting on Test Data'> 4. Predicting on Test Data </a>

# ### 1. Obtaining and Viewing the Data 
# <a id='1. Obtaining and Viewing the Data'></a>

# In[1]:


# import libraries 
import pandas as pd
import numpy as np
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

import warnings
warnings.filterwarnings("ignore")


# **1.1. Test Data**
# <a id='1.1. Test Data'></a>

# In[6]:


df_test = pd.read_csv('../input/test.csv')
df_test.info()


# In[7]:


df_test.head(2)


# In[8]:


df_test.isna().sum()/len(df_test)


# In[9]:


# why is user score stored as string?
df_test.User_Score.unique()


# In[10]:


# replace 'tbd'
df_test['User_Score'].replace(to_replace='tbd', value=np.nan, inplace=True)
# convert string values into numeric values
df_test['User_Score'] = pd.to_numeric(df_test['User_Score'])

df_test.info()


# In[11]:


df_test.duplicated().sum()


# **1.2. Train Data**
# <a id='1.2. Train Data'></a>

# In[12]:


df_train = pd.read_csv('../input/train.csv')
df_train.info()


# In[13]:


df_train.head(2)


# In[14]:


df_train.isna().sum()/len(df_train)


# In[15]:


# replace 'tbd'
df_train['User_Score'].replace(to_replace='tbd', value=np.nan, inplace=True)

# convert string values into numeric values
df_train['User_Score'] = pd.to_numeric(df_train['User_Score'])

df_train.info()


# In[16]:


df_train.duplicated().sum()


# In[17]:


plt.subplots(figsize=(8, 6))
sns.heatmap(df_train.corr(), cmap="Oranges", linewidths=0.1);


# As the columns `Publisher` and `Developer` contain way too many values and also doesn't seem to add any reasonable value to our prediction, let's drop them.

# In[18]:


df_train = df_train.drop(['Publisher', 'Developer'], axis=1)
df_test  = df_test.drop(['Publisher', 'Developer'], axis=1)


# *Back to: <a href='#Table of contents'> Table of contents</a>*
# ### 2. Preprocessing the Data 
# <a id='2. Preprocessing the Data'></a>

# **2.1. Handling `Genre` column**
# <a id='2.1. Handling Genre column'></a>

# ***Train Data***

# In[19]:


print('Train Data:')
print('------------------------------------------')
df_train.info()


# We only face 2 records with a missing value in the `Genre` column - let's drop both:

# In[20]:


df_train = df_train.dropna(subset=['Genre'])

df_train.Genre = df_train.Genre.astype('category')

df_train.Genre.value_counts()


# ***Test Data***

# In[21]:


print('Test Data:')
print('------------------------------------------')
df_test.info()


# No missing values in `Genre` at all, lucky we!

# In[22]:


df_test.Genre = df_test.Genre.astype('category')
df_test.Genre.value_counts()


# *Back to: <a href='#Table of contents'> Table of contents</a>*
# 
# **2.2. Handling `Rating` column**
# <a id='2.2. Handling Rating column'></a>

# ***Train Data***

# In[23]:


df_train.Rating.unique()


# In[24]:


df_train.Rating.value_counts()


# According to [Wikipedia](https://en.wikipedia.org/wiki/Entertainment_Software_Rating_Board), we can deal with some of rather outdated ratings by replacing them with actual labels:

# In[25]:


def value_replacement(col, to_replace, new_value):
    col.replace(to_replace, new_value, inplace=True)


# In[26]:


value_replacement(df_train.Rating, to_replace='EC', new_value='E')
value_replacement(df_train.Rating, to_replace='K-A', new_value='E')
value_replacement(df_train.Rating, to_replace='RP', new_value='None')
value_replacement(df_train.Rating, to_replace=np.nan, new_value='None')

df_train.Rating.value_counts()


# In[27]:


df_train.Rating = df_train.Rating.astype('category')


# ***Test Data***

# In[28]:


df_test.Rating.value_counts()


# In[29]:


value_replacement(df_test.Rating, to_replace='EC', new_value='E')
value_replacement(df_test.Rating, to_replace='AO', new_value='M')
value_replacement(df_test.Rating, to_replace='K-A', new_value='E')
value_replacement(df_test.Rating, to_replace='RP', new_value='None')
value_replacement(df_test.Rating, to_replace=np.nan, new_value='None')

df_test.Rating.value_counts()


# In[30]:


df_test.Rating = df_test.Rating.astype('category')


# *Back to: <a href='#Table of contents'> Table of contents</a>*
# 
# **2.3. Handling `Year_of_Release` column**
# <a id='2.3. Handling Year_of_Release column'></a>

# ***Train Data***

# In[31]:


df1 = df_train[df_train['Year_of_Release'].isna()]
df1.tail()


# In[32]:


df_train.Year_of_Release.max()


# In[33]:


df_test.Year_of_Release.max()


# In[34]:


df_train.Year_of_Release.hist(bins=20);


# It might be reasonable to bin the years into periods of roughly 5 years and when doing this, also bin the NaN values into *unknown*.

# In[35]:


# bin the year_of_release into periods_of_release
bins = [1980, 1995, 2000, 2005, 2010, 2015, 2017]
labels = ['Before 1995', '1995-2000', '2000-2005', '2005-2010', '2010-2015', '2015-2020']
df_train['Periods_of_Release'] = pd.cut(df_train['Year_of_Release'], bins=bins, labels=labels)

# create another category for the unknown release date
df_train['Periods_of_Release'].replace(to_replace=np.nan, value='Unknown', inplace=True)


# In[36]:


df_train.Periods_of_Release.value_counts()


# In[37]:


# visualize the distribution of categories
order = ['Unknown', '2015-2020', '2010-2015', '2005-2010',  '2000-2005','1995-2000', 'Before 1995']
df_train.Periods_of_Release.value_counts().loc[order].plot(kind='barh');


# In[38]:


# drop the original year related column
df_train = df_train.drop(['Year_of_Release'], axis=1)

df_train.Periods_of_Release = df_train.Periods_of_Release.astype('category')

df_train.info()


# ***Test Data***

# In[39]:


df_test['Periods_of_Release'] = pd.cut(df_test['Year_of_Release'], bins=bins, labels=labels)

# create another category for the unknown release date
df_test['Periods_of_Release'].replace(to_replace=np.nan, value='Unknown', inplace=True)

df_test = df_test.drop(['Year_of_Release'], axis=1)

df_test.Periods_of_Release = df_test.Periods_of_Release.astype('category')

df_test.info()


# *Back to: <a href='#Table of contents'> Table of contents</a>*
# 
# **2.4. Handling `User Count` column**
# <a id='2.4. Handling User Count column'></a>

# ***Train Data***

# In[40]:


df_train.info()


# In[41]:


df_train = df_train.drop(['Critic_Count', 'User_Score'], axis=1)


# In[42]:


df_train.User_Count.describe()


# In[43]:


df_train.User_Count.plot(kind='box', vert=False, xlim=(4,500));


# In[44]:


# removing outliers
df_train = df_train.drop(df_train[df_train.User_Count > 200].index, axis=0)
df_train.User_Count.describe()


# In[45]:


df_train.info()


# *a) Creating Sub-Datasets*

# In[46]:


# filter out sub_df to work with
#sub_df = df[['NA_Sales', 'JP_Sales', 'Critic_Score', 'User_Score']]
sub_df = df_train[['JP_Sales', 'Genre', 'Rating', 'User_Count']]

# split datasets
train_data = sub_df[sub_df['User_Count'].notnull()]
test_data  = sub_df[sub_df['User_Count'].isnull()]

# define X
X_train = train_data.drop('User_Count', axis=1)
X_train = pd.get_dummies(X_train)

X_test  = test_data.drop('User_Count', axis=1)
X_test  = pd.get_dummies(X_test)

# define y
y_train = train_data['User_Count']


# *b) Scaling the Features*

# In[47]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled)


# *c) Implementing Linear Regression*

# In[48]:


# import Linear Regression
from sklearn.linear_model import LinearRegression

# instantiate
linreg_user_score = LinearRegression()

# fit model to training data
linreg_user_score.fit(X_train_scaled, y_train)

# making predictions
y_test = linreg_user_score.predict(X_test_scaled)


# In[49]:


# preparing y_test
y_test = pd.DataFrame(y_test)
y_test.columns = ['User_Count']
print(y_test.shape)
y_test.head(2)


# In[50]:


# preparing X_test
print(X_test.shape)
X_test.head(2)


# *d) Concatenating Dataset*

# In[51]:


# make the index of X_test to an own dataframe
prelim_index = pd.DataFrame(X_test.index)
prelim_index.columns = ['prelim']

# ... and concat this dataframe with y_test
y_test = pd.concat([y_test, prelim_index], axis=1)
y_test.set_index(['prelim'], inplace=True)

# finally combine the new test data
test_data = pd.concat([X_test, y_test], axis=1)

# combine train and test data back to a new sub df
sub_df_new = pd.concat([test_data, train_data], axis=0, sort=True)

print(sub_df_new.shape)
sub_df_new.head(2)


# In[52]:


# drop duplicate columns in dataframe before concatening 
df_train.drop(['User_Count'], axis=1, inplace=True)
sub_df_new = sub_df_new[['User_Count']]

# concatenate back to complete dataframe
df_train_1 = pd.concat([sub_df_new, df_train], axis=1)

#print(df_train.shape)
df_train_1.head(2)


# In[53]:


df_train_1.User_Count.isna().sum()


# In[54]:


df_train_1.User_Count.describe()


# ***Test Data***

# In[55]:


df_test.info()


# In[56]:


df_test = df_test.drop(['Critic_Count', 'User_Score'], axis=1)


# In[57]:


df_test.User_Count.describe()


# *a) Creating Sub-Datasets*

# In[58]:


# filter out sub_df to work with
sub_df = df_test[['JP_Sales', 'Genre', 'Rating', 'User_Count']]

# split datasets
train_data = sub_df[sub_df['User_Count'].notnull()]
test_data  = sub_df[sub_df['User_Count'].isnull()]

# define X
X_train = train_data.drop('User_Count', axis=1)
X_train = pd.get_dummies(X_train)

X_test  = test_data.drop('User_Count', axis=1)
X_test  = pd.get_dummies(X_test)

# define y
y_train = train_data['User_Count']


# *b) Scaling the Features*

# In[59]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled)


# *c) Implementing Linear Regression*

# In[60]:


# import Linear Regression
from sklearn.linear_model import LinearRegression

# instantiate
linreg_user_score = LinearRegression()

# fit model to training data
linreg_user_score.fit(X_train_scaled, y_train)

# making predictions
y_test = linreg_user_score.predict(X_test_scaled)


# In[61]:


# preparing y_test
y_test = pd.DataFrame(y_test)
y_test.columns = ['User_Count']
print(y_test.shape)
y_test.head(2)


# In[62]:


# preparing X_test
print(X_test.shape)
X_test.head(2)


# *d) Concatenating Dataset*

# In[63]:


# make the index of X_test to an own dataframe
prelim_index = pd.DataFrame(X_test.index)
prelim_index.columns = ['prelim']

# ... and concat this dataframe with y_test
y_test = pd.concat([y_test, prelim_index], axis=1)
y_test.set_index(['prelim'], inplace=True)

# finally combine the new test data
test_data = pd.concat([X_test, y_test], axis=1)

# combine train and test data back to a new sub df
sub_df_new = pd.concat([test_data, train_data], axis=0, sort=True)

print(sub_df_new.shape)
sub_df_new.head(2)


# In[64]:


# drop duplicate columns in dataframe before concatening 
df_test.drop(['User_Count'], axis=1, inplace=True)
sub_df_new = sub_df_new[['User_Count']]

# concatenate back to complete dataframe
df_test_1 = pd.concat([sub_df_new, df_test], axis=1)

#print(df_train.shape)
df_test_1.head(2)


# In[65]:


df_test_1.User_Count.isna().sum()


# In[66]:


df_test_1.User_Count.describe()


# In[67]:


df_test_1.info()


# *Back to: <a href='#Table of contents'> Table of contents</a>*
# 
# **2.5. Handling `Critic Score` column**
# <a id='2.5. Handling Critic Score column'></a>

# ***Train Data***

# In[68]:


df_train_1.info()


# In[69]:


df_train_1.Critic_Score.describe()


# In[70]:


df_train_1.Critic_Score.plot(kind='box', vert=False, xlim=(0,100));


# In[71]:


# removing outliers
df_train_1 = df_train_1.drop(df_train_1[df_train_1.Critic_Score < 30].index, axis=0)
df_train_1.Critic_Score.describe()


# In[72]:


df_train_1.info()


# *a) Creating Sub-Datasets*

# In[73]:


# filter out sub_df to work with
#sub_df = df[['NA_Sales', 'JP_Sales', 'Critic_Score', 'User_Score']]
sub_df = df_train_1[['JP_Sales', 'Genre', 'Rating', 'Critic_Score']]

# split datasets
train_data = sub_df[sub_df['Critic_Score'].notnull()]
test_data  = sub_df[sub_df['Critic_Score'].isnull()]

# define X
X_train = train_data.drop('Critic_Score', axis=1)
X_train = pd.get_dummies(X_train)

X_test  = test_data.drop('Critic_Score', axis=1)
X_test  = pd.get_dummies(X_test)

# define y
y_train = train_data['Critic_Score']


# *b) Scaling the Features*

# In[74]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled)


# *c) Implementing Linear Regression*

# In[75]:


# import Linear Regression
from sklearn.linear_model import LinearRegression

# instantiate
linreg_user_score = LinearRegression()

# fit model to training data
linreg_user_score.fit(X_train_scaled, y_train)

# making predictions
y_test = linreg_user_score.predict(X_test_scaled)


# In[76]:


# preparing y_test
y_test = pd.DataFrame(y_test)
y_test.columns = ['Critic_Score']
print(y_test.shape)
y_test.head(2)


# In[77]:


# preparing X_test
print(X_test.shape)
X_test.head(2)


# *d) Concatenating Dataset*

# In[78]:


# make the index of X_test to an own dataframe
prelim_index = pd.DataFrame(X_test.index)
prelim_index.columns = ['prelim']

# ... and concat this dataframe with y_test
y_test = pd.concat([y_test, prelim_index], axis=1)
y_test.set_index(['prelim'], inplace=True)

# finally combine the new test data
test_data = pd.concat([X_test, y_test], axis=1)

# combine train and test data back to a new sub df
sub_df_new = pd.concat([test_data, train_data], axis=0, sort=True)

print(sub_df_new.shape)
sub_df_new.head(2)


# In[79]:


# drop duplicate columns in dataframe before concatening 
df_train_1.drop(['Critic_Score'], axis=1, inplace=True)
sub_df_new = sub_df_new[['Critic_Score']]

# concatenate back to complete dataframe
df_train_2 = pd.concat([sub_df_new, df_train_1], axis=1)

#print(df_train.shape)
df_train_2.head(2)


# In[80]:


df_train_2.Critic_Score.isna().sum()


# In[81]:


df_train_2.Critic_Score.describe()


# In[82]:


df_train_2.info()


# ***Test Data***

# In[83]:


df_test_1.info()


# In[84]:


df_test_1.Critic_Score.describe()


# *a) Creating Sub-Datasets*

# In[85]:


# filter out sub_df to work with
#sub_df = df[['NA_Sales', 'JP_Sales', 'Critic_Score', 'User_Score']]
sub_df = df_test_1[['JP_Sales', 'Genre', 'Rating', 'Critic_Score']]

# split datasets
train_data = sub_df[sub_df['Critic_Score'].notnull()]
test_data  = sub_df[sub_df['Critic_Score'].isnull()]

# define X
X_train = train_data.drop('Critic_Score', axis=1)
X_train = pd.get_dummies(X_train)

X_test  = test_data.drop('Critic_Score', axis=1)
X_test  = pd.get_dummies(X_test)

# define y
y_train = train_data['Critic_Score']


# *b) Scaling the Features*

# In[86]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled)


# *c) Implementing Linear Regression*

# In[87]:


# import Linear Regression
from sklearn.linear_model import LinearRegression

# instantiate
linreg_user_score = LinearRegression()

# fit model to training data
linreg_user_score.fit(X_train_scaled, y_train)

# making predictions
y_test = linreg_user_score.predict(X_test_scaled)


# In[88]:


# preparing y_test
y_test = pd.DataFrame(y_test)
y_test.columns = ['Critic_Score']
print(y_test.shape)
y_test.head(2)


# In[89]:


# preparing X_test
print(X_test.shape)
X_test.head(2)


# *d) Concatenating Dataset*

# In[90]:


# make the index of X_test to an own dataframe
prelim_index = pd.DataFrame(X_test.index)
prelim_index.columns = ['prelim']

# ... and concat this dataframe with y_test
y_test = pd.concat([y_test, prelim_index], axis=1)
y_test.set_index(['prelim'], inplace=True)

# finally combine the new test data
test_data = pd.concat([X_test, y_test], axis=1)

# combine train and test data back to a new sub df
sub_df_new = pd.concat([test_data, train_data], axis=0, sort=True)

print(sub_df_new.shape)
sub_df_new.head(2)


# In[91]:


# drop duplicate columns in dataframe before concatening 
df_test_1.drop(['Critic_Score'], axis=1, inplace=True)
sub_df_new = sub_df_new[['Critic_Score']]

# concatenate back to complete dataframe
df_test_2 = pd.concat([sub_df_new, df_test_1], axis=1)

#print(df_train.shape)
df_test_2.head(2)


# In[92]:


df_test_2.Critic_Score.isna().sum()


# In[93]:


df_test_2.Critic_Score.describe()


# In[94]:


df_test_2.info()


# *Back to: <a href='#Table of contents'> Table of contents</a>*
# 
# **2.6. Handling `Platform` column**
# <a id='2.6. Handling Platform column'></a>

# In[95]:


#df_train_2.Platform.value_counts()


# In[96]:


#df_test_2.Platform.value_counts()


# In[97]:


#df_train_final = df_train_final.groupby('Platform').filter(lambda x: len(x) > 100)


# In[98]:


df_train_2 = df_train_2.drop(['Platform'], axis=1)


# In[99]:


df_test_2 = df_test_2.drop(['Platform'], axis=1)


# *Back to: <a href='#Table of contents'> Table of contents</a>*
# 
# ### 3. Modeling the Train Data 
# <a id='3. Modeling the Train Data'></a>

# *Preparing Target & Features*

# In[100]:


df_training = df_train_2.drop(['Id'], axis=1)
df_training.info()


# In[101]:


# define our features 
features = df_training.drop(['NA_Sales'], axis=1)

# define our target
target = df_training[['NA_Sales']]


# *Recoding Categorical Features*

# In[102]:


# create dummy variables of all categorical features
features = pd.get_dummies(features)


# *Train-Test-Split*

# In[103]:


# import train_test_split function
from sklearn.model_selection import train_test_split

# split our data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=40)


# *Scaling the Data*

# In[104]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train) 
X_test = scaler.transform(X_test)


# *Training XGBoost*

# In[105]:


# create a baseline
booster = xgb.XGBRegressor()


# In[ ]:


from sklearn.model_selection import GridSearchCV

# create Grid
param_grid = {'n_estimators': [100, 150, 200],
              'learning_rate': [0.01, 0.05, 0.1], 
              'max_depth': [3, 4, 5, 6, 7],
              'colsample_bytree': [0.6, 0.7, 1],
              'gamma': [0.0, 0.1, 0.2]}

# instantiate the tuned random forest
booster_grid_search = GridSearchCV(booster, param_grid, cv=3, n_jobs=-1)

# train the tuned random forest
booster_grid_search.fit(X_train, y_train)

# print best estimator parameters found during the grid search
print(booster_grid_search.best_params_)


# In[106]:


# instantiate xgboost with best parameters
booster = xgb.XGBRegressor(colsample_bytree=0.6, gamma=0.2, learning_rate=0.05, 
                             max_depth=6, n_estimators=100, random_state=4)

# train
booster.fit(X_train, y_train)

# predict
y_pred_train = booster.predict(X_train)
y_pred_test  = booster.predict(X_test)


# In[107]:


# import metrics
from sklearn.metrics import mean_squared_error, r2_score

RMSE = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"RMSE: {round(RMSE, 4)}")

r2 = r2_score(y_test, y_pred_test)
print(f"r2: {round(r2, 4)}")


# In[108]:


booster.score(X_test, y_test)


# In[109]:


# plot the important features
feat_importances = pd.Series(booster.feature_importances_, index=features.columns)
feat_importances.nlargest(15).sort_values().plot(kind='barh', color='darkgrey', figsize=(10,5))
plt.xlabel('Relative Feature Importance with XGBoost');


# *Back to: <a href='#Table of contents'> Table of contents</a>*
# 
# ### 4. Predicting on Test Data
# <a id='4. Predicting on Test Data'></a>

# *Preparing Target & Features*

# In[110]:


df_testing = df_test_2.drop(['Id'], axis=1)
#df_testing = df_test_2
df_testing.info()


# In[111]:


df_testing.head()


# *Recoding Categorical Features*

# In[112]:


# create dummy variables of all categorical features
test_features = pd.get_dummies(df_testing)
test_features.shape


# *Scaling the Data*

# In[113]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(test_features)
test_features_scaled = scaler.transform(test_features) 


# *Predicting with trained XGBoost*

# In[114]:


#booster.fit(test_features_scaled)
predictions = booster.predict(test_features_scaled)
predictions = pd.DataFrame(predictions)

predictions.columns = ['Prediction']
predictions.head()


# In[115]:


df_submission = pd.merge(df_test_2, predictions, left_index=True, right_index=True)
df_submission = df_submission[['Id', 'Prediction']]
df_submission.head()


# In[116]:


df_submission.to_csv('df_submission.csv', index=False)


# In[ ]:




