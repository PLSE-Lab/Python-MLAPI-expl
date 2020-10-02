#!/usr/bin/env python
# coding: utf-8

# # Importing The Tools 

# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from xgboost import  XGBRegressor


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing The Data and take a tour in the data 

# In[ ]:


train_data = pd.read_csv("../input/train.csv",low_memory= False)
test_data = pd.read_csv("../input/test.csv",low_memory= False)
store_data = pd.read_csv("../input/store.csv",low_memory= False)
test_copy = test_data


# In[ ]:


print("Shape of Train data :", train_data.shape)
print("Shape of Test data :", test_data.shape)
print("Shape of Store data :", store_data.shape)


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


store_data.head(100)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


store_data.isnull().sum().sort_values(ascending = False)


# In[ ]:


store_data['Promo2SinceWeek'].unique()


# In[ ]:


train_data['Store'].unique()


# In[ ]:


train_data['DayOfWeek'].unique()


# In[ ]:


train_data['Open'].unique()


# In[ ]:


train_data['StateHoliday'].unique()


# In[ ]:


train_data['Promo'].unique()


# In[ ]:


train_data['Store'].unique()


# In[ ]:


store_data['CompetitionOpenSinceMonth'].unique()


# In[ ]:


print(sum(train_data["Open"] == 0))
print(sum(train_data["Open"] == 1))


# In[ ]:


print(sum(test_data["Open"] == 0))
print(sum(test_data["Open"] == 1))


# In[ ]:


print(sum(train_data["StateHoliday"] == 'a'))
print(sum(train_data["StateHoliday"] == 'b'))
print(sum(train_data["StateHoliday"] == 'c'))
print(sum(train_data["StateHoliday"] == 0))


# In[ ]:


plt.plot(train_data['DayOfWeek'],train_data['Customers'])


# In[ ]:


train_data[['Sales','Customers','Promo','SchoolHoliday']].corr(method='pearson')


# In[ ]:


train_data['Mon'] = train_data["Date"].apply(lambda x : int(x[5:7]))
train_data['Yr'] = train_data["Date"].apply(lambda x : int(x[:4]))
train_data["HolidayBin"] = train_data.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})


# In[ ]:


test_data['Mon'] = test_data["Date"].apply(lambda x : int(x[5:7]))
test_data['Yr'] = test_data["Date"].apply(lambda x : int(x[:4]))
test_data["HolidayBin"] = test_data.StateHoliday.map({"0": 0, "a": 1, "b": 1, "c": 1})


# In[ ]:


train_data = train_data.merge(store_data)
test_data =test_data.merge(store_data)


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


train_data.isnull().sum().sort_values(ascending= False)


# In[ ]:


test_data.isnull().sum().sort_values(ascending= False)


# In[ ]:


test_data[test_data['Open'].isnull()]


# In[ ]:


for i in train_data['Promo2SinceWeek'].unique() :
    print(i ,':', sum(train_data['Promo2SinceWeek'] == i ))


# In[ ]:


for i in train_data['CompetitionOpenSinceMonth'].unique() :
    print(i ,':', sum(train_data['CompetitionOpenSinceMonth'] == i ))


# In[ ]:


for i in train_data['Promo2SinceYear'].unique() :
    print(i ,':', sum(train_data['Promo2SinceYear'] == i ))


# In[ ]:


for i in train_data['CompetitionOpenSinceYear'].unique() :
    print(i ,':', sum(train_data['CompetitionOpenSinceYear'] == i ))


# # Drop Some Data

# In[ ]:


train_data = train_data.drop(['Customers', 'Store','Date','StateHoliday'],axis= 1 )
test_data = test_data.drop(['Date','StateHoliday','Store','Id'],axis= 1 )


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


sum(train_data['Open'] == 0)


# In[ ]:


train_data = train_data.drop(train_data[train_data['Open'] == 0].index.tolist())


# In[ ]:


sum(train_data['Open'] == 0)


# In[ ]:


train_data.shape


# In[ ]:


train_data[train_data['HolidayBin'].isnull()]


# # Missing Values Cleaning 

# In[ ]:


train_data['CompetitionOpenSinceMonth'] = train_data['CompetitionOpenSinceMonth'].fillna(9.0)
train_data['HolidayBin'] = train_data['HolidayBin'].fillna(0)
train_data['Promo2SinceWeek'] = train_data['Promo2SinceWeek'].fillna(40.0)
train_data['Promo2SinceYear'] = train_data['Promo2SinceYear'].fillna(2012.0)
train_data['CompetitionOpenSinceYear'] = train_data['CompetitionOpenSinceYear'].fillna(2012.0)
train_data['CompetitionDistance'] = train_data['CompetitionDistance'].fillna(train_data['CompetitionDistance'].mean())

train_data.isnull().sum().sort_values(ascending = False)


# In[ ]:


test_data['Open'] = test_data['Open'].fillna(1)
test_data['CompetitionOpenSinceMonth'] = test_data['CompetitionOpenSinceMonth'].fillna(9.0)
test_data['CompetitionDistance'] = test_data['CompetitionDistance'].fillna(train_data['CompetitionDistance'].mean())
test_data['CompetitionOpenSinceYear'] = test_data['CompetitionOpenSinceYear'].fillna(2012.0)
test_data['Promo2SinceWeek'] = test_data['Promo2SinceWeek'].fillna(40.0)
test_data['Promo2SinceYear'] = test_data['Promo2SinceYear'].fillna(2012.0)

test_data.isnull().sum().sort_values(ascending = False)


# In[ ]:


train_data.shape


# In[ ]:


sum(train_data['Sales'] < 0 )


# In[ ]:


test_data.shape


# In[ ]:


train_data.head(100)


# # Categorcal Data

# In[ ]:


categorical_train = train_data.columns.tolist()
print(categorical_train)
train_data[categorical_train].corr(method='pearson')


# ### Concatenate train and test data to be the same number of categorical labels 

# In[ ]:


train_features = train_data.drop(['Open'],axis = 1)
categorical_train = train_features.columns.tolist()
print(categorical_train)
train_data[categorical_train].corr(method='pearson')
train_features = train_data.drop(['Sales'],axis = 1)
full_features = pd.concat([train_features,test_data],ignore_index= True)
print(train_features.shape)
print(test_data.shape)


# In[ ]:


full_features.head()


# In[ ]:


full_features.shape


# In[ ]:


full_features = pd.get_dummies(full_features,columns= ['HolidayBin','Assortment','StoreType'])


# In[ ]:


full_features.shape


# In[ ]:


full_features = full_features.drop('PromoInterval',axis = 1)


# ### Split the train features and test features from full features data frame

# In[ ]:


train_features = full_features.iloc[:844392,:].values
test_data = full_features.iloc[844392:,:].values
train_sales = train_data['Sales'].values


# In[ ]:


print(train_features.shape)
print(train_sales.shape)
print(test_data.shape)


# # Get the log of Sales

# In[ ]:


#train_sales = np.log(train_sales)


# # Machine Learning Model

# In[ ]:


xgboost = XGBRegressor(learning_rate=0.009, n_estimators=500,
                                     max_depth=10, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006, random_state=42)


# In[ ]:


xgboost.fit(train_features,train_sales)


# In[ ]:


predictions = xgboost.predict(test_data)


# # Get the exp of predections

# In[ ]:


#preds = np.exp(predictions)


# # Submission File

# In[ ]:


pred_df = pd.DataFrame({"Id": test_copy["Id"], 'Sales': predictions})
pred_df.to_csv("xgboost_4_submission.csv", index=False)


# In[ ]:




