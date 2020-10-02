#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


train_df = pd.read_csv('../input/train_2016_v2.csv')


# In[3]:


# No null entries - nothing to fill
train_df.info()


# In[4]:


train_df.columns


# the training data frame, only has the logerror for the different parcel IDs along with the transaction date. 

# In[5]:


train_df.head()


# In[6]:


## there are 352 unique dates
train_df['transactiondate'].nunique()


# In[7]:


train_df['logerror'].describe()


# The values in the 4 or -4 range look unreal, given this is a log value - needs some cleaning up

# In[8]:


import seaborn as sns


# In[9]:


sns.boxplot(train_df['logerror'])


# In[10]:


train_df[train_df['logerror'] < -1].count()


# There are only a few results in this range though

# In[11]:


import re


# Creating separate columns for month, year and date

# In[12]:


train_df['year'] = train_df['transactiondate'].apply(lambda X: X.split('-')[0])


# In[13]:


train_df['month'] = train_df['transactiondate'].apply(lambda X: X.split('-')[1])
train_df['day'] = train_df['transactiondate'].apply(lambda X: X.split('-')[2])


# In[14]:


train_df.head()


# In[15]:


del train_df['transactiondate']


# In[16]:


train_df.groupby(['year', 'month'])['parcelid'].aggregate('count')


# Looks like April to september is the most crowded

# In[17]:


prop_df = pd.read_csv('../input/properties_2016.csv')


# In[18]:


prop_df.head()


# In[19]:


prop_df.columns


# In[20]:


prop_df.info()


# In[21]:


train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')


# In[22]:


train_df.head()


# In[23]:


train_df.isnull().sum()


# In[24]:


high_null_cols_df = pd.DataFrame((train_df.isnull().sum() > (train_df.shape[0] / 2)).reset_index())


# In[25]:


high_null_cols_df.columns = ['category', 'high_null']


# In[26]:


high_null_cols = []
for i in range (0, high_null_cols_df.shape[0]):
    if (high_null_cols_df.iloc[i]['high_null'] == True):
        high_null_cols.append(high_null_cols_df.iloc[i]['category'])
    else:
        pass


# In[27]:


high_null_cols


# In[28]:


train_df.drop(high_null_cols, axis=1, inplace=True)
##drop all columns with too many nulls


# In[29]:


train_df['error'] = np.exp(train_df['logerror'])


# In[30]:


train_df.isnull().sum()


# In[31]:


train_df.groupby('bedroomcnt')['error'].mean()


# In[32]:


train_df.groupby('bathroomcnt')['error'].mean()


# In[33]:


train_df['latitude_scaled'] = np.rint(train_df['latitude']/10000)


# In[34]:


train_df['longitude_scaled'] = np.rint(train_df['longitude']/100000)


# In[35]:


train_df['year'] = train_df['year'].apply(lambda X: float(X))
train_df['month'] = train_df['month'].apply(lambda X: float(X))
train_df['day'] = train_df['day'].apply(lambda X: float(X))


# In[36]:


train_df['propertycountylandusecode'].value_counts()


# 0100 seems to be the most popular code - use that to replace the null element

# In[37]:


train_df['propertycountylandusecode'].fillna(value='0100', inplace=True)


# Replacing objects with numbers

# In[38]:


prop_land_codes = train_df['propertycountylandusecode'].unique()
prop_land_codes_zf = dict(zip(prop_land_codes, range(len(prop_land_codes))))


# In[39]:


train_df['propertycountylandusecode'].replace(prop_land_codes_zf, inplace=True)


# In[40]:


prop_zone_desc = train_df['propertyzoningdesc']


# In[41]:


del train_df['propertyzoningdesc']


# In[42]:


train_df.isnull().sum()


# Building quality : 1 being the best. 2 and 3 seem to be missing entirely ?? 
#     Option 1 : fill half of NA VALUES WITH 7 AND THE OTHER HALF WITH 4
#     Option 2 : fill all of them with 7
#     Option 3 : throw the column out

# In[43]:


train_df['buildingqualitytypeid'].value_counts()


# 'calculatedbathnbr' - does not seem any different from bathroom count, delete it 

# In[44]:


del train_df['calculatedbathnbr']


# finishedsquarefeet12 means the same thing as total square feet

# In[45]:


del train_df['finishedsquarefeet12']


# How has the house area changed with time ?? 

# In[46]:


train_df['yearbuilt'].value_counts().head()


# Bathroomcnt and fullbathcnt - are these any different ? If yes, does the different matter ?? 

# In[47]:


train_df[train_df['bathroomcnt'] == train_df['fullbathcnt'] + 0]['error'].mean()


# In[48]:


train_df[train_df['bathroomcnt'] == train_df['fullbathcnt'] + 0.5]['error'].mean()


# In[49]:


train_df[train_df['bathroomcnt'] == train_df['fullbathcnt'] + 1]['error'].mean()


# Looks like it kind of does.

# In[50]:


(train_df['bathroomcnt'] - train_df['fullbathcnt']).value_counts()


# In[51]:


train_df['fullbathcnt'].fillna(train_df['bathroomcnt'], inplace=True)


# No clue what this means !! 

# In[52]:


del train_df['censustractandblock']


# In[53]:


tax_df = pd.DataFrame(train_df.groupby('roomcnt')['taxamount'].aggregate('mean').reset_index())


# In[54]:


tax_df


# In[55]:


train_df = pd.merge(train_df, tax_df, on='roomcnt', how='left')


# In[56]:


train_df['taxamount_x'].fillna(train_df['taxamount_y'], inplace=True)


# In[57]:


del train_df['taxamount_y']


# In[58]:


train_df.isnull().sum()


# In[59]:


del train_df['buildingqualitytypeid']


# In[60]:


train_df['heatingorsystemtypeid'].fillna(method='ffill', inplace=True)


# In[61]:


train_df['lot_to_house_ratio'] = train_df['lotsizesquarefeet']/train_df['calculatedfinishedsquarefeet']


# In[62]:


import matplotlib.pyplot as plt


# In[63]:


train_df['lot_to_house_ratio'].describe()


# Use the 50% value to fill empty sizes

# In[64]:


train_df['lotsizesquarefeet'].fillna(4.578765 * train_df['calculatedfinishedsquarefeet'], inplace=True)


# In[65]:


train_df.isnull().sum()


# In[66]:


train_df.groupby('unitcnt')['calculatedfinishedsquarefeet'].aggregate('mean')


# In[67]:


def return_unitcnt(X):
    if(X > 2493):
      return 3.0
    elif(X > 1976.0):
      return 2.0
    else:
       return 1.0


# In[68]:


train_df['temp'] = train_df['calculatedfinishedsquarefeet'].apply(lambda X:return_unitcnt(X))


# In[69]:


train_df['unitcnt'].fillna(train_df['temp'], inplace=True)


# In[70]:


train_df.isnull().sum()


# In[71]:


del train_df['lot_to_house_ratio']


# In[72]:


del train_df['temp']


# In[73]:


train_df['regionidcity'].value_counts().head(5)


# In[74]:


train_df['regionidcity'].fillna(12447.0, inplace=True)


# In[75]:


train_df.groupby('bedroomcnt')['calculatedfinishedsquarefeet'].aggregate('mean')


# In[76]:


train_df[train_df['calculatedfinishedsquarefeet'].isnull()]['bedroomcnt'].value_counts()


# Most of the nulls are the 0 bedroom cases

# In[77]:


def fill_sqft(X):
    if(X == 0):
        return 1914
    elif(X == 1):
        return 819
    elif(X == 2):
        return 1209
    else:
        return 1633


# In[78]:


train_df['temp'] = train_df['bedroomcnt'].apply(lambda X:fill_sqft(X))


# In[79]:


train_df['calculatedfinishedsquarefeet'].fillna(train_df['temp'], inplace=True)


# In[80]:


del train_df['temp']


# In[81]:


train_df.isnull().sum()


# In[82]:


train_df.groupby('bedroomcnt')['lotsizesquarefeet'].aggregate('mean').head(5)


# In[83]:


train_df[train_df['lotsizesquarefeet'].isnull()]['bedroomcnt'].value_counts()


# In[84]:


def lotSqFt(X):
    if(X == 0):
        return 55528
    else:
        return 21658


# In[85]:


train_df['temp'] = train_df['bedroomcnt'].apply(lambda X:lotSqFt(X))


# In[86]:


train_df['lotsizesquarefeet'].fillna(train_df['temp'], inplace=True)


# In[87]:


train_df['yearbuilt'].fillna(method='bfill', inplace=True)


# In[88]:


train_df[train_df['regionidzip'].isnull()]['regionidcounty'].value_counts()


# In[89]:


train_df[train_df['regionidcounty'] == 2061.0]['regionidzip'].value_counts().head(5)


# In[90]:


train_df[train_df['regionidcounty'] == 1286.0]['regionidzip'].value_counts().head(5)


# In[91]:


def fill_zip(X):
    if(X == 2061.0):
        return 97118.0
    else:
        return 96987.0


# In[92]:


del train_df['temp']


# In[93]:


train_df['temp'] = train_df['regionidcounty'].apply(lambda X:fill_zip(X))


# In[94]:


train_df['regionidzip'].fillna(train_df['temp'], inplace=True)


# In[95]:


import seaborn as sns


# In[96]:


sns.lmplot(data=train_df, x='taxvaluedollarcnt', y='structuretaxvaluedollarcnt')


# It looks like taxvaluedollarcnt and structuretaxvaluedollarcnt might be linearly dependent - deleting one of them

# In[97]:


del train_df['structuretaxvaluedollarcnt']


# In[98]:


tax_val_mean = train_df['taxvaluedollarcnt'].mean()


# In[99]:


train_df['taxvaluedollarcnt'].fillna(train_df['taxvaluedollarcnt'].mean(), inplace=True)


# In[100]:


train_df['landtaxvaluedollarcnt'].fillna(train_df['landtaxvaluedollarcnt'].mean(), inplace=True)


# In[101]:


train_df['age'] = 2016 - train_df['yearbuilt']


# In[102]:


del train_df['yearbuilt']


# In[103]:


from sklearn.model_selection import train_test_split


# In[104]:


X = train_df.drop(['logerror', 'error', 'temp'], axis=1)


# In[105]:


Y = train_df['error']


# In[106]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# In[107]:


from sklearn.linear_model import LinearRegression


# In[108]:


lm = LinearRegression()


# In[110]:


lm.fit(X_train, Y_train)


# In[111]:


print(lm.coef_)


# In[112]:


error_pred = lm.predict(X_test)


# In[113]:


plt.scatter(Y_test, error_pred)
plt.xlabel('Y_test')
plt.ylabel('Pred_y')
plt.show()


# In[114]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(Y_test, error_pred))
print('MSE:', metrics.mean_squared_error(Y_test, error_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, error_pred)))


# In[117]:


plt.scatter(np.log(Y_test), np.log(error_pred))
plt.xlabel('Y_test')
plt.ylabel('Pred_y')
plt.show()


# In[ ]:




