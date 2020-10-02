#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd


# In[ ]:


dataset = pd.read_csv('../input/bigmart-sales-data/Train.csv')


# In[ ]:


dataset.describe()


# In[ ]:


dataset.info()


# In[ ]:


dataset.shape


# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset.head()


# In[ ]:


print('The unique items in the dataset is',dataset.Item_Identifier.nunique())
print('Total Items in the dataset is',dataset.Item_Identifier.count())


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


sns.distplot(dataset.Item_Outlet_Sales,bins = 50)
#The sales is skewed to the left 


# In[ ]:


#squareroot transformation makes the target variable normal
sns.distplot(np.sqrt(dataset.Item_Outlet_Sales),bins = 50)


# In[ ]:


corr = dataset.corr()
corr


# In[ ]:


corr.Item_Outlet_Sales.sort_values()
#Item_MRP is most correlated and Item_visiblity is negatively correlated implies lesser visiblity more sales


# In[ ]:


fig,ax = plt.subplots(figsize = (10,8))
sns.heatmap(corr,vmax=.8)


# In[ ]:


categorical_features = dataset.select_dtypes(include=[object]).columns


# In[ ]:


categorical_features


# In[ ]:


dataset.Item_Fat_Content.value_counts().plot(kind='bar')
#LF and reg has to be replaced with low fat and regular


# In[ ]:


sns.countplot(dataset.Item_Type)
plt.xticks(rotation = 90)


# In[ ]:


sns.countplot(dataset.Outlet_Size)
#In the dataset there are more medium and small stores 


# In[ ]:


sns.countplot(dataset.Outlet_Location_Type)
#More stores in the tier 3


# In[ ]:


sns.countplot(dataset.Outlet_Type)
plt.xticks(rotation=90)
#supermarket type 1 is most and rest are very lesser in number compared to it


# In[ ]:


sns.pairplot(dataset)
#Distribution of the target variable with the numerical features


# In[ ]:


plt.scatter(dataset.Item_Visibility, dataset.Item_Outlet_Sales, alpha = 0.3)
#Item visiblity is negatively correlated viewing the distribution in the below plot


# In[ ]:


dataset.pivot_table(index = 'Item_Type',values = 'Item_Outlet_Sales',aggfunc = np.median).plot(kind='bar',figsize = (10,8)),


# In[ ]:


dataset.pivot_table(index = 'Outlet_Establishment_Year',values = 'Item_Outlet_Sales',aggfunc = np.median).plot(kind='bar')
#Year 1998 has the lowest sales otherwise rest of the values are same in the dataset


# In[ ]:


dataset.groupby('Item_Fat_Content').median()['Item_Outlet_Sales'].plot(kind='bar')


# In[ ]:


fig,axes = plt.subplots(figsize=(10,8))
sns.barplot(x=dataset.Outlet_Identifier,y = dataset.Item_Outlet_Sales,hue =dataset['Outlet_Type'])
plt.xticks(rotation = 90)


# In[ ]:


dataset.pivot_table(index = 'Outlet_Identifier',values = 'Item_Outlet_Sales',columns='Outlet_Type',aggfunc = np.median).plot(kind='bar',figsize= (12,8))
plt.xticks(rotation = 0)


# In[ ]:


dataset.pivot_table(index = 'Outlet_Size',values = 'Item_Outlet_Sales',aggfunc = np.median).plot(kind='bar')
#Medium size has more sales based on the below visualisation


# In[ ]:


dataset.pivot_table(index = 'Outlet_Type',values = 'Item_Outlet_Sales',aggfunc = np.median).plot(kind='bar')
#Supermarket type 3 has highest impact on the sales


# In[ ]:


dataset.pivot_table(index='Outlet_Location_Type', values="Item_Outlet_Sales", aggfunc=np.median).plot(kind='bar')


# In[ ]:


#Import mode function:
from scipy.stats import mode
#Determing the mode for each
outlet_size_mode = dataset.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=lambda x:x.mode())
outlet_size_mode


# In[ ]:


def impute_size_mode(cols):
    Size = cols[0]
    Type = cols[1]
    if pd.isnull(Size):
        return outlet_size_mode.loc['Outlet_Size'][outlet_size_mode.columns == Type][0]
    else:
        return Size
print ('Orignal #missing: %d'%sum(dataset['Outlet_Size'].isnull()))
dataset['Outlet_Size'] = dataset[['Outlet_Size','Outlet_Type']].apply(impute_size_mode,axis=1)
print ('Final #missing: %d'%sum(dataset['Outlet_Size'].isnull()))


# In[ ]:


#Determine the average weight per item:
#item_avg_weight = dataset.pivot_table(values='Item_Weight', index='Item_Identifier')


# In[ ]:


#Get a boolean variable specifying missing Item_Weight values
#miss_bool = dataset['Item_Weight'].isnull() 
#miss_bool


# In[ ]:


#Impute data and check #missing values before and after imputation to confirm
#dataset.loc[miss_bool,'Item_Weight'] = dataset.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight[x])


# In[ ]:


dataset['Item_Weight'][dataset['Item_Weight'].isnull()]=dataset['Item_Weight'].mean()


# In[ ]:


dataset.info()


# In[ ]:


dataset.loc[dataset['Item_Visibility']==0.000000,'Item_Visibility']=np.mean(dataset['Item_Visibility'])


# In[ ]:


dataset['Outlet_Years'] = 2013 - dataset['Outlet_Establishment_Year']
dataset['Outlet_Years'].describe()


# In[ ]:


dataset['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}, inplace= True)


# In[ ]:


dataset.drop(['Item_Identifier','Outlet_Establishment_Year','Outlet_Identifier'],axis=1,inplace = True)


# In[ ]:


X = dataset.drop(['Item_Outlet_Sales'],axis=1)
y= dataset['Item_Outlet_Sales']


# In[ ]:


X.info()


# In[ ]:


X = pd.get_dummies(X)


# In[ ]:


from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[ ]:


Regressor.fit(X_train,y_train)


# In[ ]:


y_pred = Regressor.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)


# In[ ]:


np.sqrt(mse)


# In[ ]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(Regressor,X_train,y_train,cv=10,scoring='neg_mean_squared_error')
np.mean(np.sqrt(-score))


# In[ ]:


from sklearn.linear_model import Ridge
r=Ridge(alpha=0.05,solver='cholesky')
r.fit(X_train,y_train)
predict_r=r.predict(X_test)
mse=mean_squared_error(y_test,predict_r)
r_score=np.sqrt(mse)
r_score


# In[ ]:


r=Ridge(alpha=0.05,solver='cholesky')
score=cross_val_score(Regressor,X_train,y_train,cv=10,scoring='neg_mean_squared_error')
r_score_cross=np.sqrt(-score)
np.mean(r_score_cross),np.std(r_score_cross)


# In[ ]:




