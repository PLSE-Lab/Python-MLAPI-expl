#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_data = pd.read_csv("../input/big-mart-sales-dataset/Train_UWu5bXk.csv")
test_data = pd.read_csv("../input/big-mart-sales-dataset/Test_u94Q5KV.csv")


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


print(train_data.shape)
print(test_data.shape)


# In[ ]:


train_data.info()


# In[ ]:


['Item_Fat_Content', 'Item_Type ', ' Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']


# In[ ]:


train_data.isnull().sum()


# In[ ]:


data = train_data.append(test_data, sort=False)


# In[ ]:


data.isnull().sum()


# In[ ]:


train_data.describe()


# In[ ]:


train_data.describe().columns


# In[ ]:


for i in train_data.describe().columns:
    sns.distplot(data[i].dropna())
    plt.show()


# In[ ]:


for i in train_data.describe().columns:
    sns.boxplot(data[i].dropna())
    plt.show()


# In[ ]:


sns.boxplot(data['Item_Visibility'])


# In[ ]:


data['Item_Visibility'].describe()


# In[ ]:


sns.boxplot(y=data['Item_Weight'], x=data['Outlet_Identifier'])
plt.xticks(rotation='vertical')


# In[ ]:


data['Item_Fat_Content'].value_counts()


# In[ ]:


data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat','reg':'Regular', 'low fat':'Low Fat'})


# In[ ]:


data.groupby('Item_Identifier')['Item_Weight'].mean().head(5)


# In[ ]:


for i in data.groupby('Item_Identifier')['Item_Weight'].mean().index:
    data.loc[data.loc[:,'Item_Identifier']==i, 'Item_Weight'] = data.groupby('Item_Identifier')['Item_Weight'].mean()[i]


# In[ ]:


data['Outlet_Type'].value_counts()


# In[ ]:


data.Outlet_Size[data['Outlet_Type']=='Grocery Store'].value_counts()


# In[ ]:


data.Outlet_Size[data['Outlet_Type']=='Supermarket Type2'].value_counts()


# In[ ]:


data.Outlet_Size[data['Outlet_Type']=='Supermarket Type1'].value_counts()


# In[ ]:


data.Outlet_Size[data['Outlet_Type']=='Supermarket Type3'].value_counts()


# In[ ]:


data.Outlet_Size.fillna(data.Outlet_Size[data['Outlet_Type']=='Grocery Store'].mode()[0], inplace=True)


# In[ ]:


data.Outlet_Size.fillna(data.Outlet_Size[data['Outlet_Type']=='Supermarket Type1'].mode()[0], inplace=True)


# In[ ]:


data.Outlet_Size.fillna(data.Outlet_Size[data['Outlet_Type']=='Supermarket Type2'].mode()[0], inplace=True)


# In[ ]:


data.Outlet_Size.fillna(data.Outlet_Size[data['Outlet_Type']=='Supermarket Type3'].mode()[0], inplace=True)


# In[ ]:


for i in data.groupby('Item_Identifier')['Item_Visibility'].mean().index:
    data.loc[data.loc[:,'Item_Identifier']==i, 'Item_Visibility']=data.groupby('Item_Identifier')['Item_Visibility'].mean()[i]


# In[ ]:


data['Outlet_Establishment_Year']=2013-data['Outlet_Establishment_Year']


# In[ ]:


data


# In[ ]:


data.isnull().sum()


# In[ ]:


train_data=data.dropna()


# In[ ]:


test_data=data[data['Item_Outlet_Sales'].isnull()]
test_data.drop('Item_Outlet_Sales', axis=1, inplace=True)


# In[ ]:


sns.boxplot(train_data['Item_Visibility'])


# In[ ]:


train_data['Item_Visibility'].describe()


# In[ ]:


print(test_data.shape)
print(train_data.shape)


# In[ ]:


len(train_data)


# In[ ]:


len(test_data)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


categorical_list = ['Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type','Outlet_Establishment_Year']


# In[ ]:


le = LabelEncoder()
for i in categorical_list:
    train_data[i] =le.fit_transform(train_data[i])
    train_data[i]=train_data[i].astype('category')
    test_data[i]=le.fit_transform(test_data[i])
    test_data[i]=test_data[i].astype('category')


# In[ ]:


data


# In[ ]:


test_data.head()


# In[ ]:


train_data.head()


# In[ ]:


train_data.corr()


# In[ ]:


from sklearn.linear_model import LinearRegression as LR


# In[ ]:


lm = LR(normalize=True)


# In[ ]:


lm.fit(train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),train_data['Item_Outlet_Sales'])


# In[ ]:


train_data


# In[ ]:


y_train=train_data['Item_Outlet_Sales']
X_train=train_data.drop('Item_Outlet_Sales', axis=1)


# In[ ]:


train = train_data.drop(['Item_Outlet_Sales'], axis=1)
predictions=train_data['Item_Outlet_Sales']
out=[]
LM_model=LR(normalize=True)
for i in range(len(test_data)):
    LM_fit=LM_model.fit(train.drop(['Outlet_Identifier','Item_Identifier'], axis=1), predictions)
    Output=LM_fit.predict(test_data.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)[test_data.index==i])
    out.append(Output)
    train.append(test_data[test_data.index==i])
    predictions.append(pd.Series(Output))


# In[ ]:


len(out)


# In[ ]:


len(test_data)


# In[ ]:


outp=np.vstack(out)


# In[ ]:


ansp = pd.Series(data = outp[:,0], index=test_data.index, name='Item_Outlet_Sales')


# In[ ]:


outp_df=pd.DataFrame([test_data['Item_Identifier'], test_data['Outlet_Identifier'], ansp]).T


# In[ ]:


outp_df.to_csv('UploadLMP.csv', index= False)


# In[ ]:


mod1_train_pred=lm.predict(train_data.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'], axis=1))


# In[ ]:


from sklearn import metrics
from math import sqrt


# In[ ]:


sqrt(metrics.mean_squared_error(train_data['Item_Outlet_Sales'], mod1_train_pred))/np.mean(train_data['Item_Outlet_Sales'])


# In[ ]:


from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score


# In[ ]:


ann=MLPRegressor(activation='relu',alpha=2.0,learning_rate='adaptive',warm_start=True,hidden_layer_sizes=(2500,),max_iter=1000)


# In[ ]:


ann.fit(train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1),train_data['Item_Outlet_Sales'])


# In[ ]:


ann_train_pred=ann.predict(train_data.drop(['Item_Outlet_Sales','Outlet_Identifier','Item_Identifier'],axis=1))


# In[ ]:


r2_score(train_data['Item_Outlet_Sales'],ann_train_pred)


# In[ ]:


sqrt(metrics.mean_squared_error(train_data['Item_Outlet_Sales'],ann_train_pred))/np.mean(train_data['Item_Outlet_Sales'])


# In[ ]:


ann_pred=ann.predict(test_data.drop(['Item_Identifier','Outlet_Identifier'],axis=1))


# In[ ]:


ann_ans=pd.Series(data=ann_pred,index=test_data.index,name='Item_Outlet_Sales')


# In[ ]:


ann_out=pd.DataFrame([test_data['Item_Identifier'],test_data['Outlet_Identifier'],ann_ans]).T


# In[ ]:


ann_out.to_csv('Uploadann.csv',index=False)


# In[ ]:





# In[ ]:




