#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
pd.pandas.set_option('display.max_columns',None)


# In[ ]:


data=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')


# In[ ]:


data.head()


# In[ ]:


data.drop(labels=['App','Last Updated','Current Ver','Android Ver'],axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.drop('Genres',axis=1,inplace=True)


# In[ ]:


data['Category'].value_counts()


# In[ ]:


data.head()


# In[ ]:


data['Category'].unique()


# In[ ]:


data.isnull().sum()


# In[ ]:


data['Rating'] = data['Rating'].fillna(data['Rating'].median())
data['Content Rating']=data['Content Rating'].fillna(data['Content Rating'].mode())


# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.dropna(inplace=True)


# In[ ]:


data.isnull().sum()


# In[ ]:


sns.catplot(x='Category',y='Rating',data=data.sort_values('Rating',ascending=False),kind='boxen',height=6,aspect=3)


# In[ ]:


data.head()


# In[ ]:


data.dtypes


# In[ ]:


data['Reviews']=data['Reviews'].astype(int)


# In[ ]:


data.dtypes


# In[ ]:


data.tail()


# In[ ]:


#scaling and cleaning size of installation
def change_size(size):
    if 'M' in size:
        x = size[:-1]
        x = float(x)*1000000
        return(x)
    elif 'k' == size[-1:]:
        x = size[:-1]
        x = float(x)*1000
        return(x)
    else:
        return None

data["Size"] = data["Size"].map(change_size)

#filling Size which had NA
data.Size.fillna(method = 'ffill', inplace = True)


# In[ ]:


data.head()


# In[ ]:


data['Type'].value_counts()    


# In[ ]:


#Cleaning no of installs classification
data['Installs'] = [int(i[:-1].replace(',','')) for i in data['Installs']]


# In[ ]:


data.head()


# In[ ]:


Category=data['Category']
Category=pd.get_dummies(Category,drop_first=True)


# In[ ]:


Category.head()


# In[ ]:


Category.tail()


# In[ ]:


data['Content Rating'].value_counts()


# In[ ]:


Content =data['Content Rating']
Content=pd.get_dummies(Content,drop_first=True)
Content.head()


# In[ ]:


#Cleaning prices
def price_clean(price):
    if price == '0':
        return 0
    else:
        price = price[1:]
        price = float(price)
        return price

data['Price'] = data['Price'].map(price_clean).astype(float)


# In[ ]:


Type=data['Type']
Type=pd.get_dummies(Type,drop_first=True)
Type.head()


# In[ ]:


data.head()


# In[ ]:


train_data=pd.concat([data,Category,Type,Content],axis=1)


# In[ ]:


train_data.head()


# In[ ]:


train_data.drop(labels=['Category','Type','Content Rating'],axis=1,inplace=True)


# In[ ]:


train_data.head()


# In[ ]:


train_data.columns


# In[ ]:


X=train_data.loc[:,['Reviews', 'Size', 'Installs', 'Price', 'AUTO_AND_VEHICLES',
       'BEAUTY', 'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'COMMUNICATION',
       'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FAMILY', 'FINANCE',
       'FOOD_AND_DRINK', 'GAME', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME',
       'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'MAPS_AND_NAVIGATION', 'MEDICAL',
       'NEWS_AND_MAGAZINES', 'PARENTING', 'PERSONALIZATION', 'PHOTOGRAPHY',
       'PRODUCTIVITY', 'SHOPPING', 'SOCIAL', 'SPORTS', 'TOOLS',
       'TRAVEL_AND_LOCAL', 'VIDEO_PLAYERS', 'WEATHER', 'Paid', 'Everyone',
       'Everyone 10+', 'Mature 17+', 'Teen', 'Unrated']]


# In[ ]:


y=train_data.iloc[:,0]


# In[ ]:


y.head()


# In[ ]:


X.head()


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor
reg=ExtraTreesRegressor()
reg.fit(X,y)


# In[ ]:


print(reg.feature_importances_)


# In[ ]:


plt.figure(figsize=(12,8))
feat_importances=pd.Series(reg.feature_importances_,index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rg=RandomForestRegressor()
rg.fit(X_train,y_train)


# In[ ]:


y_pred=rg.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


rg.score(X_train,y_train)


# In[ ]:


rg.score(X_test,y_test)


# In[ ]:


sns.distplot(y_test-y_pred)


# In[ ]:


from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,y_pred))
print('MSE:',metrics.mean_squared_error(y_test,y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[ ]:


metrics.r2_score(y_test,y_pred)


# In[ ]:




