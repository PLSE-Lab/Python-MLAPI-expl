#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[ ]:





# ### Loading dataset

# In[ ]:


df = pd.read_csv(r"../input/brasilian-houses-to-rent/houses_to_rent.csv")
df = df.drop(['Unnamed: 0'], axis = 1)
df


# In[ ]:


df.info()


# ### Data Cleaning
# #### Since the coloumns hoa,rent amount,property tax,fire insurance and total have string character which also turn the datatype of these colums to object type, thefore, we must have to remove these string chracters from these coloumns and change the data type too. 

# In[ ]:


df.isna().sum()
df['hoa'] = df['hoa'].str.replace(r'\D', '')

df['rent amount']=df['rent amount'].str.replace(r'\D', '')

df['property tax']=df['property tax'].str.replace(r'\D', '')

df['fire insurance']=df['fire insurance'].str.replace(r'\D', '')

df['total']=df['total'].str.replace(r'\D', '')

df['floor'] = df['floor'].replace('-',np.nan)

####################################################################
df['hoa']=pd.to_numeric(df['hoa'])
df['hoa']= df['hoa'].fillna(df['hoa'].median()).astype('int')

df['rent amount']=pd.to_numeric(df['rent amount'])
df['rent amount'] = df['rent amount'].fillna(df['rent amount'].median()).astype('int')

df['property tax']=pd.to_numeric(df['property tax'])
df['property tax'] = df['property tax'].fillna(df['property tax'].median()).astype('int')

df['fire insurance']=pd.to_numeric(df['fire insurance'])
df['fire insurance'] = df['fire insurance'].fillna(df['fire insurance'].median()).astype('int')

df['floor']= df['floor'].fillna(df['floor'].median()).astype('int')

df['total'] = df['total'].fillna(df['total'].median()).astype('int')

df.info()
df.head()


# ### **Label Encoding**
# As it is obvious that coloumns animal and furniture are categorical variables so we have to convert these variables into numerical values by using labelencoder method

# In[ ]:


from sklearn.preprocessing import LabelEncoder
la = LabelEncoder()
df['animal'] = la.fit_transform(df['animal'])
df['furniture'] =la.fit_transform(df['furniture'])
df.info()


# ### Pairplot visualization

# In[ ]:


sns.pairplot(df)


# ### Correlation analysis 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
corrmat = df.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, annot=True);


# #### Heatmap of correlation matrix indicates that property tax,hoa,rent amount and fire insurance are highly correlated to total,respectively

# ### Distribution plot
# #### By having a look at the pairplot, we can suggest that some of the features shows skewness in the data. Lets check them individually

# In[ ]:


f, axes = plt.subplots(2, 2,figsize=(10,10))
sns.set(style = 'darkgrid')
sns.despine(left=True)
sns.distplot(df['hoa'],ax=axes[0,0])
sns.distplot(df['rent amount'],ax=axes[0,1])
sns.distplot(df['fire insurance'],ax=axes[1, 0])
sns.distplot(df['total'],ax=axes[1, 1])

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)


# In[ ]:


sns.distplot(df['area'])


# In[ ]:


df_new = df.copy()
#df_new


# #### In order to distribute the data normally, we can use log function and then choose an IQR in order to overcome outliers

# In[ ]:


df['area'] = np.log1p(df['area'])
df.hoa = np.log1p(df.hoa)
df['rent amount'] = np.log1p(df['rent amount'])
df['property tax'] = np.log1p(df['property tax'])
df['fire insurance'] = np.log1p(df['fire insurance'])
df.total = np.log1p(df.total)
f, axes = plt.subplots(2, 2,figsize=(10,10))
sns.set(style = 'darkgrid')
sns.despine(left=True)
sns.distplot(df['hoa'],ax=axes[0,0])
sns.distplot(df['rent amount'],ax=axes[0,1])
sns.distplot(df['fire insurance'],ax=axes[1, 0])
sns.distplot(df['total'],ax=axes[1, 1])


plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)


# In[ ]:


sns.distplot(df['area'])


# In[ ]:


df=df[(df['area'] >= 3) & (df['area'] <= 7 )    &(df['hoa'] >= 5) & (df['hoa'] <= 9 )      & (df['property tax'] >= 2) & (df['property tax'] <= 9 )      & (df['total'] >= 7) & (df['total'] <= 11 )]
#df


# In[ ]:


f, axes = plt.subplots(2, 2,figsize=(10,10))
sns.set(style = 'darkgrid')
sns.despine(left=True)
sns.distplot(df['hoa'],ax=axes[0,0])
sns.distplot(df['rent amount'],ax=axes[0,1])
sns.distplot(df['fire insurance'],ax=axes[1, 0])
sns.distplot(df['total'],ax=axes[1, 1])


plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)


# In[ ]:


sns.distplot(df['area'])


# #### The data looks like perfectly distributed which can be seen by their distribution plots and by pairplot diagram as shown in below

# In[ ]:


f, axes = plt.subplots(2, 2,figsize=(10,10))
sns.set(style = 'darkgrid')

g = sns.scatterplot("total", "hoa", data=df,ax=axes[0, 0]
                  
                  
                  )
g = sns.scatterplot("total", "rent amount", data=df,
                  ax=axes[0, 1]
                  
                  )
g = sns.scatterplot("total","property tax", data=df,
                  ax=axes[1, 0]
                  
                  )
g = sns.scatterplot("total","fire insurance", data=df,
                  ax=axes[1, 1]
                  
                  )
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)


# In[ ]:


g = sns.scatterplot("total","area", data=df)


# In[ ]:


sns.pairplot(df)


# #### Since the data has both catagorical and continous features lets plot them individualy 

# ### Boxplots

# In[ ]:


y = df.total
plt.figure(figsize=(10,10))
sns.boxplot(x="city", y="total", palette=["m", "g"], data=df)
plt.title('City and Total Price')


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x="floor", y="total", palette=["m", "g"], data=df)
plt.title('Floor and Total Price')


# In[ ]:


df.columns


# In[ ]:


y = df.total
plt.figure(figsize=(10,10))
sns.boxplot(x="parking spaces", y="total", palette=["m", "g"], data=df)
plt.title('Parking space and Total Price')


# In[ ]:


y = df.total
plt.figure(figsize=(10,10))
sns.boxplot(x="furniture", y="total", palette=["m", "g"], data=df)
plt.title('Furniture and Total Price')


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x="animal", y="total", palette=["m", "g"], data=df)
plt.title('Animal and Total Price')


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x="bathroom", y="total", palette=["m", "g"], data=df)
plt.title('Bathroom and Total Price')


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x="rooms", y="total", palette=["m", "g"], data=df)
plt.title('rooms and Total Price')


# In[ ]:


sns.pairplot(df, x_vars=['area','hoa','rent amount','property tax','fire insurance']             , y_vars='total', kind='reg')


# ### Linear Regression
# #### Linear regression techinque attempts to model the relationship between two variables by fitting a linear equation to observed data. As described before our data has both catagorical as well as continuos variables lets build a model first without catagorical features and then with both features and see the performance of the model.

# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score


# In[ ]:


df.info()
df=df.astype(float)
df.info()


# In[ ]:


x = df.drop(['total','city','rooms','bathroom','parking spaces','floor','animal','furniture'],axis='columns')
y = df.total


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=0)
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
print(model.intercept_)
print(model.coef_)
model.score(X_train,y_train)


# In[ ]:


y_predicted = model.predict(X_test)
#y_predicted


# In[ ]:


cross_val_score(model,X_train,y_train)


# In[ ]:


x2 = df.drop(['total'],axis='columns')
y2 = df.total
X_train,X_test,y_train,y_test = train_test_split(x2,y2,train_size=0.7,random_state=0)
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
print(model.intercept_)
print(model.coef_)
print(model.score(X_train,y_train))
cross_val_score(model,X_train,y_train)


# #### As we can see that having all featured coloums in our training dataset the performance of the model slightly improves with accuracy score of 0.9912.
