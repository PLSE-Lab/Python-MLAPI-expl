#!/usr/bin/env python
# coding: utf-8

# =======================================================================================================
# 
# 
# **In this post we are going to explore dataset of car prices to predict the best price a car can get.The price of car depends on many factors**
# * Make and Model of the car
# * Type of Transmission
# * Fuel type and so on
# 
# ![Image Credit : Unsplash](https://images.unsplash.com/photo-1505691730847-e62da813ed0e?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=889&q=80)
# 
# =======================================================================================================
# 
# Lets look at our data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

sns.set_style({ 
    'axes.spines.bottom': False,
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'xtick.bottom': False,
    'ytick.left': False
})


# In[ ]:


data = pd.read_csv('../input/car data.csv')
data.head()


# In[ ]:


data.isnull().sum()


# **Lucky there are no missing values**

# In[ ]:


data.info()


# In[ ]:


data.describe(include='all')


# In[ ]:


sns.pairplot(data)


# **Now Lets Check few Hypothesis**
# 
# * **Hypothesis - 1 :**Less Driven cars have high selling price
# * **Hypothesis - 2 :**Latest Cars will have high selling price
# * **Hypothesis - 3 :**Automatic Transmission Cars have high selling price
# 
# Lets check our hypothesis
# 
# ## Hypothesis - 1

# In[ ]:


fig,ax1 = plt.subplots(figsize=(15,10))
sns.scatterplot(x='Kms_Driven',y='Selling_Price',data=data,ax=ax1)


# **First Hypothesis turns out to be true**
# 
# ## Hypothesis - 2

# In[ ]:


data['Y_S_L'] = 2019 - data.Year
data.head()


# In[ ]:


sns.catplot(x='Y_S_L',y='Selling_Price',data=data,kind='point',height=8,aspect=2)
sns.despine(left=True,bottom=True)


# **Second hypothesis is also true**
# 
# ## Hypothesis - 3

# In[ ]:


data.Transmission.value_counts()


# In[ ]:


data.loc[:,['Transmission','Selling_Price']].sort_values(by=['Selling_Price'],ascending =False)['Transmission'].head(15).value_counts().plot.pie(figsize=(15,15),subplots=True, autopct='%.1f%%',explode=[0,.08],shadow=True)


# **Top 15 cars with high sale price have Automatic transmission.That proves our 3rd hypothesis**
# 
# **Now Lets get in to prediction**

# In[ ]:


le = LabelEncoder()
df = pd.get_dummies(data['Fuel_Type'],prefix='FT',drop_first=True)
data['Seller_Type'] = le.fit_transform(data['Seller_Type'])
data['Transmission'] = le.fit_transform(data['Transmission'])
data = pd.concat([data,df],axis=1)
data.drop(['Fuel_Type'],axis=1,inplace=True)
data.head()


# In[ ]:


fig,ax1 = plt.subplots(figsize=(15,8))
sns.heatmap(data.corr(),annot=True,ax=ax1,cmap=sns.cm.vlag,cbar=False)


# **The Selling Price has Strong Correlation with Year and Present Price , Diesel Fuel Type of the car**
# 
# **I am gonna create 2 models**
# * **Model with all features**
# * **Model with Strong Correlated features**

# In[ ]:


def gen_model(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)
    lr = LinearRegression()
    
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    
    coeffecients = pd.DataFrame(lr.coef_,X.columns)
    coeffecients.columns = ['Coeffecient']
    print(f' Coefficients : \n {coeffecients} \n')
    
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    print(f'Mean Squared Error of Test Set : {mse}')
    print(f'Root Mean Square Error of Test Set : {rmse}')
    
    yt_pred = lr.predict(X_train)
    tmse = mean_squared_error(y_test,y_pred)
    trmse = np.sqrt(mse)
    print(f'Mean Squared Error of Train Set : {tmse}')
    print(f'Root Mean Square Error of Train Set : {trmse}')
    

    fig,ax1 = plt.subplots(figsize=(15,8))
    fig = sns.scatterplot(y_test,y_pred,ax=ax1)
    plt.xlabel('Y true')
    plt.ylabel('Y predicted')
    plt.title('True vs Predicted')
    plt.show(fig)
    
    fig,ax1 = plt.subplots(figsize=(15,8))
    fig = sns.distplot((y_test-y_pred),ax=ax1);
    plt.title('Residual Distrubution')
    plt.show(fig)


# 
# ## Model-1

# In[ ]:


#train test_split
X = data.drop(['Car_Name','Selling_Price'],axis = 1)
y = data['Selling_Price']


# In[ ]:


gen_model(X,y)


# 
# ## Model-2

# In[ ]:


#train test_split
X = data.drop(['Car_Name','Selling_Price','Kms_Driven','Seller_Type','Transmission',
               'Owner', 'Y_S_L','FT_Petrol'],axis = 1)
y = data['Selling_Price']


# In[ ]:


gen_model(X,y)


# In[ ]:




