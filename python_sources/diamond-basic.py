#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


diamond=pd.read_csv('/kaggle/input/diamonds/diamonds.csv')


# In[ ]:


diamond.head()


# In[ ]:


diamond.columns


# Here, I just wanted to check the the no. of cuts,colors,clarity

# In[ ]:


def uniques(feature):
    un=diamond[feature].unique()
    unno=diamond[feature].nunique()
    return (un,unno)


# In[ ]:


uniques('cut')


# In[ ]:


uniques('color')


# In[ ]:


uniques('clarity')


# In[ ]:


del diamond['Unnamed: 0']


# In[ ]:


diamond


# In[ ]:


diamond.describe()


# We have a cleaned dataset, but the problem here is the dimension can't be zero for the follwings diamonds.it makes them unreliable. we can see that in "min". we have to check them and replace.in this case, as they seem to be in very small in percentage,I just droped them.

# In[ ]:


z=sum(diamond["x"]==0)
print("no of unreliable length:",z)
print("no of unrealaible width:{}".format(sum(diamond["y"]==0)))
print("no of unrealaible depth:{}".format(sum(diamond["z"]==0)))


# In[ ]:


diamond[['x','y','z']] = diamond[['x','y','z']].replace(0,np.NaN)


# In[ ]:


diamond.info()


# In[ ]:


def missing_percentage(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])


# In[ ]:


missing_percentage(diamond)


# In[ ]:


diamond.dropna(inplace=True)


# In[ ]:


diamond.shape


# In[ ]:


sns.heatmap(diamond.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# I just wanted to check the correlation between the features, we can see that dimensions are more correlated with the price of the diamond.you can view it clearly with the visuals.

# In[ ]:


plt.figure(figsize=(10,6))
sns.pairplot(diamond)


# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(diamond.corr(),linewidths=1,cmap='RdYlGn',annot=True)


# The green region as correlated you can refer them with the pairplot figure.

# In[ ]:


sns.countplot(x='cut',data=diamond,palette='rainbow')


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x='cut',y='price',data=diamond,palette='winter')


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x='color',y='price',data=diamond,palette='winter')


# In[ ]:


plt.figure(figsize=(10,10))
sns.boxplot(x='clarity',y='price',data=diamond,palette='winter')


# As seen above these cut,clarity,color contains many outliers.They don't seem to have very high effect on price. We have to encode the strings.

# In[ ]:


encoded=pd.get_dummies(diamond)


# In[ ]:


encoded


# In[ ]:


end=encoded.drop(['carat','depth','table',
       'price','x', 'y', 'z'],axis=1)


# In[ ]:


diamond_data=pd.concat([diamond,end],axis=1).drop(['cut', 'color', 'clarity'],axis=1)


# In[ ]:


diamond_data


# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(diamond_data.corr(),linewidths=1,cmap='RdYlGn')


# Train the model.

# In[ ]:


x = diamond_data.drop(['price'],axis=1)
y = diamond_data['price']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
coeff_df


# Here, we can see the effect of features on the price of the diamond, here carat seems to be playing a major role.

# In[ ]:


plt.scatter(y_test,predictions)


# In[ ]:


from sklearn import metrics


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:


print("accuracy: ",(lm.score(X_test,y_test)))


# Thankyou!!

# In[ ]:




