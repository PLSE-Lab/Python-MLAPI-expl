#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


data=pd.read_csv("../input/USA_Housing.csv")


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.dropna(how="any",axis=0)


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


import seaborn as sns
sns.pairplot(data)


# In[ ]:


plt.show()


# In[ ]:


sns.distplot(data['Price'])


# In[ ]:


plt.show()


# In[ ]:


x=data["Avg. Area Income"]
y=data["Avg. Area House Age"]
x.head()


# In[ ]:


y.head()


# In[ ]:


sns.regplot(x="Avg. Area Income",y="Price",data=data)
plt.show()


# In[ ]:


data.head()


# In[ ]:


sns.regplot(x="Avg. Area House Age",y="Price",data=data)


# In[ ]:


plt.show()


# In[ ]:


sns.regplot(x="Avg. Area Number of Rooms",y="Price",data=data)
plt.show()


# In[ ]:


sns.regplot(x="Avg. Area Number of Bedrooms",y="Price",data=data)
plt.show()


# In[ ]:


sns.regplot(x="Area Population",y="Price",data=data)
plt.show()


# In[ ]:


sns.regplot(x="Price",y="Price",data=data)
plt.show()


# In[ ]:


sns.jointplot(x="Avg. Area House Age",y="Price",kind="reg",data=data)
plt.show()


# In[ ]:


sns.jointplot(x="Avg. Area Number of Rooms",y="Price",kind="reg",data=data)


# In[ ]:


plt.show()


# In[ ]:


data.corr()


# In[ ]:


corr=data.corr()
corr.nlargest(7,'Price')["Price"]


# In[ ]:


fig = plt.figure(figsize = (10,7))
sns.heatmap(data.corr(), annot = True,cmap = "coolwarm")
plt.show()


# In[ ]:


X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Area Population']]
y=data[['Price']]


# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


X_train, X_test, Y_train, Y_test=train_test_split(X,y,test_size = 0.4, random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


sv=LinearRegression()
sv.fit(X_train,Y_train)
sv.predict(X_train)


# In[ ]:


sv.score(X_train,Y_train)


# In[ ]:


test_score=sv.score(X_test,Y_test)
test_score


# In[ ]:


sv.coef_


# In[ ]:


print(sv.intercept_)


# In[ ]:





# In[ ]:


coef = pd.DataFrame(sv.coef_,X.columns, columns = ['coeffcient'])
coef


# In[ ]:


predict = sv.predict(X_test)
predict


# In[ ]:





# In[ ]:




