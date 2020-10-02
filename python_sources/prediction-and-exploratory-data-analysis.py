#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv("../input/kc_house_data.csv")
data.head()


# In[ ]:


data.columns


# In[ ]:


count_per_bedrooms = pd.value_counts(data['bedrooms'])
print (count_per_bedrooms)
count_per_bedrooms.plot(kind='bar')


# In[ ]:


df_6 = data[data['bedrooms'] == 6]
df_5 = data[data['bedrooms'] == 5]
df_4 = data[data['bedrooms'] == 4]
df_3 = data[data['bedrooms'] == 3]
df_2 = data[data['bedrooms'] == 2]
df_1 = data[data['bedrooms'] == 1]


# In[ ]:


print (data.shape)
print(data.isnull().any())
data.describe()


# In[ ]:


print("Mean Price of 6 bedrooms House :", df_6['price'].mean())
print("Mean Price of 5 bedrooms House :", df_5['price'].mean())
print("Mean Price of 4 bedrooms House :", df_4['price'].mean())
print("Mean Price of 3 bedrooms House :", df_3['price'].mean())
print("Mean Price of 2 bedrooms House :", df_2['price'].mean())
print("Mean Price of 1 bedrooms House :", df_1['price'].mean())


# In[ ]:


#Correlation Matrix
import seaborn as sns
corr = data.corr()
corr = (corr)
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
sns.plt.title('Heatmap of Correlation Matrix')
corr


# In[ ]:


# Kernel Density Plot
fig = plt.figure(figsize=(15,4),)
ax=sns.lmplot(x="yr_built", y="price", data=data)


# In[ ]:


def convert_to_year(date_in_some_format):
    date_as_string = str(date_in_some_format)
    year_as_string = date_in_some_format[0:4] 
    return int(year_as_string)
data['year'] = data['date'].apply(convert_to_year)


# In[ ]:


print(data.columns)


# In[ ]:


print(data.columns)
def add(x):
    y = int(x[0])+ int(x[1])
    return y
df = data[['sqft_living','sqft_lot']]
data['sqft'] = df.apply(add, axis = 1)
data['sqft'].head()


# In[ ]:


import math
def norm_price(price):
    price = math.log(price)
    price = price/(1.0+price)
    
    return price
data['price'] = data['price'].apply(norm_price)
data['price'].head()


# In[ ]:


data.columns
important_feature = [ 'bedrooms', 'bathrooms','sqft_living','sqft_lot',
                     'floors', 'waterfront', 'view', 'condition', 'grade',
                     'sqft_above', 'sqft_basement', 'yr_renovated',
                     'lat', 'long', 'sqft_living15', 'sqft_lot15']
X = data[important_feature]
Y = data['price']


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = 0.30, random_state=3)


# In[ ]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
print ("No. of rows considered as training : %d " % len(x_train))
print ("No. of rows considered as test : %d " % len(x_test))

x_train.shape
y_train.shape


# In[ ]:


accuracy = regr.score(x_test,y_test)  # .score(x_test, y_test)
print ("Accuracy achieved : {}%".format(int(round(accuracy * 100))))


# In[ ]:


from sklearn import linear_model
reg = linear_model.Lasso(alpha = 0.1)
reg.fit(x_train, y_train)


# In[ ]:


accuracy = reg.score(x_test,y_test)
print ("Accuracy achieved : {}%".format(int(round(accuracy * 100))))


# In[ ]:


reg.coef_


# In[ ]:


l = reg.coef_.data.tolist()


# In[ ]:


feat = []
for i in range(len(l)):
    if l[i] > 0:
        feat.append(important_feature[i])
feat    


# In[ ]:


X = data[feat]
Y = data['price']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size = 0.30, random_state=3)


# In[ ]:


from sklearn import tree
reg = tree.DecisionTreeRegressor()
reg.fit(x_train, y_train)


# In[ ]:


accuracy = reg.score(x_test,y_test)
print ("Accuracy achieved : {}%".format(int(round(accuracy * 100))))


# In[ ]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
print ("No. of rows considered as training : %d " % len(x_train))
print ("No. of rows considered as test : %d " % len(x_test))

x_train.shape
y_train.shape


# In[ ]:


accuracy = regr.score(x_test,y_test)  # .score(x_test, y_test)
print ("Accuracy achieved : {}%".format(int(round(accuracy * 100))))


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

X_new = SelectKBest(f_regression, k=10).fit_transform(X, Y)
X_new.shape


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_new, Y, test_size = 0.30, random_state=3)


# In[ ]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
print ("No. of rows considered as training : %d " % len(x_train))
print ("No. of rows considered as test : %d " % len(x_test))

print(x_train.shape)
y_train.shape


# In[ ]:


accuracy = regr.score(x_test,y_test)  # .score(x_test, y_test)
print ("Accuracy achieved : {}%".format(int(round(accuracy * 100))))


# In[ ]:




