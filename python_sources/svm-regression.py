#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()


# df.head() will give us the details fo top 5 rows of every cooloumn. We can use df.tail() to get the last 5 rows and similiary df.head(10) to get top 10 rows.
# 
# The data is about cars and we need to predict the price of car using the above data
# 
# We will be using Decision Tree to get the price of the car.

# In[ ]:


df.dtypes


# dtypes gives the data type of coloumn

# In[ ]:


df.describe()


# In the above dataframe all the coloumns are not numeric. So we will consider only those coloumn whose values are in numeric and will make all numeric to float.

# In[ ]:


df.dtypes
for x in df:
    if df[x].dtypes == "int64":
        df[x] = df[x].astype(float)
        print (df[x].dtypes)


# Preparing the Data As with the classification task, in this section we will divide our data into attributes and labels and consequently into training and test sets. We will create 2 data set,one for price while the other (df-price). Since pur dataframe has many data in object format, for this analysis we are removing all the coloumn with object type and for all NaN value we are removig that row

# In[ ]:


df = df.select_dtypes(exclude=['object'])
df=df.fillna(df.mean())
X = df.drop('price',axis=1)
y = df['price']


# Here the X variable contains all the columns from the dataset, except 'Price' column, which is the label. The y variable contains values from the 'Price' column, which means that the X variable contains the attribute set and y variable contains the corresponding labels.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Training SVM

# In[ ]:


from sklearn.svm import SVR


# We will create an object svr using the impo function SVM.We will use the kernel as linear.

# In[ ]:


svr = SVR(kernel = 'linear',C = 1000)


# in order to work in an efficient manner we will standardize our data.SVM works on distance of points so its necessary that all our data should be of same standard.

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


sc= StandardScaler().fit(X_train)


# In[ ]:


X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_test_std


# Now our data has been standarised.

# In[ ]:


svr.fit(X_train_std,y_train)
y_test_pred = svr.predict(X_test_std)


# In[ ]:


y_train_pred = svr.predict(X_train_std)


# lets check our predicted values

# In[ ]:


y_test_pred


# Time to check modal performance

# In[ ]:


from sklearn import metrics

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_train,y_train_pred)


# In[ ]:


r2_score(y_test,y_test_pred)


# Our R sqrt score for test data is 0.72 and for train data is 0.85 which is a good value.

# In[ ]:


import seaborn as sns
plt.figure(figsize=(5, 7))


ax = sns.distplot(y, hist=False, color="r", label="Actual Value")
sns.distplot(y_test_pred, hist=False, color="b", label="Fitted Values" , ax=ax)


plt.title('Actual vs Fitted Values for Price')


plt.show()
plt.close()


# The above is the graph between the actual and predicted values

# In[ ]:




