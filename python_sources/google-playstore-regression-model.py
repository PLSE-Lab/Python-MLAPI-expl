#!/usr/bin/env python
# coding: utf-8

# ## 1- Importing libraries and data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df_google= pd.read_csv('../input/google-play-store-cleaned-data/Goog_out1.csv')
df_google


# ### - My cleaned google play store data "from previous notebook"
# #### LINK: https://www.kaggle.com/taghredsalah199/google-store-cleaning-data-before-modeling
# ****

# In[ ]:


df_google.info()


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap( df_google.isnull() , yticklabels=False ,cbar=False )


# In[ ]:


df_google=df_google.fillna(value=df_google['Size'].mean())
df_google=df_google.fillna(value=df_google['Installs'].mean())
# Fill nan value with the mean of col


# In[ ]:


df_google=df_google.drop('Unnamed: 0', axis=1)


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap( df_google.isnull() , yticklabels=False ,cbar=False )


# ## No missing data at all !

# ## 2- Data visualization
# 

# In[ ]:


figure= plt.figure(figsize=(10,10))
sns.heatmap(df_google.corr(), annot=True)
#To show the correlation between variables


# In[ ]:


figure= plt.figure(figsize=(10,10))

sns.distplot(df_google['Rating']) #My predicted col


# In[ ]:


sns.pairplot(df_google)


# In[ ]:


sns.lmplot(x='Rating',y='Size',data=df_google)


# # 3- The Regression Model :
# ### Here I will compare between Linear Regression model and KNN model and see what we get!

# ## 3-A) Linear Regression :

# In[ ]:


x= df_google[['Reviews','Size','Installs']]
y=df_google['Rating']


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=108)


# In[ ]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()


# In[ ]:


LR.fit(x_train,y_train)


# In[ ]:


# The coefficients
print('Coefficients: \n', LR.coef_)


# In[ ]:


prediction = LR.predict(x_test)
plt.scatter(y_test,prediction) #NOT CORRECT MODEL


# ## - Linear Regression Evaluation
# 

# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.mean_absolute_error(y_test,prediction)


# In[ ]:


metrics.mean_squared_error(y_test,prediction)


# In[ ]:


np.sqrt(metrics.mean_squared_error(y_test,prediction))


# ## Very High error!!

# ## 3-B) KNN Regression:

# In[ ]:


df_google_KNN= df_google.drop(['App','Category','Type','Price','Content Rating','Genres','Last Updated','Current Ver','Android Ver'], axis=1)
#Select the numiric cols only


# ### Apply standard scaller for all numeric features::

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Data=df_google_KNN.drop('Rating',axis=1)


# In[ ]:


Data


# In[ ]:


scaler.fit(Data)
scaled_features= scaler.transform(df_google_KNN.drop('Rating',axis=1))


# In[ ]:


df_accepted_feat= pd.DataFrame(scaled_features,columns=df_google_KNN.columns[1:])
df_accepted_feat


# ### Train_test_split :

# In[ ]:


from sklearn.model_selection import train_test_split
x= df_accepted_feat
y=df_google_KNN['Rating']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=109)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
knn= KNeighborsRegressor(n_neighbors=3)
knn.fit(x_train,y_train)
pred= knn.predict(x_test)


# ## Evaluation of KNN Regression Model:

# In[ ]:


from sklearn import metrics
metrics.mean_absolute_error(y_test,pred)


# In[ ]:


metrics.mean_squared_error(y_test,pred)


# In[ ]:


np.sqrt(metrics.mean_squared_error(y_test,pred))


# ## Low Error than Linear Reggression!
# 

# ## ((Choose the best K-Value))

# In[ ]:


error_rate= []
for i in range(1,50):
    knn=KNeighborsRegressor(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i= knn.predict(x_test)
    error_rate.append(np.mean(pred_i!= y_test))


# In[ ]:


plt.figure(figsize=(12,8))
plt.plot(range(1,50), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red',markersize=8)
plt.title('Error rate vs K values')
plt.xlabel('K Values')
plt.ylabel('Error rate')


# ## The point from 0 to 6 has a low error rate!
