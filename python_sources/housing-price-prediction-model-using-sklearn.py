#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading the data set using pandas function

# In[ ]:


data_df_train = pd.read_csv('../input/train.csv')


# In[ ]:


data_df_train.columns


# # out come of the data set is saleprice, therefore y will be out label

# In[ ]:


y = data_df_train['SalePrice']


# In[ ]:


y.describe()


# In[ ]:


sns.distplot(y, bins=40)


# ### There are 81 features in this dataset, to identify the important dataset, seaborn library is very useful. using seaborn heatmap
# ### correlation,  selecting the top 8-10 features that are really contributing to the saleprince

# In[ ]:


corr_map = data_df_train.corr()
plt.subplots(figsize=(15, 15))
sns.heatmap(corr_map)


# ### Selecting the top 10 features that have direct correlation with output

# In[ ]:


data_df_train.corr().nlargest(10, 'SalePrice')['SalePrice'].index


# In[ ]:


cols = data_df_train.corr().nlargest(8, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data_df_train[cols].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


x = data_df_train[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
       'TotalBsmtSF', '1stFlrSF','FullBath', 'TotRmsAbvGrd']]


# In[ ]:


x.head()


# # Data Scaling 

# In[ ]:


x_min = x.min()
x_range = x.max()- x_min
x_scaled = x/x_range


# # To find out any null values in the selected features

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(x.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# # Training the model

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)


# In[ ]:


from sklearn.linear_model import LinearRegression
classifier = LinearRegression()
classifier.fit(X_train, y_train)


# # Testing the model

# In[ ]:


y_predict_test = classifier.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict_test})
df1 = df.head(25)


# In[ ]:


df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[ ]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict_test))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict_test))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict_test)))


# In[ ]:




