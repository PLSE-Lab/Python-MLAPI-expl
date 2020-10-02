#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Introduction
# Greetings! This is an automatically-generated kernel with starter code demonstrating how to read in the data and begin exploring. Click the blue "Edit Notebook" or "Fork Notebook" button at the top of this kernel to begin editing.
# In this section we will do some exploratory analysis and regressive linear analysis by using global-market-sales data. This dataset consists of four files which are related one to another by particular columns. We will start to compile one to another and see the track record of the company from these data and at the end we will try to create a regression linear analysis for some variables within this dataset.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# set the figure size for the seaborn graphic. 

# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})


# # **Import the Data**

# In[ ]:


factmarketsales = pd.read_excel('../input/global-market-sales/FactMarketSales.xlsx')
orders = pd.read_excel('../input/global-market-sales/Orders.xlsx')
products = pd.read_excel('../input/global-market-sales/Products.xlsx')
shippings = pd.read_excel('../input/global-market-sales/Shippings.xlsx')


# Compile all Data in Dataset with the OrderCode and ProductId as the key for the merging

# In[ ]:


df_all = factmarketsales.merge(orders,on='OrderCode').merge(products,on='ProductId').merge(shippings,on='OrderCode')
df_all.head(5)


# # Gain some graphical insight

# In[ ]:


#heatmap of the profit by sub category
df_pivot = df_all.pivot_table(values='Profit',index=['Category','SubCategory'],columns='OrderPriority',aggfunc='sum')
sns.heatmap(df_pivot,annot=True)


# In[ ]:


#bar cart for the sales and profit for each order priority
by_priority = df_all.groupby(['OrderPriority']).sum()[['Sales','Profit']]
by_priority.plot(kind='bar')


# In[ ]:


#distribution plot of sales
sns.distplot(df_all['Sales'])


# In[ ]:


#jointplot between saes and profit
sns.jointplot(data=df_all,x='Profit',y='Sales',kind='reg')


# In[ ]:


#Profit and sales value by shipping region
by_region = df_all.groupby(['ShippingRegion']).sum()[['Profit','Sales']]
by_region.plot(kind='bar')


# In[ ]:


#number of sales divided by region for each category
sns.barplot(x="Sales", y="ShippingRegion", hue="Category",data=df_all, palette="coolwarm")


# In[ ]:


#amount of profit obtained by region for each category
sns.barplot(x="Profit", y="ShippingRegion", hue="Category",data=df_all, palette="coolwarm")


# # Start linear regression analysis
Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the Profit column. We will toss out the non corelated columns because they have no correlation for the performance of the Profit.
# In[ ]:


#X and y arrays
X = df_all[['Sales','Shipping Cost','Discount']]
y = df_all['Profit']


# **Train Test Split**
# 
# Now let's split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# **Creating and Training the Model**

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()
lm.fit(X_train,y_train)


# # Model Evaluation

# Below are the coefficients that represent correlation between profit, shipping cost and dicount variable.

# In[ ]:


print(lm.intercept_)


# In[ ]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# # Prediction from Our Model

# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


plt.scatter(y_test,predictions)


# In[ ]:


sns.distplot((y_test-predictions),bins=100);


# # Regression Evaluation Metric

# In[ ]:


from sklearn import metrics


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




