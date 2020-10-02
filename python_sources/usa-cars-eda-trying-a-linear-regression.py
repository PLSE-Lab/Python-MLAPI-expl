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


# # **1- Loading the data into a data frame:**
# 

# In[ ]:


df = pd.read_csv("../input/usa-cers-dataset/USA_cars_datasets.csv")
df.head()


# In[ ]:


df.tail()


# # 2- Dropping irrelevant columns :

# Since all of the cars are from the United States, I think the "Country" column doesn't mean much.
# And I also believe that the "Condition" column has nothing to do with the price of the car, and I can't find what it could add to our model. So in this step, we are going to delete these two columns.

# In[ ]:


df = df.drop(['country', 'condition'], axis = 1)
df.head()


# # 3- Finding duplicate rows :

# Some Datasets have some duplicate data which might be disturbing, In this step we will try to find the duplicate rows and remove them.

# In[ ]:


df.shape


# In[ ]:


duplicate_rows_df = df[df.duplicated()]
duplicate_rows_df.shape


# As we can see, there are no duplicate rows in this dataset

# # 4- Finding missing Values and Outliers :

# **A- Missing Values :**

# In[ ]:


df.count()


# As we can notice that all the columns have 2499 rows, we can therefore conclude that there are no missing values

# **B- Outliers :**

# In[ ]:


df.describe()


# In the table above, we were able to find rows with the value "0" as a price or as mileage which cannot be correct, so it is clear that we have some incomplete rows.

# In[ ]:


df[df['price']==0].count()


# In[ ]:


import seaborn as sns
sns.set(color_codes=True)
sns.boxplot(x=df['price'])


# In[ ]:


df[df['mileage']==0].count()


# In[ ]:


sns.boxplot(x=df['mileage'])


# As we can notice from the boxplots, we don't only have very low points but we also have a lot of high points.
# 
# In order to detect and remove outliers we are going to use a technique called "IQR score technique".

# In[ ]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[ ]:


df = df[~((df < (Q1-1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape


# As we can see, we have removed 336 rows, which represents 13% of our original dataset.

# In[ ]:


df[df['price']==0].count()


# In[ ]:


df[df['mileage']==0].count()


# We haven't deleted all rows where the price is equal to 0, but let's just say that 3 rows is better then 43.

# # 5- Visualizations :

# In[ ]:


import matplotlib.pyplot as plt #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Plotting a Histogram
df['brand'].value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Number of cars by brand")
plt.ylabel("Number of cars")
plt.xlabel("Brand");


# In[ ]:


df['state'].value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Number of cars by state")
plt.ylabel("Number of cars")
plt.xlabel("State");


# In[ ]:


df['color'].value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Number of cars by color")
plt.ylabel("Number of cars")
plt.xlabel("Color");


# In[ ]:


df['year'].value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Number of cars by Year")
plt.ylabel("Number of cars")
plt.xlabel("Year");


# In[ ]:


# Finding the relations between the variables
plt.figure(figsize=(20,10))
c= df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c


# In the above heat map we know that the price feature depends mainly on the year and the mleage.

# In[ ]:


# Plotting a scatter plot
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['year'], df['price'])
ax.set_xlabel('Year')
ax.set_ylabel('Price')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['mileage'], df['price'])
ax.set_xlabel('Mileage')
ax.set_ylabel('Price')
plt.show()


# In[ ]:


x= np.array(df['mileage'])
y= np.array(df['price'])
plt.plot(x, y, 'o')
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)


# From these scatter plots we can conclude that there is a linear regression between the price and the mileage.

# # 6- Linear Regression : 

# **A- Applying the model :**

# In this step we will try to use a linear regression between the "mileage" and the "price" in the aime to estimate the price.

# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

X = df['mileage'].values.reshape(-1,1)
y = df['price'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# **B- Evaluation of the Model**

# Let's compare between the real and the predicted prices from the test data after applying the linear regression.

# In[ ]:


y_pred = regressor.predict(X_test)
df_test = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df_test


# In[ ]:


df1 = df_test.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# From the table and the bar graphe above, we can see that the difference is much or less significent. to be more precise in evaluating this model we are going to calculate the MAE, the MSE and the RMS

# In[ ]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('10% of Mean Price:', df['price'].mean() * 0.1)


# You can see that the value of root mean squared error is 8690, which is much greater than 10% of the mean value which is 1916. This means that our algorithm was not very accurate

# *I hope you found this usefull, I will appreciate it if you could help me with your advises and I will be more than happy if you could upvote this. Thanks :) *

# 
