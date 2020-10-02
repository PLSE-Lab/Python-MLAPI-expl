#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score,accuracy_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')


# In[ ]:


df.head()


# Firstly, let's see more about the data

# In[ ]:


df.info()


# It can be seen that there are 53940 entries with 11 columns 

# Let's first work on finding the outliers in the data, we will use boxplot to visvualize if there is any outlier

# In[ ]:


sns.boxplot(data=df)


# It can be observed from the boxplot that there is no outlier<br>
# Even though there are certain outliers for the price column , still we can consider them as there is a cluster of data values outside the whiskers

# Now , next step can be to detect if there is any Missing value in the data

# In[ ]:


df.isnull().sum()


# Ah, great we don't have any missing value. That is great.

# There is a column named 'Unnamed' , this is probably generated when reading the csv, this can be dropped

# In[ ]:


df.drop(columns = 'Unnamed: 0', axis=1, inplace=True)


# In[ ]:


df.head()


# **Encoding Categorical Features**
# The color coloumn contains a categorical data with ordinal values which represent diamond colour from J (worst) to D (best)<br>
# The approach that can be used here is to use OrdinalEncoder <br>
# Let's take the unique values from the color column

# In[ ]:


df.head()


# In[ ]:


enc = OrdinalEncoder()
df['color'] = enc.fit_transform(df[['color']])
df.head()


# Look's great, we have now the color coloumn in the required format. *with all integre values*

# There are other columns also which include Ordinal Values, these are 'cut' and 'clarity'.<br>
# Using OrdinalEncoding is not a good option as the algorithm does not be able to justify the ordering without the domain knowledge<br>
# Therefore, we can procedd with manual mapping for these colums.
# * cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)
# * clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

# **Getting the unique values from the cut column**

# In[ ]:


df['cut'].unique()


# Let's create a dictionary to map the values for the 'cut' column

# In[ ]:


cut_map = {'Ideal':0,'Premium':1,'Very Good':2,'Good':3,'Fair':4}
df['cut'] = df['cut'].map(cut_map)


# In[ ]:


df.head()


# Similary let's create a dictionary to map the values for the 'clarity' column, (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

# In[ ]:


clarity_map = {'IF':0,'VVS1':1,'VVS2':2,'VS1':3,'VS2':4, 'SI1':5,'SI2':6,'I1':7}
df['clarity'] = df['clarity'].map(clarity_map)


# In[ ]:


df.head()


# * Now that all the features are correctly converted and categorized, the next step is scaling the dat.
# * The available option for scaling the data are *Normalising* and *Standardizing*
# * For this dataset let's consider using Normalisaton

# In[ ]:


scaling_df = preprocessing.MinMaxScaler()
df[['depth','table']] = scaling_df.fit_transform(df[['depth','table']])


# In[ ]:


df.head()


# **The Model looks good enough to go with the model implementation**

# In[ ]:


df_features = df.drop(columns = 'price')
df_target = df['price']


# Splitting the data in train and test 

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(df_features,df_target, test_size = 0.2, random_state = 2)


# ****Creating the regression model on the data generated

# In[ ]:


models = [] # Creating a list to store the models
accuracy_score = [] # Creating a list to store the model accuracy


# **Creating a function to calculate the accuracy of different models**

# In[ ]:


def model_performance(model,model_name,features = X_train,target = y_train, test_features = X_test, true_values = y_test):
    models.append(model_name)
    model.fit(features,target)
    y_pred = model.predict(test_features)
    accuracy = r2_score(true_values,y_pred)
    accuracy_score.append(accuracy)
    print(accuracy)
    
    


# **Linear Regression Model**

# In[ ]:


reg = LinearRegression()
model_performance(reg, 'Linear Regression')


# **Decision Tree Regressor**

# In[ ]:


reg = DecisionTreeRegressor()
model_performance(reg, 'Decision Tree Regression')


# In[ ]:


reg = KNeighborsRegressor()
model_performance(reg, 'KNN Regression')


# In[ ]:


reg = GaussianNB()
model_performance(reg, 'Naive Bayes Regression')


# In[ ]:


reg = Ridge()
model_performance(reg, 'Ridge Regression')


# In[ ]:


reg = Lasso()
model_performance(reg, 'Lasso Regression')


# In[ ]:


reg = ElasticNet()
model_performance(reg, 'ElasticNet Regression')


# In[ ]:


reg = LinearSVR()
model_performance(reg, 'LinearSVR Regression')


# In[ ]:


reg = RandomForestRegressor(n_estimators = 10, random_state = 42)
model_performance(reg, 'Random Forest Regression')


# In[ ]:


reg = AdaBoostRegressor(n_estimators = 100)
model_performance(reg, 'AdaBoost Regression')


# In[ ]:


reg = GradientBoostingRegressor(n_estimators = 100, random_state = 42, max_depth=4)
model_performance(reg, 'GradientBoosting Regression')


# **It's time to compare the performance of all the models used to predict the accuracy**

# In[ ]:


model_comparision_df = pd.DataFrame({'Regression Model': models, 'R2 Accuracy Score': accuracy_score})
model_comparision_df.sort_values(by = 'R2 Accuracy Score', ascending= False)


# **Plotting the comparision on the bar plot**

# In[ ]:


sns.barplot(x = 'R2 Accuracy Score', y='Regression Model', data = model_comparision_df)


# **The best perfroming models for us here are GradientBoosting folowed by Random Forest**
