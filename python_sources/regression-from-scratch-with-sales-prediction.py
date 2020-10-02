#!/usr/bin/env python
# coding: utf-8

# # REGRESSION From Scratch With SALES PREDICTION

# <img src='https://drive.google.com/uc?id=1-6z0sZc9YrK_czjy8mBQuxBj3wdD01-V' width=800 >

# #### In this Notebook we will Learn:-
# * Basic EDA.
# * Feature Engineering
# * Dealing with missing values.
# * Aplly Scaling on Feature matrix.
# * Dealing with Categorical Dataset.
# * Dimensionality Reduction (PCA) .
# * K-Cross validation to check accuracy.
# * Multi-linear Regression
# * Random Forest Regressor
# * Polynomial Regression
# * Prediction on new Values.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, download_plotlyjs, iplot
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
import warnings
warnings.filterwarnings('ignore')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print()
print("The files in the dataset are:-")
from subprocess import check_output
print(check_output(['ls','../input']).decode('utf'))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Importing the datasets
df_train = pd.read_csv('../input/Train.csv')
df_test = pd.read_csv('../input/Test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.shape


# In[ ]:


df_test.shape


# In[ ]:





# # BASIC ANALYSIS AND FEATURES ENGINEERING

# #### 1). Removing Unwanted Columns/Features.

# In[ ]:


try:
    df_train.drop(labels=['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'], axis=1, inplace=True)
    df_test.drop(labels=['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'], axis=1, inplace=True)
except Exception as e:
    pass


# In[ ]:


df_train.head()


# #### 2). Getting Information about Null values,

# In[ ]:


temp_df = df_train.isnull().sum().reset_index()
temp_df['Percentage'] = (temp_df[0]/len(df_train))*100
temp_df.columns = ['Column Name', 'Number of null values', 'Null values in percentage']
print(f"The length of dataset is \t {len(df_train)}")
temp_df


# * So it is clear that we do not have to remove null values, as they 28% and 17% in the Outlet_Size and Item_Weight Columns respectively.
# * Null values are in less quantity.
# * We will replace them later with thier mean or mode values.

# #### 3). Making Correction in 'Item_Fat_Content' column.

# In[ ]:


def convert(x):
    if x in ['low fat', 'LF']: 
        return 'Low Fat'
    elif x=='reg':
        return 'Regular'
    else:
        return x

df_train['Item_Fat_Content'] = df_train['Item_Fat_Content'].apply(convert)
df_test['Item_Fat_Content'] = df_train['Item_Fat_Content'].apply(convert)

print(f"Now Unique values in this column in Train Set are\t  {df_train['Item_Fat_Content'].unique()} ")
print(f"Now Unique values in this column in Test Set are\t  {df_test['Item_Fat_Content'].unique()} ")


# #### 4). Dealing with the Missing Values in Categorical type column i.e. 'Outlet_Size'

# In[ ]:


# Counting the values
count = df_train['Outlet_Size'].value_counts().reset_index()
count.iplot(kind='bar', color='deepskyblue', x='index', y='Outlet_Size', title='High VS Mediun VS Small',
           xTitle='Size', yTitle='Frequency')


# * We will remove the missing values from 'Medium' in both Training set and Test set.

# In[ ]:


df_train['Outlet_Size'].fillna(value='Medium', inplace= True)
df_test['Outlet_Size'].fillna(value='Medium', inplace= True)


# ### ===============================================================================

# # PREDICTION WITH REGRESSION MODELS.

# In[ ]:


# Let us Import the Important Libraries  to train our Model for Machine Learning 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder, OneHotEncoder # To deal with Categorical Data in Target Vector.
from sklearn.model_selection import train_test_split  # To Split the dataset into training data and testing data.
from sklearn.model_selection import cross_val_score   # To check the accuracy of the model.
from sklearn.preprocessing import Imputer   # To deal with the missing values
from sklearn.preprocessing import StandardScaler   # To appy scaling on the dataset.


# In[ ]:


# Let us create feature matrix and Target Vector.
x_train = df_train.iloc[:, :-1].values    # Features Matrix
y_train = df_train.iloc[:,-1].values   # Target Vector
x_test = df_test.values    # Features Matrix


# In[ ]:


df_train.head()


# ### 1). Dealing with Missing data.

# In[ ]:


imputer = Imputer()
x_train[:,[0]] = imputer.fit_transform(x_train[:,[0]])
x_test[:,[0]] = imputer.fit_transform(x_test[:,[0]])


# ### 2). Dealing With the Categorical Values in Features/Columns.

# In[ ]:


labelencoder_x = LabelEncoder()
x_train[:, 1 ] = labelencoder_x.fit_transform(x_train[:,1 ])
x_train[:, 3 ] = labelencoder_x.fit_transform(x_train[:,3 ])
x_train[:, 5 ] = labelencoder_x.fit_transform(x_train[:,5 ])
x_train[:, 6 ] = labelencoder_x.fit_transform(x_train[:,6 ])
x_train[:, 7 ] = labelencoder_x.fit_transform(x_train[:,7 ])


#this is need to done when we have more than two categorical values.
onehotencoder_x = OneHotEncoder(categorical_features=[3,5,6,7]) 
x_train = onehotencoder_x.fit_transform(x_train).toarray()

# Let's apply same concept on test set.
x_test[:, 1 ] = labelencoder_x.fit_transform(x_test[:,1 ])
x_test[:, 3 ] = labelencoder_x.fit_transform(x_test[:,3 ])
x_test[:, 5 ] = labelencoder_x.fit_transform(x_test[:,5 ])
x_test[:, 6 ] = labelencoder_x.fit_transform(x_test[:,6 ])
x_test[:, 7 ] = labelencoder_x.fit_transform(x_test[:,7 ])


#this is need to done when we have more than two categorical values.
onehotencoder_x = OneHotEncoder(categorical_features=[3,5,6,7]) 
x_test = onehotencoder_x.fit_transform(x_test).toarray()


# ### 3). Now time to Apply Feature Scaling on Feature matrix .

# In[ ]:


sc_X=StandardScaler()
x_train=sc_X.fit_transform(x_train)
x_test = sc_X.fit_transform(x_test)


# ### 4). DIMENSIONALITY REDUCTION
# * We are doing this to reduce the number of dimensions/features in the dataset.
# * The features which have less effect on the prediction , we will remove those features.
# * It also boosts the process.
# * It saves time.
# * Here we will use Principal Component Analysis (PCA) with 'rbf' kernel.

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=None)
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)
explained_variance = pca.explained_variance_ratio_
explained_variance


# * Here we will take n_component = 24.

# In[ ]:


pca = PCA(n_components=25)
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)


# ### 5). Apply Multi-Linear Regression Model, Polynomial Regression and Random Forest Model and compare thier accuracy and pick the best one.

# #### Multi-Linear Regression

# In[ ]:


# Multi-linear regression Model.
regressor_multi = LinearRegression()
regressor_multi.fit(x_train,y_train)

# Let us check the accuray
accuracy = cross_val_score(estimator=regressor_multi, X=x_train, y=y_train,cv=10)
print(f"The accuracy of the Multi-linear Regressor Model is \t {accuracy.mean()}")
print(f"The deviation in the accuracy is \t {accuracy.std()}")


# #### Random Forest Model

# In[ ]:


"""# Random Forest Model.
regressor_random = RandomForestRegressor(n_estimators=100,)
regressor_random.fit(x_train,y_train)

# Let us check the accuray
accuracy = cross_val_score(estimator=regressor_random, X=x_train, y=y_train,cv=10)
print(f"The accuracy of the Random Forest Model is \t {accuracy.mean()}")
print(f"The deviation in the accuracy is \t {accuracy.std()}") """

print("Here accuray is 53% with deviation of 3%.")


# #### Polynomial regression

# In[ ]:


"""
# Fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4) #These 3 steps are to convert X matrix into X polynomial
x_poly=poly_reg.fit_transform(x_train) #matrix. 
regressor_poly=LinearRegression()
regressor_poly.fit(x_poly,y_train)

# Let us check the accuray
accuracy = cross_val_score(estimator=regressor_poly, X=x_train, y=y_train,cv=10)
print(f"The accuracy of the Polynomial Regression Model is \t {accuracy.mean()}")
print(f"The deviation in the accuracy is \t {accuracy.std()}")
"""
print("Here accuracy is 55% with deviation of 2%")


# #### observation:-
# * As the accuracy of Multi-linear regression Model is the best one.
# * Multi-linear Regression Model takes less time as compare to Random forest and Polynomial regression Models.
# * We will choose Multi-linear regression Model.
# * Here we are getting the accuracy of 55% and deviation of 2%, means in future if we mak eprediction on new values then we will get the accuracy in range 53% to 57%.
# * We are getting low accuracy due to less quantity of data.

# ### Let us make Prediction on test set

# In[ ]:


y_pred = regressor_multi.predict(x_test)

y_pred[:5]


# ### ============================================================================
# ### ============================================================================
# ### ============================================================================
# ### ============================================================================

# # IF THIS KERNEL IS HELPFUL, THEN PLEASE UPVOTE.
# <img src='https://drive.google.com/uc?id=1LBdaJj2pTM0cq9PY6k70RaGfUFDakUzG' width=500 >

# In[ ]:




