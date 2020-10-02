#!/usr/bin/env python
# coding: utf-8

# # REGRESSION From Scratch With BOSTON HOUSE PRICE PREDICTION

# <img src='https://drive.google.com/uc?id=1rxqFy4bsnYH325VpZwjZVeKfgKa1b4oS' width=1000 >

# ### In this Notebook we will Learn:-
# * Basic EDA.
# * Aplly Scaling on Feature matrix.
# * Dimensionality Reduction (PCA) .
# * K-Cross validation to check accuracy.
# * Multi-linear Regression
# * Polynomial Regression
# * Support Vector Regressor (SVR)
# * Decision Tress Regressor 
# * Random Forest Regressor

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


# Importing the dataset.
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('../input/housing.csv', delim_whitespace=True, names=names)


# In[ ]:


df.head()


# * In this dataset MEDV is our Target, we have to predict the values of MEDV on the basis of all other variables.
# * MEDV: Median value of owner-occupied homes in $1000s 
# * Our task is to predict the value of MEDV.

# In[ ]:


df.info()


# * There is no null values in the dataset and all values are in their proper format.
# * Dataset is in its proper format, so we will go to regression model.

# In[ ]:


df.corr().iplot(kind='heatmap', )


# #### Observation:-
# * These are correlation matrix between all variables.
# * The valriables which are highly correlated with MEDV, we need to select only those variable to make prediction.
# * But this thing we will do with the help of dimensionalty reduction algorithm (PCA). 

# # REGRESSION:-

# ### Data Preprocessing = 
#                      * In this we will follow 4 steps, MCSS.
#                      * M = dealing with Missing data
#                      * C = Dealing with the categorical dataset.
#                      * S = Splitting of dataset.
#                      * S = Scaling of the dataset.
# * As there is no missing values in the dataset.
# * There is no categorical values in the dataset.
# * As dataset is very small there is no need to split the dataset.
# * Before applying the dimensionalty reduction algorithm (PCA), we will follow 1 step i.e. Scaling of dataset.

# In[ ]:


# Importing of Useful libraries from sklearn library.
from sklearn.preprocessing import StandardScaler   # For Scaling the dataset
from sklearn.model_selection import train_test_split    # For Splitting the dataset
from sklearn.linear_model import LinearRegression      # For Linear regression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# In[ ]:


# Let us Create Feature matrix and Target Vector.
x_train = df.iloc[:,:-1].values
y_train = df.iloc[:,-1].values


# #### Scaling of Feature matrix.

# In[ ]:


sc_X=StandardScaler()
x_train=sc_X.fit_transform(x_train)


# #### Dimensionalty Reduction by PCA.
# * We are doing this to reduce the number of dimensions/features in the dataset.
# * The features which have less effect on the prediction , we will remove those features.
# * It also boosts the process.
# * It saves time.
# * Here we will use Principal Component Analysis (PCA) with 'rbf' kernel.

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=None)
x_train = pca.fit_transform(x_train)

explained_variance = pca.explained_variance_ratio_
explained_variance


# In[ ]:


print(f"The sum of initial 5 values is \t {0.47+0.11+0.09+0.06+0.06} , which is very good." )
print("So we will choose 5 number of features and reduce our training feature matrix to 5 features/columns. ")


# In[ ]:


pca = PCA(n_components=5)
x_train = pca.fit_transform(x_train)


# In[ ]:


def all_models():    
    # Multi-linear regression Model. 
    regressor_multi = LinearRegression()
    regressor_multi.fit(x_train,y_train)
    # Let us check the accuray
    accuracy = cross_val_score(estimator=regressor_multi, X=x_train, y=y_train,cv=10)
    print(f"The accuracy of the Multi-linear Regressor Model is \t {accuracy.mean()}")
    print(f"The deviation in the accuracy is \t {accuracy.std()}")
    print()
    
    # Polynomial Regression
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg=PolynomialFeatures(degree=4) #These 3 steps are to convert X matrix into X polynomial
    x_poly=poly_reg.fit_transform(x_train) #matrix. 
    regressor_poly=LinearRegression()
    regressor_poly.fit(x_poly,y_train)
    # Let us check the accuray
    accuracy = cross_val_score(estimator=regressor_poly, X=x_train, y=y_train,cv=10)
    print(f"The accuracy of the Polynomial Regression Model is \t {accuracy.mean()}")
    print(f"The deviation in the accuracy is \t {accuracy.std()}")
    print()
    
    # Random Forest Model
    regressor_random = RandomForestRegressor(n_estimators=100,)
    regressor_random.fit(x_train,y_train)
    # Let us check the accuray
    accuracy = cross_val_score(estimator=regressor_random, X=x_train, y=y_train,cv=10)
    print(f"The accuracy of the Random Forest Model is \t {accuracy.mean()}")
    print(f"The deviation in the accuracy is \t {accuracy.std()}")
    print()
    
    # SVR 
    regressor_svr = SVR(kernel='rbf')
    regressor_svr.fit(x_train, y_train)
    # Let us check the accuracy
    accuracy = cross_val_score(estimator=regressor_svr, X=x_train, y=y_train,cv=10)
    print(f"The accuracy of the SVR Model is \t {accuracy.mean()}")
    print(f"The deviation in the accuracy is \t {accuracy.std()}")
    print()
    
    # Decision Tress Model
    regressor_deci = DecisionTreeRegressor()
    regressor_deci.fit(x_train, y_train)
    # Let us check the accuracy
    accuracy = cross_val_score(estimator=regressor_deci, X=x_train, y=y_train,cv=10)
    print(f"The accuracy of the Decision Tree Model is \t {accuracy.mean()}")
    print(f"The deviation in the accuracy is \t {accuracy.std()}")
    
    

    


# In[ ]:


# Let us run all models together. If we have large dataset then we will not run all models together.
# Then we will run one model at a time, otherwise your processor will struck down.
all_models()


# #### Observation:-
# * The best model is Multi-linear Regression.
# * In multi-linear Regressio, we are getting the accuracy of 31% and deviation of 54%.
# * The accuracy we  are getting is not that much good due many factors like less quantity of dataset, data not collected properly, something  wrong at the time of web scraping.

# # IF THIS KERNEL IS HELPFUL, THEN PLEASE UPVOTE.
# <img src='https://drive.google.com/uc?id=17o_bxPmndgdL9Y6PPdcIXOBsJ0jizDNG' width=500 >

# In[ ]:




