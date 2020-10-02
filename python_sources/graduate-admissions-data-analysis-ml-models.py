#!/usr/bin/env python
# coding: utf-8

# # Kaggle's Graduate Admissions dataset

# ## You can find information about the content in the following Kaggle's links:
# ## https://www.kaggle.com/mohansacharya/graduate-admissions

# In[6]:

import os
import pandas as pd
import numpy as np
import random as rdm

# for plotting
import matplotlib.pyplot as plt
import seaborn as sn
#% matplotlib inline


# In[7]:


# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)


# In[8]:
filename = os.listdir('../input')
filename[1]
# Reading the Graduates dataset
data = pd.read_csv('../input/'+filename[1])
data.head()
# In[9]:


# A quick look at the rows and columns
print(data.shape)
data.head()


# In[10]:


data.columns.values.tolist()


# In[11]:


Id = "Serial No."
target = "Chance of Admit "


# ## Data Analysis

# ### Missing values
# 
# Let's go ahead and find out which variables of the dataset contain missing values

# In[12]:


# make a list of the variables that contain missing values
vars_with_na = [var for var in data.columns if data[var].isnull().sum()>=1]

# print the variable name and the percentage of missing values
for var in vars_with_na:
    print(var, np.round(data[var].isnull().mean(), 3),  ' % missing values')


# No missing values

# ### Numerical variables
# 
# Let's go ahead and find out what numerical variables we have in the dataset

# In[13]:


# list of numerical variables
num_vars = [var for var in data.columns if data[var].dtypes != 'O']

print('Number of numerical variables: ', len(num_vars))

# visualise the numerical variables
data[num_vars].head()


# From the above view of the dataset, we notice the variable Serial No., which will not be used to make our predictions, as there is one different value of the variable per each row, it will not add to our prediction. See below:

# In[14]:


print('Number of Customers: ', len(data[Id].unique()))
print('Number of rowns in the Dataset: ', len(data))


# #### Discrete variables
# 
# Let's go ahead and find which variables are discrete, i.e., show a finite number of values

# In[15]:


#  list of discrete variables
discrete_vars = [var for var in num_vars if len(data[var].unique())<20 and var not in [Id]]

print('Number of discrete variables: ', len(discrete_vars))


# In[16]:


# let's visualise the discrete variables
data[discrete_vars].head()


# Let's go ahead and analyse their contribution to the target.

# In[17]:


def analyse_discrete(df, var, target):
    df = df.copy()
    df.groupby(var)[target].median().plot.bar()
    plt.title(var)
    plt.ylabel(target)
    plt.show()
    
for var in discrete_vars:
    analyse_discrete(data, var, target)


# It can be seen that all of theses variables are strongly related to the target variable

# #### Continuous variables
# 
# Let's go ahead and find the distribution of the continuous variables. We will consider continuous all those that are not temporal or discrete variables in our dataset.

# In[18]:


# list of continuous variables
cont_vars = [var for var in num_vars if var not in discrete_vars+[Id]]

print('Number of continuous variables: ', len(cont_vars))


# In[31]:


# let's visualise the continuous variables
data[cont_vars].head()


# In[19]:


# Let's go ahead and analyse the distributions of these variables
def analyse_continous_vars(df, var):
    df = df.copy()
    df[var].hist(bins=20)
    plt.ylabel('Customers')
    plt.xlabel(var)
    plt.title(var)
    plt.show()
    
for var in cont_vars:
    analyse_continous_vars(data, var)


# We see that all of the above variables, are normally distributed, including the target variable. For linear models to perform best, we need to account for non-Gaussian distributions. 

# #### Outliers

# In[20]:


# let's make boxplots to visualise outliers in the continuous variables 

def find_outliers(df, var):
    df = df.copy()
    
    # log does not take negative values, so let's be careful and skip those variables
    if 0 in data[var].unique():
        pass
    else:
        #df[var] = np.log(df[var])
        df.boxplot(column=var)
        plt.title(var)
        plt.ylabel(var)
        plt.show()
    
for var in cont_vars:
    if var != target:
        find_outliers(data, var)


# No outliers

# ### Categorical variables
# 
# Let's go ahead and analyse the categorical variables present in the dataset.

# In[21]:


### Categorical variables
id_vars = [Id]

cat_vars = [var for var in data.columns if ((data[var].dtypes=='O') and (var not in id_vars))]

print('Number of categorical variables: ', len(cat_vars))


# No categorical variables

# * It seems that the Data is ready to be processed, no feature engineering is required

# ## Model Building

# In[22]:


columns = data.columns.values.tolist()
columns


# In[24]:


sn.pairplot(data, x_vars=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP','LOR ','CGPA','Research'], y_vars='Chance of Admit ', height=7, aspect=0.7, kind='reg')


# * There is evident linear correlation with the target vaiable

# ### Obtaining X and Y

# In[25]:


# Columns of the dataset
columns = data.columns.values.tolist()
columns


# In[26]:


# x Columns , and y columns 
X = data.iloc[:, data.columns != target]
X = X.iloc[:, X.columns != Id]
Y = data.iloc[:, data.columns == target]


# In[27]:


X.head()


# ### Scaling

# In[28]:


from sklearn.preprocessing import MinMaxScaler


# In[29]:


scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
scaler.fit(X)
sc_vars = X.columns.values.tolist()
X = pd.DataFrame(scaler.transform(X),columns=sc_vars)


# ### Feature Selection

# In[30]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# In[31]:


lr = LinearRegression(n_jobs = 2)


# In[32]:


#feature selecting
rfe = RFE(lr,3)
rfe = rfe.fit(X, Y.values.ravel())


# In[33]:


# Selected variables
X.columns[rfe.support_]


# ### Building the Linear model

# In[34]:


# Getting the train and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X[X.columns[rfe.support_]], Y, test_size = 0.3, random_state = 1)


# In[35]:


# Creating the model
lmodel = LinearRegression(n_jobs=2)
lmodel.fit(X_train,np.array(Y_train).ravel())


# In[36]:


# Getting the R2 score 
lmodel.score(X_train,np.array(Y_train).ravel())


# ####  Metrics

# In[37]:


from sklearn import metrics
from math import sqrt


# In[38]:


Y_test = np.array(Y_test).ravel()


# In[39]:


Y_pred = lmodel.predict(X_test)
print("Testing. MSE:",metrics.mean_squared_error(Y_test, Y_pred))
print("Testing. RMSE:",sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print("Testing. R2:",metrics.r2_score(Y_test, Y_pred))


# In[40]:


Y_pred_train = lmodel.predict(X_train)
print("Training. MSE:",metrics.mean_squared_error(Y_train, Y_pred_train))
print("Training. RMSE:",sqrt(metrics.mean_squared_error(Y_train, Y_pred_train)))


# * MSE and RMSE from the training and testing sets are similar, suggesting No overfitting

# #### Final Linear Model

# In[41]:


# Final model 
pd.DataFrame(list(zip(X.columns[rfe.support_],np.transpose(lmodel.coef_))))


# In[42]:


# intercept 
print(lmodel.intercept_)


# ### Exploring a Polynomial model

# In[43]:


# Training & Testing polynomial models
for d in range(2,5):
    poly = PolynomialFeatures(degree=d)
    X_data = poly.fit_transform(X_train)
    lmodel = LinearRegression(n_jobs=2)
    lmodel.fit(X_data,np.array(Y_train).ravel())
    print("Degree:",d)
    
    Y_pred_train = lmodel.predict(X_data)
    print("Training prediction. MSE:",metrics.mean_squared_error(Y_train, Y_pred_train))
    print("Training prediction. RMSE:",sqrt(metrics.mean_squared_error(Y_train, Y_pred_train)))
    print("Trainining prediction. R2",metrics.r2_score(Y_train, Y_pred_train))
    
    X_data = poly.fit_transform(X_test)
    lmodel = LinearRegression(n_jobs=2)
    lmodel.fit(X_data,np.array(Y_test).ravel())
    Y_pred = lmodel.predict(X_data)
    print("Test. Prediction. MSE:",metrics.mean_squared_error(Y_test, Y_pred))
    print("Test. Prediction. RMSE:",sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
    print("Test. Prediction. R2:",metrics.r2_score(Y_test, Y_pred))


# * The polynomial option seems to adjust better with higher degrees, however, given the complexity of a polynomyal model, and the fact that the Linear Regression prediction is at a R2 of 82%, I would suggest to stay with the simpler model

# ### Exploring other option

# #### Random forests

# In[44]:


from sklearn.ensemble import RandomForestRegressor


# In[45]:


# x Columns , and y columns 
X = data.iloc[:, data.columns != target]
X = X.iloc[:, X.columns != Id]
Y = data.iloc[:, data.columns == target]


# In[46]:


forest = RandomForestRegressor(n_jobs=2, oob_score=True, n_estimators=100,min_samples_leaf=10, min_samples_split=10,random_state =1,
                             max_leaf_nodes=None)
forest.fit(X,Y.values.ravel())


# In[48]:


forest.oob_score_


# * After testing several parameter options the Random trees did not overcome the Linear model
