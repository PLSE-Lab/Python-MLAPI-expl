#!/usr/bin/env python
# coding: utf-8

# ## Individual Fish Weight Estimation from Body Size
# <font color = 'blue'>
# Content:
# 
# 1. [LOAD AND CHECK DATA](#1)
# 1. [VARIABLE DESCRIPTION](#2)
#     * [Categorical Variable](#3)
#     * [Numerical Variable](#4)
# 1. [BASIC DATA ANALYSIS](#5)
# 1. [OUTLIER DETECTION](#6)
# 1. [MISSING VALUE](#8)
#     * [Find Missing Value](#8)
#     * [Fill Missing Value](#8)
# 1. [VISUALIZATION](#9)
# 1. [MACHINE LEARNING](#10)
#     1. [Regression Models](#11)
#         1. [Lineer Regression](#12)
#         1. [Multiple Lineer Regression](#13)
#         1. [Polinomial Lineer Regression](#14)
#         1. [Decision Tree Regression](#15)
#         1. [Random Forest Regression](#16)
#     1. [Evaluation Regression Models](#17)
# 1. [CONCLUSION](#18)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import seaborn as sns

from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id = "1"></a><br>
# # LOAD AND CHECK DATA

# In[ ]:


data = pd.read_csv('../input/fish-market/Fish.csv')
print(plt.style.available)
plt.style.use('ggplot')


# In[ ]:


data.head()


# Let's analyze the features:
# 
# In order to understand the meanings of the features, we need to know about fish measurements. Following figures are explanatory.
# 
# ![1](http://fishionary.fisheries.org/wp-content/uploads/2014/04/scup_lengths.png)
# (figure from: https://fishionary.fisheries.org/wp-content/uploads/2014/04/scup_lengths.png; date accessed: 09.05.2020)
# 
# ![2](https://www.researchgate.net/profile/Harrison_Charo-Karisa/publication/40123354/figure/fig6/AS:669473098977283@1536626233163/Body-measurements-taken-on-each-fish-total-length-TL-standard-length-SL-body-depth_W640.jpg)
# (figure from: https://www.researchgate.net/profile/Harrison_Charo-Karisa/publication/40123354/figure/fig6/AS:669473098977283@1536626233163/Body-measurements-taken-on-each-fish-total-length-TL-standard-length-SL-body-depth_W640.jpg; date accessed: 09.05.2020)
# 

# As seen in the figures above biological measurements of a fish and the coreesponding feature in the data are listed below:
# (Actually there is not enough explanations about the length1, length2 and length3 features of the data, but they are assumed as listed)
# * SL: Standard Length---Length1
# * FL: Fork Length--------Length2
# * TL: Total Length-------Length3
# * BT: Body Thickness----Height
# * BD: Body Depth--------Width
# 
# Now I change the column names according to the explanation

# In[ ]:


data.columns = ['Species', 'Weight', 'SL', 'FL', 'TL', 'BD', 'BT']
data.head()


# Now we can continue to analize the data

# In[ ]:


data.info()


# Only the Species feature is object and the others are float.
# It is also seen that there are no missing values. Lets double check it

# In[ ]:


print(str('Is there any NaN value in the dataset: '), data.isnull().values.any())


# In[ ]:


data.describe()


# <a id = "2"></a><br>
# # VARIABLE DESCRIPTION
# 
# 1. Species: The species of fish
# 1. Weight: Weight of fish is grams
# 1. SL: Standard length of fish in cm
# 1. FL: Fork length of fish in cm
# 1. TL: Total length of fish in cm
# 1. BD: Body depth (height) of fish in cm
# 1. BT: Body thickness (width) of fish in cm
# 

# <a id = "3"></a><br>
# ## Categorical Variables
# * Species

# In[ ]:


sp = data['Species'].value_counts()
sp = pd.DataFrame(sp)
sp.T


# In[ ]:


sns.barplot(x=sp.index, y=sp['Species']);
plt.xlabel('Species')
plt.ylabel('Counts of Species')
plt.show()


# <a id = "4"></a><br>
# ## Numerical Variables
# * Weight
# * SL
# * FL
# * TL
# * BD
# * BT

# <a id = "5"></a><br>
# # BASIC DATA ANALYSIS

# Correlation of the features:

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(data[["Weight", "SL", "FL", "TL", "BD", "BT",]].corr(), annot = True)
plt.show()


# It is obvious that the different lengths of fish are correlated. But our aim is to estimate the weigth of the fish from the other features. so the most important thing is the correlation of weigth with the other features. Weigth is correlated with all types of length as well as BT and BD.

# Does the heat map change according to the species of fish?

# In[ ]:


def corr(species):
    data1 = data[data['Species'] == species]
    fig, ax = plt.subplots(figsize=(5,5)) 
    sns.heatmap(data1[["Weight", "SL", "FL", "TL", "BD", "BT",]].corr(), annot = True)
    plt.title("Correlation heat map of {} ".format(species))
    plt.show()


# In[ ]:


species_list = list(data['Species'].unique())
for s in species_list:
    corr(s)


# In[ ]:


g = sns.pairplot(data, kind='scatter', hue='Species')
g.fig.set_size_inches(10,10)


# <a id = "6"></a><br>
# # OUTLIER DETECTION

# In[ ]:


def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers


# In[ ]:


data.loc[detect_outliers(data,["Weight", "SL", "FL", "TL", "BD", "BT"])]


# Index number 142, 143 and 144 are the outliers of this data. We may think about dropping these lines. But before that I will divide the data I will take only the Pike fish and check for the outliers again.

# In[ ]:


df_pike = data[data['Species'] == 'Pike']
df_pike.loc[detect_outliers(df_pike,["Weight", "SL", "FL", "TL", "BD", "BT"])]


# As you see, there are no outliers for Pike fish. The main reason of the difference is there are 7 different species in the data and each has different body sizes. So we should check for outliers for each spesies.

# In[ ]:


species_list = list(data['Species'].unique())
print(species_list)


# In[ ]:


df_s = data[data['Species'] == 'Bream']
df_s.loc[detect_outliers(df_s,["Weight", "SL", "FL", "TL", "BD", "BT"])]


# No outliers in Bream

# In[ ]:


df_s = data[data['Species'] == 'Roach']
df_s.loc[detect_outliers(df_s,["Weight", "SL", "FL", "TL", "BD", "BT"])]


# There are 2 outliers in Roach. I will drop these.

# In[ ]:


df_s = data[data['Species'] == 'Whitefish']
df_s.loc[detect_outliers(df_s,["Weight", "SL", "FL", "TL", "BD", "BT"])]


# No outliers in Whitefish

# In[ ]:


df_s = data[data['Species'] == 'Parkki']
df_s.loc[detect_outliers(df_s,["Weight", "SL", "FL", "TL", "BD", "BT"])]


# No outliers in Parkki

# In[ ]:


df_s = data[data['Species'] == 'Perch']
df_s.loc[detect_outliers(df_s,["Weight", "SL", "FL", "TL", "BD", "BT"])]


# No outliers in Perch

# In[ ]:


df_s = data[data['Species'] == 'Smelt']
df_s.loc[detect_outliers(df_s,["Weight", "SL", "FL", "TL", "BD", "BT"])]


# There are 2 outliers in Roach. I will drop these.

# In[ ]:


data1 = data.drop([35, 54, 157,158])
data1.info()


# <a id = "8"></a><br>
# # Missing Value
#     * Find Missing Value
#     * Fill Missing Value

# In[ ]:


data1.columns[data1.isnull().any()]


# There are no missing values. So we don't need to fill the missing data. If you are keen to learn about how to fill the missing values, you can check my other kernels here
# 
# https://www.kaggle.com/albatros1602/albatros-titanic-eda/edit/run/32433379

# <a id = "9"></a><br>
# # Visualization
# In this kernel I'll not get deep in data visualization. In order to understand the relations between the features of the dataset we usse some visualization tools. If you are keen to learn about how to visualise the data, you can check my other kernels here 
# 
# https://www.kaggle.com/albatros1602/albatros-titanic-eda/edit/run/32433379
# 
# https://www.kaggle.com/albatros1602/visualization-for-hearth-disease-prediction
# 
# https://www.kaggle.com/albatros1602/a-quick-comparison-of-covid-19-cases-in-nyc-vs-la
# 

# <a id = "10"></a><br>
# # MACHINE LEARNING
# I will use sklearn as ML library

# <a id = "11"></a><br>
# ## Regression Models
# As we analyzed above, the correlation of the features vary according to the species. So it is not logical to make a single regression model to fit on every fish species. We have to create different regression models for each species unless we can not make a close estimation.
# 
# In this study I will try to show you how to work with following regression models:
# * Lineer Regression
# * Multiple Lineer Regression
# * Polinomial Lineer Regression
# * Decision Tree Regression
# * Random Forest Regression
# * Evaluation Regression Models
# 

# In[ ]:


data1.head()


# <a id = "12"></a><br>
# ### Lineer Regression
# In this model I will estimate the weight of Bream from TL.
# The formula for this type of regression is:
# 
# y = b0 + b1*x

# In[ ]:


df_bream = data1[data1['Species'] == 'Bream']

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

x = df_bream.TL.values.reshape(-1,1)
y = df_bream.Weight.values.reshape(-1,1)

linear_reg.fit(x,y)
y_head = linear_reg.predict(x)

plt.scatter(df_bream.TL,df_bream.Weight)
plt.plot(x,y_head,color= "black")
plt.xlabel("Total Length")
plt.ylabel("Weight")
plt.show()


# Lets check the model for a bream of which the TL=36 cm

# In[ ]:


print('The weight of a 36 cm Bream is: ', linear_reg.predict([[36]]), 'grams')


# <a id = "13"></a><br>
# ### Multiple Lineer Regression
# 
# In this regression I will estimate the weight of Bream according to TL and BD
# 
# The formula for this type of regression is:
# 
# y = b0 + b1*x1 + b2*x2
# 
# y: Dependent (Target variable
# 
# b0: constant
# 
# b1,b2: coefficient
# 
# X1,X2: Independent variables
# 

# ### Separate Dependant and Independant Variables

# In[ ]:


y = df_bream['Weight'] # Dependant Var
X = df_bream.iloc[:,[4,5]]


# ### Divide Data into Train and Test Data

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[ ]:


print('Samples in the test and train datasets are:')
print('X_train: ', np.shape(X_train))
print('y_train: ', np.shape(y_train))
print('X_test: ', np.shape(X_test))
print('y_test: ', np.shape(y_test))


# In[ ]:


ML_reg = LinearRegression()
ML_reg.fit(X_train, y_train)


# Now lets find out the missing constant (b0) and coefficient (b1,b2) values of the formula given above.

# In[ ]:


print('y = ' + str('%.2f' % ML_reg.intercept_) + ' + ' + str('%.2f' % ML_reg.coef_[0]) + '*X1 ' + ' + ' + str('%.2f' % ML_reg.coef_[1]) + '*X2 ')


# Lets check the model for a bream of which the TL=31 cm and BD=12 cm

# In[ ]:


print('The weight of a TL=31cm and BD=12cm Bream is: ', ML_reg.predict(np.array([[30,11.52]])), 'grams')


# <a id = "14"></a><br>
# ### Polinomial Lineer Regression
# In this regression I will estimate the weight of Bream according to TL
# 
# The formula for this type of regression is:
# 
# y = b0 + b1X^1 + b2X^2
# 
# y: Dependent (Target) variable
# 
# b0: constant
# 
# b1,b2: coefficient
# 
# X1,X2: Independent variables

# I will use the same data that we used for linear regression

# In[ ]:


x = df_bream.TL.values.reshape(-1,1)
y = df_bream.Weight.values.reshape(-1,1)

linear_reg.fit(x,y)
y_head = linear_reg.predict(x)

from sklearn.preprocessing import PolynomialFeatures
PL_reg = PolynomialFeatures(degree = 2)

x_polynomial = PL_reg.fit_transform(x)

L_reg = LinearRegression()
L_reg.fit(x_polynomial,y)
y_head2 = L_reg.predict(x_polynomial)


# Lets visualize the difference between linear and polynomial regressions. There is a very slight difference in this study.

# In[ ]:


plt.scatter(df_bream.TL,df_bream.Weight)
plt.plot(x,y_head,color= "orange")
plt.plot(x,y_head2,color= "green")
plt.xlabel("Total Length")
plt.ylabel("Weight")
plt.show()


# <a id = "15"></a><br>
# ### Decision Tree Regression
# This model splits the data into leaves and makes regression for each leaf

# In[ ]:


data1.head()


# In[ ]:


x1 = data1.iloc[:,1].values.reshape(-1,1)
y1 = data1.iloc[:,4].values.reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x1,y1)

x1_ = np.arange(min(x1), max(x1), 0.01).reshape(-1,1)
y1_head = tree_reg.predict(x1_)


# In[ ]:


plt.scatter(x1,y1, color = "red")
plt.plot(x1_,y1_head,color = "green")
plt.xlabel("Total Length")
plt.ylabel("Weight")
plt.show()


# <a id = "16"></a><br>
# ### Random Forest Regression
# This regression model is similar to Decision Tree. We will chose the number of trees to run estimation model.
# 
# Lets use the same data that I used for Decision Tree.

# In[ ]:


x2 = data1.iloc[:,1].values.reshape(-1,1)
y2 = data1.iloc[:,4].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(x2,y2)

x2_ = np.arange(min(x2),max(x2),0.01).reshape(-1,1)
y2_head = rf.predict(x2_)

plt.scatter(x2,y2,color = "red")
plt.plot(x2_,y2_head,color = "green")
plt.xlabel("Total Length")
plt.ylabel("Weight")
plt.show()


# <a id = "17"></a><br>
# ### Evaluation Regression Models
# Lets evaluate the models and see how trustworthy they are. 

# ### Evaluation of Multiple Linear Regression Model 

# In[ ]:


# Separate variables
y = df_bream['Weight']
X = df_bream.iloc[:,[4,5]]

# Divide dataset for train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Regression model
ML_reg = LinearRegression()
ML_reg.fit(X_train, y_train)

#Predict weight values from train dataset
y_head = ML_reg.predict(X_train)


# #### Analyze the success of the model

# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_train, y_head)


# #### Cross Validation

# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score_train = cross_val_score(ML_reg, X_train, y_train, cv=4, scoring='r2')
print(cross_val_score_train)


# #### Average of Cross Validation

# In[ ]:


cross_val_score_train.mean()


# #### Predict weights by means of test dataset

# In[ ]:


y_pred = ML_reg.predict(X_test)
print(r2_score(y_test, y_pred))


# ### Visualize the predictions

# In[ ]:


plt.scatter(X_test['TL'], y_test, color='red', alpha=0.4) #Real data
plt.scatter(X_test['TL'], y_pred, color='blue', alpha=0.4) #Predicted data
plt.xlabel('Total Length in cm')
plt.ylabel('Weight of the fish in grams')
plt.title('Linear Regression Model for Weight Estimation');


# ### Evaluation of Random Forest Regression Model

# In[ ]:


# Separate variables
yRF = data1.iloc[:,1].values.reshape(-1,1)
XRF = data1.iloc[:,4].values.reshape(-1,1)

# Divide dataset for train and test
from sklearn.model_selection import train_test_split
XRF_train, XRF_test, yRF_train, yRF_test = train_test_split(XRF, yRF, test_size=0.2, random_state=1)

# Regression model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(XRF_train,yRF_train)

#Predict weight values from train dataset
yRF_head = rf.predict(XRF_train)


# #### Analyze the success of the model

# In[ ]:


r2_score(yRF_train, yRF_head)


# #### Cross Validation

# In[ ]:


from sklearn.model_selection import cross_val_score
cross_val_score_train = cross_val_score(rf, XRF_train, yRF_train, cv=10, scoring='r2')
print(cross_val_score_train)


# #### Average of Cross Validation

# In[ ]:


cross_val_score_train.mean()


# #### Predict weights by means of test dataset

# In[ ]:


yRF_pred = rf.predict(XRF_test).reshape(-1,1)
print(r2_score(yRF_test, yRF_pred))


# ### Visualize the predictions

# In[ ]:


plt.scatter(XRF_test, yRF_test, color='red', alpha=0.4) #Real data
plt.scatter(XRF_test, yRF_pred, color='blue', alpha=0.4) #Predicted data
plt.xlabel('Total Length in cm')
plt.ylabel('Weight of the fish in grams')
plt.title('Random Forest Regression Model for Weight Estimation');


# <a id = "18"></a><br>
# # Conclusion
# 
# For this dataset, linear regression models fit very well as long as the dataset is divided into species. Otherwise the data of Pike (which is a bigger fish) or Smelt (which is a smaller fish) will lead the model to a false prediction. 
# 
# On the other hand rnadom forest regression model splits the data and make predictions from the data located at each leaf, and gives a better result.
# 
# You can try the models with different species or different features and compare the results.
# 
# I hope you enjoy it. Your comments are appreciated.
# 
# Ihsan
# 
# To make this study I got inspired from:
# 
# https://www.kaggle.com/akdagmelih/multiplelinear-regression-fish-weight-estimation
# 
# https://www.kaggle.com/kanncaa1/dataiteam-titanik-eda
# 
# https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners
