#!/usr/bin/env python
# coding: utf-8

# # **About the dataset:**
# 
# This dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass.
# 
# There are 14 attributes in each case of the dataset. They are:
# 
# * CRIM - per capita crime rate by town
# * ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# * INDUS - proportion of non-retail business acres per town.
# * CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# * NOX - nitric oxides concentration (parts per 10 million)
# * RM - average number of rooms per dwelling
# * AGE - proportion of owner-occupied units built prior to 1940
# * DIS - weighted distances to five Boston employment centres
# * RAD - index of accessibility to radial highways
# * TAX - full-value property-tax rate per \$10,000
# * PTRATIO - pupil-teacher ratio by town
# * B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# * LSTAT - % lower status of the population
# * MEDV - Median value of owner-occupied homes in $1000's
# 
# Variable #14 seems to be censored at 50.00 (corresponding to a median price of \$50,000); Censoring is suggested by the fact that the highest median price of exactly \$50,000 is reported in 16 cases, while 15 cases have prices between \$40,000 and $50,000, with prices rounded to the nearest hundred.
# 
# Our goal is to select the valiables which predicts the MEDV best, also to suggest a machine learning model to predict MEDV.

# In[ ]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score as cvs


# In[ ]:


#importing dataset
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv('/kaggle/input/boston-house-prices/housing.csv', delimiter=r'\s+', names=column_names)


# In[ ]:


#Top 5 rows of dataset
dataset.head()


# In[ ]:


#Shape of dataset (rows, columns)
dataset.shape


# In[ ]:


#describing the dataste to see distribution of data
dataset.describe()


# From the above distribution we can see that:
# 1.   Variable 'ZN' is 0 for 25th and 50th percentile that will result in skweed data. This is a result of 'ZN' being a conditional variable.
# 2.   Also for variable 'CHAS' it's 0 for 25th, 50th and 75th percentile that will also show us that data is highly skweed. This is a result of 'CHAS' being a categorical data, contaning vaules 0 and 1 only.
# 
# Another important fact we can derive form above description is tha max value of 'MEDV' which is 50, goes along with the original data description which says : Variable #14 seems to be censored at 50.00 (corresponding to a median price of $50,000)
# 
# For a start we can derive an asumption that 'ZN' and 'CHAS' variables may not be useful in predicting MEDV as they will result in biased model, so let's remove them.

# In[ ]:


#removing variables 'ZN' and 'CHAS' form data
dataset = dataset.drop(['ZN', 'CHAS'], axis=1)


# # **Checking null values**

# In[ ]:


dataset.isnull().sum()


# There are no null values in our dataset

# # **Checking and treating outliers in the data**

# In[ ]:


#Plotting boxplots to see if there are any outliers in our data (considering data betwen 25th and 75th percentile as non outlier)
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(15, 5))
ax = ax.flatten()
index = 0
for i in dataset.columns:
  sns.boxplot(y=i, data=dataset, ax=ax[index])
  index +=1
plt.tight_layout(pad=0.4)
plt.show()


# Columns CRIM, RM, DIS, PTRATIO, B, LSTAT and MEDV have outliers.

# In[ ]:


#checking percentage/ amount of outliers
for i in dataset.columns:
  dataset.sort_values(by=i, ascending=True, na_position='last')
  q1, q3 = np.nanpercentile(dataset[i], [25,75])
  iqr = q3-q1
  lower_bound = q1-(1.5*iqr)
  upper_bound = q3+(1.5*iqr)
  outlier_data = dataset[i][(dataset[i] < lower_bound) | (dataset[i] > upper_bound)] #creating a series of outlier data
  perc = (outlier_data.count()/dataset[i].count())*100
  print('Outliers in %s is %.2f%% with count %.f' %(i, perc, outlier_data.count()))
  #----------------------code below is for comming sections----------------------
  if i == 'B':
    outlierDataB_index = outlier_data.index
    outlierDataB_LB = dataset[i][(dataset[i] < lower_bound)]
    outlierDataB_UB = dataset[i][(dataset[i] > upper_bound)]
  elif i == 'CRIM':
    outlierDataCRIM_index = outlier_data.index
    outlierDataCRIM_LB = dataset[i][(dataset[i] < lower_bound)]
    outlierDataCRIM_UB = dataset[i][(dataset[i] > upper_bound)]
  elif i == 'MEDV':
    lowerBoundMEDV = lower_bound
    upperBoundMEDV = upper_bound


# Variable 'CRIM' and 'B' have high percentage of outlier data which can adversely affect the accuracy of our model.
# 
# To get rid of this we can either drop the observations or replace with some apporach like mean or median. But dropping all the outlier observations is not a good idea as we will be left with very fewer observations due to higher percentage of outliers to train our model on, also if we replace such a big percentage of the outliers with some approach (mean, median...etc.) then it might result into less accurate or biased model.
# 
# We can use an alternative : let's drop the extreme outliers and replace the remaning by some approach (mean, median.....etc.)

# In[ ]:


dataset2 = dataset.copy() # I copied the data in another variable just for an ease of coding, but this is not required


# In[ ]:


#removing extreme outliers form B and CRIM (removing those observations)
removed=[]
outlierDataB_LB.sort_values(ascending=True, inplace=True)
outlierDataB_UB.sort_values(ascending=False, inplace=True)
counter=1
for i in outlierDataB_LB.index:
  if counter<=19:
    dataset2.drop(index=i, inplace=True)
    counter+=1
    removed.append(i)
for i in outlierDataB_UB.index:
  if counter<=38:
    dataset2.drop(index=i, inplace=True)
    counter+=1
    removed.append(i)
for i in outlierDataB_LB.index:
  if counter<=38 and i not in removed:
    dataset2.drop(index=i, inplace=True)
    counter+=1
    removed.append(i)


outlierDataCRIM_LB.sort_values(ascending=True, inplace=True)
outlierDataCRIM_UB.sort_values(ascending=False, inplace=True)
counter=1
for i in outlierDataCRIM_LB.index:
  if counter<=16 and i not in removed:
    dataset2.drop(index=i, inplace=True)
    counter+=1
    removed.append(i)
for i in outlierDataCRIM_UB.index:
  if counter<=33 and i not in removed:
    dataset2.drop(index=i, inplace=True)
    counter+=1
    removed.append(i)
for i in outlierDataCRIM_LB.index:
  if counter<=33 and i not in removed:
    dataset2.drop(index=i, inplace=True)
    counter+=1
    removed.append(i)

dataset2.shape


# We have dropped 71 observations from our dataset, now we are left with 435 observations and 12 columns.
# 
# **Now replacing the remaning outliers with mean of each variable.**

# In[ ]:


dataset3 = dataset2.copy() # I copied the data in another variable just for an ease of coding, but this is not required


# In[ ]:


#replacing remaning outliers by mean
for i in dataset.columns:
  dataset.sort_values(by=i, ascending=True, na_position='last')
  q1, q3 = np.nanpercentile(dataset[i], [25,75])
  iqr = q3-q1
  lower_bound = q1-(1.5*iqr)
  upper_bound = q3+(1.5*iqr)
  mean = dataset3[i].mean()
  if i != 'MEDV':
    dataset3.loc[dataset3[i] < lower_bound, [i]] = mean
    dataset3.loc[dataset3[i] > upper_bound, [i]] = mean
  else:
    dataset3.loc[dataset3[i] < lower_bound, [i]] = mean
    dataset3.loc[dataset3[i] > upper_bound, [i]] = 50


# Below is the description of our dataset after treating the outliers:

# In[ ]:


dataset3.describe()


# # **Selecting the features which can predict MEDV the best**

# **Using p-Value to to select the optimal features:**
# 
# Dropping all the variables whose p-value is less than significance level of 0.05 using backward elimination method

# In[ ]:


#independent variable(X) and dependent variable(Y)
X = dataset3.iloc[:, :-1]
Y = dataset3.iloc[:, 11]


# In[ ]:


#Feature selection using P-Value/ Backward elimination
def BackwardElimination(sl, w):
    for i in range(0, len(w.columns)):
        regressor_OLS = sm.OLS(endog=Y, exog=w).fit()
        max_pvalue = max(regressor_OLS.pvalues)
        pvalues = regressor_OLS.pvalues
        if max_pvalue > SL:
            index_max_pvalue = pvalues[pvalues==max_pvalue].index
            w = w.drop(index_max_pvalue, axis = 1) #delete the valriable for that p value
    return w,pvalues,index_max_pvalue

SL = 0.05
ones = np.ones((435,1))  #adding a columns of ones to X as it is required by statsmodels library
W = X
W.insert(0, 'Constant', ones, True)
W_optimal = W.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11]]

W_optimal,pvalues,index_max_pvalue = BackwardElimination(SL, W_optimal)
X = W_optimal.drop('Constant', axis=1)


# In[ ]:


#remaning variabls after backward elimination
X.columns


# **Using pearson correlation to remove any highly correlated independent variables to avoid multicollinearity :**

# In[ ]:


#Ploting heatmap using pearson correlation among independent variables
plt.figure(figsize=(8, 8))
ax = sns.heatmap(X.corr(method='pearson').abs(), annot=True, square=True)
plt.show()


# From above correlation heatmap we can see that:
# 1.   TAX and RAD are highly correlated with score 0.86. As per my personal understandig RAD (index of accessibility to radial highways) will be more important in predicting MEDV as commpared to TAX (full-value property-tax rate per $10,000), so I am considering to drop TAX
# 2.   DIS and NOX are highly correlated with score 0.75. As per my personal understandig DIS (weighted distances to five Boston employment centres) will be more important in predicting MEDV as commpared to NOX (nitric oxides concentration (parts per 10 million)), so I am considering to drop NOX
# 
# 

# In[ ]:


#dropping TAX and NOX
X.drop('TAX', axis=1, inplace=True)
X.drop('NOX', axis=1, inplace=True)

#remaning columns after removing multicollinearity
X.columns


# **Checking correlation of remaning independent variables with MEDV using Pearson correlation method**

# In[ ]:


#now checking correlation of each variable with MEDV by pearson method and dropping the one with least correlation with MEDV
for i in X.columns:
  corr, _ = pearsonr(X[i], Y)
  print(i,corr)


# We can see that DIS and RAD are least correlated to MEDV, so dropping DIS and RAD

# In[ ]:


X.drop(['DIS', 'RAD'], axis=1, inplace=True)


# In[ ]:


#remaning variables/ features that can predict the MEDV most
X.columns


# From the above feature selection process we conclude that features *RM*, *PTRATIO* and *LSAT* can alone predict MEDV the best

# # **Machine learning**
# 
# This is a regression a problem as we have to predict a continous (non catagorical) value.
# 
# Implementing regression machine learning models to out our dataset (using the remaing independent variables) to predict MEDV

# In[ ]:


#spliting data into traning set and test set
X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.2, random_state=0)


# **Linear regression model:**

# In[ ]:


linear = lr()
linear.fit(X_train, Y_train)
Y_pred = linear.predict(X_test)
Y_compare_linear = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
Y_compare_linear.head() #displaying the comparision btween actual and predicted values of MEDV


# **Polynomial regression model:**

# In[ ]:


polyRegressor = PolynomialFeatures(degree=3)
X_train_poly = polyRegressor.fit_transform(X_train)
X_test_poly = polyRegressor.fit_transform(X_test)
poly = lr()
poly.fit(X_train_poly, Y_train)
Y_pred = poly.predict(X_test_poly)
Y_compare_poly = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
Y_compare_poly.head() #displaying the comparision btween actual and predicted values of MEDV


# **Support vector regression model:**

# In[ ]:


svr = SVR(kernel= 'poly', gamma='scale')
svr.fit(X_train,Y_train)
Y_pred = svr.predict(X_test)
Y_compare_svr = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
Y_compare_svr.head() #displaying the comparision btween actual and predicted values of MEDV


# **Decission tree regression model:**

# In[ ]:


rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train,Y_train)
Y_pred = rf.predict(X_test)
Y_compare_randomforrest = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
Y_compare_randomforrest.head() #displaying the comparision btween actual and predicted values of MEDV


# **K-Nearest Neighbour regression model:**

# In[ ]:


knn = KNeighborsRegressor(n_neighbors=13)
knn.fit(X_train,Y_train)
Y_pred = knn.predict(X_test)
Y_compare_knn = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
Y_compare_knn.head() #displaying the comparision btween actual and predicted values of MEDV


# **Plotting compariasion of actual and predicted values of MEDV that we got using different machine learning models**

# In[ ]:


fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(25, 4))
ax = ax.flatten()
Y_compare_linear.head(10).plot(kind='bar', title='Linear Regression', grid=True, ax=ax[0])
Y_compare_poly.head(10).plot(kind='bar', title='Polynomial Regression', grid=True, ax=ax[1])
Y_compare_svr.head(10).plot(kind='bar', title='Support Vector Regression', grid=True, ax=ax[2])
Y_compare_randomforrest.head(10).plot(kind='bar', title='Random Forrest Regression', grid=True, ax=ax[3])
Y_compare_knn.head(10).plot(kind='bar', title='KNN Regression', grid=True, ax=ax[4])
plt.show()


# **Scores (R squared) of different machine learning models using K-fold cross validation:**

# In[ ]:


print('According to R squared scorring method we got below scores for out machine learning models:')
modelNames = ['Linear', 'Polynomial', 'Support Vector', 'Random Forrest', 'K-Nearest Neighbour']
modelRegressors = [linear, poly, svr, rf, knn]
models = pd.DataFrame({'modelNames' : modelNames, 'modelRegressors' : modelRegressors})
counter=0
score=[]
for i in models['modelRegressors']:
  if i is poly:
    accuracy = cvs(i, X_train_poly, Y_train, scoring='r2', cv=5)
    print('Accuracy of %s Regression model is %.2f' %(models.iloc[counter,0],accuracy.mean()))
    score.append(accuracy.mean())
  else:
    accuracy = cvs(i, X_train, Y_train, scoring='r2', cv=5)
    print('Accuracy of %s Regression model is %.2f' %(models.iloc[counter,0],accuracy.mean()))
    score.append(accuracy.mean())
  counter+=1


# In[ ]:


pd.DataFrame({'Model Name' : modelNames,'Score' : score}).sort_values(by='Score', ascending=True).plot(x=0, y=1, kind='bar', figsize=(15,5), title='Comparison of R2 scores of differnt models', )
plt.show()


# From the above visualiation we can summarise that Random Forrest (r2 = 0.72) machine learning model gives the best score and we can use it to predict the values of MEDV the best.
# 
# However other models like Polynomial (r2 = 0.64) regression model and KNN (r2 = 0.64)regression model also have comparable score to Random Forrest and hence can also be used to make predictions of MEDV.

# # **Final summary**
# 
# **From above data engineering and machine learning techniques we can conclude that:**
# 
# 1.   Features RM, PTRATIO and LSAT are alone capable of predicting MEDV to a good accuracy
# 2.   Random Forrest regression model (with 100 estimators) can be considered as a good model for predictiong MEDV using the above mentioned three features.
# 3.   However Polynomial and KNN regression models can also be used as an alternative to Random Forrest.
# 4.   Linear and Support Vector regression models shows the least r2 score, which can be considered bad models for predicting MEDV.
# 
# I would like to close it by mentioning an important fact, that no Data Science technique is perfect and there is always scope for imporvement.

# **Please comment your suggestions**
# 
# **Please upvote if this notebook is helpful**
