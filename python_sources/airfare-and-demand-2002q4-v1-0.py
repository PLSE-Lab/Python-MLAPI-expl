#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# hide warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# for expanding dataframe and displaying all columns
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


# # Part 1

# ## Load the data using Python Pandas library

# In[ ]:


# loading data from dat file to a dataframe
df = pd.read_csv('../input/airfare-and-demand-2002/airq402.dat', sep='\s+', header=None)


# In[ ]:


# displaying top 5 rows and we can observe we dont have column labels
df.head()


# In[ ]:


# renaming columns
df.columns = ['City1', 'City2', 'Average_Fare0', 'Distance', 'Average Weekly Passengers', 'Market Leading Airline', 
           'Market_Share1', 'Average_Fare1', 'Low Price Airline', 'Market_Share2', 'Average_Fare2']


# In[ ]:


# top 5 rows after renaming columns
df.head()


# ## Exploratory Data Analysis of the data

# In[ ]:


df.info()


# #### We can observe from above information that we don't have any null values present in the dataframe.

# In[ ]:


df.describe()


# From the above data description, we can see there may be a chance of presence of outliers as from the values in 25th and 75th quartile.
# 
# We will draw boxplot in next section to verify this.

# #### let's write some methods to plot our data.

# In[ ]:


# pair plot function
def plot_pair(df):
    fig=plt.figure(figsize=(64,64))
    sns.pairplot(df)
    plt.show()


# In[ ]:


# box plot function
def plot_box(df):
    plt.figure(figsize=(25, 30))
    i=1
    for each in columns:
        plt.subplot(3, 3, i)
        sns.boxplot(y = each,data = df)
        i+=1
    plt.show()


# In[ ]:


# Correlation plot function
def plot_corr(df):
    plt.figure(figsize=(20, 14))
    sns.heatmap(df.corr(), cmap='YlGnBu', annot = True)
    plt.show()


# In[ ]:


# # Scatter plot function
# def plot_scatter(X,y):
#   plt.figure(figsize=(25, 30))
#   i=1
#   for each in [2,3,5,6,8,9]:
#     plt.subplot(3,2, i)
#     sns.scatter(X.iloc[:,each], y, alpha=0.5)
#     plt.show()


# In[ ]:


columns = ['Average_Fare0', 'Distance', 'Average Weekly Passengers', 'Market_Share1', 'Average_Fare1', 'Market_Share2', 'Average_Fare2']


# In[ ]:


plot_pair(df)
plt.show()


# The above plot clearly indicates the relationship between pair of variables along with data disribution of each variable at diagonal.
# - You can see that average_fair1 and average_fare2 have approximately high linear relation with our target variable average_fare0
# 
# We will futher investigate into these relationship  during HeatMap of Correlation Matrix

# In[ ]:


plot_corr(df)


# As you can see from the above correlation Matrix, how numerical variables are related to each other.
# - Average fare 1 and average fare 2 are highly correlated with out target variable average fare0
# - Average weekly passenger, Market Share2, Market share 1 are negative correlated with our Target variable.

# In[ ]:


plot_box(df)


# From the boxplot above, you can see there are some outliers in average fair2 , average fare 0 and weekly passenger.
# 
# These can be due to high demand of festive season or any other reason.
# 
# To impute or perform any action on these we will need more insights in the dataset

# In[ ]:


leading_airline_group = df.groupby(['Market Leading Airline'])['City1'].size()
leading_airline_group.plot.bar(figsize=(18,8))
plt.show()


# From the above bar plot we can observe **WN** is the Market Leading Airline for most of the trips, further followed by **DL** and **AA** after it.

# In[ ]:


low_price_airline_group = df.groupby(['Low Price Airline'])['City1'].size()
low_price_airline_group.plot.bar(figsize=(18,8))
plt.show()


# From the above bar plot we can observe **WN** is also the Low Price Airline for most of the trips, further followed by **DL** and **AA** after it.

# ## Part 2: Preparing data for Modeling

# In[ ]:


# Printing top 5 rows
df.head()


# #### Let's seperate our target variable from predictor variables.
# 
# We will store target variable in y, 
# while predictor varibles in X

# In[ ]:


X = df.drop('Average_Fare0' , axis = 1)
y = df.Average_Fare0


# #### Generating Dummy variables of Non-Numerical Categorical columns.
# As our machine learning model can only process numerical data we have to process our non Numerical columns
# Note- Assuming that, origin city and destination city also plays crucial role while deciding the airfare, as well as the airlines too(few of the airlines can be top service provider, so their price can be slightly higher, or they are making more profit, so they can afford to minimize the price to provide better service)

# In[ ]:


X = pd.get_dummies(X)
X.head()


# ## Modeling
# 
# In this section we will perform modeling on our dataset.
# We will use different algorithms to model our data and than later pick the best one.
# 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error , make_scorer


# #### Splitting our dataset into Train and Test set uing 70:30 ratio

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.7,random_state = 42)


# In[ ]:


#printing length
len(y_train),len(X_train)


# #### Let's scale our data to reduce model complexity and increase performance

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


# creating standardscaler object
scaler = StandardScaler()

#Scaling and Transforming our training Dataframe
X_train = scaler.fit_transform(X_train)


# In[ ]:


# We don't want our test set to learn from training Data so, we are will just transform it
X_test = scaler.transform(X_test)


# ### Model Evaluation Functions
# writing functions for performance evaluation of our models

# In[ ]:


kfolds = KFold(n_splits=3, shuffle=True, random_state=42)

# model scoring and validation function
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y,scoring="r2",cv=kfolds))
    return (rmse)

# rmsle scoring function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# ### Creating Linear Regression Model with all features

# In[ ]:


linear_model_rfe = LinearRegression()
linear_model_rfe.fit(X_train,y_train)


# In[ ]:


## Train Accuracy
y_train_pred=linear_model_rfe.predict(X_train)

## Test Accuracy
y_pred=linear_model_rfe.predict(X_test)

rmsle(y_test,y_pred), rmsle(y_train,y_train_pred)


# As we can see the difference of Mean_Squared_error between Train Score and test Score is huge.
# Our model has overfitted on train set.

# In[ ]:


#### Let's verify this by evaluating r-squared score

from sklearn.metrics import r2_score
r2_score(y_train,y_train_pred)


# **95.8 % r2_score** on trainData

# In[ ]:


y_pred = linear_model_rfe.predict(X_test)
r2_score(y_test,y_pred)


# In[ ]:


### Also Cross Validation Score
cv_rmse(linear_model_rfe,X)


# In[ ]:


linear_train_score = linear_model_rfe.score(X_train,y_train)
linear_test_score = linear_model_rfe.score(X_test, y_test)
linear_train_score , linear_test_score


# #### *As we can see the performance of our baseline Model is very Bad on test Set indicating OverFitting* 
# #### We will do feature selection using VIF

# ## Part 3

# ### Feature Selection

# In[ ]:


from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


# Selection top 20 features using RFE
select = RFE(linear_model_rfe, 20 ,step=1)
select = select.fit(X_train,y_train)


# In[ ]:


# Ranking features based on their relevancy
select.ranking_


# In[ ]:


# Zipping column names, ranking and support
list(zip(X.columns,select.support_,select.ranking_))


# In[ ]:


col = X.columns[select.support_]
col


# In[ ]:


X_train_rfe = X[col]


# In[ ]:


# Checking VIF of Each predictor Variable
vif = pd.DataFrame()
vif['Features']=X_train_rfe.columns
vif['VIF'] = [ variance_inflation_factor(X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
vif.sort_values(by = 'VIF' , ascending=False)


# ### We can observe here, feature given using RFE are not quite significant

# #### Creating a baseline linear Regression model and fitting the data

# In[ ]:


# training and test data with top 20 features given by VIF
xx = pd.DataFrame(X_train, columns= X.columns)[col]
xt = pd.DataFrame(X_test, columns= X.columns)[col]


# In[ ]:


lm = LinearRegression()
lm.fit(xx,y_train)


# In[ ]:


## Train Accuracy
y_train_pred=lm.predict(xx)

## Test Accuracy
y_pred=lm.predict(xt)

rmsle(y_test,y_pred), rmsle(y_train,y_train_pred)


# As we can see the difference of Mean_Squared_error between Train Score and test Score is huge.
# Our model has overfitted on train set.

# #### Let's verify this by evaluating r-squared score

# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_train,y_train_pred)


# **45.99 % r2_score** on trainData

# In[ ]:


y_pred = lm.predict(xt)
r2_score(y_test,y_pred)


# In[ ]:


### Also Cross Validation Score
cv_rmse(lm,X)


# In[ ]:


linear_train_score = lm.score(xx,y_train)
linear_test_score = lm.score(xt, y_test)
linear_train_score , linear_test_score


# #### *As we can see the performance of our baseline Model is very Bad on test Set indicating OverFitting, and even performance on train data is also not good* 
# #### We will create fit Regularized models next to simplify our model and increase the test accuracy
# 

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.01, max_iter=10e5)
rr = Ridge(alpha=0.01)


# In[ ]:


rr.fit(X_train, y_train)
Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)
Ridge_train_score,Ridge_test_score


# **WE are getting r2Score of 98.47 and 97.85 percent respectively**
# Thus ridge is performing much better than simple linear Regression
# 
# 

# In[ ]:


lasso.fit(X_train, y_train)
Lasso_train_score = lasso.score(X_train,y_train)
Lasso_test_score = lasso.score(X_test, y_test)
Lasso_train_score,Lasso_test_score


# **WE are getting r2Score of 98.46 and 97.88 percent respectively thus lasso is outperforming Ridge on Test Set marginally**

# In[ ]:


plt.figure(figsize = (16,10))
plt.plot(rr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Ridge Regression')
plt.plot(lasso.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='red',label='Lasso Regression')

plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# ### Printing Model Coefficients as indicator of Feature importance in predicting the outcome Variable

# In[ ]:


coeff_df = pd.DataFrame(lasso.coef_, X.columns, columns=['Coefficient'])  
coeff_df['absolute'] = coeff_df.Coefficient.abs()
coeff_df.head()


# In[ ]:


coeff_df = coeff_df.sort_values(by = 'absolute', ascending = False)

# Printing top 10 variables with highest importance/impact on label per unit change
coeff_df.Coefficient.head(10)


# You can see the most important features by the magnitude of Coefficient are given above

# ## Observations

# - There were few outliers in few of the columns data, but we can ignore them as these outliers may be because of surge pricing when demand is high or during festive season.
# - As we can observe, not just continuous variables which he had plays the significant role for improving the model accuracy, but also few of the origin and destination cities and market leading airlines
# - Linear Regression model was not quite a good fit for this type of data, as it heavily overfitted on train data but accuracy on test data was very low.
# - Using VIF also, the feature which we were getting for Linear Regression model was not quite significant, so we just can't blindly trust output of VIF, because the top 20 features which we selected using RFE technique, was not making much sense
# - Ridge Regression model was performing way better than Linear Regression model, but we fitted all data into it.
# - Lasso Regression proves to be best for this type of data, as it automatically do the relevant feature selection also, and based on business understanding, the top 10 features given by Lasso model are very much relevant
# 
