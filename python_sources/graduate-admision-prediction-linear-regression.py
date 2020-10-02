#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing necessary libraries for avoiding warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# import necessary libraries for importing and understanding the data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# importing the necessary libraries for model building
import sklearn
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ### Data Inspection

# In[ ]:


# Reading the data into dataframes
admission_dataframe =  pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')


# In[ ]:


# analyzing the data
admission_dataframe.head()


# - In the above dataframe Chance of Admit is the target variable and rest other are predictor variables.

# In[ ]:


# Checking the shape of the dataframe
admission_dataframe.shape


# In[ ]:


# checking the info about the columns of the dataframe
admission_dataframe.info()


# - From the above information we can see that there exists no null values in the columns.
# - Also we can see that the data types of the columns have been correctly mapped to their respective data types.
# - All the colunmns are numerical variables expect the column Research which can ordinal categorical variable which is encoded with 1's and 0's
# - Since the target variable is quantitative variable we use regression techniques to predict the value.

# In[ ]:


# analyzing the summary statistics of the numerical columns of the dataframe
admission_dataframe[['GRE Score', 'TOEFL Score','CGPA', 'SOP', 'LOR ', 'University Rating' ]].describe()


# - From the above statistical description of the numerical column we can have an idea about the descriptive statistics value information of the numerical columns in the data.
# - Upon observing the quantile distribution of the columns we can most of the columns have almost normal distribution spread as their median values and the max values are close to each other.

# In[ ]:


# checking the columns present in the dataframe
admission_dataframe.columns


# In[ ]:


# analyzing the array of the values of the data
admission_dataframe.values


# ### EDA

# In[ ]:


# analyzing the distributions of numerical columns of the dataframe
plt.figure(figsize = (10,5)) #(Width, height) in figsize
plt.subplot(2,3,1)
sns.distplot(admission_dataframe['GRE Score'])
plt.subplot(2,3,2)
sns.distplot(admission_dataframe['TOEFL Score'])
plt.subplot(2,3,3)
sns.distplot(admission_dataframe['CGPA'])
plt.subplot(2,3,4)
sns.distplot(admission_dataframe['SOP'])
plt.subplot(2,3,5)
sns.distplot(admission_dataframe['University Rating'])
plt.subplot(2,3,6)
sns.distplot(admission_dataframe['LOR '])
plt.tight_layout()
plt.show()


# - From the distributions we can see that the distributions are centered across their mean value mostly and the distributions are almost normal distributions though they have some spikes in distribution.

# In[ ]:


# analyzing the pair plot distibutions of the numerical variables
sns.pairplot(admission_dataframe[['GRE Score', 'TOEFL Score', 'CGPA', 'SOP', 'LOR ', 'University Rating' ,'Chance of Admit ']])


# - The reason for plotting pair plots is we can get to know whether there exists amy linear relationship between the target variable and the predictors variables. Knowing so, we will get to confirmation whether we can built linear regression model using these predictor variables.
# - From the above visualization we can see that there exists some positive correlation between the taregt variable and the predictor variables. Hence the data in hand is perfectly suitable for building a linear regression predictive model.
# - At the same time, the predictor variables are also strongly correlated with each other. Which again raises the issue of multicollinearity which can be handled during modelling.<br><br>
# The pair plot just determines the scatter plot visualizations between pairs of numerical variables. In order to understand the quantified relationships between the variables we can check for heatmap.

# In[ ]:


# plotting the heat map between the numerical variables
matrix = admission_dataframe[['GRE Score', 'TOEFL Score', 'CGPA', 'SOP', 'LOR ', 'University Rating' ,'Chance of Admit ']].corr()
# plotting the heatmap using seaborn
sns.heatmap(matrix, annot = True, linecolor= 'white', linewidths= 1)


# - From the heat map we have an quantified correlations between pairs of numerical variables of the data.
# - We can see that predictor variables are strongly correlated among each others.
# - Also there exists strong positive correlation between target and the predictor variables.

# In[ ]:


# analyzing the variation between categorical variables in relation with target variables
plt.figure(figsize=(7,5))
sns.boxplot(admission_dataframe['Research'], admission_dataframe['Chance of Admit '])
plt.tight_layout()
plt.show()


# - From the above plot we can see that variation Research variable in relation with target variable. The median value of people having done Research is higher when compared to people who didn't perform any Research activity. Thus people having done Reseach have higher chnaces getting a admission.

# ###  Data Preparation

# In[ ]:


# analyzing the dataframe again
admission_dataframe.head()


# __The steps involved in Data Preparation as follows :__
# - Train Test Split of the data
# - Rescaling of the features in order to bring all the feature variables into single scale for better interpretation of variables in relation with target variable over other variables
# - Since we don't have any categorical variables we no need to perform one hot encoding or the dummy encoding process.

# In[ ]:


# Performing Train Test split
df_train, df_test = train_test_split(admission_dataframe, train_size = 0.70, test_size = 0.30, random_state = 100)


# In[ ]:


# checking the shapes of train and test dataframe
print(df_train.shape, df_test.shape)


# In[ ]:


# Dropping the column serial number as it will be no use in modelling
df_train.drop('Serial No.', axis = 1, inplace = True)
df_test.drop('Serial No.', axis =1, inplace = True)


# In[ ]:


# checking the df_train
df_train.head()


# In[ ]:


# Scaling the features of the dataframe

# initiating the scaler object
scaler = MinMaxScaler()

# fittting and transforming the data on top of the scaler object for the training data set
df_train[::] = scaler.fit_transform(df_train)


# In[ ]:


# checking the dataframe after scaling the variables
df_train.head()


# In[ ]:


# checking the describe of the dataframe to check whether scaling has been done or not
df_train.describe()


# - Since we have used the min-max scaler we can see that all the max values of the variables are 1.0. Since we have fitted and transformed on the same training dataset we can see max values as 1.0.

# - Let's start the model building process using the Manual selection method. In which we will be using top down approach or the backward selection method. In which we will be considering all the features at a time and perform manual feature elimination step by step taking into consideration of p values and VIF values of the coefficients.
# - Before building the model we need to create X_train and y_train out of df_train dataset.

# In[ ]:


# creating X_train and y_train
y_train = df_train.pop('Chance of Admit ')
X_train =  df_train


# In[ ]:


# checking X_train 
X_train.head()


# In[ ]:


# checking y_train
y_train.head()


# ### Model Building

# #### First Model

# In[ ]:


# adding constant to X_train as statsmodel in built doesn't add constant
X_train_sm = sm.add_constant(X_train)

# creating the model object
lr_model_1 = sm.OLS(y_train, X_train_sm)

# fitting the model on top of the data
lr_model_1 = lr_model_1.fit()

# checking the summary statistics of the model
lr_model_1.summary()


# - from the above summary statistics of the model we can see that there are some features which are insignificant as per their p values and also the R square value si terribly high which is an indication that model has overfitted on the training dataset rather than identifying generalised patterns in the traning dataset. In order to avoid overfitting problem we need to make the model light by removing some of the insignificant features from the model so that model will be kind of balanced between the bias and variance.

# In[ ]:


# checking the VIF values

# creating the dataframe of the VIF values
VIF = pd.DataFrame()

# adding column to the VIF dataframe for columns of the X_train
VIF['features'] = X_train.columns

# adding column for the VIF values of the features
VIF['VIF_value'] =  [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

# sorting the dataframe based on VIF_value
VIF.sort_values('VIF_value', ascending= False, inplace = True)

# checking the dataframe
VIF


# __Steps involved in Manual Feature Elimination are:__
# - check for the variables having high vif and high p values so that they can be dropped
# - check for the variables having high p value and low vif so that they can be dropped
# - check for the variables having low p value and high vif which can be dropped once the variables having high p value and low vif are dropped. Doing so, the variables having high vif value will be dropped upon rebuilding the model.

# - As the above summary statistics of the model and VIF values we can see that SOP feature has high p value and realtively low vif value when compared to other features. Hence it is better to drop first the insignificant features so that the features having vif values can evidence the decrease in vif value.

# In[ ]:


# dropping SOP from X_train
X_train.drop('SOP', axis = 1, inplace = True)


# #### Model 2

# In[ ]:


# adding constant to X_train as statsmodel in built doesn't add constant
X_train_sm = sm.add_constant(X_train)

# creating the model object
lr_model_2 = sm.OLS(y_train, X_train_sm)

# fitting the model on top of the data
lr_model_2 = lr_model_2.fit()

# checking the summary statistics of the model
lr_model_2.summary()


# - From tha above summary statistics still we can find some of the features which are insignificant. Also the R square value of the model still considerably high. Let's look at the VIF values of the features before dropping the features.

# In[ ]:


# checking the VIF values

# creating the dataframe of the VIF values
VIF = pd.DataFrame()

# adding column to the VIF dataframe for columns of the X_train
VIF['features'] = X_train.columns

# adding column for the VIF values of the features
VIF['VIF_value'] =  [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

# sorting the dataframe based on VIF_value
VIF.sort_values('VIF_value', ascending= False, inplace = True)

# checking the dataframe
VIF


# - As per above summary statistics and the VIF values we can see that there are some features having low p values and high vif values and also features having high p value and low vif values. But as per the call we proceed to drop the features having high p value and low vif first and then again look at the features having vif values after rebuilding the model.

# In[ ]:


# Dropping Univeristy Rating from X_train
X_train.drop('University Rating', axis = 1, inplace = True)


# #### Model 3

# In[ ]:


# adding constant to X_train as statsmodel in built doesn't add constant
X_train_sm = sm.add_constant(X_train)

# creating the model object
lr_model_3 = sm.OLS(y_train, X_train_sm)

# fitting the model on top of the data
lr_model_3 = lr_model_3.fit()

# checking the summary statistics of the model
lr_model_3.summary()


# - From the above summary statistics we can see that all the features are significant as per their p values. But still the R square value is very high which is an indication that the model is still overfitting on the training dataset. Let's look at the VIF values before taking any call for dropping a particular feature.

# In[ ]:


# checking the VIF values

# creating the dataframe of the VIF values
VIF = pd.DataFrame()

# adding column to the VIF dataframe for columns of the X_train
VIF['features'] = X_train.columns

# adding column for the VIF values of the features
VIF['VIF_value'] =  [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

# sorting the dataframe based on VIF_value
VIF.sort_values('VIF_value', ascending= False, inplace = True)

# checking the dataframe
VIF


# - We can see that some of the features having insignificant p values. Let's drop such features having high p value and low vif values.

# In[ ]:


# Dropping CGPA from X_train
X_train.drop('GRE Score', axis = 1, inplace = True)


# #### Model 4

# In[ ]:


# adding constant to X_train as statsmodel in built doesn't add constant
X_train_sm = sm.add_constant(X_train)

# creating the model object
lr_model_4 = sm.OLS(y_train, X_train_sm)

# fitting the model on top of the data
lr_model_4 = lr_model_4.fit()

# checking the summary statistics of the model
lr_model_4.summary()


# - From above summary statistics we can see that all the features are significant as per teir p values. But the R square value is still on a higher side. Having such high value will be problem for the model prediction on a unseen dataset. Let's look at the VIF values before taking any call for dropping the features so as to make the model light.

# In[ ]:


# checking the VIF values

# creating the dataframe of the VIF values
VIF = pd.DataFrame()

# adding column to the VIF dataframe for columns of the X_train
VIF['features'] = X_train.columns

# adding column for the VIF values of the features
VIF['VIF_value'] =  [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

# sorting the dataframe based on VIF_value
VIF.sort_values('VIF_value', ascending= False, inplace = True)

# checking the dataframe
VIF


# - We can still see some of the features having high VIF values. Let's drop the feature having highest VIF in order to make the model light and free from overfitting.

# In[ ]:


# Dropping TOEFL Score from X_train
X_train.drop('CGPA', axis = 1, inplace = True)


# #### Model 5

# In[ ]:


# adding constant to X_train as statsmodel in built doesn't add constant
X_train_sm = sm.add_constant(X_train)

# creating the model object
lr_model_5 = sm.OLS(y_train, X_train_sm)

# fitting the model on top of the data
lr_model_5 = lr_model_5.fit()

# checking the summary statistics of the model
lr_model_5.summary()


# - From the above summary statistics we can see that all the features are statistically significant.

# In[ ]:


# checking the VIF values

# creating the dataframe of the VIF values
VIF = pd.DataFrame()

# adding column to the VIF dataframe for columns of the X_train
VIF['features'] = X_train.columns

# adding column for the VIF values of the features
VIF['VIF_value'] =  [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

# sorting the dataframe based on VIF_value
VIF.sort_values('VIF_value', ascending= False, inplace = True)

# checking the dataframe
VIF


# - We can see that features are significant as per p values. In order to make the model further light let's drop the feature having high vif value.

# In[ ]:


# Dropping Research from X_train
X_train.drop('TOEFL Score', axis = 1, inplace = True)


# #### Model 6

# In[ ]:


# adding constant to X_train as statsmodel in built doesn't add constant
X_train_sm = sm.add_constant(X_train)

# creating the model object
lr_model_6 = sm.OLS(y_train, X_train_sm)

# fitting the model on top of the data
lr_model_6 = lr_model_6.fit()

# checking the summary statistics of the model
lr_model_6.summary()


# - From above summary statistics we can see that all the features are significant as per their p values. Let's look at their VIF values.

# In[ ]:


# checking the VIF values

# creating the dataframe of the VIF values
VIF = pd.DataFrame()

# adding column to the VIF dataframe for columns of the X_train
VIF['features'] = X_train.columns

# adding column for the VIF values of the features
VIF['VIF_value'] =  [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

# sorting the dataframe based on VIF_value
VIF.sort_values('VIF_value', ascending= False, inplace = True)

# checking the dataframe
VIF


# - As per summary statistics and vif values we can see that features are significant as per their p values and vif values which an indication that there doesn't exists any multicollinearity issue in the model. In order to check whether the model we have built was able to pick generalised patterns in the data or not we need to check on the test data set. If the performace of the model on the test dataset as well is close to train dataset performance then we can say that the model is kind of generalized but not an best model.

# ### Residual Analysis

# In[ ]:


# Making predictions on the train dataset
y_train_pred = lr_model_6.predict(X_train_sm)


# In[ ]:


# analyzing the y_train_pred
y_train_pred.head()


# In[ ]:


# checking the distribution of the error terms
res = y_train - y_train_pred
sns.distplot(res)


# - From the above distribution we can see that the error terms are more of the sort normally distributed hence we have satisfied the assumption of error terms hould be normally distributed with mean 0 and some standard deviation. Also we have satisfied the assumption of homosadacity which says that there should be constant variance in the error terms from the above distribution.

# In[ ]:


# checking for the assumption whether there exists any pattern in the error terms
plt.figure()
sns.scatterplot(y_train, res)
plt.show()


# - The reason for plotting the error terms in relation with either y_train or y_test is to check whether there exists any visible pattern in the error terms or not. If there exists any visible pattern in the error terms distribution then it is can indication that the model has failed to include some of the explanatory variables and in such case there is a requirement of rebuilding the model. If there exists no visible pattern in the error terms distribution then it is an indication that the model has captured all the explanatory features and has left behind random noise in the error terms distribution. From the above distribution we can see that there doesn't exists any fixed pattern in error terms distribution.

# ### Making Predictions

# In[ ]:


# analyzing the df_test
df_test.head()


# In[ ]:


# scaling the test data using the scaler object defined for scaling of train dataset
df_test[::] = scaler.transform(df_test)


# In[ ]:


# analyzing the df_test after scaling
df_test.head()


# In[ ]:


# creating X_test and y_test
y_test =  df_test.pop('Chance of Admit ')
X_test = df_test


# In[ ]:


# analyzing X_test
X_test.head()


# In[ ]:


# analyzing y_test
y_test.head()


# In[ ]:


# modifying the X_test using the features of the model build
X_test = X_test[X_train.columns]


# In[ ]:


# adding constant to X_test
X_test_sm = sm.add_constant(X_test)


# In[ ]:


# making predictions using model
y_test_pred = lr_model_6.predict(X_test_sm)


# In[ ]:


# checking the r2_score on the predictions made on the test set
r2_score_test = r2_score(y_test, y_test_pred)
r2_score_test


# In[ ]:


# checking the r2_score on the train set again
r2_score_train = r2_score(y_train, y_train_pred)
r2_score_train


# - From the above R square values on the train and test we can see that model is able to perform slightly well on the unseen dataset when compared to the trained dataset.

# In[ ]:


# checking the MSE and RMSE as well on the test data set
mean_squared_error_test =  mean_squared_error(y_test, y_test_pred)
print('The MSE value on test dataset {}'.format(mean_squared_error_test))
RMSE_test = np.sqrt(mean_squared_error_test)
print('The RMSE value on test dataset {}'.format(RMSE_test))


# In[ ]:


# checking the MSE and RMSE as well on the train data set
mean_squared_error_train =  mean_squared_error(y_train, y_train_pred)
print('The MSE value on train dataset {}'.format(mean_squared_error_train))
RMSE_train = np.sqrt(mean_squared_error_train)
print('The RMSE value on train dataset {}'.format(RMSE_train))


# - From the above mean square error and root mean square error values we can see that the values are considerably small which is an inidcation that there doesn't exists much of the difference between the true values and the predicted values.

# ### Task 2

# - Considering last 100 observations of the validation set and indicating the error value 

# In[ ]:


# analyzing the error between y_test and y_test_pred
error = y_test - y_test_pred
error.head()


# In[ ]:


# creating a dataframe having y_last and the error terms
last_error_datafarme = pd.DataFrame()

# adding column for y_last
last_error_datafarme['True values'] = y_test

# adding error terms as a column
last_error_datafarme['Error values'] = error

# restricting the observation to be last 100 of the validation set
last_error_datafarme = last_error_datafarme.tail(100)

# checking the dataframe
last_error_datafarme.head()


# In[ ]:


# checking the shape of the last_error_dataframe
last_error_datafarme.shape


# - Hence we have mentioned the error caused between the actual and predicted values by the model for the last 100 data points of the vadlidation dataset.

# ### Task 3

# - Assign RMSE values for the last 100 observations of the test dataset

# In[ ]:


# checking the dataframe for which last 100 observations error has been defined
last_error_datafarme.head()


# In[ ]:


# adding a column for computing RMSE for each data point
last_error_datafarme['RMSE'] = last_error_datafarme['Error values'].apply(lambda x: np.sqrt(x**2))


# In[ ]:


# checking the dataframe after addition of column
last_error_datafarme.head()


# - Generally RMSE metric is calculated to evaluate the goodness of fit of the model. Lower the RMSE the better the model. RMSE value indicates the variance between the predicted values and the actual values. Lower RMSE value indicates that the model has fit closely to the actual data points and there exists minimum difference between the actual and the predicted data points. But when needed to calculate the RMSE value for indiviual data point we would be computing by taking the root of the square error of that particular data point.
