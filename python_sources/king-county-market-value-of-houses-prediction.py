#!/usr/bin/env python
# coding: utf-8

# #### Objective: 
# The main objective of this assessment is to predict the `SaleDollarCnt` or the `Current Market Value` of a house.

# ### Step 1: Importing Libraries/Packages
# 
# I have used `numpy` and `pandas` which are basic libraries for Data Science in Python. Next, I have used various packages from the `sklearn` library for completing this assessment. I have used `train_test_split` for splitting the data into train and test and `KFold` for performing K-fold cross validation. Along with this, I have used `LinearRegression`, `Support Vector Regression`, `AdaBoostRegressor` and `RandomForestRegressor` for modeling purposes.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier


# ### Step 2: Loading the Data

# In[ ]:


#Loading the training data into a variable named data
data = pd.read_csv('../input/king-county-housing-dataset/train.csv')

#Loading the test data into a variable named test
test = pd.read_csv('../input/king-county-housing-dataset/test.csv')


# ### Step 3: Data Cleaning and Analysis

# In[ ]:


data.shape


# Here, I have dropped the following columns -
# 1. `Usecode` and `censusblockgroup` - I have dropped these columns as every observation has the same usecodes and censusblockgroups and it won't be useful towards predicting the `SaleDollarCnt`
# 2. `Latitude` and `Longitude` - Both these columns give approximately the same information as `ZoneCodeCounty` and given the time constraints, exploring this was not possible.
# 3. `BGMedYearBuilt` - The year when the house is made is more important than that of the block and hence eliminated this column.
# 4. `BGPctKids` - This column was dropped as it had a high correlation to `BGMedAge`. If the `BGMedAge` is on the higher side, the `BGPctKids` is on the lower side and vice versa and hence dropped this.
# 5. `ViewType` - Given that the test data can contain different views which are not there in the original data, I decided to not take this into account.

# In[ ]:


#Dropping the following columns in both data and test
data.drop(['Usecode', 'censusblockgroup', 'Latitude', 'Longitude', 
           'BGMedYearBuilt', 'BGPctKids', 'ViewType'], axis=1, inplace = True)
test.drop(['Usecode', 'censusblockgroup', 'Latitude', 'Longitude', 
           'BGMedYearBuilt', 'BGPctKids', 'ViewType'], axis=1, inplace = True)


# For cleaning the data, first I checked how many NA's are present in the data and observed that `GarageSquareFeet`, `BGMedHomeValue` and `BGMedRent` had NA's.

# In[ ]:


#Looking at the number of NA's in each column
data.isna().sum()


# To clean the NA's in `GarageSquareFeet`, I replaced the values with 0 - The reason for this is that a home has an NA in the `GarageSquareFeet`, it means that they do not have a garage and this can simply be considered as 0 square feet.

# In[ ]:


#Filled the NA's in GarageSquareFeet with 0 in data and test
data['GarageSquareFeet'] = data['GarageSquareFeet'].fillna(0)
test['GarageSquareFeet'] = test['GarageSquareFeet'].fillna(0)


# The NA's in `BGMedHomeValue` are replaced with the mean of the column as most of the values fall into the middle range and also becasue there are only 6 NA's which is a very small number in comparison to the entire data.

# In[ ]:


#Filled the NA's in BGMedHomeValue with the mean of the BGMedHomeValue
data['BGMedHomeValue'] = data['BGMedHomeValue'].fillna(data['BGMedHomeValue'].mean())
test['BGMedHomeValue'] = test['BGMedHomeValue'].fillna(test['BGMedHomeValue'].mean())


# There are 2631 NA's in `BGMedRent` and these cannot be directly replaced with the mean as it is a significant part(~20%) of the data. Thus, I have considered all the observations that are not NA's in the training set and the observations with NA's in the validation set. I have run a linear regression model to predict the missing NA's. I have calculated the mean absolute error and the root mean square error for this model which show that the model is working well. This has been done similarly for the test dataset as well.

# In[ ]:


#Making the train data with no NA's
train = data[(data.BGMedRent.notna())]
#Making the validation set with the observations that have NA's
validation = data[data.BGMedRent.isna()]

#Dividing the validation set into target and explanatory variables
validation_y = validation['BGMedHomeValue'].reset_index(name='BGMedHomeValue').drop(columns='index')
validation_x = validation.loc[:, validation.columns != 'BGMedRent'].drop(columns=['TransDate', 
                                                                                  'PropertyID', 
                                                                                  'SaleDollarCnt', 'ZoneCodeCounty'])

#Dividing the training set into target and explanatory variables
y = train['BGMedHomeValue'].reset_index(name='BGMedHomeValue').drop(columns='index')
x = train.loc[:, train.columns != 'BGMedRent'].drop(columns=['TransDate', 
                                                             'PropertyID', 
                                                             'SaleDollarCnt', 'ZoneCodeCounty'])

#Using train-test-split to divide the training data into train and validation
x_train, x_validation, y_train, y_validation = train_test_split(x, y)

#Using Linear Regression
reg = LinearRegression()
reg.fit(x_train, y_train)
predict = reg.predict(x_validation)

print('The mean absolute error is:', metrics.mean_absolute_error(y_validation, predict))
print('The root mean square error is:', np.sqrt(metrics.mean_squared_error(y_validation, predict)))

#Predicting the values of the validation set
predict_validation = reg.predict(validation_x)
indexes = data[data['BGMedRent'].isna()].index.values

#Putting the values in the indexes where the NA's are present
for i, element in enumerate(indexes):
    data.iloc[element, 12] = predict_validation[i]


# In[ ]:


#Making the train data with no NA's
train = test[(test.BGMedRent.notna())]
#Making the validation set with the observations that have NA's
validation = test[test.BGMedRent.isna()]

#Dividing the validation set into target and explanatory variables
validation_y = validation['BGMedHomeValue'].reset_index(name='BGMedHomeValue').drop(columns='index')
validation_x = validation.loc[:, validation.columns != 'BGMedRent'].drop(columns=['TransDate', 
                                                                                  'PropertyID', 
                                                                                  'SaleDollarCnt', 'ZoneCodeCounty'])

#Dividing the training set into target and explanatory variables
y = train['BGMedHomeValue'].reset_index(name='BGMedHomeValue').drop(columns='index')
x = train.loc[:, train.columns != 'BGMedRent'].drop(columns=['TransDate', 
                                                             'PropertyID', 
                                                             'SaleDollarCnt', 'ZoneCodeCounty'])

#Using train-test-split to divide the training data into train and validation
x_train, x_validation, y_train, y_validation = train_test_split(x, y)

#Using Linear Regression
reg = LinearRegression()
reg.fit(x_train, y_train)
predict = reg.predict(x_validation)

print('The mean absolute error is:', metrics.mean_absolute_error(y_validation, predict))
print('The root mean square error is:', np.sqrt(metrics.mean_squared_error(y_validation, predict)))

#Predicting the values of the validation set
predict_validation = reg.predict(validation_x)
indexes = test[test['BGMedRent'].isna()].index.values

#Putting the values in the indexes where the NA's are present
for i, element in enumerate(indexes):
    test.iloc[element, 12] = predict_validation[i]


# Checked if there are any NA's present in the data.

# In[ ]:


data.isna().sum()


# I made a new column named `Month` to see if the months have any correlation with the `SaleDollarCnt`.

# In[ ]:


#Using the pandas datetime format to get the month
data['Month'] = pd.to_datetime(data['TransDate']).dt.strftime('%m')


# Having made a new column, I grouped the data by the month to get the count of each month. I observed that the month data is very skewed as January, February, March, October, November and December are not part of the data and thus it is better to not take it into consideration. Hence, month is dropped from the data.

# In[ ]:


#Displaying the count of each month
data.groupby(data.Month).size().reset_index(name='Count')


# In[ ]:


#Dropping month
data.drop(['Month'], inplace=True, axis=1)


# I have used the corr function to find the Pearson correlation of all variables with `SaleDollarCnt`

# In[ ]:


#Correlation
data.corr().head()


# ### Step 4: Data Modeling (Prediction)

# In this step, I have taken the `SaleDollarCnt` as the predictor (target variable) and have taken the rest of the data as the explanatory variable. For the explanatory variables, I have dropped `TransDate` because it doesn't make a difference whatever date it is because the data is skewed. `PropertyID` is dropped as it is a unique identifier and isn't important for predictions. I have also dropped `ZoneCodeCounty` because it has ~170 unique counties and there is no guarantee that the test data will have the same 170 unique counties and thus it is better to drop and not consider it for now. For this problem, I have normalized the explanatory variables using mean normalization. I have stored the mean and standard deviation in mean_x and std_x and then converted it to a numpy array.

# In[ ]:


#Making the target variable as SaleDollarCnt
y = data['SaleDollarCnt']
#Making the rest of it as explanatory
x = data.loc[:, data.columns != 'SaleDollarCnt'].drop(columns=['TransDate', 'PropertyID', 'ZoneCodeCounty'])
x1 = test.loc[:, test.columns != 'SaleDollarCnt'].drop(columns=['TransDate', 'PropertyID', 'ZoneCodeCounty'])

#Stored the mean and standard deviations of all the columns
mean_x = x.mean()
std_x = x.std()

mean_x1 = x1.mean()
std_x1 = x1.std()

print(std_x1, std_x)

#Used mean normalization
x = ((x - mean_x) / std_x).to_numpy()


# In this problem, I have used `Random Forest Regressor` to do the perdiction of the data. I initially tried using `Linear Regression` and `Support Vector Regression` and `AdaBoostRegression`, but `Random Forest Regressor` gave the best result among the four. In this model, I have set the number of estimators to 1000 and set the random state to 42. I then used Kfold Cross Validation with 10 splits. The reason I used this is to make sure my training data is not overfitting. In Kfold cross validation, the data is split into 10 parts and then each part is selected one at a time to be the validation data and rest of the data is set to training data. Thus, in each iteration, I fitted the model and stored the predicted outcome in lists of `AAPE` and `MAPE`. I stored it into a list because I could then take the mean and median of those lists respectively. This is how I got the error rate of my regression model and could decide that this model is fitting in really well.

# In[ ]:


#Two empty lists are initialized for storing the results
aape, mape = [], []

#Used linear_regession
linear_reg = LinearRegression()

#Kfold cross validation is used with 10 splits
cv = KFold(n_splits=10, random_state=42, shuffle=False)

#Going over every iteration and fitted and predicted the values in every iteration
for train_index, validation_index in cv.split(x):
    x_train, x_validation, y_train, y_validation = x[train_index], x[validation_index], y[train_index], y[validation_index]
    linear_reg.fit(x_train, y_train)
    predict_linear_reg = linear_reg.predict(x_validation)
    aape.append((abs(predict_linear_reg - y_validation)/y_validation).mean())
    mape.append((abs(predict_linear_reg - y_validation)/y_validation).median())
    
print('AAPE for linear regression is:', np.mean(aape))
print('MAPE for linear regression is:', np.median(mape))


# In[ ]:


#Two empty lists are initialized for storing the results
aape, mape = [], []

#Used support vector regession
support_reg = SVR(kernel='poly', degree=5, gamma='scale')

#Kfold cross validation is used with 10 splits
cv = KFold(n_splits=10, random_state=42, shuffle=False)

#Going over every iteration and fitted and predicted the values in every iteration
for train_index, validation_index in cv.split(x):
    x_train, x_validation, y_train, y_validation = x[train_index], x[validation_index], y[train_index], y[validation_index]
    support_reg.fit(x_train, y_train)
    predict_support_reg = support_reg.predict(x_validation)
    aape.append((abs(predict_support_reg - y_validation)/y_validation).mean())
    mape.append((abs(predict_support_reg - y_validation)/y_validation).median())
    
print('AAPE for support vector regression is:', np.mean(aape))
print('MAPE for support vector regression is:', np.median(mape))


# In[ ]:


#Two empty lists are initialized for storing the results
aape, mape = [], []

#Used adaboost_regession
adaboost_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=200, learning_rate=0.8)

#Kfold cross validation is used with 10 splits
cv = KFold(n_splits=10, random_state=42, shuffle=False)

#Going over every iteration and fitted and predicted the values in every iteration
for train_index, validation_index in cv.split(x):
    x_train, x_validation, y_train, y_validation = x[train_index], x[validation_index], y[train_index], y[validation_index]
    adaboost_reg.fit(x_train, y_train)
    predict_adaboost_reg = adaboost_reg.predict(x_validation)
    aape.append((abs(predict_adaboost_reg - y_validation)/y_validation).mean())
    mape.append((abs(predict_adaboost_reg - y_validation)/y_validation).median())
    
print('AAPE for Adaboost regression is:', np.mean(aape))
print('MAPE for Adaboost regression is:', np.median(mape))


# In[ ]:


#Two empty lists are initialized for storing the results
aape, mape = [], []

#Used randomforest
random_forest = RandomForestRegressor(n_estimators = 1000, random_state=42)

#Kfold cross validation is used with 10 splits
cv = KFold(n_splits=10, random_state=42, shuffle=False)

#Going over every iteration and fitted and predicted the values in every iteration
for train_index, validation_index in cv.split(x):
    x_train, x_validation, y_train, y_validation = x[train_index], x[validation_index], y[train_index], y[validation_index]
    random_forest.fit(x_train, y_train)
    predict_random_forest = random_forest.predict(x_validation)
    aape.append((abs(predict_random_forest - y_validation)/y_validation).mean())
    mape.append((abs(predict_random_forest - y_validation)/y_validation).median())
    
print('AAPE for Random forest regression is:', np.mean(aape))
print('MAPE for Random forest regression is:', np.median(mape))


# Using the above model, I predicted the values of the final test set. For this I dropped `SaleDollarCnt`, `TransDate`, `PropertyID` and `ZoneCodeCounty` as done earlier in the training data. I normalized this data using the mean and standard deviation of the training data as the test data is not revealed and using the mean and standard deviation of training data gives a good estimate. 

# ### Step 5: Error Analysis
# In this assessment, I initially used `Linear Regression` to do the modeling of data and found the error rate to be good not but not the best. I wanted to give a shot at other models for regression and used `Support Vector Regression` next, and found the error rate to be higher than the `Linear Regression` model. I then tried using `AdaboostRegression` which gave me a really bad error rate. Then, after all this I used `Random Forest` to perform the regression and found the best results using this. 
# TO avoid the bias-variance trade off, I used `K-fold cross validation` to make sure that the data doesn't overfit or underfit. Though the one thing I observed is that `Random Forest` is among the slowest algorithms and Adaboost is a lot faster.

# ### Step 6: Future Work
# 
# 1. Due to the time constraint, I couldn't work on the zone codes or the latitude/longitudes properly. Using geospatial data, it could have been possible to get the location of the blocks and this could have been used to understand the blocks which were nearer to grocery shops/local stores. The homes near these blocks would ideally cost more and this could be used as a parameter.
# 
# 2. The seasonal trends were not analyzed because of the skewness of the data. If the data would have been evenly spread out, it could have been used to understand the seasons when there were higher and lower transactions. This could then be an important factor towards prediction.
# 
# 3. If there were different censuscodeblocks and usecodes, the model could be trained on that parameter as the usecode or censusblockcode a house is in could turn out to be important.
# 
# 4. If there was more domain knowledge on a particular censusblock to understand the views present in a block, I could have used this knowledge to be sure that the test data doesn't contain any extra or lesser views.
