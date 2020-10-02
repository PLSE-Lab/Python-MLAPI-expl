#!/usr/bin/env python
# coding: utf-8

# # Sales Prediction

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model 
from sklearn import neural_network 
from sklearn import ensemble

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Objective

# Using the features provided, we have to predict the weekly sales for the each department in each store.

# ## Reading the Data

# In[ ]:


train_set = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
test_set = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip')
stores = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')
features = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip')


# ## Analyzing the Data

# The first step we are going to do is to take a quick glance in the data we have. After that we can see some aspects such as, the number of instances, the number of features, how many missing values we have, etc.

# In[ ]:


test_set.info()


# In[ ]:


train_set.info()


# In[ ]:


stores.info()


# In[ ]:


features.info()


# ### Analyzing the Missing Values

# At this time, we can see that only the features set has missing values. So before merging it with the others sets, let's do some analysis.

# In[ ]:


#Selecting the features which have missing values
features_nan = features.iloc[:, (features.isna().sum() > 0).values].columns

#Calculating the percentage of the missing values
percentage_nan = (features.isna().sum()/features.shape[0])*100


#plotting those values
fig, ax = plt.subplots(figsize=(10, 5))
ax.axhline(y=50, color="red", linestyle="--")
ax.bar(features_nan, percentage_nan[features_nan].values)
ax.set_ylabel('Percentage of Missing Values', fontsize=13)
ax.set_xlabel('Features which have Missing Values', fontsize=13)
ax.set_title('Features set Missing Values Analysis', fontsize=16)
#ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")


# Looking at the graph we can observe that all the MarkDown features have more than a half of its values as missing values, what may be a problem during the training. However, let's analyse if the missing values corresponds to dates and stores that are contained into the training and testing sets. For that purpose, we are going to merge the sets using the inner mode.

# In[ ]:


training_data = train_set.merge(stores).merge(features).sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)
test_data = test_set.merge(stores).merge(features).sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)

del stores, features, train_set, test_set

Y = training_data['Weekly_Sales']


# In[ ]:


training_data.info()


# In[ ]:


test_data.info()


# In[ ]:


#Selecting the features which have missing values
training_data_nan = training_data.iloc[:, (training_data.isna().sum() > 0).values].columns

#Calculating the percentage of the missing values
percentage_nan = (training_data.isna().sum()/training_data.shape[0])*100


#plotting those values
fig, ax = plt.subplots(figsize=(10, 5))
ax.axhline(y=50, color="red", linestyle="--")
ax.bar(training_data_nan, percentage_nan[training_data_nan].values)
ax.set_ylabel('Percentage of Missing Values', fontsize=13)
ax.set_xlabel('Features which have Missing Values', fontsize=13)
ax.set_title('New Training Set Missing Values Analysis', fontsize=16)
#ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")


# In[ ]:


#Selecting the features which have missing values
test_data_nan = test_data.iloc[:, (test_data.isna().sum() > 0).values].columns

#Calculating the percentage of the missing values
percentage_nan = (test_data.isna().sum()/test_data.shape[0])*100


#plotting those values
fig, ax = plt.subplots(figsize=(10, 5))
ax.axhline(y=50, color="red", linestyle="--")
ax.bar(test_data_nan, percentage_nan[test_data_nan].values)
ax.set_ylabel('Percentage of Missing Values', fontsize=13)
ax.set_xlabel('Features which have Missing Values', fontsize=13)
ax.set_title('New Test Set Missing Values Analysis', fontsize=16)
#ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")


# We can see that after the merge, the CPI and Unemployment did not affect the training data and affected a little bit the test data. However, the MarkDown features affected a lot the training data and this will be taken into account in a further analysis.

# ### Classifying the features

# Now let's classify the features into categorical and numerical.

# In[ ]:


all_features = training_data.columns
categorical_features = ['Store', 'Dept', 'Date', 'IsHoliday', 'Type']
target_value = 'Weekly_Sales'
numeric_features = all_features.drop(categorical_features)
numeric_features = numeric_features.drop(target_value)


# ## Numerical Data

# ### Analyzing Histograms

# Histograms are useful tools that can give to us informations such as Range and Skewness for numerical data. So, let's plot the histogram for the numerical data.

# In[ ]:


training_data[numeric_features].hist(figsize=(12,8))
plt.tight_layout()
plt.show()


# Through these histograms, we can see that these features have different scales and they also have some skewness. So, we may need a further transformation which minimizes theses effects.

# ### Weekly Sales x Numerical Features

# ### CPI

# In[ ]:


#Selecting the CPI values
CPI = training_data['CPI'].values
Weekly_Sales = training_data['Weekly_Sales'].values

#plotting the relation
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(CPI, Weekly_Sales)
ax.set_ylabel('Weekly Sales', fontsize=13)
ax.set_xlabel('CPI', fontsize=13)


# ### Fuel Price

# In[ ]:


#Selecting the Fuel Price values
Fuel_Price = training_data['Fuel_Price'].values
Weekly_Sales = training_data['Weekly_Sales'].values

#plotting the relation
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Fuel_Price, Weekly_Sales)
ax.set_ylabel('Weekly Sales', fontsize=13)
ax.set_xlabel('Fuel Price', fontsize=13)


# ### MarkDown1

# In[ ]:


#Selecting the MArkDown1 values
Mkd1 = training_data['MarkDown1'].values
Weekly_Sales = training_data['Weekly_Sales'].values

#plotting the relation
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Mkd1, Weekly_Sales)
ax.set_ylabel('Weekly Sales', fontsize=13)
ax.set_xlabel('MarkDown1', fontsize=13)


# ### MarkDown2

# In[ ]:


#Selecting the MarkDown2 values
Mkd2 = training_data['MarkDown2'].values
Weekly_Sales = training_data['Weekly_Sales'].values

#plotting the relation
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Mkd2, Weekly_Sales)
ax.set_ylabel('Weekly Sales', fontsize=13)
ax.set_xlabel('MarkDown2', fontsize=13)


# ### MarkDown3

# In[ ]:


#Selecting the MarkDown3 values
Mkd3 = training_data['MarkDown3'].values
Weekly_Sales = training_data['Weekly_Sales'].values

#plotting the relation
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Mkd3, Weekly_Sales)
ax.set_ylabel('Weekly Sales', fontsize=13)
ax.set_xlabel('MarkDown3', fontsize=13)


# ### MarkDown4

# In[ ]:


#Selecting the MarkDown4 values
Mkd4 = training_data['MarkDown4'].values
Weekly_Sales = training_data['Weekly_Sales'].values

#plotting the relation
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Mkd4, Weekly_Sales)
ax.set_ylabel('Weekly Sales', fontsize=13)
ax.set_xlabel('MarkDown4', fontsize=13)


# ### MarkDown5

# In[ ]:


#Selecting the CPI values
Mkd5 = training_data['MarkDown5'].values
Weekly_Sales = training_data['Weekly_Sales'].values

#plotting the relation
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Mkd5, Weekly_Sales)
ax.set_ylabel('Weekly Sales', fontsize=13)
ax.set_xlabel('MarkDown5', fontsize=13)


# ### Size

# In[ ]:


#Selecting the Size values
Size = training_data['Size'].values
Weekly_Sales = training_data['Weekly_Sales'].values

#plotting the relation
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Size, Weekly_Sales)
ax.set_ylabel('Weekly Sales', fontsize=13)
ax.set_xlabel('Size', fontsize=13)


# ### Temperature

# In[ ]:


#Selecting the Temperature values
Temperature = training_data['Temperature'].values
Weekly_Sales = training_data['Weekly_Sales'].values

#plotting the relation
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Temperature, Weekly_Sales)
ax.set_ylabel('Weekly Sales', fontsize=13)
ax.set_xlabel('Temperature', fontsize=13)


# ### Unemployment

# In[ ]:


#Selecting the Unemployment values
Unemployment = training_data['Unemployment'].values
Weekly_Sales = training_data['Weekly_Sales'].values

#plotting the relation
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Unemployment, Weekly_Sales)
ax.set_ylabel('Weekly Sales', fontsize=13)
ax.set_xlabel('Unemployment', fontsize=13)


# Here we can see a low correlation between the numerical features and the Weekly Sales. Also, we can see that most of the feature values result in Weekly Sales less than 300'000 dolars, what allow us to infer that those higher values are related with the other ones.

# ## Correlation Matrix

# Additionally to the graphical analysis, let's calculate the correlation matrix of those features.

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(training_data[np.append(numeric_features, target_value)].corr(method='pearson'), annot=True, 
            fmt='.2f', cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")


# ## Categorical Data

# ## Histogram Analysis

# In[ ]:


sns.set(font_scale=1)
plt.figure(figsize=(20, 5))
sns.countplot(training_data['Store'], color='gray')


# In[ ]:


sns.set(font_scale=1)
plt.figure(figsize=(20,5))
sns.countplot(training_data['Dept'], color='gray')


# In[ ]:


sns.set(font_scale=1)
plt.figure(figsize=(20, 5))
chart = sns.countplot(training_data['Date'], color='gray')
chart.set(xticklabels=[])


# In[ ]:


sns.set(font_scale=1)
plt.figure(figsize=(20, 5))
sns.countplot(training_data['IsHoliday'], color='gray')


# In[ ]:


sns.set(font_scale=1)
plt.figure(figsize=(20, 5))
sns.countplot(training_data['Type'], color='gray')


# The first three features have a high level of uniformity, which is good for training. The others are right skewed. In the date histogram the xlabels were hided due to the fact that there are a lot of labels, what impairs viewing.

# ### Weekly Sales X Categorical Features

# ### Stores

# In[ ]:


#Selecting the Stores values
Stores = training_data['Store'].values
Weekly_Sales = training_data['Weekly_Sales'].values

#plotting the relation
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Stores, Weekly_Sales)
ax.set_ylabel('Weekly Sales', fontsize=13)
ax.set_xlabel('Store', fontsize=13)


# There are some stores that have a higher weely sales than others. This fact may be associated with other atributes such as size and type.

# ### Dept

# In[ ]:


#Selecting the Dept values
Dept = training_data['Dept'].values
Weekly_Sales = training_data['Weekly_Sales'].values

#plotting the relation
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Dept, Weekly_Sales)
ax.set_ylabel('Weekly Sales', fontsize=13)
ax.set_xlabel('Dept', fontsize=13)


# As expected, there are some departments that sell more than others. This may be associated with the fact that there are some products that are highly attractive to the public even with a high price, so they associate a high price with a high sales amount. We do not have the names of these departments, but that may be a hypothesis.

# ### Date

# In[ ]:


#Selecting the Date values
Date = training_data['Date'].values
Weekly_Sales = training_data['Weekly_Sales'].values

#plotting the relation
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Date, Weekly_Sales)
ax.set_ylabel('Weekly Sales', fontsize=13)
ax.set_xlabel('Date', fontsize=13)
ax.set_xticklabels([])


# Here we can see that there are four dates which the weekly sales are much greater than the others. These dates may be holidays. Let's evaluate this Hypothesis.

# In[ ]:


great_sales = training_data[training_data['Weekly_Sales'] > 300000]
great_sales[['Date', 'Dept', 'IsHoliday']]


# As we can see, our hypothesis was correct. The weekly sales are highly increased by the Thanksgiving holiday. It is important to cite that the Christmas also increase the sales, despite not being classified as a holiday in its week. We can also note that the department that is affected by these holidays is the 72 and 7 departments, which may be related with gifts. 

# ### IsHoliday

# In[ ]:


#Selecting the IsHoliday values
IsHoliday = training_data['IsHoliday'].values
Weekly_Sales = training_data['Weekly_Sales'].values

#plotting the relation
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(IsHoliday, Weekly_Sales)
ax.set_ylabel('Weekly Sales', fontsize=13)
ax.set_xlabel('IsHoliday', fontsize=13)


# As explained above, the holidays increase the weekly sales, so this result is expected.

# ### Type

# In[ ]:


#Selecting the Tyoe values
Type = training_data['Type'].values
Weekly_Sales = training_data['Weekly_Sales'].values

#plotting the relation
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(Type, Weekly_Sales)
ax.set_ylabel('Weekly Sales', fontsize=13)
ax.set_xlabel('Type', fontsize=13)


# The Stores with type A and B have higher weekly sales. This may be indicate some correlation with other features such as Size, Temperature and Unemployment.

# ## Encoding the Categorical Variables

# After analyzing the features and before doing some feature engineering, we are going to encode the categorical features, due to the fact that the machine learning algorithms just can understand numbers. So, the encoding step is just a transformation which receives a categorical variable and returns a code number.

# ### Encoding Date Format

# The First variable we are goingo to encode is the date. As stated in the [data description section](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data), the date is actually representing just the week and not an specific day. So we are going to encode the date as two additional features, Week and Year.

# In[ ]:


#Training_Data
training_data['Week'] = pd.to_datetime(training_data['Date']).dt.week
training_data['Year'] = pd.to_datetime(training_data['Date']).dt.year
training_data = training_data.drop(columns='Date')

#Test_Data
test_data['Week'] = pd.to_datetime(test_data['Date']).dt.week
test_data['Year'] = pd.to_datetime(test_data['Date']).dt.year

#We are going to need for the submission
Date_list = [str(x) for x in test_data['Date']]

test_data = test_data.drop(columns='Date')

#The Categorical_feature has changed
categorical_features = ['Store', 'Dept', 'Year', 'Week', 'IsHoliday', 'Type']


# ### Encoding Type and IsHoliday Features

# The Type feature has three possible values: 'A', 'B' and 'C'.
# The IsHoliday feature has two possible values: 'Yes' and 'No'.
# 
# The IsHoliday will be encoded as the traditional 'Yes': 1, 'No': 0, used for binary variables, while the Type feature will be encoded as 'A':0, 'B':1, 'C':2. This is not the best encoder due to the fact that the Machine Learning model may interpret an order in that sequence, but it has some interesting characteristics which are: (i) simplicity and that (ii) the model do not add new features, which is highly desireble because can avoid increasing the overfitting. So, for this moment let's use this encoder as our first try.

# In[ ]:


Types = np.unique(training_data['Type'])
TypeOrdinal = preprocessing.LabelEncoder()
TypeOrdinal.fit(Types)
training_data['Type'] = TypeOrdinal.transform(training_data['Type'])
test_data['Type'] = TypeOrdinal.transform(test_data['Type'])

Holidays = np.unique(training_data['IsHoliday'])
IsHolidayOrdinal = preprocessing.LabelEncoder()
IsHolidayOrdinal.fit(Holidays)
training_data['IsHoliday'] = IsHolidayOrdinal.transform(training_data['IsHoliday'])
test_data['IsHoliday'] = IsHolidayOrdinal.transform(test_data['IsHoliday'])


# The Store and Dept features are categorical features which are encoded as an Ordinal Variable. This is also not the best encoder because it gives the sense that the instances of those features have some order, what is not true. But to avoid using other encoders that add new features, such as the OneHotEncoder, we will keep that for the moment and we are going to evaluate later. 

# ### Filling the Nan Values

# As we saw before, the training data has some missing values in the the MarkDown1-5 featues, and the test data has some missing values in the MarkDown1-5, CPI and Unemployment. As these features have skewed data, we are going to fill the missing values with the median, this avoids that the values with high occurence appears even more, that what could happen with approaches like the mean, in this case.

# In[ ]:


training_data = training_data.fillna(training_data.median())
test_data = test_data.fillna(test_data.median())


# ## Scaling the Variables

# As the majority of our algorithms use optimization methods which are based on the gradient calculation and as we saw before that our features have scales, it is a good approach to do a transformation that minimizes thoses scales differences. So, we are going to do a data standardization which removes the mean and scales to the unit variance. This approach is suitable for gradient based optimization algorithms.

# In[ ]:


#Separating Between features and target value
Y = training_data[target_value]
X = training_data.drop(columns=target_value)

#Removing the Weekly Sales
training_data_features = X.columns
features_without_holiday = training_data_features.drop('IsHoliday') #We are not going to scale the holliday

scaler = preprocessing.StandardScaler().fit(training_data[features_without_holiday])
X_test = test_data.copy()

X[features_without_holiday] = scaler.transform(X[features_without_holiday])
X_test[features_without_holiday] = scaler.transform(test_data[features_without_holiday])

X.head()

#training_data_scaled = training_data.copy()
#training_data_scaled = training_data_scaled.drop(columns='Weekly_Sales')
#test_data_scaled = test_data.copy()


# ### Correlation Matrix of the Categorical Features

# Now that we did the encoding part, let's see the correlation between all these features and the sales.

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(training_data.corr(method='pearson'), annot=True, 
            fmt='.2f', cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal")


# ## Feature Selection

# As some features have high correlation with the others, we can drop them before the training step because this may help us to avoid the overfitting. So, we are going to drop the Fuel Price, which has high correlation with the Year. We will keep the size and the type, which have high correlation with each other, because they have a relatively high correlation (if compared with the others).
# In addition to that, the MarkDown features have a lot of missing values and a low correlation with the weekly sales, so we are also going to drop them.

# In[ ]:


features_drop = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Fuel_Price']

X = X.drop(columns=features_drop)
X_test = X_test.drop(columns=features_drop)

X.head()


# In[ ]:


X_test


# ## Validation Set

# For evaluating the hyperparameters we are going to use the Hold-Out Validation strategy. This approach selects a part of the training set for training and the rest for validation. This approach is less precise than the Cross-Validation, but is faster due to the fact that you do not need to train and test over n train/validation sets.

# In[ ]:


X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, Y, 
                                                                                test_size = 0.2, random_state = 42)


# ## Model Selection

# ### Defining the metric

# In[ ]:


def WMAE(holiday, y_hat, y):
    W = np.ones(y_hat.shape)
    
    W[holiday == 1] = 5
    
    metric = (1/np.sum(W))*np.sum(W*np.abs(y-y_hat))
    
    return metric


# Initially we are going to try the linear models due to the fact that they are simple and fast.

# ### Ridge Regression Model

# In[ ]:


#List of alphas to be tested
alphas = np.logspace(-6, 6, 13)
WMAE_list_val = []
WMAE_list_tra = []

for i in alphas:
    RidgeModel = linear_model.Ridge(alpha=i)
    RidgeModel.fit(X_train, y_train)
    
    y_hat_val = RidgeModel.predict(X_validation)
    y_hat_tra = RidgeModel.predict(X_train)
    
    wError_val = WMAE(X_validation['IsHoliday'], y_hat_val, y_validation)
    wError_tra = WMAE(X_train['IsHoliday'], y_hat_tra, y_train)
    
    WMAE_list_val.append(wError_val)
    WMAE_list_tra.append(wError_tra)


# In[ ]:


#Ploting the Training and Validation Curves
plt.plot(alphas, WMAE_list_tra, label='Training Set')
plt.plot(alphas, WMAE_list_val, label='Validation Set')
plt.xlabel('Alpha')
plt.ylabel('Weighted Mean Absolute Error')
plt.title('Ridge Regression Error Analysis')
plt.legend()
plt.show()

#Selecting the alpha
WMAE_min = np.amin(WMAE_list_val)
alpha_min = alphas[WMAE_list_val == WMAE_min]

print("Alpha: " + str(alpha_min))
print("Minimum WMAE: " + str(WMAE_min))


# As we saw, the linear models do not have enough "flexibility" for this data set. That can be seen through the previous graph which shows the error in the training and test set and as the training set error was very high, the linear model underfitted the data. So, we need models that can map functions that are more non-linear. Then, let's try another class of Machine Learning models which is the Ensemble models. 

# ### Random Forest

# In[ ]:


#Params
n_estimators = [10, 50, 80, 100, 150, 200]
WMAE_list_val = []
WMAE_list_tra = []

for i in n_estimators:
    print('Number of Estimators: ', i)
    RF = ensemble.RandomForestRegressor(n_estimators = i, n_jobs = -1, random_state = 42)
    RF.fit(X_train, y_train)
    
    y_hat_val = RF.predict(X_validation)
    y_hat_tra = RF.predict(X_train)
    
    wError_val = WMAE(X_validation['IsHoliday'], y_hat_val, y_validation)
    wError_tra = WMAE(X_train['IsHoliday'], y_hat_tra, y_train)
    
    WMAE_list_val.append(wError_val)
    WMAE_list_tra.append(wError_tra)


# In[ ]:


#Ploting the Training and Validation Curves
plt.plot(n_estimators, WMAE_list_tra, label='Training Set')
plt.plot(n_estimators, WMAE_list_val, label='Validation Set')
plt.xlabel('Number of Estimators')
plt.ylabel('Weighted Mean Absolute Error')
plt.title('Random Forest Regression Error Analysis')
plt.legend()
plt.show()

#Selecting the alpha
WMAE_min = np.amin(WMAE_list_val)
n_estimators_min = n_estimators[np.argmax(WMAE_list_val == WMAE_min)]

print("Number of Estimators: " + str(n_estimators_min))
print("Minimum WMAE: " + str(WMAE_min))


# In[ ]:


#Params
max_depth = [10, 30, 50, 70, 90]
WMAE_list_val = []
WMAE_list_tra = []

for i in max_depth:
    print('Max Depth: ', i)
    RF = ensemble.RandomForestRegressor(n_estimators = n_estimators_min, max_depth = i, 
                                        n_jobs = -1, random_state = 42)
    RF.fit(X_train, y_train)
    
    y_hat_val = RF.predict(X_validation)
    y_hat_tra = RF.predict(X_train)
    
    wError_val = WMAE(X_validation['IsHoliday'], y_hat_val, y_validation)
    wError_tra = WMAE(X_train['IsHoliday'], y_hat_tra, y_train)
    
    WMAE_list_val.append(wError_val)
    WMAE_list_tra.append(wError_tra)


# In[ ]:


#Ploting the Training and Validation Curves
plt.plot(max_depth, WMAE_list_tra, label='Training Set')
plt.plot(max_depth, WMAE_list_val, label='Validation Set')
plt.xlabel('Max Depth')
plt.ylabel('Weighted Mean Absolute Error')
plt.title('Random Forest Regression Error Analysis')
plt.legend()
plt.show()

#Selecting the alpha
WMAE_min = np.amin(WMAE_list_val)
max_depth_min = max_depth[np.argmax(WMAE_list_val == WMAE_min)]

print("Max Depth: " + str(max_depth_min))
print("Minimum WMAE: " + str(WMAE_min))


# ### Neural Network

# In[ ]:


#List of architectures to be tested
architectures = [(30, 30, 10, 10), (30, 30, 10, 10, 5, 5), (30, 30, 10, 5, 5, 5)]

WMAE_list_val = []
WMAE_list_tra = []

for i in architectures: 
    print('Architecture: ', i)
    MLP = neural_network.MLPRegressor(hidden_layer_sizes = i, max_iter=10000, random_state=42)
    MLP.fit(X_train, y_train)

    y_hat_val = MLP.predict(X_validation)
    y_hat_tra = MLP.predict(X_train)
    
    wError_val = WMAE(X_validation['IsHoliday'], y_hat_val, y_validation)
    wError_tra = WMAE(X_train['IsHoliday'], y_hat_tra, y_train)
    
    WMAE_list_val.append(wError_val)
    WMAE_list_tra.append(wError_tra)


# In[ ]:


#Ploting the Training and Validation Curves
plt.plot(np.arange(0, len(architectures)), WMAE_list_tra, label='Training Set')
plt.plot(np.arange(0, len(architectures)), WMAE_list_val, label='Validation Set')
plt.xlabel('Architectures')
plt.ylabel('Weighted Mean Absolute Error')
plt.title('MLP Regression Error Analysis')
plt.legend()
plt.show()

#Selecting the alpha
WMAE_min = np.amin(WMAE_list_val)
architecture_min = architectures[np.argmax(WMAE_list_val == WMAE_min)]

print("Architecture: " + str(architecture_min))
print("Minimum WMAE: " + str(WMAE_min))


# ## Evaluating the Results

# After training the three models (linear, ensemble and neural networks), we can see that for the tested hyperparameters, the random forest (RF) was the one that got the best performance. That can be seen due to the fact that the RF resulted in the smallest training and testing errors. In addition to that the RF was faster than the neural networks. 
# 
# Let's try to improve even more that model. We have seen that the Random Forest had some overfitting, so we are going to remove some features and see if this improve the model. We have chosen those values because they have very low correlation with the Weekly Sales. Others features also have low correlation, but we judged that they have theoretical relation, as IsHoliday, for example which was shown that has a significant impact on the Sales

# In[ ]:


#Params
new_features_remove = ['CPI', 'Unemployment', 'Temperature']
WMAE_list_val = []
WMAE_list_tra = []

for i in new_features_remove:
    print('Removing: ', i)
    
    X_train = X_train.drop(columns=i)
    X_validation = X_validation.drop(columns=i)
    
    RF = ensemble.RandomForestRegressor(n_estimators = n_estimators_min, max_depth = max_depth_min, 
                                        n_jobs = -1, random_state = 42)
    RF.fit(X_train, y_train)
    
    y_hat_val = RF.predict(X_validation)
    y_hat_tra = RF.predict(X_train)
    
    wError_val = WMAE(X_validation['IsHoliday'], y_hat_val, y_validation)
    wError_tra = WMAE(X_train['IsHoliday'], y_hat_tra, y_train)
    
    WMAE_list_val.append(wError_val)
    WMAE_list_tra.append(wError_tra)


# In[ ]:


#Ploting the Training and Validation Curves
plt.plot(new_features_remove, WMAE_list_tra, label='Training Set')
plt.plot(new_features_remove, WMAE_list_val, label='Validation Set')
plt.xlabel('Features Removed')
plt.ylabel('Weighted Mean Absolute Error')
plt.title('Random Forest Error Analysis')
plt.legend()
plt.show()

#Selecting the alpha
WMAE_min = np.amin(WMAE_list_val)
#new_features_remove_min = new_features_remove[np.argmax(WMAE_list_val == WMAE_min)]

#print("After " + str(architecture_min) + ' we got minimum error')
print("Minimum WMAE: " + str(WMAE_min))


# So, we have achieved a WMA Error of 1518 on the validation set which seems to be a good value. Therefore, let's apply that model to our test set.

# ## Using the Model on the Test Data

# In[ ]:


RF = ensemble.RandomForestRegressor(n_estimators = n_estimators_min, max_depth=max_depth_min, n_jobs = -1)
RF.fit(X_train, y_train)

X_test = X_test.drop(columns=new_features_remove)
y_hat_test = RF.predict(X_test)


# ## Creating the submission File

# In[ ]:


Store_list = [str(x) for x in test_data['Store']]
Dept_list = [str(x) for x in test_data['Dept']]

id_list = []
for i in range(len(Store_list)):
    id_list.append(Store_list[i] + '_' + Dept_list[i] + '_' + Date_list[i])

Output = pd.DataFrame({'id':id_list, 'Weekly_Sales':y_hat_test})
Output.to_csv('submission.csv', index=False)

Output


# ## Further Improvements

# There are some approaches that we can take in order to decrease even more this error:
# 
# **1. Try deeper and bigger neural networks architectures:** The Neural Networks model seems to be promissing for this data set. We know that the NN can model functions that are highly non-linear, so we think that increasing the NN can overcome that underfitting problem.
# 
# **2. Try different encoders**
# 
# **3. Try different scale transformation**

# **If you have some doubt of suggestion, please let me know. Thanks!!**
