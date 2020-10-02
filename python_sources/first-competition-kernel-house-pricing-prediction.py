#!/usr/bin/env python
# coding: utf-8

# # About the Kernel

# This notebook consist of Exploratory Data Analysis(EDA) and Sales Price prediction with different Regression Models for House Price prediction competition .
# 
# This kernel is easily understandable to the beginner(like me). I tried to explain everything as simple and straightforward to my best wisdom.
# 
# I have tried different model and submitted score based of all models , in this way we can see different models/approach and how to fine tune them. With the latest model score is **0.16303**
# 
# Please provide comment if you have any kind of suggestions ao that I can improve this kernel and if you like in someway then please upvote the kernel.
# 
# **Note** : *This is Work In Progress Notebook , will be updating on regular basis till the time will get good model and satisfactory scores.*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # 1. Getting the Data

# Let's import the data and check columns data type and total records

# In[ ]:


#importing libraries and data set
import seaborn as sns
import matplotlib.pyplot as plt
dataset=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


#Checking total number records
dataset.shape


# In[ ]:


#Checking all columns
dataset.columns


# In[ ]:


dataset.dtypes.unique()


# # 2. Selecting independent variables(features)

# From the provided columns we have to select few columns as independent variables or features on which we can train our model

# In[ ]:


dataset.head()


# In[ ]:


#Checking columns data types
#String data type
len(dataset.select_dtypes(include=['O']).columns)


# In[ ]:


#Integer data type
len(dataset.select_dtypes(include=['int64']).columns)


# In[ ]:


#Float data type
len(dataset.select_dtypes(include=['float64']).columns)


# In[ ]:


#Getting Correlation Coefficient of sale price with other numerical data
saleprice_corr=dataset.corr()['SalePrice']
saleprice_corr


# I have done detailed data analysis for this data set and it's present in this [link](https://docs.google.com/spreadsheets/d/1IyfMnTl4g8JUpI6N_l8QlPB42uwygR9-m-i1z5KciJ8/edit?usp=sharing), if anyone intrested can take a look
# 
# I took some help from this Kaggle [Kernel](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) for data analysis.
# 
# I have created the spreadsheet with the following columns:
# 
# Variable - Variable name
#     
# Type - Identification of the variables' type. There are two possible values for this field: 'numerical' or 'categorical'. By 'numerical' we mean variables for which the values are numbers, and by 'categorical' we mean variables for which the values are categories
# 
# Category - Identification of the variables category . We can define three possible segments: building, space or location. When we say 'building', we mean a variable that relates to the physical characteristics of the building (e.g. 'OverallQual'). When we say 'space', we mean a variable that reports space properties of the house (e.g. 'TotalBsmtSF'). Finally, when we say a 'location', we mean a variable that gives information about the place where the house is located (e.g. 'Neighborhood')
# 
# Expected Effect on Sale Price - Our expectation about the variable influence in 'SalePrice'. We can use a categorical scale with 'High', 'Medium' and 'Low' as possible values. We can look at each variable and try to understand their meaning and relevance
# 
# Correlation Coefficient - Consist of correlation cofficient value for numerical variable
# 
# Conclusion - Our conclusions about the importance of the variable, after we give a quick look at the data and cofficenent
# 
# Note - Any general comments realted to variables

# Based on above analysis I have come up with following variables as a posisble independent variables (features)
# 'Neighborhood','OverallQual','YearBuilt','ExterCond','TotalBsmtSF','GrLivArea'.
# I have selected these variables from all of the segments (building, space and location) with 'High' expected effect on Sales Price.
# Now , I will check the relation bewtween all these variables

# In[ ]:


#Creating independent variables data frame X
X=dataset[['Neighborhood','OverallQual','YearBuilt','ExterCond','TotalBsmtSF','GrLivArea','SalePrice']]


# In[ ]:


#Verifying the relation between GrLivArea and SalePrice
plt.scatter(X['GrLivArea'],X['SalePrice'])


# In[ ]:


#Verifying the relation between TotalBsmtSF and SalePrice
plt.scatter(X['TotalBsmtSF'],X['SalePrice'])


# In[ ]:


#Verifying the relation between OverallQual and SalePrice
plt.scatter(X['OverallQual'],X['SalePrice'])


# In[ ]:


#Verifying the relation between YearBuilt and SalePrice
plt.scatter(X['YearBuilt'],X['SalePrice'])


# In[ ]:


#Creating pair plot along with categorical variable 'ExterCond' to get relation with Sale Price and other variables
sns.pairplot(X,hue='ExterCond')


# In[ ]:


#Creating pair plot along with categorical variable 'Neighborhood' to get relation with Sale Price and other variables
sns.pairplot(X,hue='Neighborhood')


# With above scatter plots and pair plots we can say that numerical variables have linear relationship with Sale Price and Categorical variables also have some relationship with Sale Price , will dig more on Categorical variables
# 
# Note: If we observe graph between 'OverallQual' and 'SalePrice' we can see that though it's a numerical variable it's actually a  categorical variable and on checking the variable description it's pretty evident that 'OverallQual' is a categorical variable only
# 
# Let's analyze relationship between Categorical variables and Sale price with the help of box plots

# In[ ]:


#Box plot between 'OverallQuality' and 'Sales Price'
sns.boxplot(x=X['OverallQual'],y=X['SalePrice'],palette='rainbow')


# In[ ]:


#Box plot between 'ExterCond' and 'Sales Price'
sns.boxplot(x=X['ExterCond'],y=X['SalePrice'],palette='rainbow')


# In[ ]:


#Box plot between 'Neighborhood' and 'Sales Price'
plt.figure(figsize=(20,10))
sns.boxplot(y=X['Neighborhood'],x=X['SalePrice'],palette='rainbow')


# With these plots we can see that out of three categorical variables only 'OverallQual' is having linear relationship with 'SalePrice'(When Quality increase Sales Price also increase) and other two variables i.e. 'Neighborhood' and 'ExtCond' does not show any linear relationship. Based on this we can skip these two variables from our feature list
# 
# Now, we have independent variables (features) 'OverallQual','YearBuilt','TotalBsmtSF','GrLivArea' and from above analysis we know that 'OverallQual','TotalBsmtSF','GrLivArea' are linearly related to our dependent variable('SalePrice') for 'YearBuilt' let's create box plot graph with 'SalePrice' to see the relation ship more clearly between these two

# In[ ]:


#Box plot between 'Year built' and 'Sales Price' to check sales price across years
plt.figure(figsize=(20, 10))
sns.boxplot(x=X['YearBuilt'],y=X['SalePrice'],palette='rainbow')


# Here, we can see that 'YearBuilt' and 'SalePrice' are kind of linearly related as Sales price increases over the years.From scatter and box plots for 'OverallQual' and 'YearBuilt' we can say that they are kind of linearly related with Sales Price
# 
# Now , lets consider other numerical variable with high correlation cofficent which I have skipped during my initial data analysis because these variables does not look relevant to me. But due to high correlation cofficent numbers let's consider them now , and these variables are
# 'FullBath','1stFlrSF',
# 'YearRemodAdd','GarageCars','GarageArea'
# 
# On checking data description and correlation coffecient it looks like that 'YearRemodAdd'//'YearBuilt' , 'TotalBsmtSF' //'1stFlrSF' ,'GarageCars'//'GarageArea' variables are kind of identical variables only .If we practially think about these variables, we can conclude that they give almost the same information so multicollinearity can occurs
# So we can consider only 'FullBath','GarageCars' in new list and ignore the remaining of them

# In[ ]:


#Adding new independent Variables in X
X=dataset[['FullBath','OverallQual','YearBuilt','TotalBsmtSF','GrLivArea','GarageCars','SalePrice']]


# In[ ]:


#Verifying the relation between FullBath and SalePrice
plt.scatter(X['FullBath'],X['SalePrice'])


# In[ ]:


#Verifying the relation between FullBath and SalePrice
plt.scatter(X['GarageCars'],X['SalePrice'])


# With above two graphs we can say that 'FullBath' and 'GarageCars' have linear relationship with Sale Price. 'GarageCars' have an exception(last value '4') which can be outliner value
# 
# With this updated list of dependent variables let's create a new pair plot graph to see relationship between all variables

# In[ ]:


#Create pairplot with new list
sns.pairplot(X)


# If we see last row of Sale Price with each independent variables we will find out that all dependent are somewhat linearly related to Sale Price(dependent variable) and all these variables have high correlation coefficient with dependent variable
# 

# # 3. Missing Values

# Let's check missing values in our independent and dependent variables

# In[ ]:


#Checking missing values 
total_missing_values_X=X.isnull().sum().sort_values(ascending=False)
total_missing_values_X


# There is no missing values in our independent and dependent variables so we are good for model fitting , but before that let's look into Outliers

# # 4. Outliers

# If we see our graph between 'GirLivArea' ,'TotalBsmtSF' and 'GarageCars' with 'SalePrice' we can see that there are few points which are out of the crowd (not following the trend of regression) and we can consider them as outliers .There can be logical reason behind these outliers , like for 'GirLivArea' there are two observation where Sale price decreased with high above ground living area so posisble reason can be that these properties are located in outskirts or agriculture area or can be anything and we are not sure about these so we will treat them as outliers and delete them

# In[ ]:


#Checking first two values in 'GirLivArea' for outliers
X.sort_values(by='GrLivArea',ascending=False)[:2]


# In[ ]:


#Drop outliers rows from the data set
X=X.drop(1298)
X=X.drop(523)


# For 'TotalBsmtSF' only one outlier was there which was at index '1298' and same got deleted with 'GrLivArea'.
# 
# Now For 'GarageCars' on looking at graph we can say when 'GarageCars' value is 4 then sale price is dropped and does not follow the trend , so we will drop the rows where Garage car value is 4

# In[ ]:


#Get index for records with GarageCars as 4
indexNames=X[X['GarageCars'] == 4].index


# In[ ]:


#Drop the records for GarageCars as 4
X=X.drop(indexNames)


# Now we will create pair plot one more time to check outliers are coming or not

# In[ ]:


#Create pairplot with new list
sns.pairplot(X)


# Now there is not much outliers coming in our data . We can see few point in 'TotalBsmtSF' but they are pretty much following the trend so we will leave them and we can start modelling our data

# # 5. Data Modeling

# In[ ]:


#Split independent and dependent variables
X=dataset[['FullBath','OverallQual','YearBuilt','TotalBsmtSF','GrLivArea','GarageCars']]
y=dataset[['SalePrice']]


# In[ ]:


# Fitting Multiple linear regression to the data set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)


# In[ ]:


#Predicting the train set results
y_pred=regressor.predict(X)


# In[ ]:


#Converting y from series to array , to generate a graph for comparision with y_pred
y=y.values


# In[ ]:


#Rounding off the y_pred 
y_pred=y_pred.round()


# In[ ]:


#Converting 2 dimensional y and y_pred array into single dimension 
y=y.ravel()
y_pred=y_pred.ravel()
y_pred


# In[ ]:


#Creating data frame for y and y_pred to create line plot
df=pd.DataFrame({"y":y,"y_pred":y_pred})
sns.lineplot(data=df)


# From above graph it's evident that our predicted Sale value is matching with actual sale value  but with few mismatch values . It means there is a scope of remodelling here.
# Let's get some statistics related to this model

# On calculating 'P' values for every variable we can say that for 'FullBath' it's pretty high (0.418 which is > significant value of .05) so we can drop this feature and remodel our model Note: Removing feature/independent varaible based on 'P' value is called process of Backward Elimination

# In[ ]:


#Removing 'FullBath' from list of independent variables
#X=dataset[['OverallQual','YearBuilt','GrLivArea','TotalBsmtSF','GarageCars']]


# Commneting above line of code as after removing 'FullBath' also there is no difference in prediction so we decided to keep it as it is highly corelated 

# In[ ]:


#Creating new regressor object and fitting the model
regressor_new=LinearRegression()
regressor_new.fit(X,y)
y_pred_new=regressor_new.predict(X)


# In[ ]:


#Rounding off the y_pred_new
y_pred_new=y_pred_new.round()


# In[ ]:


#Converting 2 dimensional y and y_pred array into single dimension 
y_pred_new=y_pred_new.ravel()


# In[ ]:


#Creating data frame for y ,y_pred,y_pred_new to create line plot
df=pd.DataFrame({"y":y,"y_pred":y_pred,"y_pred_new":y_pred_new})
sns.lineplot(data=df)


# Green and Blue lines are almost overlapping each other (only very few exception , need to zoom for that) , but this is on train data set. We need to make our prediction on test set, so will repeat these steps on test set and verify our findings
# 
# As mentioned earlier , removing 'FullBath' is not making much differnce in prediction so we will keep it in our feature list
# 

# # 6. Prediction from Test Data

# Let's summarize what we have done till now
#     1. Data import
#     2. Data analysis
#     3. Selection of independent variables
#     4. Verifying any missing values
#     5. Removal of Outliers
#     6. Data modeling
#     7. Prediction
# 
# To get prediction from test data we have to perform step 4 and 5 and fit our model on test data

# In[ ]:


#Get test data 
dataset_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[ ]:


#Create X_test and fetching id in different frame
X_test=dataset_test[['FullBath','OverallQual','YearBuilt','TotalBsmtSF','GrLivArea','GarageCars']]
y_test_id=dataset_test[['Id']]
X_test.head()


# In[ ]:


#Checking missing value in test data set
total_missing_values_X_test=X_test.isnull().sum().sort_values(ascending=False)
total_missing_values_X_test


# In[ ]:


#Checking the missing Garage Cars record
X_test[X_test['GarageCars'].isnull()]


# In[ ]:


#Checking the missing Total Bsmt SF record
X_test[X_test['TotalBsmtSF'].isnull()]


# We can use Imputer library to take care of missing value but in this scenario only one value is missing in both columns so we will update that with most frequent value and mean value in 'Garage Cars' and 'TotalBsmtSF' respectively.
# 
# For 'Garage Cars' we  can say that value '2' is most common in test data so we will replace the missing with this value

# In[ ]:


#Updating Garage Cars to 2 at missing value index
X_test.at[1116,'GarageCars'] = 2


# In[ ]:


#Verifying the missing value in Garage Cars
X_test[X_test['GarageCars'].isnull()]


# Now we will check 'TotalBsmtSF' mean and update the same to missing value

# In[ ]:


#Fetching 'TotalBsmtSF' information
X_test['TotalBsmtSF'].describe()


# In[ ]:


#Updating the missing value to mean value
X_test.at[660,'TotalBsmtSF'] = 1046.12


# In[ ]:


#Verifying the missing value in TotalBsmtSF
X_test[X_test['TotalBsmtSF'].isnull()]


# In[ ]:


#Checking missing value in test data set again
total_missing_values_X_test=X_test.isnull().sum().sort_values(ascending=False)
total_missing_values_X_test


# Now we can see there is no missing value in test data set , let's deal with outliers now.

# In[ ]:


#Visualize test data
sns.pairplot(X_test)


# Based on our train set data and above pairplot, we will check top values in 'GirLivArea' and 'TotalBsmtSF'

# In[ ]:


X_test.sort_values(by='GrLivArea',ascending=False)[:2]


# In[ ]:


X_test.sort_values(by='TotalBsmtSF',ascending=False)[:2]


# In[ ]:


#We can drop the outliers but our submission csv needs 1459 records 
#X_test=X_test.drop(1089)


# In[ ]:


#Creating predictions based on X_test
y_test_pred=regressor_new.predict(X_test)


# In[ ]:


#Converting 2 dimensional y_test_pred into single dimension 
y_test_pred=y_test_pred.ravel()


# In[ ]:


#Rounding off the values
y_test_pred=y_test_pred.round()


# In[ ]:


#Converting Id into array
y_test_id=y_test_id.values


# In[ ]:


#Converting 2 dimensional y_test_id into single dimension 
y_test_id=y_test_id.ravel()


# In[ ]:


#Creating Submission dataframe from id and predecited Sale price
submission_df=pd.DataFrame({"Id":y_test_id,"SalePrice":y_test_pred})


# In[ ]:


#Setting index as Id Column
submission_df.set_index("Id")


# In[ ]:


#Converting into CSV file for submission
submission_df.to_csv("submission_1.csv",index=False)


# # 7. K-Fold Techniques

# In[ ]:


#Apply K-fold in current model to check model accuracy
from sklearn.model_selection import cross_val_score
accuracies_linreg_model = cross_val_score(estimator = regressor, X = X, y = y, cv = 10)


# In[ ]:


#Checking accuracies for 10 fold in linear regression model
accuracies_linreg_model


# In[ ]:


#Checking Mean and Standard Deviation between accuracies
accuracies_linreg_model.mean()
accuracies_linreg_model.std()


# Mean Accuracy is coming close to 76% and standard Devaition is also not that much (~8.4%) , but our score with this model is ~ 0.50 we need to improve this
# We need to try another mode  , let's try with Random Forest Regression Model

# # 8. Random Forest Regression
# 

# In[ ]:


#Creating new Regressor model for Random forest regression
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
rf_regressor.fit(X, y)


# In[ ]:


# Predicting from test set with this new model
y_test_rf_pred=rf_regressor.predict(X_test)


# In[ ]:


#Converting 2 dimensional y_test_pred into single dimension 
y_test_rf_pred=y_test_rf_pred.ravel()


# In[ ]:


#Rounding off the values
y_test_rf_pred=y_test_rf_pred.round()


# In[ ]:


#Creating Submission dataframe from id and predecited Sale price
submission_rf_df=pd.DataFrame({"Id":y_test_id,"SalePrice":y_test_rf_pred})
#Setting index as Id Column
submission_rf_df.set_index("Id")


# In[ ]:


#Converting into CSV file for submission
submission_rf_df.to_csv("submission_2.csv",index=False)


# Score is improved with this submission , with linear regression model score(RMSE) was 0.51 and with this Random Forest regressor model it is 0.16955.
# Let's fine tune this model with Hyperparameter tuning

# # 7. Hyperparameter Tuning

# For implementing Grid Search in Random Forest Regressor model I have refered this [blog](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[ ]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X,y)


# In[ ]:


#Checking Best params
rf_random.best_params_


# In[ ]:


# Predicting from test set with this new model
y_test_rf_random_pred=rf_random.predict(X_test)


# In[ ]:


#Converting 2 dimensional y_test_pred into single dimension 
y_test_rf_random_pred=y_test_rf_random_pred.ravel()
#Rounding off the values
y_test_rf_random_pred=y_test_rf_random_pred.round()


# In[ ]:


#Creating Submission dataframe from id and predecited Sale price
submission_rf_random_df=pd.DataFrame({"Id":y_test_id,"SalePrice":y_test_rf_random_pred})
#Setting index as Id Column
submission_rf_random_df.set_index("Id")


# In[ ]:


#Converting into CSV file for submission
submission_rf_random_df.to_csv("submission_3.csv",index=False)


#  *Score is improved with this submission , before hyper tuning of parameters it was **0.16955** and now it is **0.16379***

# # 8. XGBoost Regressor

# In[ ]:


#importing required library and creating XGboost Regressor model
from xgboost import XGBRegressor
xgboost_regressor=XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
xgboost_regressor.fit(X,y)


# In[ ]:


# Predicting from test set with this new model
y_test_xgb_pred=xgboost_regressor.predict(X_test)


# In[ ]:


#Converting 2 dimensional y_test_pred into single dimension 
y_test_xgb_pred=y_test_xgb_pred.ravel()
#Rounding off the values
y_test_xgb_pred=y_test_xgb_pred.round()


# In[ ]:


#Creating Submission dataframe from id and predecited Sale price
submission_xgb_df=pd.DataFrame({"Id":y_test_id,"SalePrice":y_test_xgb_pred})
#Setting index as Id Column
submission_xgb_df.set_index("Id")


# In[ ]:


#Converting into CSV file for submission
submission_xgb_df.to_csv("submission_4.csv",index=False)


#  *Score is improved with this submission , before XGBoost it was **0.16379** and now it is **0.16303***
