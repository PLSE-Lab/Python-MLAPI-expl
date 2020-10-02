#!/usr/bin/env python
# coding: utf-8

# A bit of a precursor before we begin. This shall be my first attempt at writing a very basic and simple walkthrough of a Machine Learning (ML) model on Kaggle. I hope for this kernel to be a well written and clear approach to ML, so much so that an absolute beginner may read through and understand what is going on. As such, this kernel will include little to no exploratory data analysis (EDA) nor feature engineering, but will focus more so on the absolute basics of training and running an ML model. Enjoy!!

# To begin we will first load in our training and testing datasets that are conveniently given to us from Kaggle. We shall do this using Pandas, which is a very useful Python library that can be thought of as a more robust form of Microsoft Excel (Pandas dataframes (df) are very similar to spreadsheets in Excel). After loading in our train and test data, we will use pd.concat to combine our train and test datasets so that we can perform some data cleanup on both datasets at the same time. We will also drop 'SalePrice' from our training data as well as creating an housing_label dataframe with just SalePrice to use as our feature target (label) in our ML model to test how well our model performs. 

# In[ ]:


import pandas as pd 

train = pd.read_csv('../input/train.csv')
train_df = train.drop('SalePrice', axis=1)
test_df = pd.read_csv('../input/test.csv')
housing_labels = train['SalePrice']
data_full = pd.concat([train_df, test_df])

data_full.head()


# Now let us do some very basic data cleanup. First let us view which columns in our dataset have NaN values.

# In[ ]:


data_NaN = data_full.isnull().sum()
data_NaN  = data_NaN[data_NaN>0]
data_NaN.sort_values(ascending=False)


# We will now fill our columns that contain NaN values with 0 inplace of the NaN. This is very rudimentary for taking care of missing values, and in the future it is recommended to fill NaN values with something along the line of median or mean, as well as encoding any categorical data. But as this is a very basic walkthrough, I will be using a very basic method of filling in NaN values.

# In[ ]:


data_full['PoolQC'].fillna(0, inplace=True)        
data_full['MiscFeature'].fillna(0, inplace=True)
data_full['Alley'].fillna(0, inplace=True)   
data_full['Fence'].fillna(0, inplace=True)         
data_full['FireplaceQu'].fillna(0, inplace=True)     
data_full['LotFrontage'].fillna(0, inplace=True)      
data_full['GarageFinish'].fillna(0, inplace=True)     
data_full['GarageYrBlt'].fillna(0, inplace=True)     
data_full['GarageQual'].fillna(0, inplace=True)       
data_full['GarageCond'].fillna(0, inplace=True)      
data_full['GarageType'].fillna(0, inplace=True)       
data_full['BsmtExposure'].fillna(0, inplace=True)     
data_full['BsmtCond'].fillna(0, inplace=True)         
data_full['BsmtQual'].fillna(0, inplace=True)          
data_full['BsmtFinType2'].fillna(0, inplace=True)     
data_full['BsmtFinType1'].fillna(0, inplace=True)      
data_full['MasVnrType'].fillna(0, inplace=True)        
data_full['MasVnrArea'].fillna(0, inplace=True)        
data_full['MSZoning'].fillna(0, inplace=True)           
data_full['BsmtFullBath'].fillna(0, inplace=True)       
data_full['BsmtHalfBath'].fillna(0, inplace=True)       
data_full['Utilities'].fillna(0, inplace=True)          
data_full['Functional'].fillna(0, inplace=True)         
data_full['Exterior2nd'].fillna(0, inplace=True)        
data_full['Exterior1st'].fillna(0, inplace=True)        
data_full['SaleType'].fillna(0, inplace=True)           
data_full['BsmtFinSF1'].fillna(0, inplace=True)         
data_full['BsmtFinSF2'].fillna(0, inplace=True)         
data_full['BsmtUnfSF'].fillna(0, inplace=True)          
data_full['Electrical'].fillna(0, inplace=True)         
data_full['KitchenQual'].fillna(0, inplace=True)        
data_full['GarageCars'].fillna(0, inplace=True)         
data_full['GarageArea'].fillna(0, inplace=True)         
data_full['TotalBsmtSF'].fillna(0, inplace=True)        

data_full.isnull().sum()


# An extremely basic but useful data analysis tool to use is something know as correlation. A correlation value relates how likely one variable is to follow another variable, ranging from values -1 to 1. A value of -1 means that if our variable we are correlating against moves up, the variable with a -1 correlation(r) value will move in the opposite direction which is in this case move down. A 1 value would mean that both variables move in the same direction.  A correlation table is very useful to see relationships between different variables in our dataset. 
# (We will be using our training dataset to view the correlation of features against our target, SalePrice)

# In[ ]:


corr_matrix = train.corr()
corr_matrix['SalePrice'].sort_values(ascending=False)


# Now lets create a Pipeline to transform our data so that we may use it with our ML model. A Pipeline simply uses our data as input and scales it and convert it to a numpy array as Scikit-Learn cannot use Pandas dataframes as inputs. We also must scale our data so that our numerical values are present on the same scale, which will help boost the performance of our ML model.  We will also create a DataSelector class which we will use to choose the relevant features we would like to use with our model. The code for this may seem overwhelming at first, especially for those new to coding, but please bare with me.  We will then use our pipeline to fit our data and are now ready for our ML model testing!!
# (As we have done no categorical encoding, you must choose features that are either int64 or float64, any columns with datatype=object will not work)
# For our example I have chosen 3 basic features to use: LotArea, 1stFlrSF, 2ndFlrSF.

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from warnings import filterwarnings
filterwarnings('ignore') #Use this to get rid of the DataConversion warning concerning converting int64 data to float64 data.

class DataSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

features = data_full[['LotArea', '1stFlrSF', '2ndFlrSF']]
features_selected = list(features)

pipeline = Pipeline([
    ('selected_features', DataSelector(features_selected)),
    ('scaler', StandardScaler())
])

housing_prepared = pipeline.fit_transform(train_df)
housing_prepared


# We are now ready to test some models! Since this is a regression task, we must use a regression ML model (instead of a classifier). The difference between regression and classification is that regression is used to predict a NUMBER(in this case SalePrice), while classification is used to predict a CLASS(e.g. "Yes or No", "Alive or Dead", etc.). 
# 
# First we must split our housing_prepared data into a training set(housing_prepared_train) and a testing testing set(housing_prepared_test). We do this so that we can test how well our models do on predicting sale price. The test_size parameter refers to our testing set being 20% of our full training set, letting us see which model performs best. 
# 
# To use our ML models(Linear Regression, Decision Tree Regression, Random Forest Regression) we must fit our training data to our model. We then use this model to predict our SalePrice using our testing data. We will then use a performance metric to see how well our model predicts SalePrice.Our performance metric we will use is Root Mean Squared Error(RMSE). The model with the lowest RMSE will be the model we shall use.

# In[ ]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

housing_prepared_train, housing_prepared_test, housing_labels_train, housing_labels_test = train_test_split(housing_prepared, housing_labels, test_size=0.2, random_state=33)

#LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared_train, housing_labels_train)
lin_housing_predictions = lin_reg.predict(housing_prepared_test)
lin_mse = mean_squared_error(housing_labels_test, lin_housing_predictions)
lin_rmse = np.sqrt(lin_mse)

#DecisionTreeRegression
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared_train, housing_labels_train)
tree_reg_predictions = tree_reg.predict(housing_prepared_test)
tree_mse = mean_squared_error(housing_labels_test, tree_reg_predictions)
tree_rmse = np.sqrt(tree_mse)

#RandomForestRegression
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared_train, housing_labels_train)
forest_reg_predictions = forest_reg.predict(housing_prepared_test)
forest_mse = mean_squared_error(housing_labels_test, forest_reg_predictions)
forest_rmse = np.sqrt(forest_mse)

print(lin_rmse)
print(tree_rmse)
print(forest_rmse)


# Our Random Forest Regression model had the lowest RMSE score so that is the model we will use. We will now use our pipeline to prepare our test data and make our final predictions. After this we will create a new pandas dataframe that is ready for submission!!

# In[ ]:


test_prepared = pipeline.fit_transform(test_df)

final_model = RandomForestRegressor()
final_model.fit(housing_prepared, housing_labels)
final_predictions = final_model.predict(test_prepared)

my_submission = pd.DataFrame({'Id': test_df.Id, 'SalePrice': final_predictions})
my_submission.head()

my_submission.to_csv('submission.csv', index=False)


# As this is my first kernel on Kaggle I would very much appreciate any feedback! I have only begun learning python and data science around a month ago and kaggle has been and extremely valuable learning resource. I know this kernel is very basic and lacking things such as GridSearchCV and CrossValidation, and in the future I will most likely create an updated kernel for these things, but for my first kernel I felt the need to create a very basic and simplistic model for beginners to utilize and hopefully learn from. Thank you and good luck Kaggling!
