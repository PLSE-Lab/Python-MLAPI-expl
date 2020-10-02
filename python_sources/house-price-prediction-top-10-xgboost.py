#!/usr/bin/env python
# coding: utf-8

# # Solution For House Price Contest
# 
# ![](https://media1.tenor.com/images/286156bd33ce64d69f6a2367557392b5/tenor.gif?itemid=10804810)
# 
# This is my first contest submission. It includes data imputation (manual and automatic), one-hot variable encoding, and random forest model (XGBoosted)
# 
# While this solution was good enough for an top ~10% entry, I was limited in the features that I was able to use by when I chose to one-hot encode variables. Ideally I would have one-hot encoded the catagorical variables early on, and with the test and training data combined into a single DataFrame. Because I didn't do that, there were some variables missing from the Test data set, that would throw errors and need to be left out. 

# # Imports:

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# # Load Data:

# In[ ]:


iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)

test_file_path = '../input/test.csv'
test_data = pd.read_csv(test_file_path)


# # Explore Missing Data

# In[ ]:


# Display vars that have 1 or more missing data points
print('Training Data:')
print(home_data.isnull().sum().sort_values(ascending=False).head(20))# display top categories with missing data
print('---------------------------')
print('Test Data:')
print(test_data.isnull().sum().sort_values(ascending=False).head(35)) # check in Test data too so you don't build model around vars that don't exist


# # Split home_data and test_data into quantitative and qualitative (will be imputed differently)

# In[ ]:


home_quan = list( home_data.loc[:,home_data.dtypes != 'object'].drop('Id',axis=1).columns.values )
home_qual = list( home_data.loc[:,home_data.dtypes == 'object'].columns.values )

test_quan = list( test_data.loc[:,test_data.dtypes != 'object'].drop('Id',axis=1).columns.values )
test_qual = list( test_data.loc[:,test_data.dtypes == 'object'].columns.values )

home_data_imputed = home_data.copy()
test_data_imputed = test_data.copy()


# # Impute Data:

# In[ ]:


# Impute quantitative data
# Some Values probably should be 0, i.e. if there is no garage it should have 0 SF area...   
home_data_imputed.BsmtHalfBath.fillna(0, inplace=True)
home_data_imputed.BsmtFullBath.fillna(0, inplace=True)
home_data_imputed.GarageArea.fillna(0, inplace=True)
home_data_imputed.GarageCars.fillna(0, inplace=True)    
home_data_imputed.TotalBsmtSF.fillna(0, inplace=True)   
home_data_imputed.BsmtUnfSF.fillna(0, inplace=True)     
home_data_imputed.BsmtFinSF2.fillna(0, inplace=True)    
home_data_imputed.BsmtFinSF1.fillna(0, inplace=True)

test_data_imputed.BsmtHalfBath.fillna(0, inplace=True)
test_data_imputed.BsmtFullBath.fillna(0, inplace=True)
test_data_imputed.GarageArea.fillna(0, inplace=True)
test_data_imputed.GarageCars.fillna(0, inplace=True)    
test_data_imputed.TotalBsmtSF.fillna(0, inplace=True)   
test_data_imputed.BsmtUnfSF.fillna(0, inplace=True)     
test_data_imputed.BsmtFinSF2.fillna(0, inplace=True)    
test_data_imputed.BsmtFinSF1.fillna(0, inplace=True)

# Others we will fill with the mean using Imputer:
quan_imputer = Imputer()
home_quan_imputed = quan_imputer.fit_transform(home_data_imputed[home_quan]) # ... this changes it from DataFrame to ndarray
test_quan_imputed = quan_imputer.fit_transform(test_data_imputed[test_quan])

home_quan_imputed = pd.DataFrame(data=home_quan_imputed,columns=home_quan) #This converts back to a dataframe
test_quan_imputed = pd.DataFrame(data=test_quan_imputed,columns=test_quan)

# Impute Qualitative data (I don't have SimpleImputer... so we will do manually for selected vars)

# Filling missing values for categorical features# Filli 
home_data_imputed.GarageCond.fillna('NA', inplace=True)    # Replace with NA
home_data_imputed.GarageQual.fillna('NA', inplace=True)       
home_data_imputed.GarageType.fillna('NA', inplace=True)          
home_data_imputed.BsmtCond.fillna('NA', inplace=True)        
home_data_imputed.BsmtQual.fillna('NA', inplace=True)        
home_data_imputed.Functional.fillna(home_data_imputed.Functional.mode()[0], inplace=True)  # Replace with mode 
home_data_imputed.SaleType.fillna(home_data_imputed.SaleType.mode()[0], inplace=True)                
home_data_imputed.KitchenQual.fillna(home_data_imputed.KitchenQual.mode()[0], inplace=True)        
home_data_imputed.Electrical.fillna(home_data_imputed.Electrical.mode()[0], inplace=True) 

test_data_imputed.GarageCond.fillna('NA', inplace=True)    # Replace with NA
test_data_imputed.GarageQual.fillna('NA', inplace=True)       
test_data_imputed.GarageType.fillna('NA', inplace=True)          
test_data_imputed.BsmtCond.fillna('NA', inplace=True)        
test_data_imputed.BsmtQual.fillna('NA', inplace=True)        
test_data_imputed.Functional.fillna(test_data_imputed.Functional.mode()[0], inplace=True)  # Replace with mode 
test_data_imputed.SaleType.fillna(test_data_imputed.SaleType.mode()[0], inplace=True)                
test_data_imputed.KitchenQual.fillna(test_data_imputed.KitchenQual.mode()[0], inplace=True)        
test_data_imputed.Electrical.fillna(test_data_imputed.Electrical.mode()[0], inplace=True)
test_data_imputed.MSZoning.fillna(test_data_imputed.MSZoning.mode()[0], inplace=True) #whoops, missed this originally

# Manually Encode Neighborhood data?


# convert qual data back into dataframe
home_qual_imputed = home_data_imputed[home_qual]
test_qual_imputed = test_data_imputed[test_qual]

# Combine imputed quan, qual data into single data frames... already edited home_data_imputed?


# # Manually Encode A Couple of Variables

# In[ ]:


# Lets also Manually encode some importand variables that may have too many dimensions for one-hot encoding

# Neighborhood:
plt.figure(figsize=(12,7))
sns.boxplot(x = home_data_imputed['Neighborhood'], y = home_data_imputed['SalePrice'])

# We'll work on this more later...


# # Split Training Data into target  (sale price) and explanatory vars

# In[ ]:


y = home_data_imputed.SalePrice


# # Let's also try transforming our target variable, since it's not normally distributed..

# In[ ]:


# Distribution Plot:
sns.distplot(home_data_imputed['SalePrice'], fit = norm)
(mu, sigma) = norm.fit(home_data_imputed['SalePrice'])
print('mu = {:,.2f} and sigma = {:,.2f}'.format(mu, sigma))
# Q-Q plot:
fig = plt.figure()
res = stats.probplot(home_data_imputed['SalePrice'], plot=plt)
plt.show()

# Apply Log transform to SalePrice (y in training data)
y = np.log(y)
# Distribution Plot
sns.distplot(y, fit = norm)
(mu, sigma) = norm.fit(y)
print('transformed mu = {:,.2f} and transformed sigma = {:,.2f}'.format(mu, sigma))
# QQ Plot
fig = plt.figure()
res = stats.probplot(y, plot=plt)
plt.show()

y.head(10)


# In[ ]:


# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath',             'BedroomAbvGr', 'TotRmsAbvGrd', 'YearRemodAdd', 'OverallCond',             'OverallQual', 'BsmtFinSF1', 'GarageCars', 'HalfBath','GrLivArea',             'TotalBsmtSF', 'BsmtFinSF2', 'BsmtFullBath', 'Fireplaces', 'PoolArea'] # maybe include , 'YrSold'

cat_features = ['BldgType', 'CentralAir', 'BsmtQual', 'Street', 'MSZoning',                 'BsmtExposure','FireplaceQu','PavedDrive','Neighborhood',                'GarageType', 'LotConfig']; #, 'LotShape', 'LotConfig', 'Electrical', 'Functional', 'GarageType', , 'BsmtQual', 'BsmtCond', 'KitchenQual', 'Street', 'Condition',,'HouseStyle'

all_features = features + cat_features # add new categorical features (so as not to break the sections before)
X = home_data_imputed.copy()[all_features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

test_X = test_data_imputed.copy()[all_features]


# # One-hot Encode Categorical Variables:

# In[ ]:


# One Hot Encode the Categorical Variables
# Originally, I only one-hot encoded vars that would result in <10 new input cols, however, to include neighborhood, I ended up bumping up that number to 30...
low_cardinality_cols_train = [cname for cname in train_X.columns if 
                                train_X[cname].nunique() < 30 and
                                train_X[cname].dtype == "object"] # eligible to be one-hot encoded if it is type object and <10 unique values
numeric_cols_train = [cname for cname in train_X.columns if 
                                train_X[cname].dtype in ['int64', 'float64']] # numerical data if integer or float

my_cols_train = low_cardinality_cols_train + numeric_cols_train
#my_cols_val = low_cardinality_cols_val + numeric_cols_val
#print(my_cols_train)

train_predictors = train_X[my_cols_train]
val_predictors = val_X[my_cols_train] #no need to include any columns that don't exist in training set?

# One-hot Encode Categorical data for trianing and validation sets. include only data that exists in both
one_hot_train_predictors = pd.get_dummies(train_predictors)
one_hot_val_predictors = pd.get_dummies(val_predictors)
one_hot_train_X, one_hot_val_X = one_hot_train_predictors.align(one_hot_val_predictors,
                                                                   join='inner', 
                                                                    axis=1)
one_hot_X = one_hot_train_X.append(one_hot_val_X) # Train from this!!! (with y)

# One Hot Encode Test data for contest submission:
test_predictors = test_X[my_cols_train]
one_hot_test_predictors = pd.get_dummies(test_predictors)


# # XG Boost Model and Validation

# In[ ]:


# Initial Pass:
my_XGB_model = XGBRegressor()
my_XGB_model.fit(one_hot_train_X, train_y, verbose=False)

# make predictions
XGB_predictions = my_XGB_model.predict(one_hot_val_X)
XGB_predictions = np.exp(XGB_predictions)
# Print MAE for initial XGB model
XGB_mae = mean_absolute_error(XGB_predictions, np.exp(val_y))
print("Validation MAE for XGBoost Model : " + str(XGB_mae))
      
# Additional Passes
my_XGB_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_XGB_model.fit(one_hot_train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(one_hot_val_X, val_y)], verbose=False)
XGB_predictions = my_XGB_model.predict(one_hot_val_X)
XGB_predictions = np.exp(XGB_predictions)
XGB_mult_mae = mean_absolute_error(XGB_predictions, np.exp(val_y))
print("Validation MAE for multi-pass XGBoost Model : " + str(XGB_mult_mae))

# Predict SalePrice on Test Data:
final_predictions = my_XGB_model.predict(one_hot_test_predictors)
final_predictions = np.exp(final_predictions)

print('\n\n ---------------------------------------- \n\n')
print(final_predictions)


# #  Write output CSV file of predictions from test data for contest submission

# In[ ]:


output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': final_predictions})
output.to_csv('submission.csv', index=False)


# # Test Your Work
# After filling in the code above:
# 1. Click the **Commit and Run** button. 
# 2. After your code has finished running, click the small double brackets **<<** in the upper left of your screen.  This brings you into view mode of the same page. You will need to scroll down to get back to these instructions.
# 3. Go to the output tab at top of your screen. Select the button to submit your file to the competition.  
# 4. If you want to keep working to improve your model, select the edit button. Then you can change your model and repeat the process.
# 
# Congratulations, you've started competing in Machine Learning competitions.
# 
# # Continuing Your Progress
# There are many ways to improve your model, and **experimenting is a great way to learn at this point.**
# 
# The best way to improve your model is to add features.  Look at the list of columns and think about what might affect home prices.  Some features will cause errors because of issues like missing values or non-numeric data types. 
# 
# Level 2 of this course will teach you how to handle these types of features. You will also learn to use **xgboost**, a technique giving even better accuracy than Random Forest.
# 
# 
# # Other Courses
# The **[Pandas course](https://kaggle.com/Learn/Pandas)** will give you the data manipulation skills to quickly go from conceptual idea to implementation in your data science projects. 
# 
# You are also ready for the **[Deep Learning](https://kaggle.com/Learn/Deep-Learning)** course, where you will build models with better-than-human level performance at computer vision tasks.
# 
# ---
# **[Course Home Page](https://www.kaggle.com/learn/machine-learning)**
# 
# **[Learn Discussion Forum](https://kaggle.com/learn-forum)**.
# 
