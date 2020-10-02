#!/usr/bin/env python
# coding: utf-8

# This notebook seek to serve as my test to be accepted at RISTEK FASILKOM UI 2020
# 
# I am fairly new to Data Science, and have depended on the guides made public by other Kagglers. Namely I have adopted much of my data exploration techniques from a certain topic of kaggle. However I have also processed the data using my own understanding, and at certain points have chosen to make different decisions from the guides that already on the topics.
# 
# **Disclaimers** As this is my first time publishing a notebook on Kaggle I hope you find this notebook helpful in some ways. If you find any areas for imporvement, please feel free to suggest new approaches I may adopt. I welcome all feedbacks.
# 
# **Content Outline** My approach can be largely cagtegorised into 4 major steps: STEP 1: IMPORTING LIBRARIES AND DATASET STEP 2 : EXPLORATORY DATA ANALYSIS ON TRAINING SET STEP 3: DATA PRE-PROCESSING AND FEATURE ENGINEERING ON COMBINED DATASET STEP 4: XGBOOST MODELING WITH PARAMETER TUNING
# 
# Skuy!

# #**Import File**

# In[ ]:


import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system("ls '../input'")
get_ipython().system("ls '../input/ipricristik20'")


# #**STEP 1: IMPORTING LIBRARIES AND DATASET**

# I believe this step begins like all other kaggle submission, importing the relevant libaries for data science challanges, and of course importing of train and test data set.

# In[ ]:


#STEP 1: IMPORTING LIBRARIES AND DATASET

#import some necessary librairies
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_validate, GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns

color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


# In[ ]:


# Importing dataset from kaggle

train = pd.read_csv('../input/ipricristik20/train.csv')
test = pd.read_csv('../input/ipricristik20/test.csv')


# #**STEP 2: EXPLORATORY DATA ANALYSIS ON TRAIN DATASET**

# Personally, I find that it is important to understand the problem that we're trying to solve right from the start. In this particular problem, we are to predict an independent variable 'price' based ona ling list of dependent variables. The first step for me, would then be understand more about the 'price' variable - We're interested in what's the count of data and how are they distributed?

# In[ ]:


train['price'].describe()


# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# In[ ]:


print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

# Save the 'Id' column
train_ID = train['id']
test_ID = test['id']

# Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("id", axis = 1, inplace = True)
test.drop("id", axis = 1, inplace = True)

#check again the data size after dropping the 'id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))


# Dropping Id and put it in a variable.

# In[ ]:


sns.distplot(train['price'] , fit=norm);

#Now plot the distribution
plt.ylabel('Frequency')
plt.title('price distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['price'], plot=plt)
plt.show()


# So based on initial analysis, we can see a total count of 3304 data count in the labelled train set, centred around the mean of ~893,211. However we will not say that the data is normally distributed, and have demonstrated positive skewness. The probability plot is a technique picked up from a guide - normally distributed data should be following the diagonal line closely.
# 
# So the next step would be to resolve the skewness and normalise our target variable. This is important as most machine learning techniques are either built on, or simply works better with normally distributed data. A simple technique would be to apply log transformation to resolve the skewness.

# In[ ]:


#Log transformation

#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["price"] = np.log1p(train["price"])

#Check the new distribution 
sns.distplot(train['price'] , fit=norm);

plt.ylabel('Frequency')
plt.title('price distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['price'], plot=plt)
plt.show()


# There! Now that we have a normally distributed target variable, the next step would be to explore the remaining variables. Let's begin with numerical features.
# 
# As our dataset has a plethora of independet variable, feature selection is more critical than feature engineering in this particular problem. Thankfully, we can use seaborn to plot a correlation matrix. Seaborn not only helps us to identify 2 important things:
# 
# 
# 
# 1.   Correlation between numerical features and our target variable
# 2.   Correlation between numerical features and other key features
# 
# You can choose to plot the entire features map, but personally I find it overwhelming
# 
# 

# In[ ]:


#Correlation map to see how features are correlated with price
corrmat = train.corr()
plt.subplots(figsize=(12,12))
sns.heatmap(corrmat, vmax=0.9, square=True)


# This is the best way to see all the correlation between attributes. There are few attributes that got my attentions. Secret_2 and secret_9 are always the same so we will drop them off. Then you can see that relationsheep between distances are so strong that it could means multicollinearity. we can conclude that they give almost the same information so multicollinearity really occurs. And then, there are many data that similar in value like secret_7 and item_3.

# In[ ]:


data = train.corr()["price"].sort_values()[::-1]
plt.figure(figsize=(12, 8))
sns.barplot(x=data.values, y=data.index)
plt.title("Correlation with price")
plt.xlim(-0.2, 1)
plt.show()


# We can see all the relatable data from the biggest to the smallest. The 5 biggest are item_2, secret_4, facility_1, item_1, secret_7

# Then we continue to drop all the outlier on numerical attributes. But not on the boolean attributes..

# In[ ]:


#deleting outliers points by index --> room_size
var = 'room_size'
temp = pd.concat([train[var], train['price']], axis=1)
temp.plot.scatter(x=var, y='price')
temp.sort_values(by = var, ascending = True)

train = train.drop(train[train['price'] < 12].index, axis=0)


# I think it's varies to every person for how many data that you think is an outlier and not. For me this room_size attributes has one outlier on the bottom. 

# In[ ]:


#deleting outliers points by index --> room_size
var = 'distance_poi_A2'
temp = pd.concat([train[var], train['price']], axis=1)
temp.plot.scatter(x=var, y='price')
temp.sort_values(by = var, ascending = True)

# train = train.drop(train[(train['price'] > 14.5) & train['distance_poi_A2'] > 10000].index, axis=0)
# train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_A2'] > 25000].index, axis=0)

train = train.drop(train[train['distance_poi_A2'] > 12500].index, axis=0)


# Here you can see many outliers. I think every think many dots are concentrated on distance that less than 13000. The rest are outliers.

# In[ ]:


#deleting outliers points by index --> room_size
var = 'distance_poi_A1'
temp = pd.concat([train[var], train['price']], axis=1)
temp.plot.scatter(x=var, y='price')
temp.sort_values(by = var, ascending = True)

train = train.drop(train[train['distance_poi_A1'] > 16250].index, axis=0)


# I think outliers that every dots on the distance morethan 16000.

# In[ ]:


#deleting outliers points by index --> room_size
var = 'distance_poi_A3'
temp = pd.concat([train[var], train['price']], axis=1)
temp.plot.scatter(x=var, y='price')
temp.sort_values(by = var, ascending = True)

# train = train.drop(train[(train['price'] > 14.5) & train['distance_poi_A3'] > 10000].index, axis=0)
# train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_A3'] > 30000].index, axis=0)

train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_A3'] > 10000].index, axis=0)
train = train.drop(train[train['distance_poi_A3'] > 15000].index, axis=0)


# In[ ]:


#deleting outliers points by index --> room_size
var = 'distance_poi_B3'
temp = pd.concat([train[var], train['price']], axis=1)
temp.plot.scatter(x=var, y='price')
temp.sort_values(by = var, ascending = True)

# train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_B3'] > 15000].index, axis=0)
# # train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_A2'] > 30000].index, axis=0)

train = train.drop(train[train['distance_poi_B3'] > 13000].index, axis=0)


# In[ ]:


#deleting outliers points by index --> room_size
var = 'distance_poi_B4'
temp = pd.concat([train[var], train['price']], axis=1)
temp.plot.scatter(x=var, y='price')
temp.sort_values(by = var, ascending = True)

# train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_B4'] > 15000].index, axis=0)
# train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_A2'] > 30000].index, axis=0)

train = train.drop(train[train['distance_poi_B4'] > 14000].index, axis=0)


# In[ ]:


#deleting outliers points by index --> room_size
var = 'longitude'
temp = pd.concat([train[var], train['price']], axis=1)
temp.plot.scatter(x=var, y='price')
temp.sort_values(by = var, ascending = True)

# train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_B4'] > 15000].index, axis=0)
# train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_A2'] > 30000].index, axis=0)

train = train.drop(train[train['longitude'] < 110.28].index, axis=0)


# In[ ]:


# #deleting outliers points by index --> room_size
# var = 'latitude'
# temp = pd.concat([train[var], train['price']], axis=1)
# temp.plot.scatter(x=var, y='price')
# temp.sort_values(by = var, ascending = True)

# # train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_B4'] > 15000].index, axis=0)
# # train = train.drop(train[(train['price'] > 14.0) & train['distance_poi_A2'] > 30000].index, axis=0)

# train = train.drop(train[train['longitude'] < -7.88].index, axis=0)


# #**STEP 3: DATA PRE-PROCESSING AND FEATURE ENGINEERING ON COMBINED DATASET**

# Firstly, we'll combine the train and test dataset, so that any data transformation will be applied to all data uniformly. Once done, we'll need to find out what exactly are the missing data we need to handle

# In[ ]:


#STEP 3: DATA PRE-RPOCESSING AND FEATURE ENGINEERING ON COMBINED DATASET

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.price.values
all_data = pd.concat((train, test)).reset_index(drop=True)
# all_data.drop(['price'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# In[ ]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)


# In[ ]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# There are multiple methods to handle missing data:
# 
# 
# *   Fill missing data with or 0 (Common for Numerical Features)
# *   Fill missing data with 'None' ( Commong for Categorical Features)
# 
# *   Fill missing data with Mean (Commong for Numerical Features)
# *   Fill missing data with Mode (Common for Categorical Features)
# 
# *   Drop the Row (Common for rare occurances among data)
# *   Drop the Column (Common for features with large percentage of missing data)
# 
# *   Replace with any other value you deem logical
# 
# In our case, our data doesn't have that many missing value. So putting mode on boolean features  and mean on numerical features will just do good.
# 
# 
# 
# 
# 
# 
# 
# 

# Because secret_2 and secret_9 always has the same value, it's better just to drop them.

# In[ ]:


#karena secret_2 dan secret_9 semuanya 1 dan 0. Lebih baik di drop saja.
all_data = all_data.drop(['secret_2', 'secret_9'],axis=1)


# Dikarenakan secret_5 dan item_4 kemungkinan memiliki data yang sama, lebih baik kita drop salah satunya

# In[ ]:


all_data = all_data.drop(['item_4'],axis=1)


# Dikarenakan item_3 dan secret_7 kemungkinan memiliki data yang sama lebih baik kita drop saja secret_7nya

# In[ ]:


all_data = all_data.drop(['secret_7'],axis=1)


# Dikarenakan secret_10 sama room_size kemungkinan memiliki data yang sama, lebih baik kita drop salah satunya gue sih maunya secret_10

# In[ ]:


all_data = all_data.drop(['secret_10'],axis=1)


# Ini nyoba2 doang2

# In[ ]:


all_data = all_data.drop(['distance_poi_A4', 'distance_poi_A6', 'distance_poi_A5', 'distance_poi_B2', 'distance_poi_B1'],axis=1)


# Semua yang distance dari poi A1 ampe b4 diisi pake rata2

# In[ ]:


for col in ('distance_poi_A1', 'distance_poi_A2', 'distance_poi_A3', 'distance_poi_B3', 'distance_poi_B4','room_size'):
    all_data[col] = all_data[col].fillna((all_data[col].mean()))


# In[ ]:


all_data['room_size'] = all_data['room_size'].fillna((all_data['room_size'].mean()))


# Semua selain latitude longitude, diisi dengan mode

# In[ ]:


for col in ('facility_1', 'facility_2', 'facility_3','facility_4', 'facility_5', 'female', 'male', 'item_1', 'item_2', 'item_3', 'item_5','secret_1','secret_3','secret_4','secret_5','secret_6','secret_8'):
    all_data[col] = all_data[col].fillna((all_data[col].mode()[0]))


# Latitude longitude diisi dengan interpolate

# In[ ]:


all_data['latitude'] = all_data['latitude'].abs()

#change the latitude value to positive.


# In[ ]:


for col in ('longitude', 'latitude'):
    all_data[col] = all_data[col].fillna((all_data[col].interpolate(method='linear')))


# In[ ]:


#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# Now that all missing data have been handled, we'll take a look at the distribution pattern for some of our critical features. As mentioned at the beginning of this journal, most machine learning techniques work better with normalised data. I'll be using the same technique to identify and correct for skewness for every numerical features.

# Skewed Features

# In[ ]:



# Analysing and normalising target variable
var = 'room_size'
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)

# Applying log transformation to resolve skewness
all_data[var] = np.log1p(all_data[var])
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)

# Analysing and normalising target variable
var = 'distance_poi_A1'
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)

# Applying log transformation to resolve skewness
all_data[var] = np.log1p(all_data[var])
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)

# Analysing and normalising target variable
var = 'distance_poi_A2'
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)

# Applying log transformation to resolve skewness
all_data[var] = np.log1p(all_data[var])
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)

# Analysing and normalising target variable
var = 'distance_poi_A3'
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)

# Applying log transformation to resolve skewness
all_data[var] = np.log1p(all_data[var])
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)

# Analysing and normalising target variable
var = 'distance_poi_B3'
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)

# Applying log transformation to resolve skewness
all_data[var] = np.log1p(all_data[var])
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)

# Analysing and normalising target variable
var = 'distance_poi_B4'
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)

# Applying log transformation to resolve skewness
all_data[var] = np.log1p(all_data[var])
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)

# Analysing and normalising target variable
var = 'longitude'
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)

# Applying log transformation to resolve skewness
all_data[var] = np.log1p(all_data[var])
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)

# Analysing and normalising target variable
var = 'latitude'
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)

# Applying log transformation to resolve skewness
all_data[var] = np.log1p(all_data[var])
sns.distplot(all_data[var], fit=norm);
fig = plt.figure()
res = stats.probplot(all_data[var], plot=plt)


# In the final step of data pre-processing, we need to ensure that all faetures are holding numerical values, so that we can run them through the XGBRegressor model. We need to do the following:
# 
# *   Recast any numerical features that are actually categorical
# *   Conduct Label Encoding for ordinal features
# *   Conduct OneHot Encoder for remaining categorical features
# 
# But we don't have to do anything categorycal in this data.
# 
# 
# 

# #**STEP 4: XGBOOST MODELING WITH PARAMETER TUNING**

# For this journal, I have chosen a single model of XGBRegressor. The approach I have adopted for parameter-tuning can be found here:
# https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
# 
# I have gone with the common 'Mean Absolute Error' as the measuring metric, and will be applying a 5-fold cross-validation technique for training. Before we begin, we'll do a train_test_split, and load them into DMatrix (data format required for XGB models).

# In[ ]:


#STEP 4: XGBOOST MODELING WITH PARAMETER TUNING

#Creating train_test_split for cross validation
X = all_data.loc[all_data['price']>0]
X = X.drop(['price'], axis=1)
y = all_data[['price']]
y = y.drop(y.loc[y['price'].isnull()].index, axis=0)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=8)

#Creating DMatrices for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


# Next we'll set the initial parameters for the model. I have followed the logics laid out in this guide for the setting of original parameters:
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# 
# And with the initial parameters, we'll determine the ieal 'num_boost_round' and set a baseline MAE to beat. This way, we'll know whether the parameters we're tuning will indeed be resulting in a lower MAE.

# In[ ]:


#Setting initial parameters
params = {
    # Parameters that we are going to tune.
    'max_depth':5,
    'min_child_weight': 1,
    'eta':0.3,
    'subsample': 0.80,
    'colsample_bytree': 0.80,
    'reg_alpha': 0,
    'reg_lambda': 0,
    # Other parameters
    'objective':'reg:squarederror',
}


#Setting evaluation metrics - MAE from sklearn.metrics
params['eval_metric'] = "mae"

num_boost_round = 5000

#Begin training of XGB model
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=50
)

print("Best MAE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))

#replace num_boost_round with best iteration + 1
num_boost_round = model.best_iteration+1

#Establishing baseline MAE
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=8,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=50
)
cv_results
cv_results['test-mae-mean'].min()


# We can see that the baseline MAE to beat is ~0.215986. And initial num_boost_round will be set as 16. From here on, we will begin parameter tuning in 3 phases:
# 
# *   Tuning max_depth & min_child_weight (Tree-specific parameter)
# *   Tuning subsample & colsample (Tree-specific parameter)
# 
# *   Tuning reg_alpha & reg_lambda (Regularisation parameter)
# *   Tuning eta (Learning Rate)
# 
# By using gridsearch technique, we can narrow down on various values we want to test for each phase. Once the ideal value is determined, we need to update the parameters and recalibrate num_boost_round.
# 
# 
# 
# 

# In[ ]:


#Parameter-tuning for max_depth & min_child_weight (First round)
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in range(0,12,2)
    for min_child_weight in range(0,12,2)
]

min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=8,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=50
    )
    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))


#Parameter-tuning for max_depth & min_child_weight (Second round)
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in [3,4,5]
    for min_child_weight in [3,4,5]
]

min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=8,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=50
    )
    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))


#Parameter-tuning for max_depth & min_child_weight (Third round)
gridsearch_params = [
    (max_depth, min_child_weight)
    for max_depth in [3]
    for min_child_weight in [i/10. for i in range(30,50,2)]
]

min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(
                             max_depth,
                             min_child_weight))
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=8,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=50
    )
    # Update best MAE
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))


# In[ ]:


#Updating max_depth and mind_child_weight parameters
params['max_depth'] = 3
params['min_child_weight'] = 3.0


# In[ ]:


#Recalibrating num_boost_round after parameter updates
num_boost_round = 5000

model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=50
)

print("Best MAE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))

#replace num_boost_round with best iteration + 1
num_boost_round = model.best_iteration+1


# In[ ]:


#Parameter-tuning for subsample & colsample (First round)
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(3,11)]
    for colsample in [i/10. for i in range(3,11)]
]

min_mae = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=8,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=50
    )
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

#Parameter-tuning for subsample & colsample (Second round)
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/100. for i in range(80,100)]
    for colsample in [i/100. for i in range(70,90)]
]

min_mae = float("Inf")
best_params = None
# We start by the largest values and go down to the smallest
for subsample, colsample in reversed(gridsearch_params):
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
    # We update our parameters
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=8,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=50
    )
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (subsample,colsample)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))


# In[ ]:


#Updating subsample and colsample parameters
params['subsample'] = 0.85
params['colsample'] = 0.88


# In[ ]:


#Recalibrating num_boost_round after parameter updates
num_boost_round = 5000

model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")],
    early_stopping_rounds=50
)

print("Best MAE: {:.2f} with {} rounds".format(
                 model.best_score,
                 model.best_iteration+1))

#replace num_boost_round with best iteration + 1
num_boost_round = model.best_iteration+1


# In[ ]:


#Parameter-tuning for reg_alpha & reg_lambda
gridsearch_params = [
    (reg_alpha, reg_lambda)
    for reg_alpha in [1e-5, 1e-4, 1e-3, 1e-2, 0.1]
    for reg_lambda in [1e-5, 1e-4, 1e-3, 1e-2, 0.1]
]

min_mae = float("Inf")
best_params = None

for reg_alpha, reg_lambda in gridsearch_params:
    print("CV with reg_alpha={}, reg_lambda={}".format(
                             reg_alpha,
                             reg_lambda))
    # We update our parameters
    params['reg_alpha'] = reg_alpha
    params['reg_lambda'] = reg_lambda
    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=8,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=50
    )
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (reg_alpha,reg_lambda)
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))


# In[ ]:


#Updating reg_alpha and reg_lambda parameters
params['reg_alpha'] = 0.0001
params['reg_lambda'] = 1e-05


# In[ ]:


#Resetting num_boost_round to 5000
num_boost_round = 5000

#Parameter-tuning for eta
min_mae = float("Inf")
best_params = None
for eta in [0.3, 0.2, 0.1, 0.05, 0.01, 0.005]:
    print("CV with eta={}".format(eta))

    params['eta'] = eta
    cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=8,
            nfold=5,
            metrics=['mae'],
            early_stopping_rounds=50
          )
    # Update best score
    mean_mae = cv_results['test-mae-mean'].min()
    boost_rounds = cv_results['test-mae-mean'].idxmin()
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = eta
print("Best params: {}, MAE: {}".format(best_params, min_mae))


# In[ ]:


params['eta'] = 0.01


# Now that all ideal parameters are found, let us recap:
# 
# params = { 'max_depth':3, 'min_child_weight': 3.0, 'eta':0.005, 'subsample': 0.85, 'colsample_bytree': 0.89, 'reg_alpha': 0.0001, 'reg_lambda': 0.01, }

# In[ ]:


model = xgb.train(
    params,
    dtrain,
    num_boost_round=5000,
    evals=[(dtest, "Test")],
    early_stopping_rounds=50
)

num_boost_round = model.best_iteration + 1
best_model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtest, "Test")]
)

mean_absolute_error(best_model.predict(dtest), y_test)


# In[ ]:


testdf = all_data.loc[all_data['price'].isnull()]
testdf = testdf.drop(['price'],axis=1)
sub = pd.DataFrame()
sub['id'] = test_ID
testdf = xgb.DMatrix(testdf)

y_pred = np.expm1(best_model.predict(testdf))
sub['price'] = y_pred

sub.to_csv('submission.csv', index=False)


# In[ ]:




