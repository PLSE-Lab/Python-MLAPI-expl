#!/usr/bin/env python
# coding: utf-8

# # Kaggle Competition - Tanzania Waterpoint Prediction

# In this kernel, I will walkthorough the procedure of how I achieved over 80% accuracy in the prediction of Tanzania's waterpoint. It's important to recognize before proceding that the goal of this prediction is to maximize accuracy so I will not discuss the interpreation of each of the features or the statistical significance of each of them.

# ### The Problem
# 
# Can you predict which water pumps are faulty?
# Using data from Taarifa and the Tanzanian Ministry of Water, can you predict which pumps are functional, which need some repairs, and which don't work at all? Predict one of these three classes based on a number of variables about what kind of pump is operating, when it was installed, and how it is managed. A smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.

# ### Data Overview

# **Features**
# 
#         Your goal is to predict the operating condition of a waterpoint for each record in the dataset. You are provided the following set of information about the waterpoints:
# 
#       amount_tsh : Total static head (amount water available to waterpoint) - numeric
#       date_recorded : The date the row was entered - datetime
#       funder : Who funded the well
#       gps_height : Altitude of the well
#       installer : Organization that installed the well
#       longitude : GPS coordinate
#       latitude : GPS coordinate
#       wpt_name : Name of the waterpoint if there is one
#       num_private :
#       basin : Geographic water basin
#       subvillage : Geographic location
#       region : Geographic location
#       region_code : Geographic location (coded)
#       district_code : Geographic location (coded)
#       lga : Geographic location
#       ward : Geographic location
#       population : Population around the well
#       public_meeting : True/False
#       recorded_by : Group entering this row of data
#       scheme_management : Who operates the waterpoint
#       scheme_name : Who operates the waterpoint
#       permit : If the waterpoint is permitted
#       construction_year : Year the waterpoint was constructed
#       extraction_type : The kind of extraction the waterpoint uses
#       extraction_type_group : The kind of extraction the waterpoint uses
#       extraction_type_class : The kind of extraction the waterpoint uses
#       management : How the waterpoint is managed
#       management_group : How the waterpoint is managed
#       payment : What the water costs
#       payment_type : What the water costs
#       water_quality : The quality of the water
#       quality_group : The quality of the water
#       quantity : The quantity of water
#       quantity_group : The quantity of water
#       source : The source of the water
#       source_type : The source of the water
#       source_class : The source of the water
#       waterpoint_type : The kind of waterpoint
#       waterpoint_type_group : The kind of waterpoint
# 
# 
# **Labels**
# 
# There are three possible values:
# 
#       functional : the waterpoint is operational and there are no repairs needed
#       functional needs repair : the waterpoint is operational, but needs repairs
#       non functional : the waterpoint is not operational

# The first step is to import the data and make sure the import is the desired one. 

# ### Importing packages and files

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('seaborn-whitegrid')
import seaborn as sns


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)


# In[ ]:


train_feature_url = '../input/train_features.csv'
test_feature_url = '../input/test_features.csv'
train_labels_url = '../input/train_labels.csv'


# In[ ]:


train_F = pd.read_csv(train_feature_url, index_col='id')
test_F = pd.read_csv(test_feature_url, index_col='id')
train_L = pd.read_csv(train_labels_url, index_col='id')

train_F.shape, test_F.shape, train_L.shape  #,  train_F_C.shape, test_F_C.shape, # 


# We have 39 features and 1 target variable. I will do some further investigation to see if I can get more features.

# ### Feature Cleanup and Engineering

# In[ ]:


train_F.dtypes


# **Finding Columns with Null Values for later**

# In[ ]:


null_list = []

for col in train_F.columns:
  if train_F[col].isnull().sum() > 0:
    null_list.append(col)
    
null_list.append('recorded_by')
null_list.append('date_recorded')
    
null_list


# I will do some datatype conversion to exploit characteristics of variables that are not in the original feature.

# **Converting the date variable into a datetime object and splitting it into pieces**

# In[ ]:


def convert_datetime(df, col):
  df[col] = pd.to_datetime(df[col])
  df['day_of_week'] = df[col].dt.weekday_name 
  df['year'] = df[col].dt.year
  df['month'] = df[col].dt.month 
  df['day'] = df[col].dt.day 
  
  return None


# In[ ]:


convert_datetime(train_F, 'date_recorded')
convert_datetime(test_F, 'date_recorded')

train_F.dtypes


# **Converting numeric columns to category type**

# In[ ]:


train_F['region_code'] = train_F['region_code'].astype('category')
test_F['region_code'] = test_F['region_code'].astype('category')
train_F['district_code'] = train_F['district_code'].astype('category')
test_F['district_code'] = test_F['district_code'].astype('category')
train_F['wpt_name'] = train_F['wpt_name'].astype('category')
test_F['wpt_name'] = test_F['wpt_name'].astype('category')
train_F['ward'] = train_F['ward'].astype('category')
test_F['ward'] = test_F['ward'].astype('category')


train_F.dtypes


# 

# I will use a function to divide the training and testing data into numerical and catagorical data.

# #### Splitting into Numeric and Nonnumeric variables

# In[ ]:


def df_split(df):
  numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
  df_num = df.select_dtypes(include=numerics)
  df_cat = df.drop(df_num, axis = 'columns')
  print (df.shape, df_num.shape, df_cat.shape)
  return df_num, df_cat
  


# In[ ]:


train_F_num, train_F_cat = df_split(train_F)

test_F_num, test_F_cat = df_split(test_F)


# I will now look at the numeric data using the pandas describe() method to find any flaws and fix them as best as possible.

# #### Feature Engineering in the numeric sets

# In[ ]:


train_F_num.describe().T


# In[ ]:


train_F_num['construction_year'].loc[train_F_num['construction_year'] == 0] = train_F_num['year']
test_F_num['construction_year'].loc[test_F_num['construction_year'] == 0] = test_F_num['year']

train_F_num.describe().T


# **Creating new variables "distance" and "distance3D"**

# In[ ]:


mean_lat_train = train_F_num['latitude'].mean()
mean_long_train = train_F_num['longitude'].mean()
mean_lat_test = test_F_num['latitude'].mean()
mean_long_test = test_F_num['longitude'].mean()


train_F_num['distance'] = np.sqrt((train_F_num['longitude'] - mean_long_train)**2 + (train_F_num['latitude'] - mean_lat_train)**2)
test_F_num['distance'] = np.sqrt((test_F_num['longitude'] - mean_long_test)**2 + (test_F_num['latitude'] - mean_lat_test)**2)

train_F_num['distance3d'] = np.sqrt((train_F_num['gps_height']**2 + train_F_num['longitude'] - mean_long_train)**2 + (train_F_num['latitude'] - mean_lat_train)**2)
test_F_num['distance3d'] = np.sqrt((test_F_num['gps_height']**2 + test_F_num['longitude'] - mean_long_test)**2 + (test_F_num['latitude'] - mean_lat_test)**2)


train_F_num.describe().T


# I will now convert the categorical variables into dummy variable dataframes using the pandas get_dummies() method. In the last step, I will concat these into one large dummy variable dataframe.

# #### Converting Categorical variables to Dummy Variables

# In[ ]:


for col in train_F_cat.columns:
  print (col, train_F_cat[col].nunique())


# In[ ]:


null_list


# #### Creating Dummy variables out of the less unique (<=125 categories) categorical variables.

# In[ ]:


cols_kept = []

for col in train_F_cat.columns:
  if col not in null_list:
    if train_F_cat[col].nunique() <= 125:
      cols_kept.append(col)
    
print (len(cols_kept))
    
cols_kept


# In[ ]:


small_cat_train = train_F_cat[cols_kept]
small_cat_test = test_F_cat[cols_kept]

small_cat_train.shape, small_cat_test.shape


# In[ ]:


def dummy_df(category_df):
  df_dummy = pd.DataFrame()
  for col in category_df.columns:
    df_dummy = pd.concat([df_dummy, pd.get_dummies(category_df[col], drop_first=True, prefix = 'Is')], axis='columns')
  return df_dummy


# In[ ]:


df_dumb_train = dummy_df(small_cat_train)
df_dumb_test = dummy_df(small_cat_test)


# In this process, there is an occasional mismatch between the number of training and testing features which will cause problems with the model down the road. I will fix those issues here.

# #### Check to see the Train and Test Features have the same columns in the same order.

# In[ ]:


df_dumb_train.shape, df_dumb_test.shape


# In[ ]:


a = list(df_dumb_train.columns.values)

print(a)
print(len(a))


# In[ ]:


b = list(df_dumb_test.columns.values)

print(b)
print(len(b))


# In[ ]:


a == b


# In[ ]:


def ex_cols(a,b):
  ex_a = []
  ex_b = []
  for i in range(0,len(a)):
    if a[i] not in b:
      ex_a.append(a[i])
  for j in range(0,len(b)):
    if b[j] not in a:
      ex_b.append(b[j])
  return ex_a, ex_b

ex_a, ex_b = ex_cols(a,b)

ex_a,ex_b


# In[ ]:


for col in df_dumb_train.columns:
  if col in ex_a:
    del df_dumb_train[col]

for col in df_dumb_test.columns:
  if col in ex_b:
    del df_dumb_test[col]

df_dumb_train.shape, df_dumb_test.shape


# In[ ]:


c = list(df_dumb_train.columns.values)
d = list(df_dumb_test.columns.values)

c == d


# In[ ]:


for i in range(0,len(c)):
  if c[i] != d[i]:
    print("No match")
    


# In the last pre-modeling step for the features, I recombine the feature engineered numeric and categorical matrices.

# #### Combining to form the Feature Matrices

# In[ ]:


X_train = pd.concat([train_F_num,df_dumb_train],axis='columns')
X_test = pd.concat([test_F_num,df_dumb_test],axis='columns')

X_train.shape, X_test.shape


# In[ ]:


X_train.head()


# This is a check to see that there are no strange values that cause errors in the prediction. 

# In[ ]:


np.any(np.isnan(X_train)), np.any(np.isnan(X_test))


# Now that the feature matrices are complete, I start work on the target variable vector to make confirm it is working properly.

# ### Target Variable

# In[ ]:


train_L.head()


# In[ ]:


train_L['status_group'] = train_L['status_group'].astype('category')

train_L.dtypes, train_L.shape


# In[ ]:


y_train = train_L['status_group']

y_train.value_counts()


# The first guess in a prediction is the majority class baseline. It's the simplest model and the result of this prediction serves as a reference point against future predictions. For the majority class baseline, I am predicting against the actual test target using the mode of the training target. I achieved **53.75%** accuracy using this baseline.

# ### Majority Class Baseline - First,Guess

# In[ ]:


majority_class = y_train.mode()[0]

print(majority_class)

y_pred = pd.DataFrame(np.full(shape=len(X_test), fill_value = majority_class))


# In[ ]:


temp = X_test.reset_index()

y_pred['id'] = temp['id'].values
y_pred.rename(columns={0:'status_group'}, inplace=True)
y_pred.set_index('id', inplace=True)

y_pred.head()

print(y_pred.shape)


# In[ ]:


# from google.colab import files

# y_pred.to_csv('majority_class.csv')
# files.download('majority_class.csv')


# Now that I have a baseline accuracy, I want to improve on that. Looking at the training target variable, one can see it has three values. Therefore, I will need a *multinomial* classifier.
# 
# My choice of classifiers are *Multinomial Logistic Regression* and *Random Forest*.
# 
# **Documentation**:
# 
# [Logistic Regression](http://https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
# 
# [Random Forest](http://https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
# 
# **Background:**
# 
# [Logistic Regression - Wikipedia](http://https://en.wikipedia.org/wiki/Logistic_regression)
# 
# [Random Forest - Wikipedia](http://https://en.wikipedia.org/wiki/Random_forest)

# ### Prediction using better classifiers

# **Classifiers**

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier


clf_rf = RandomForestClassifier(n_estimators=100, 
                                max_depth=34,
                                min_samples_split = 17,
                                min_samples_leaf = 1,
                                criterion = 'gini', 
                                max_features = 6, 
                                oob_score = True, 
                                random_state=237)

clf_lr = LogisticRegression(random_state=237, solver='lbfgs', multi_class='multinomial', max_iter=1000)


# ### Validation with just the training set split

# In[ ]:


def quick_eval(X,y, clf):
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.preprocessing import RobustScaler
  from sklearn.metrics import accuracy_score

  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True, random_state=237)
  print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
  
  scaler = StandardScaler()
#   scaler = RobustScaler()
  X_train_clf = scaler.fit_transform(X_train)
  X_test_clf = scaler.transform(X_test)
  
  clf.fit(X_train_clf, y_train)
  
  y_pred_train = clf.predict(X_train_clf)
  
  y_pred = clf.predict(X_test_clf)
  
  
  return accuracy_score(y_train,y_pred_train), accuracy_score(y_test, y_pred)


# In[ ]:


# %%time

quick_eval(X_train, y_train, clf_rf)


# In order to improve on the accuracy, I will use a Randomized Grid Search to find the tuning parameters that maximize accuracy.

# **Tuning the parameters - Randomized Grid Search - Random Forest (est. runtime 5 hours)**

# In[ ]:


# %%time

# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint as sp_randint
# # parameters for GridSearchCV
# # specify parameters and distributions to sample from
# param_dist = {"n_estimators": [100, 200,300],
#               "max_features": sp_randint(5, 9),
#               "max_depth": [18,22,26,30,34,38],
#               "min_samples_split": sp_randint(8, 32),
#               "min_samples_leaf": sp_randint(1, 20)              
#              }
# # run randomized search
# n_iter_search = 200
# random_search = RandomizedSearchCV(clf_rf, param_distributions=param_dist,
#                                    n_iter=n_iter_search)

# random_search.fit(X_train, y_train)


# In[ ]:


# def report(results, n_top=5):
#     for i in range(1, n_top + 1):
#         candidates = np.flatnonzero(results['rank_test_score'] == i)
#         for candidate in candidates:
#             print("Model with rank: {0}".format(i))
#             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                   results['mean_test_score'][candidate],
#                   results['std_test_score'][candidate]))
#             print("Parameters: {0}".format(results['params'][candidate]))
#             print("")
            
# report(random_search.cv_results_)            


# In this cross-validation stage, I confirm that the results are consistent.

# ### Cross Validation

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler

scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()

X_train_clf = scaler.fit_transform(X_train)
X_test_clf = scaler.transform(X_test)


X_train_clf.shape, y_train.shape, X_test_clf.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.model_selection import cross_validate\n\nfrom sklearn.metrics import accuracy_score\n\n\nscores = cross_validate(clf_rf,\n                        X_train_clf,y_train, \n                        scoring = 'accuracy', cv=5) \n")


# In[ ]:


pd.DataFrame(scores)


# The results of the cross-validation show consistent results as well as minimal overfit to the training model. With a Random Forest classifier, one of the problems with it usually is overfit to the training set which causes massive differences between training and testing accuracy. As this accuracy gap is the smallest I've experienced, I'm keeping this model. 

# ### Creating the Prediction Vector

# In[ ]:


def predictor(X_train, X_test, y_train, clf):
  from sklearn.preprocessing import StandardScaler
  from sklearn.preprocessing import RobustScaler
 
  from sklearn.metrics import accuracy_score
  

  y_pred = pd.DataFrame()
  
  temp_test = X_test.reset_index()
  y_id = temp_test['id']
#   scaler = StandardScaler()
  scaler = RobustScaler()
  
  X_train_clf = scaler.fit_transform(X_train)

  X_test_clf = scaler.transform(X_test)
  clf.fit(X_train_clf, y_train)
  
  y_pred_train = clf.predict(X_train_clf)
  
  print (f'\nThe accuracy score of the training set is {round(accuracy_score(y_train, y_pred_train), 5)}\n')
  
  prediction = pd.DataFrame(clf.predict(X_test_clf))
  
  y_pred = pd.concat([y_id, prediction], axis='columns')

  y_pred.rename(columns={0:'status_group'}, inplace=True)
  
  y_pred.set_index('id', inplace=True)
  
  return y_pred


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndf = predictor(X_train, X_test, y_train, clf_rf)\n')


# This is the final check to see if the prediction vector is formatted correctly and has the correct shape. 

# ### Sanity Check

# In[ ]:


df.head()


# In[ ]:


df['status_group'].value_counts()


# In[ ]:


df.shape


# ### Submission Download

# In[ ]:


# from google.colab import files

# df.to_csv('submission.csv')
# files.download('submission.csv')


# ### Discussion

# This was a walkthrough through the process of taking a dataset and using it to achieve a high accuracy score. As stated earlier, this is not a discussion of the significance of the features but rather the accuracy of the model with the classifier as a whole. I started with a quick investigation and data cleanup followed by feature engineering. From there, I formatted the training target variable and used that to get a majority class baseline prediction vector. I then introduced the Logistic Regression and Random Forest Classifers, choosing to use the Random Forest for its improved accuracy. Using a Randomized Grid Search, I fine tuned the parameters of the Random Forest Classifer to achieve low overfit and greater accuracy. I cross validated the results to demonstrate the consistency of the results with each other and with the previous results. Finally with this tuned model, I used it to predict the test target vector using the test features matrix. 
# 
# From this process, I saw that the random forest classifier beat the multinomial logistic regression classifier by about 10-15% but among the drawbacks to using a random forest classifier are 1) overfit to the training set and 2) lack of interpretability of the features. However, since the stated goal was to achieve maximum accuracy, the choice of the random forest classifier is acceptable to me. 
