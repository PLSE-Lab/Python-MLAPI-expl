#!/usr/bin/env python
# coding: utf-8

# **Importing Libraries**

# In[ ]:


# Start Python Imports
import math, time, random, datetime

# Data Manipulation
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
import missingno
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from imblearn.over_sampling import SMOTE

# Machine learning
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import confusion_matrix

# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


claims = pd.read_csv("../input/warranty-claims/train.csv")


# In[ ]:


claims.head()


# In[ ]:


claims.shape


# EXPLORATORY DATA ANALYSIS

# In[ ]:


claims.loc[(claims.State == "UP"), "State"] = "Uttar Pradesh"  ## Replacing UP with Uttar Pradesh


# In[ ]:


claims.loc[(claims.Purpose == "claim"), "Purpose"] = "Claim"  ## Replacing claim with Claim


# In[ ]:


claims.loc[(claims.State == "Telengana"), "City"] = "Hyderabad 1"   ## Separating hyderbad among two states. like Andhra Pradesh = Hyderbad, Telengana = Hyderabad 1


# In[ ]:


claims.info()


# #Histograms of continous variables

# In[ ]:


claims.Product_Age.plot.hist()


# In[ ]:


claims.Claim_Value.plot.hist()


# In[ ]:


claims.Call_details.plot.hist()


# #checking mean median count of continous various variables in dataset 

# In[ ]:


claims.describe()


# #plot for missing values 

# In[ ]:


missingno.matrix(claims, figsize = (30,10))


# 11# Alternatively, you can see the number of missing values like this

# In[ ]:


claims.isnull().sum()


# In[ ]:


claims.shape


# #To perform our data analysis, let's create two new dataframes
# 

# In[ ]:


claims_bin = pd.DataFrame() # for discretised continuous variables
claims_con = pd.DataFrame() # for continuous variables


# ..# Different data types in the dataset

# In[ ]:


claims.dtypes


# # 1st column : Region

# In[ ]:


# How many missing values does Region have?
claims.Region.isnull().sum() 


# In[ ]:


# unique value counts
claims.Region.value_counts()


# In[ ]:


#regions distribution 
fig = plt.figure(figsize=(20,7))
sns.countplot(y='Region', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['Region'] = claims['Region']
claims_con['Region'] = claims['Region']


# # 2nd column : State

# In[ ]:


# How many missing values does State have?
claims.State.isnull().sum()


# In[ ]:


#unique value count
claims.State.value_counts()


# In[ ]:


#States distribution 
fig = plt.figure(figsize=(20,7))
sns.countplot(y='State', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['State'] = claims['State']
claims_con['State'] = claims['State']


# # 3rd column : Area

# In[ ]:


# How many missing values does Area have?
claims.Area.isnull().sum()


# In[ ]:


#unique values count
claims.Area.value_counts()


# In[ ]:


#States distribution 
fig = plt.figure(figsize=(15,2))
sns.countplot(y='Area', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['Area'] = claims['Area']
claims_bin['Area'] = np.where(claims_bin['Area'] == 'Urban', 1, 0) # change Area to 1 for Urban and 0 for Rural
claims_con['Area'] = claims['Area']


# # 4th column : City

# In[ ]:


# How many missing values does State have?
claims.City.isnull().sum() 


# In[ ]:


# unique value count
claims.City.value_counts()


# In[ ]:


#City distribution 
fig = plt.figure(figsize=(20,7))
sns.countplot(y='City', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['City'] = claims['City']
claims_con['City'] = claims['City']


# # 5th column : Consumer_profile

# In[ ]:


# How many missing values does Consumer_profile have?
claims.Consumer_profile.isnull().sum()


# In[ ]:


#unique value count
claims.Consumer_profile.value_counts()


# In[ ]:


#Consumer_profile distribution 
fig = plt.figure(figsize=(10,4))
sns.countplot(y='Consumer_profile', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['Consumer_profile'] = claims['Consumer_profile']
claims_bin['Consumer_profile'] = np.where(claims_bin['Consumer_profile'] == 'Business', 1, 0) # change Consumer profile to 1 for Business and 0 for Personal
claims_con['Consumer_profile'] = claims['Consumer_profile']


# # 6th column : Product_category

# In[ ]:


# How many missing values does Product_category have?
claims.Product_category.isnull().sum() 


# In[ ]:


# unique value count
claims.Product_category.value_counts()


# In[ ]:


#Product_category distribution 
fig = plt.figure(figsize=(10,4))
sns.countplot(y='Product_category', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['Product_category'] = claims['Product_category']
claims_bin['Product_category'] = np.where(claims_bin['Product_category'] == 'Entertainment', 1, 0) # change Product_category to 1 for Entertainment and 0 for Household
claims_con['Product_category'] = claims['Product_category']


# # 7th column : Product_type

# In[ ]:


# How many missing values does Product_type have?
claims.Product_type.isnull().sum()


# In[ ]:


#unique value count
claims.Product_type.value_counts()


# In[ ]:


#Product_category distribution 
fig = plt.figure(figsize=(10,4))
sns.countplot(y='Product_type', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['Product_type'] = claims['Product_type']
claims_bin['Product_type'] = np.where(claims_bin['Product_type'] == 'TV', 1, 0) # change Product_type to 1 for TV and 0 for AC
claims_con['Product_type'] = claims['Product_type']


# # 8th column : AC_1001_Issue

# In[ ]:


# How many missing values does AC_1001_Issue have?
claims.AC_1001_Issue.isnull().sum()


# In[ ]:


claims.AC_1001_Issue.value_counts()


# In[ ]:


#AC_1001_Issue distribution 
fig = plt.figure(figsize=(10,2))
sns.countplot(y='AC_1001_Issue', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['AC_1001_Issue'] = claims['AC_1001_Issue']
claims_con['AC_1001_Issue'] = claims['AC_1001_Issue']


# # 9th column : AC_1002_Issue

# In[ ]:


# How many missing values does AC_1002_Issue have?
claims.AC_1002_Issue.isnull().sum()


# In[ ]:


#unique value count
claims.AC_1002_Issue.value_counts()


# In[ ]:


#AC_1002_Issue distribution 
fig = plt.figure(figsize=(10,2))
sns.countplot(y='AC_1002_Issue', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['AC_1002_Issue'] = claims['AC_1002_Issue']
claims_con['AC_1002_Issue'] = claims['AC_1002_Issue']


# # 10th column : AC_1003_Issue

# In[ ]:


# How many missing values does AC_1003_Issue have?
claims.AC_1003_Issue.isnull().sum()


# In[ ]:


#Unique value count
claims.AC_1003_Issue.value_counts()


# In[ ]:


#AC_1003_Issue distribution 
fig = plt.figure(figsize=(10,2))
sns.countplot(y='AC_1003_Issue', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['AC_1003_Issue'] = claims['AC_1003_Issue']
claims_con['AC_1003_Issue'] = claims['AC_1003_Issue']


# # 11th column : TV_2001_Issue

# In[ ]:


# How many missing values does TV_2001_Issue have?
claims.TV_2001_Issue.isnull().sum()


# In[ ]:


# unique value count
claims.TV_2001_Issue.value_counts()


# In[ ]:


#TV_2001_Issue distribution 
fig = plt.figure(figsize=(10,2))
sns.countplot(y='TV_2001_Issue', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['TV_2001_Issue'] = claims['TV_2001_Issue']
claims_con['TV_2001_Issue'] = claims['TV_2001_Issue']


# # 12th column : TV_2002_Issue

# In[ ]:


# How many missing values does TV_2002_Issue have?
claims.TV_2002_Issue.isnull().sum()


# In[ ]:


#unique value count
claims.TV_2002_Issue.value_counts()


# In[ ]:


#TV_2002_Issue distribution 
fig = plt.figure(figsize=(10,2))
sns.countplot(y='TV_2002_Issue', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['TV_2002_Issue'] = claims['TV_2002_Issue']
claims_con['TV_2002_Issue'] = claims['TV_2002_Issue']


# # 13th column : TV_2003_Issue

# In[ ]:


# How many missing values does TV_2003_Issue have?
claims.TV_2003_Issue.isnull().sum()


# In[ ]:


#unique value count
claims.TV_2003_Issue.value_counts()


# In[ ]:


#TV_2003_Issue distribution 
fig = plt.figure(figsize=(10,2))
sns.countplot(y='TV_2003_Issue', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['TV_2003_Issue'] = claims['TV_2003_Issue']
claims_con['TV_2003_Issue'] = claims['TV_2003_Issue']


# # 14th column : Claim_Value

# In[ ]:


# How many missing values does Claim_Value have?
claims.Claim_Value.isnull().sum()


# In[ ]:


claims["Claim_Value"].mean() #14051.15 rs
claims["Claim_Value"].median() #10000 rs


# In[ ]:


#imputed NA values or missing values with median of claim_value variable
claims["Claim_Value"].fillna(10000,inplace=True)


# In[ ]:


# How many different values of Claim_Value are there?
fig = plt.figure(figsize=(30,20))
sns.countplot(y="Claim_Value", data=claims);


# In[ ]:


# How many unique kinds of Claim_Value are there?
print("There are {} unique values in Claim_Value.".format(len(claims.Claim_Value.unique())))


# In[ ]:


# Add Claim Value to sub dataframes
claims_bin['Claim_Value'] = pd.cut(claims['Claim_Value'], bins=5) # discretised into 5 categories
claims_con['Claim_Value'] = claims['Claim_Value'] 


# #Function to create count and distribution visualisations

# In[ ]:


def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):
    """
    Function to plot counts and distributions of a label variable and 
    target variable side by side.
    ::param_data:: = target dataframe
    ::param_bin_df:: = binned dataframe for countplot
    ::param_label_column:: = binary labelled column
    ::param_target_column:: = column you want to view counts and distributions
    ::param_figsize:: = size of figure (width, height)
    ::param_use_bin_df:: = whether or not to use the bin_df, default False
    """
    if use_bin_df: 
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=bin_df);
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column], 
                     kde_kws={"label": "Fraud"});
        sns.distplot(data.loc[data[label_column] == 0][target_column], 
                     kde_kws={"label": "Genuine"});
    else:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=data);
        plt.subplot(1, 2, 2)
        sns.distplot(data.loc[data[label_column] == 1][target_column], 
                     kde_kws={"label": "Fraud"});
        sns.distplot(data.loc[data[label_column] == 0][target_column], 
                     kde_kws={"label": "Genuine"});


# In[ ]:


# What do our Claim Value bins look like?
claims_bin.Claim_Value.value_counts()


# In[ ]:


# Visualise the Claim Value bin counts as well as the Claim_Value distribution versus Fraud.
plot_count_dist(data=claims,
                bin_df=claims_bin,
                label_column='Fraud', 
                target_column='Claim_Value', 
                figsize=(20,10), 
                use_bin_df=True)


# # 15th column : Service_Centre

# In[ ]:


# How many missing values does Service_Centre have?
claims.Service_Centre.isnull().sum()


# In[ ]:


#unique value count 
claims.Service_Centre.value_counts()


# In[ ]:


#Service_centre distribution 
fig = plt.figure(figsize=(10,2))
sns.countplot(y='Service_Centre', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['Service_Centre'] = claims['Service_Centre']
claims_con['Service_Centre'] = claims['Service_Centre']


# # 16th column : Product_Age

# In[ ]:


# How many missing values does Product_Age have?
claims.Product_Age.isnull().sum()


# In[ ]:


# How many different values of Product_Age are there?
fig = plt.figure(figsize=(30,20))
sns.countplot(y="Product_Age", data=claims);


# In[ ]:


# How many unique kinds of Product_Age are there?
print("There are {} unique values in Product_Age.".format(len(claims.Product_Age.unique())))


# In[ ]:


# Add Product_Age to sub dataframes
claims_bin['Product_Age'] = pd.cut(claims['Product_Age'], bins=5) # discretised
claims_con['Product_Age'] = claims['Product_Age'] 


# In[ ]:


# What do our Product_Age bins look like?
claims_bin.Product_Age.value_counts()


# In[ ]:


# Visualise the Product_Age bin counts as well as the Product_Age distribution versus Fraud.
plot_count_dist(data=claims,
                bin_df=claims_bin,
                label_column='Fraud', 
                target_column='Product_Age', 
                figsize=(20,10), 
                use_bin_df=True)


# # 17th column : Purchased_from

# In[ ]:


# How many missing values does Purchased_from have?
claims.Purchased_from.isnull().sum()


# In[ ]:


#unique value count
claims.Purchased_from.value_counts()


# In[ ]:


#Purchased_from distribution 
fig = plt.figure(figsize=(20,7))
sns.countplot(y='Purchased_from', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['Purchased_from'] = claims['Purchased_from']
claims_con['Purchased_from'] = claims['Purchased_from']


# # 18th column : Call_details

# In[ ]:


# How many missing values does Call_details have?
claims.Call_details.isnull().sum()


# In[ ]:


# How many different values of Call_details are there?
fig = plt.figure(figsize=(30,15))
sns.countplot(y="Call_details", data=claims);


# In[ ]:


# How many unique kinds of Call_details are there?
print("There are {} unique values in Call_details.".format(len(claims.Call_details.unique())))


# In[ ]:


# Add Call_details to sub dataframes
claims_bin['Call_details'] = pd.cut(claims['Call_details'], bins=5) # discretised
claims_con['Call_details'] = claims['Call_details'] 


# In[ ]:


# What do our Call_details bins look like?
claims_bin.Call_details.value_counts()


# In[ ]:


# Visualise the Call_details bin counts as well as the Call_details distribution versus Fraud.
plot_count_dist(data=claims,
                bin_df=claims_bin,
                label_column='Fraud', 
                target_column='Call_details', 
                figsize=(20,10), 
                use_bin_df=True)


# # 19th column : Purpose

# In[ ]:


# How many missing values does Purpose have?
claims.Purpose.isnull().sum()


# In[ ]:


#Unique value count
claims.Purpose.value_counts()


# In[ ]:


#purpose distribution 
fig = plt.figure(figsize=(10,3))
sns.countplot(y='Purpose', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['Purpose'] = claims['Purpose']
claims_con['Purpose'] = claims['Purpose']


# # 20th column : Fraud

# In[ ]:


# How many people fraudulent?
fig = plt.figure(figsize=(10,3))
sns.countplot(y='Fraud', data=claims);


# In[ ]:


# adding this to subset dataframes
claims_bin['Fraud'] = claims['Fraud']
claims_con['Fraud'] = claims['Fraud']


# In[ ]:


claims_bin.head()


# #Feature scaling

# In[ ]:


# One-hot encode binned variables
one_hot_cols = claims_bin.columns.tolist()
one_hot_cols.remove('Fraud')
claims_bin_enc = pd.get_dummies(claims_bin, columns=one_hot_cols)


# In[ ]:


claims_bin_enc.head()


# In[ ]:


claims_con.head()


# In[ ]:


# One hot encode the categorical columns individually
claims_Region_one_hot = pd.get_dummies(claims_con['Region'],prefix='region')
claims_State_one_hot = pd.get_dummies(claims_con['State'],prefix='state')
claims_Area_one_hot = pd.get_dummies(claims_con['Area'],prefix='area')
claims_City_one_hot = pd.get_dummies(claims_con['City'],prefix='city')
claims_Conpro_one_hot = pd.get_dummies(claims_con['Consumer_profile'],prefix='consumer_profile')
claims_Procat_one_hot = pd.get_dummies(claims_con['Product_category'],prefix='product_category')
claims_Protyp_one_hot = pd.get_dummies(claims_con['Product_type'],prefix='product_type')
claims_Servc_one_hot = pd.get_dummies(claims_con['Service_Centre'],prefix='serrvice_centre')
claims_Purfrm_one_hot = pd.get_dummies(claims_con['Purchased_from'],prefix='purchased_from')
claims_Purpose_one_hot = pd.get_dummies(claims_con['Purpose'],prefix='purpose')


# In[ ]:


# Combine the one hot encoded columns with claims_con_enc
claims_con_enc = pd.concat([claims_con, 
                        claims_Region_one_hot, 
                        claims_State_one_hot, 
                        claims_Area_one_hot, 
                        claims_City_one_hot,
                        claims_Conpro_one_hot, 
                        claims_Procat_one_hot,
                        claims_Protyp_one_hot, 
                        claims_Servc_one_hot,
                        claims_Purfrm_one_hot, 
                        claims_Purpose_one_hot,], axis=1)


# In[ ]:


# Drop the original categorical columns (because now they've been one hot encoded)
claims_con_enc = claims_con_enc.drop(['Region', 'State', 'Area','City',
                                      'Consumer_profile','Product_category',
                                      'Product_type','Service_Centre',
                                      'Purchased_from','Purpose'], axis=1)


# In[ ]:


# Let's look at claims_con_enc
claims_con_enc.head(20)


# In[ ]:


# Seclect the dataframe we want to use first for predictions
selected_claims = claims_con_enc
selected_claims.head()


# Balancing data

# In[ ]:


x_sm = pd.DataFrame.copy(selected_claims)
y_sm = x_sm.pop('Fraud')
sm = SMOTE(random_state =101)
x_train, y_train = sm.fit_sample(x_sm,y_sm)
x_train.shape, y_train.shape


# converting array to DataFrame

# In[ ]:


x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)


# changing column names

# In[ ]:


vname = x_sm.columns
x_train.columns = vname


# loading test data

# In[ ]:


test = pd.read_csv("../input/warranty-claims/test_1.csv")


# Preprocessing for test data as train

# In[ ]:


missingno.matrix(test, figsize = (30,10))


# In[ ]:


test.isnull().sum()


# In[ ]:


test.head(20)


# In[ ]:


## Replacing UP with Uttar Pradesh 
test.loc[(test.State == "UP"), "State"] = "Uttar Pradesh"


# In[ ]:


## Replacing claim with Claim
test.loc[(test.Purpose == "claim"), "Purpose"] = "Claim"


# In[ ]:


## Separating hyderbad among two states. like Andhra Pradesh = Hyderbad, Telengana = Hyderabad 1
test.loc[(test.State == "Telengana"), "City"] = "Hyderabad 1"


# In[ ]:


# Deleting first column
test.drop(["Unnamed: 0"],inplace=True,axis=1) 


# In[ ]:


test.head()


# Imputing NA values

# In[ ]:


test.Claim_Value.isnull().sum()
test["Claim_Value"].median() #median = 10000 Rs
test["Claim_Value"].fillna(10000,inplace=True) # imputed with median 


# In[ ]:


test.isnull().sum()


# creating dummies

# In[ ]:


# One hot encode the categorical columns individually
test_Region_one_hot = pd.get_dummies(test['Region'],prefix='region')
test_State_one_hot = pd.get_dummies(test['State'],prefix='state')
test_Area_one_hot = pd.get_dummies(test['Area'],prefix='area')
test_City_one_hot = pd.get_dummies(test['City'],prefix='city')
test_Conpro_one_hot = pd.get_dummies(test['Consumer_profile'],prefix='consumer_profile')
test_Procat_one_hot = pd.get_dummies(test['Product_category'],prefix='product_category')
test_Protyp_one_hot = pd.get_dummies(test['Product_type'],prefix='product_type')
test_Servc_one_hot = pd.get_dummies(test['Service_Centre'],prefix='serrvice_centre')
test_Purfrm_one_hot = pd.get_dummies(test['Purchased_from'],prefix='purchased_from')
test_Purpose_one_hot = pd.get_dummies(test['Purpose'],prefix='purpose')


# In[ ]:


# Combine the one hot encoded columns with test
x_test = pd.concat([test, 
                    test_Region_one_hot, 
                    test_State_one_hot, 
                    test_Area_one_hot, 
                    test_City_one_hot,
                    test_Conpro_one_hot, 
                    test_Procat_one_hot,
                    test_Protyp_one_hot, 
                    test_Servc_one_hot,
                    test_Purfrm_one_hot, 
                    test_Purpose_one_hot,], axis=1)


# In[ ]:


# Drop the original categorical columns (because now they've been one hot encoded)
x_test = x_test.drop(['Region', 'State', 'Area','City',
                      'Consumer_profile','Product_category',
                      'Product_type','Service_Centre',
                      'Purchased_from','Purpose'], axis=1)


# In[ ]:


x_test.head()


# In[ ]:


x_test.isnull().sum()


# # Building models

# 1) Logistoc Regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
acc_log


# 2) SVM

# In[ ]:


svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)
acc_svc


# 3) KNN

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)
acc_knn


# 4) Gaussian Naive Bayes

# In[ ]:


gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)
acc_gaussian


#  5) Perceptron

# In[ ]:


perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_test)
acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)
acc_perceptron


# 6) Linear SVC

# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_test)
acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)
acc_linear_svc


# 7) Stochastic Gradient Descent

# In[ ]:


sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_test)
acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)
acc_sgd


# 8) Decision Tree

# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree


# 9) Random Forest

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
acc_random_forest


# # Model Evalution

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# Feature Importance

# In[ ]:


def feature_importance(model, data):
    """
    Function to show which features are most important in the model.
    ::param_model:: Which model to use?
    ::param_data:: What data to use?
    """
    fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': data.columns})
    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
    _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
    return fea_imp
    #plt.savefig('catboost_feature_importance.png')


# In[ ]:


# Plot the feature importance scores
feature_importance(decision_tree, x_train)


# # Submission

# In[ ]:


# Create a list of columns to be used for the predictions
wanted_test_columns = x_train.columns
wanted_test_columns


# In[ ]:


# Make a prediction using the Decision Tree on the wanted columns
predictions = decision_tree.predict(x_test[wanted_test_columns])


# In[ ]:


# Our predictions array is comprised of 0's and 1's (Fraud or Genuine)
predictions[:20]


# In[ ]:


final = pd.read_csv("../input/warranty-claims/test_1.csv")
final.head()


# In[ ]:


# Create a submisison dataframe and append the relevant columns
submission = pd.DataFrame()
submission['Id'] = final['Unnamed: 0']
submission['Prediction1'] = predictions # our model predictions on the test dataset
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)

