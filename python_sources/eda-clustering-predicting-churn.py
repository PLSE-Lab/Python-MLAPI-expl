#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# First things first. Let's take a look at the columns to check for missing or weird values in the dataset.

# In[ ]:


telecom_data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
telecom_data.info()


# No missing values , a very very clean dataset. Not the most real world conditions but it makes our job easier. 
# 
# Let's take a look at the first 5 rows.
# 

# In[ ]:


telecom_data.head()


# I see that unlike all the other binary categorical features , senior citizen has alreayd been encoded. I'm going to reverse that for the visualization part of this notebook as It'll be taken care of during the pre-processing stage with all the other categorical columns.

# In[ ]:


telecom_data = telecom_data.replace( { 'SeniorCitizen': { 0: 'No', 1:'Yes' } } )


# I'm gonna write a function that accepts the name of a column (feature) as it's argument and groups the dataframe on that feature giving us the total amount of churned users with respect to each value in that feature and what percent of churned users does each value constitue for.

# In[ ]:


def categorical_segment(column_name:str) -> 'grouped_dataframe':
    segmented_df = telecom_data[[column_name, 'Churn']]
    segmented_churn_df = segmented_df[segmented_df['Churn'] == 'Yes']
    grouped_df = segmented_churn_df.groupby(column_name).count().reset_index().rename(columns = {'Churn':'Churned'})
    total_count_df = segmented_df.groupby(column_name).count().reset_index().rename(columns = {'Churn':'Total'})
    merged_df = pd.merge(grouped_df, total_count_df, how = 'inner', on = column_name)
    merged_df['Percent_Churned'] = merged_df[['Churned','Total']].apply(lambda x: (x[0] / x[1]) * 100, axis=1) 
    return merged_df

categorical_columns_list = list(telecom_data.columns)[1:5] + list(telecom_data.columns)[6:18]

grouped_df_list = []

for column in categorical_columns_list:
    grouped_df_list.append( categorical_segment( column ) )
    
grouped_df_list[0]


# **Churn by categorical features**
# 
# Wrting a loop that goes through each categorical feature and calls the above function to get a grouped dataframe for that feature and visualizes it using a stacked bargraph. 

# In[ ]:


import matplotlib.pyplot as plt 
for i , column in enumerate(categorical_columns_list):
    fig, ax = plt.subplots(figsize=(13,5))
    plt.bar(grouped_df_list[i][column] , [ 100 - i for i in grouped_df_list[i]['Percent_Churned'] ],width = 0.1, color = 'g')
    plt.bar(grouped_df_list[i][column],grouped_df_list[i]['Percent_Churned'], bottom =  [ 100 - i for i in grouped_df_list[i]['Percent_Churned'] ],
            width = 0.1, color = 'r')
    plt.title('Percent Churn by ' + column)
    plt.xlabel(column)
    plt.ylabel('Percent Churned')
    plt.legend( ('Retained', 'Churned') )
    plt.show()


# Some interesting observations:
# * A higher percentage of senior citizens churned over people who were not senior citizens 
# * More people with partners churned over people who did not have partners. Same for people with dependents 
# * The highest churn by internet service is accounted for by Fiber Optic connection users over people who use the good ol DSL and people not using the internet service. Possibly something to do with price, I would assume.
# 
# Feel free to take a gander at all the other plots to see what you find interesting. 

# **Churn by numerical features**
# 
# Wrting a function like above that groups by numerical features and using a similar for loop to plot one of my favorite plots namely violin plots to visualie churn by  monthly charges and tenure.

# In[ ]:


def continous_var_segment(column_name:str) -> 'segmented_df':
    segmented_df = telecom_data[[column_name, 'Churn']]
    segmented_df = segmented_df.replace( {'Churn': {'No':'Retained','Yes':'Churned'} } )
    segmented_df['Customer'] = ''
    return segmented_df

continous_columns_list = [list(telecom_data.columns)[18]] + [list(telecom_data.columns)[5]]


continous_segment_list = []

for var in continous_columns_list:
    continous_segment_list.append( continous_var_segment(var) )
    
import seaborn as sns
sns.set('talk')

for i, column in enumerate( continous_columns_list ):
    fig, ax = plt.subplots(figsize=(8,11))
    sns.violinplot(x = 'Customer', y = column, data = continous_segment_list[i], hue = 'Churn', split = True)
    plt.title('Churn by ' + column)
    plt.show()


# Some Interesting observations:
# * Not suprisingly most retained customers have monthly charges concentrated around 20 Dollars  while most churned customers had monthly charges between 80 - 100 Dollars.
# * Most of that customers that churned, churned within approximately the first 5-7 months of joining the service while a lot of the retained customers have been around for upwards of 60-65 months ( 5 years and up).

# Normalizing  tenure and monthly charges and using K-means clustering to cluster churned customers based on them.

# In[ ]:


from sklearn.cluster import KMeans 
from sklearn.preprocessing import MinMaxScaler

monthlyp_and_tenure = telecom_data[['MonthlyCharges','tenure']][telecom_data.Churn == 'Yes']

scaler = MinMaxScaler()
monthly_and_tenure_standardized = pd.DataFrame( scaler.fit_transform(monthlyp_and_tenure) )
monthly_and_tenure_standardized.columns = ['MonthlyCharges','tenure']

kmeans = KMeans(n_clusters = 3, random_state = 42).fit(monthly_and_tenure_standardized)

monthly_and_tenure_standardized['cluster'] = kmeans.labels_

fig, ax = plt.subplots(figsize=(13,8))
plt.scatter( monthly_and_tenure_standardized['MonthlyCharges'], monthly_and_tenure_standardized['tenure'],
           c = monthly_and_tenure_standardized['cluster'], cmap = 'Spectral')

plt.title('Clustering churned users by monthly Charges and tenure')
plt.xlabel('Monthly Charges')
plt.ylabel('Tenure')


plt.show()


# Seems our algorithm found one tight and 2 semi-loose clusters to group churned users by:
# * *Customers with low monthly charges and low tenure*:  Could have been a temporary connection  for them or people looking for very minimal service who found a service provider offering even lower charges for basic services and churned quickly despite low monthly charges.
# * *Customers with high monthly charges and low tenure:* The heaviest concentration of churned users. The most common churned users who were possibly unhappy with the prices and stayed for a little while before quickly leaving the service provider for better , cheaper options.
# * *Customers with high monthly charges and high tenure:* The most interesting group of churned users. They might have stayed initially despite high prices becuase they either thought the service was worth the price or simply due to lack of better alternatives and churned after a while in contrast with most other churned users who churned pretty quickly in their tenure. 
# 
# The interesting observation here is that most churned users with low monthly charges churned pretty quickly. There is a very small concentration of churned users who had low prices in the high tenure zone. This usually points to very dissatisfied customers or customers who were looking for temporary service providers at the time. 
# 

# Pre-processing the data using label encoding and one hot encoding to get it ready for machine learning algorithms.

# In[ ]:


telecom_data_filtered = telecom_data.drop(['TotalCharges','customerID'], axis = 1)

def encode_binary(column_name:str):
    global telecom_data_filtered
    telecom_data_filtered = telecom_data_filtered.replace( { column_name: { 'Yes': 1 , 'No': 0 } }  )
    

binary_feature_list = list(telecom_data_filtered.columns)[1:4] + [list(telecom_data_filtered.columns)[5]] + [list(telecom_data_filtered.columns)[15]]  + [list(telecom_data_filtered.columns)[18]]
    
for binary_feature in binary_feature_list:
    encode_binary(binary_feature)
    

telecom_data_processed = pd.get_dummies( telecom_data_filtered, drop_first = True )

telecom_data_processed.head(10)


# Not an very imbalanced dataset in terms of the positive and negative target variable.

# In[ ]:


telecom_data.Churn.value_counts()


# Importing necessary libraries and writing a function that makes it easily to assess and visualize model perfomance using sklearn metrics and matplotlib 

# In[ ]:


import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sklearn.metrics as metrics
get_ipython().run_line_magic('matplotlib', 'inline')

X = np.array( telecom_data_processed.drop( ['Churn'] , axis = 1 ) )
y = np.array( telecom_data_processed['Churn'] )

X_train,X_test,y_train,y_test = train_test_split( X, y , test_size = 0.2 , random_state = 42 )

def get_metrics( model ):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    y_actual = y_test 
    print()
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
    print()
    print('Accuracy on unseen hold out set:' , metrics.accuracy_score(y_actual,y_pred) * 100 , '%' )
    print()
    f1_score = metrics.f1_score(y_actual,y_pred)
    precision = metrics.precision_score(y_actual,y_pred)
    recall = metrics.recall_score(y_actual,y_pred)
    score_dict = { 'f1_score':[f1_score], 'precision':[precision], 'recall':[recall]}
    score_frame = pd.DataFrame(score_dict)
    print(score_frame)
    print()
    fpr, tpr, thresholds = metrics.roc_curve( y_actual, y_prob[:,1] ) 
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot( fpr, tpr, 'b-', alpha = 0.5, label = '(AUC = %.2f)' % metrics.auc(fpr,tpr) )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend( loc = 'lower right')
    plt.show()


# Model 1: Random Forest 
# * Tuning min_samples_split, min_samples_leaf, max_features and max_depth using Grid Search with 5 fold cross validation to get a good bias vs variance tradeoff from our model
# 
#     

# In[ ]:


rf = RandomForestClassifier( n_estimators = 20, n_jobs=-1, max_features = 'sqrt', random_state = 42 )
param_grid1 = {"min_samples_split": np.arange(2,11), 
              "min_samples_leaf": np.arange(1,11)}
rf_cv = GridSearchCV( rf, param_grid1, cv=5, iid = False )
rf_cv.fit(X_train,y_train)
print( rf_cv.best_params_ )
print( rf_cv.best_score_ )


# In[ ]:


rf = RandomForestClassifier( n_estimators = 20, n_jobs = -1, min_samples_split = 2, min_samples_leaf = 8, random_state = 2 )
param_grid2 = {'max_depth': np.arange(9,21),
              'max_features':['sqrt','log2']}
rf_cv = GridSearchCV(rf, param_grid2, cv=5, iid = False)
rf_cv.fit(X_train,y_train)
print( rf_cv.best_params_ )
print( rf_cv.best_score_ )


# **Note:** 
# * If you think about it from the perspective of a big telecom company , it would be more important to catch most of the postive cases, that is the users who might churn and provide incentives to them to make them stay than it is to catch the customers who will not be churning for now. This means there is a higher cost associated with misclassifying a churned user as retained than there is for missclassifying a retained user as churned. As a result for all models from here on I will focus on improving the recall to maximize the amount of positive cases our model catches. For this I would specify a higher class weight for the positive class '1' and lower very slightly the weight associated with the negative class '0'. This would most likely lower the overall accuracy of the model but as any good data scientist knows, accuracy is not everything. 
# * I will plot the ROC curve and specify the AUC for each model to see how well they were able to separate the two classes.

# In[ ]:


rf = RandomForestClassifier( n_estimators = 1000, max_features = 'log2', max_depth = 11, min_samples_split = 2, 
                          min_samples_leaf = 8, n_jobs = -1 , random_state = 42, class_weight = {0:0.95, 1:2})
rf.fit(X_train,y_train)
print('Training Accuracy:',rf.score(X_train,y_train)*100,'%')
get_metrics(rf)


# In the unseen/validation set our random forest model was able to capture 75.6% of all positive cases and has an AUC of 0.87

# Model 2: Logistic Regression 

# In[ ]:


model_pipeline = Pipeline( steps = [( 'normalizer', MinMaxScaler() ), 
                                   ( 'log_reg', LogisticRegression( penalty = 'l2', random_state = 42 ) ) ] )
param_dict = dict( log_reg__C = [0.001, 0.01, 0.1, 1, 10, 100])
estimator = GridSearchCV( model_pipeline, param_grid = param_dict, cv = 5, n_jobs = -1, iid = False )
estimator.fit(X_train,y_train)
print(estimator.best_params_)
print(estimator.best_score_)


# In[ ]:


model_pipeline = Pipeline( steps = [( 'normalizer', MinMaxScaler() ), 
                                   ( 'log_reg', LogisticRegression( penalty = 'l2', C = 100, random_state = 42, class_weight = {0:.95 , 1:2} ) ) ] )
model_pipeline.fit(X_train,y_train)
print('Training Accuracy:',model_pipeline.score(X_train,y_train)*100,'%')
get_metrics(model_pipeline)


# Our logistic regression preformed slighly better than our ensemble method by catching 76.6% of all postive cases in our unseen/validation set but had a very slighly lower AUC of 0.86

# Model 3: SVC 

# In[ ]:


svc_pipeline = Pipeline( steps = [( 'normalizer', MinMaxScaler() ), 
                                   ( 'svc', SVC(random_state = 42, probability = True) ) ] )
params = [0.001, 0.01, 0.1, 1, 10]
param_dict = dict( svc__C = params, svc__gamma = params)
estimator = GridSearchCV( svc_pipeline, param_grid = param_dict, cv = 5, n_jobs = -1, iid = False )
estimator.fit(X_train,y_train)
print(estimator.best_params_)
print(estimator.best_score_)


# In[ ]:


svc = SVC(C = 1, gamma = 0.01, class_weight = {0:1, 1:2}, random_state = 42, probability = True)
svc.fit(X_train,y_train) 
print('Training Accuracy:',svc.score(X_train,y_train)*100,'%')
get_metrics(svc)


# Our SVC performed worse than the logistic regression and random forest model by catching 71.8% of the positive cases in the unseen/validation set and had an AUC of 0.84. 

# Since our logistic regression and random forest model perfomed more or less the same, It's a good idea to go with the simpler and more intepretable model that is the logistic regression for the choice of model. 
# 
# Now let's take a look under the hood of our logistic regression model to see what it has learned by plotting the various coefficients it assigned to all the features. Taking a look at what our model has learned is also a very good way to assess a model before you deploy it for any kind of production.

# In[ ]:


coeff = model_pipeline.named_steps['log_reg'].coef_.flatten()
coeff_updated = np.append( coeff[:8], [sum(coeff[8:10]), sum(coeff[10:12]), sum(coeff[12:14]),sum(coeff[14:16]), 
                         sum(coeff[16:18]), sum(coeff[18:20]), sum(coeff[20:22]), sum(coeff[22:24]), sum(coeff[24:26]), sum(coeff[26:]) ] )
columns = ['SeniorCitizen', 'Partner', 'Dependents', 'Tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'Gender', 'MultipleLines',
          'InternetService', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies', 'Contract', 'PaymentMethod']
fig, ax = plt.subplots(figsize=(50,20))
plt.plot(columns, coeff_updated, c = 'yellow', marker='o', linewidth = 6, linestyle='dashed', markersize=20, mfc = 'red')
plt.title('Coefficients Learned by the Logistic Regression Model')
plt.ylabel('Coeff')
plt.xlabel('Features')
plt.show()


# Our model assigned very high coefficients to Monthly Charges, Tenure and Internet service which was something I was expecting and surprisingly a high coefficient for contract as well, while all the other coeffcients were pushed close to 0 by our L2 regularization parameter. Although I used C=100 which is considered a pretty large value which means less regularization we can clearly see what our logistic regression model learned.

# Thank you very much if you took a look at my Kernel. I love feedback and I thrive on criticism. If I did something wrong or stupid or if you simply have nay suggestions for stuff I can improve on, no one would be more happy than me to hear about it. 
# Cheers :) 

# In[ ]:




