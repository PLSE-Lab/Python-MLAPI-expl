#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This Playground competition will give you the opportunity to try different encoding schemes for different algorithms to compare how they perform.
# Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

# This is a fork of my original code "Categorical Feature Encoding Challenge". Here i am trying to do more better by re-visting the data.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Here we are just importing which are important for starting with.. and will add-on once I need more when reaching towards Modeling and Prdiction.


# In[ ]:


# Get File Path
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_data = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')
sample_submission = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')
test_data = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')


# In[ ]:


train_data.head()


# In[ ]:


print(train_data.shape, test_data.shape, sample_submission.shape)


# In[ ]:


test_data['id'].head()


# In[ ]:


test_data['id'].tail()


# In[ ]:


sample_submission['id'].head()


# In[ ]:


train_data.info()


# In[ ]:


train_data.describe()


# In[ ]:


train_data.describe(include='all')


# In[ ]:


train_data.describe(include=[np.object])


# Observation from above : 
# There are multiple categorical variables which are as follows
# * bin_3, bin_4  :- binary cols (T & F and Y & N respectively)
# * nom_0 - nom_4 :- nominal columns ( with no order)
# * nom_5 - nom_9 :- nominal columns with high cardinality
# * ord_1 - ord_5 :- Ordered columns
# 
# We have to use different ways to treat these columns and convert them into numerical data

# In[ ]:


# write a function to get the distinct value in each categorical value
def get_Unique_Values(list_cat_var) :
    cat_dict = dict()
    for i in list_cat_var:
        cat_dict[i] = list(train_data[i].unique())
    return cat_dict


# In[ ]:


print(get_Unique_Values(['bin_3', 'bin_4'])) 
print('-'*80)
print(get_Unique_Values(['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'])) 
print('-'*80)
#print(get_Unique_Values(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'])) 
print('-'*80)
print(get_Unique_Values(['ord_1', 'ord_2'])) 
print('-'*80)
print(get_Unique_Values(['ord_3', 'ord_4'])) 
print('-'*80)
print(get_Unique_Values(['ord_5'])) 
print('-'*80)
print(get_Unique_Values(['day', 'month'])) 


# # Encoding techniques
# * bin_3, bin_4  :- Convert T/F & Y/N to 1/0
# * nom_0 - nom_4 :- Encode using One hot encoding
# * nom_5 - nom_9 :- Target encode them as they are high cardinal variables
# * ord_1, ord_2  :- Convert into numerical order using hard coded values as Label encoder might not be able to understand the order
# * ord_3, ord_4  :- Encode using ascii as they are alphabetical values
# * ord_5         :- Separate two alphabets and then do encoding using ascii.
# 
# Taken reference from https://www.kaggle.com/ruchibahl18/categorical-data-encoding-techniques#Encoding-techniques 

# In[ ]:


# Binary encoding
train_data['bin_3'] = [0 if x == 'F' else 1 for x in train_data['bin_3']]
train_data['bin_4'] = [0 if x == 'N' else 1 for x in train_data['bin_4']]

test_data['bin_3'] = [0 if x == 'F' else 1 for x in test_data['bin_3']]
test_data['bin_4'] = [0 if x == 'N' else 1 for x in test_data['bin_4']]

print(get_Unique_Values(['bin_3', 'bin_4'])) 


# In[ ]:


#Hard coded Label encoding
print(get_Unique_Values(['ord_1', 'ord_2'])) 

train_data['ord_1'] = [0 if x == 'Novice' else 1 if x == 'Contributor' else 2 if x == 'Expert' else 3 if x == 'Master' else 4 for x in train_data['ord_1']]
train_data['ord_2'] = [0 if x == 'Freezing' else 1 if x == 'Cold' else 2 if x == 'Warm' else 3 if x == 'Hot' else 4 if x == 'Boiling Hot' else 5 for x in train_data['ord_2']]

test_data['ord_1'] = [0 if x == 'Novice' else 1 if x == 'Contributor' else 2 if x == 'Expert' else 3 if x == 'Master' else 4 for x in test_data['ord_1']]
test_data['ord_2'] = [0 if x == 'Freezing' else 1 if x == 'Cold' else 2 if x == 'Warm' else 3 if x == 'Hot' else 4 if x == 'Boiling Hot' else 5 for x in test_data['ord_2']]

print('------- After Hard coded Label Encoding -------')
print(get_Unique_Values(['ord_1', 'ord_2'])) 


# In[ ]:


# for ord_5, as mentioned will separate the value into two new features, and then will drop ord_5.
train_data["ord_5a"]=train_data["ord_5"].str[0]
train_data["ord_5b"]=train_data["ord_5"].str[1]
train_data.drop(['ord_5'], axis=1, inplace = True)

test_data["ord_5a"]=test_data["ord_5"].str[0]
test_data["ord_5b"]=test_data["ord_5"].str[1]
test_data.drop(['ord_5'], axis=1, inplace = True)

print(get_Unique_Values(['ord_5a', 'ord_5b'])) 


# In[ ]:


import string
print('ASCII for a : ', string.ascii_letters.index('a'))
print('ASCII for A : ', string.ascii_letters.index('A'))
print('ASCII for E : ', string.ascii_letters.index('E'))
print('ASCII for j : ', string.ascii_letters.index('j'))


# In[ ]:


# ascii encoding for ord_3, ord_4, ord_5a, ord_5b
train_data['ord_3'] = train_data['ord_3'].apply(lambda x: string.ascii_letters.index(x))
train_data['ord_4'] = train_data['ord_4'].apply(lambda x: string.ascii_letters.index(x))
train_data['ord_5a'] = train_data['ord_5a'].apply(lambda x: string.ascii_letters.index(x))
train_data['ord_5b'] = train_data['ord_5b'].apply(lambda x: string.ascii_letters.index(x))

test_data['ord_3'] = test_data['ord_3'].apply(lambda x: string.ascii_letters.index(x))
test_data['ord_4'] = test_data['ord_4'].apply(lambda x: string.ascii_letters.index(x))
test_data['ord_5a'] = test_data['ord_5a'].apply(lambda x: string.ascii_letters.index(x))
test_data['ord_5b'] = test_data['ord_5b'].apply(lambda x: string.ascii_letters.index(x))

print(get_Unique_Values(['ord_3', 'ord_4'])) 
print('-'*80)
print(get_Unique_Values(['ord_5a', 'ord_5b'])) 


# In[ ]:


train_data.head()


# In[ ]:


# Lets transform the Categorical Features ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'] into Number using get_dummies function (One Hot Encoding)
print(train_data.shape)
print('-'*20)
train_data = pd.get_dummies(train_data, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], drop_first=True)

test_data = pd.get_dummies(test_data, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], drop_first=True)


# In[ ]:


print(train_data.shape)
#train_data.describe(include = 'object')
train_data.columns


# In[ ]:


#Leave one out encoding high cardinal variables
high_cardinal_vars = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
high_cardinal_vars


# Taken Reference from Quora:
# 
# One would be to cluster them based on the response; you can sort them by response, then split them however you like; perhaps let a fairly shallow decision tree handle it. Now you have far fewer categories.
# 
# Another is to target encode them. Replace each category in a variable with the mean response given that category. Now you have 1 continuous feature instead of a bunch of categories.
# 
# Another is to group them by frequency. The most frequent categories may dominate, and the least frequent may be numerous but each have few samples. You can e.g. leave the top 5 categories alone and group the rest into a new category. Now you have 6 categorical variables.

# In[ ]:


train_data['nom_5'].describe()


# In[ ]:


#from category_encoders import TargetEncoder, HashingEncoder, LeaveOneOutEncoder
#trgt_encoder = TargetEncoder(cols=high_cardinal_vars, smoothing=0, return_df=True)
#hashing_encoder = HashingEncoder(cols = high_cardinal_vars)
# loo_encoder = LeaveOneOutEncoder(cols=high_cardinal_vars)
#train_data = loo_encoder.fit_transform(train_data.drop(['target'], axis = 1), train_data['target'])


# In[ ]:


# taken reference from https://maxhalford.github.io/blog/target-encoding-done-the-right-way/ 
def calc_smooth_mean(df, by, on, m):# df => Data Frame; by => column on which encoding is required; on => target field; m => weight
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return df[by].map(smooth)


# In[ ]:


train_data['nom_5'] = calc_smooth_mean(train_data, by='nom_5', on='target', m=8)
train_data['nom_6'] = calc_smooth_mean(train_data, by='nom_6', on='target', m=8)
train_data['nom_7'] = calc_smooth_mean(train_data, by='nom_7', on='target', m=8)
train_data['nom_8'] = calc_smooth_mean(train_data, by='nom_8', on='target', m=8)
train_data['nom_9'] = calc_smooth_mean(train_data, by='nom_9', on='target', m=8)
 
test_data['nom_5'] = calc_smooth_mean(train_data, by='nom_5', on='target', m=8)
test_data['nom_6'] = calc_smooth_mean(train_data, by='nom_6', on='target', m=8)
test_data['nom_7'] = calc_smooth_mean(train_data, by='nom_7', on='target', m=8)
test_data['nom_8'] = calc_smooth_mean(train_data, by='nom_8', on='target', m=8)
test_data['nom_9'] = calc_smooth_mean(train_data, by='nom_9', on='target', m=8)    


# In[ ]:


train_data.columns


# In[ ]:


print(get_Unique_Values(['nom_5'])) 


# In[ ]:


#print( train_data['nom_5'].value_counts().keys().tolist() )
#print( train_data['nom_5'].value_counts().tolist() )

#print( train_data['nom_5'].value_counts().to_frame() )
# nm1 = train_data['nom_5'].value_counts().to_frame() 
# # print(nm1.dtypes)
# print(nm1['nom_5'].head())
# print('TOTAL ', nm1.count())
# # print('MORE THAN 2000 : ', nm1.loc[nm1['nom_5'] > 2000].count())
# # print('MORE THAN 1000 : ', nm1[(nm1['nom_5'] > 1000) ].count()  )
# # print('BETWEEN 1000 and 2000 : ', nm1[(nm1['nom_5'] > 1000) & (nm1['nom_5'] < 2000)].count())
# # print('LESS THAN 1000 : ', nm1[(nm1['nom_5'] < 1000) ].count())
# print(nm1[(nm1['nom_5'] < 1000) ].index)


# In[ ]:


# df = pd.DataFrame({'FIELD_1': ['f710fca39', '1fd0233cd', '005dd4ce3', '5331f98fb', '005dd4ce3', 'f710fca39', 'eb0004a0b'], 
#                          'B': [400        , 500        , 600        , 700        , 800        , 900        , 111]})
# new_df = pd.DataFrame({'CNT': [225, 150, 80, 230],'ID': ['f710fca39', '1fd0233cd', '5331f98fb', '005dd4ce3']})
# new_df.set_index('ID', inplace=  True)
# print(df)
# print(new_df)
# print('-'*50)
# # Ask is : Update the column "FIELD_1" as 2 if the CNT is more than 200 in new_df; 1 when the CNT is >100 and <200; 0 when CNT <100.
# print(df.loc[df['FIELD_1'].isin(new_df.index & new_df['CNT'] > 200 ),'FIELD_1']  )

# print(df)


# In[ ]:


# train_nom5 = train_data['nom_5'].value_counts().to_frame() 
# test_nom5 = test_data['nom_5'].value_counts().to_frame() 


# In[ ]:


# #train_data['nom_5'] = [0 if x in train_nom5[(train_nom5['nom_5'] > 2000)].index else 1 if x in train_nom5[(train_nom5['nom_5'] > 1000) & (train_nom5['nom_5'] < 2000)].index else 2 for x in train_data['nom_5']]
# train_data['nom_5'] = 
# train_data['nom_5'].head(10)


# In[ ]:


# Lets get the % of each null values.
#total = train_data.isnull().sum().sort_values(ascending=False)
#percent_1 = train_data.isnull().sum()/train_data.isnull().count()*100
#percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
#missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
#missing_data.head(5)
# Cool.. No NaN Values in train_data


# In[ ]:


# Lets get the % of each null values.
#total = test_data.isnull().sum().sort_values(ascending=False)
#percent_1 = test_data.isnull().sum()/test_data.isnull().count()*100
#percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
#missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
#missing_data.head(5)
# Cool.. No NaN Values in test_data


# # Correlation Heatmap

# In[ ]:


#Using Pearson Correlation

plt.figure(figsize=(20,10))
cor = train_data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


#Correlation with output variable
cor_target = abs(cor["target"])

#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features

# Seems none of the numeric feature have much correlation with our target variable.
# Correlation coefficients whose magnitude are between 0.5 and 0.7 indicate variables which can be considered moderately correlated. Correlation coefficients whose magnitude are between 0.3 and 0.5 indicate variables which have a low correlation.


# In[ ]:


## Taken reference from https://www.kaggle.com/alexisbcook/categorical-variables
# #Get list of categorical variables
# s = (train_data.dtypes == 'object')
# train_data_cat_var = list(s[s].index)

# s = (test_data.dtypes == 'object')
# test_data_cat_var = list(s[s].index)

# print("Categorical variables from train_data:", train_data_cat_var)
# print("-"*30)
# print("Categorical variables from test_data:", test_data_cat_var)


# From my previous challenge... i experienced that the test and train categorical variables may have different set of values.. so better to check at first. If found we can merge the data-set and then transform them.

# In[ ]:


# list(train_data['bin_3'].unique() )
#train_data['bin_3'].value_counts() 
#train_data['bin_3'].unique().sum()
#train_data.groupby('bin_3').size()
#len(train_data['bin_3'].unique())


# In[ ]:


# write a function to get the count of distinct value in each categorical value
# def get_Unique_Count(list_cat_var) :
#     cat_dict = dict()
#     for i in list_cat_var:
#         cat_dict[i] = len(train_data[i].unique())
#     return cat_dict


# In[ ]:


# print(get_Unique_Count(list(train_data_cat_var))) 
# print(get_Unique_Count(list(test_data_cat_var))) 


# From above we see that there are some categorical variables which has more than 10 unique value such as nom_5; nom_6; nom_7; nom_8; nom_9; ord_3; ord_4; ord_5. So we will not be using these to transform.
# Will just transoform remaining as we have limited counts, also it is recommended to transform any categorical variable if the max unique is less than 15, but here we will stick max to 10.

# In[ ]:


# Dropping off un-used features.
# train_data.drop(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_3', 'ord_4', 'ord_5'], axis = 1, inplace = True)
# test_data.drop(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_3', 'ord_4', 'ord_5'], axis = 1, inplace = True)


# # Handling Categorical Features

# In[ ]:


# removing un-used features from our categorical features.
# print(len(train_data_cat_var))
# train_data_cat_var = [ele for ele in train_data_cat_var if ele not in  ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_3', 'ord_4', 'ord_5']]
# print(len(train_data_cat_var))


# In[ ]:


# print(len(test_data_cat_var))
# test_data_cat_var = [ele for ele in test_data_cat_var if ele not in  ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_3', 'ord_4', 'ord_5']]
# print(len(test_data_cat_var))


# In[ ]:


# Lets transform the Categorical Features into Number using get_dummies function (One Hot Encoding)
# final_train_data = pd.get_dummies(train_data, columns=train_data_cat_var, drop_first=True)
# print(final_train_data.shape, train_data.shape)
# final_train_data.head()


# In[ ]:


# final_test_data = pd.get_dummies(test_data, columns=test_data_cat_var, drop_first=True)
# print(final_test_data.shape, test_data.shape)
# final_test_data.head()


# # Modeling

# In[ ]:


# Defining Feature and Target.
features = train_data.drop(['target'], axis = 1).columns
target = train_data["target"]
print("Features", features)
print('--'*10)
print ("Target", target.head())


# In[ ]:


test_data.columns


# In[ ]:


# split the train_data into 2 DF's aka X_train, X_test, y_train, y_test.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data[features], target, test_size=0.2)

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[ ]:


# test_data 
X_test_df  = test_data[features].copy()
X_test_df.head()


# In[ ]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# ROC and AUR Curve related importing the libraries
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, classification_report


# In[ ]:


# # Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred_lr = logreg.predict(X_test)
#print(Y_pred_lr)


# In[ ]:


logreg_score = round(logreg.score(X_train, y_train) * 100, 2)
print("Score (LogisticRegression)", logreg_score)


# In[ ]:


logreg_accuracy_score = round(accuracy_score(y_test, Y_pred_lr) * 100, 2)
print("Accuracy Score (LogisticRegression)", logreg_accuracy_score)


# In[ ]:


logreg_confusion_matrix = confusion_matrix(y_test, Y_pred_lr)
logreg_confusion_matrix


# In[ ]:


logreg_roc_auc = roc_auc_score(y_test, Y_pred_lr)
logreg_roc_auc


# In[ ]:


# # Getting False Positive Rate (fpr); True Positive Rate (tpr) and threshold.
fpr_logreg, tpr_logreg, threshold_logreg = roc_curve(y_test,logreg.predict_proba(X_test)[:,1])
print('False Positive Rate : ', fpr_logreg)
print('True Positive Rate : ', tpr_logreg)
print('Threshold : ', threshold_logreg)


# In[ ]:


# Plotting the ROC Curve
plt.figure()
plt.plot(fpr_logreg, tpr_logreg, label = 'Logistic Regression Model (aread = %0.2f)' %logreg_roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.show()


# In[ ]:


# Support Vector Machines

# svc = SVC(gamma='auto')
# svc.fit(X_train, y_train)
# Y_pred_svc = svc.predict(X_test)


# In[ ]:


#svc_roc_auc = roc_auc_score(y_test, Y_pred_svc)
#print('ROC AUR Score for SVC Model : ', svc_roc_auc)

# Getting False Positive Rate (fpr); True Positive Rate (tpr) and threshold.
#fpr_svc, tpr_svc, threshold_svc = roc_curve(y_test,svc.predict_proba(X_test)[:,1])
#print('False Positive Rate : ', fpr_svc)
#print('True Positive Rate : ', tpr_svc)
#print('Threshold : ', threshold_svc)


# In[ ]:


# # Plotting the ROC Curve for Logistic Regression and SVC Model

# plt.figure()
# plt.plot(fpr_logreg, tpr_logreg, label = 'Logistic Regression Model (aread = %0.2f)' %logreg_roc_auc)
# plt.plot(fpr_svc, tpr_svc, label = 'SVC Model (aread = %0.2f)' %svc_roc_auc)
# plt.plot([0,1], [0,1], 'r--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc = 'lower right')
# plt.show()


# In[ ]:


# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred_knn = knn.predict(X_test)


# In[ ]:


knn_roc_auc = roc_auc_score(y_test, Y_pred_knn)
print('ROC AUR Score for KNN Model : ', knn_roc_auc)

# Getting False Positive Rate (fpr); True Positive Rate (tpr) and threshold.
fpr_knn, tpr_knn, threshold_knn = roc_curve(y_test,knn.predict_proba(X_test)[:,1])
print('False Positive Rate : ', fpr_knn)
print('True Positive Rate : ', tpr_knn)
print('Threshold : ', threshold_knn)


# In[ ]:


# Plotting the ROC Curve for Logistic Regression ; SVC ; KNN Model
plt.figure()
plt.plot(fpr_logreg, tpr_logreg, label = 'Logistic Regression Model (aread = %0.2f)' %logreg_roc_auc)
#plt.plot(fpr_svc, tpr_svc, label = 'SVC Model (aread = %0.2f)' %svc_roc_auc)
plt.plot(fpr_knn, tpr_knn, label = 'KNN Model (aread = %0.2f)' %knn_roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.show()


# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred_gnb = gaussian.predict(X_test)


# In[ ]:


gnb_roc_auc = roc_auc_score(y_test, Y_pred_gnb)
print('ROC AUR Score for Gaussian Naive Bayes Model : ', gnb_roc_auc)

# Getting False Positive Rate (fpr); True Positive Rate (tpr) and threshold.
fpr_gnb, tpr_gnb, threshold_gnb = roc_curve(y_test,gaussian.predict_proba(X_test)[:,1])
print('False Positive Rate : ', fpr_gnb)
print('True Positive Rate : ', tpr_gnb)
print('Threshold : ', threshold_gnb)


# In[ ]:


# Plotting the ROC Curve for Logistic Regression ; SVC ; KNN; Gaussian Naive Bayes Model
plt.figure()
plt.plot(fpr_logreg, tpr_logreg, label = 'Logistic Regression Model (aread = %0.2f)' %logreg_roc_auc)
#plt.plot(fpr_svc, tpr_svc, label = 'SVC Model (aread = %0.2f)' %svc_roc_auc)
plt.plot(fpr_knn, tpr_knn, label = 'KNN Model (aread = %0.2f)' %knn_roc_auc)
plt.plot(fpr_gnb, tpr_gnb, label = 'Gaussian Naive Bayes Model (aread = %0.2f)' %gnb_roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.show()


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(X_train, y_train)
Y_pred_rf = random_forest.predict(X_test)


# In[ ]:


rf_roc_auc = roc_auc_score(y_test, Y_pred_rf)
print('ROC AUR Score for Gaussian Naive Bayes Model : ', rf_roc_auc)

# Getting False Positive Rate (fpr); True Positive Rate (tpr) and threshold.
fpr_rf, tpr_rf, threshold_rf = roc_curve(y_test,random_forest.predict_proba(X_test)[:,1])
print('False Positive Rate : ', fpr_rf)
print('True Positive Rate : ', tpr_rf)
print('Threshold : ', threshold_rf)


# In[ ]:


# Plotting the ROC Curve for Logistic Regression ; SVC ; KNN; Gaussian Naive Bayes Model
plt.figure(figsize = (10, 10))
plt.plot(fpr_logreg, tpr_logreg, label = 'Log Reg Model (aread = %0.2f)' %logreg_roc_auc)
#plt.plot(fpr_svc, tpr_svc, label = 'SVC Model (aread = %0.2f)' %svc_roc_auc)
plt.plot(fpr_knn, tpr_knn, label = 'KNN Model (aread = %0.2f)' %knn_roc_auc)
plt.plot(fpr_gnb, tpr_gnb, label = 'G N Bayes Model (aread = %0.2f)' %gnb_roc_auc)
plt.plot(fpr_rf, tpr_rf, label = 'R F Model (aread = %0.2f)' %rf_roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.show()


# # Identifying best Model from above

# In[ ]:


modelling_score = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'ROC AUR Score': [0, knn_roc_auc, logreg_roc_auc, 
              rf_roc_auc, gnb_roc_auc, 0, 
              0, 0, 0]})


# In[ ]:


modelling_score.sort_values(by='ROC AUR Score', ascending=False)


# # Submission

# In[ ]:


# Predicting on actual test_data
Y_pred_test_df = random_forest.predict(X_test_df)
Y_pred_test_df 


# In[ ]:


X_test_df.head()


# In[ ]:


submission = pd.DataFrame( { 'id': X_test_df.id , 'target': Y_pred_test_df } )


# In[ ]:


print("Submission File Shape ",submission.shape)
submission.head()


# In[ ]:


submission.to_csv( '/kaggle/working/submission1.csv' , index = False )


# In[ ]:




