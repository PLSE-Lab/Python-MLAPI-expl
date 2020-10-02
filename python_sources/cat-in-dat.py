#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# # Is there a cat in your dat?
# 
# A common task in machine learning pipelines is encoding categorical variables for a given algorithm in a format that allows as much useful signal as possible to be captured.
# 
# Because this is such a common task and important skill to master, we've put together a dataset that contains only categorical features, and includes:
# 
# binary features
# low- and high-cardinality nominal features
# low- and high-cardinality ordinal features
# (potentially) cyclical features

# # Reading Data

# In[ ]:


# Read the data
train_d = pd.read_csv('../input/cat-in-the-dat/train.csv') 
test_d = pd.read_csv('../input/cat-in-the-dat/test.csv')


# # Exploring and understanding our Data

# In[ ]:


train_d.shape 


# In[ ]:


test_d.shape


# In[ ]:


train_d.head()


# In[ ]:


train_d.tail()


# In[ ]:


test_d.columns


# In[ ]:


train_d.columns


# In[ ]:


train_d.info()


# In[ ]:


train_d.describe()


# #  Is there any missing values?

# In[ ]:


train_d.isnull().sum()


# great, our data don't have any missing values

# # The number of unique values??

# In[ ]:


for col in train_d.columns[1:]:
    print(col, train_d[col].nunique())


# # Visualization and preprocessing of Binary Features :

# In[ ]:


binary_col = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

for n, col in enumerate(train_d[binary_col]): 
    plt.figure(n)
    sns.countplot(x=col, data=train_d, hue='target', palette='husl')


# In[ ]:


train_d['target'].value_counts()


# In[ ]:


sns.swarmplot(x=train_d.head(250)['bin_0'], y=train_d.head(250)['target'])


# In[ ]:


sns.swarmplot(x=train_d.head(250)['target'], y=train_d.head(250)['month'])


# In[ ]:


sns.swarmplot(x=train_d.head(250)['target'], y=train_d.head(250)['day'])


# In[ ]:


sns.swarmplot(x=train_d.head(250)['day'], y=train_d.head(250)['target'])


# In[ ]:


# Histogram 
sns.distplot(a=train_d['target'], kde=False)


# In[ ]:


sns.kdeplot(data=train_d['target'], shade=True)


# In[ ]:


bin_d = train_d[['bin_0','bin_1','bin_2','bin_3', 'bin_4']]


# In[ ]:


bin_d.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
train_df=pd.DataFrame()
label=LabelEncoder()
for col in  train_d.columns:
    if 'bin' in col:
        train_df[col]=label.fit_transform(train_d[col])
    else:
        train_df[col]=train_d[col]
        


# In[ ]:


train_df.head(3)


# In[ ]:


test_d.head(4)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
test_df=pd.DataFrame()
label=LabelEncoder()
for col in  test_d.columns:
    if 'bin' in col:
        test_df[col]=label.fit_transform(test_d[col])
    else:
        test_df[col]=test_d[col]
    


# In[ ]:


test_df.head(4) 


# In[ ]:


binary_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

for n, col in enumerate(train_df[binary_cols]): 
    plt.figure(n)
    sns.countplot(x=col, data=train_df, hue='target', palette='husl')


# In[ ]:


train_df.shape  ,  test_df.shape


# # Visualization and preprocessing of nominal Features :

# In[ ]:


nominal_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']


# In[ ]:


for n, col in enumerate(train_df[nominal_cols]): 
    plt.figure(n)
    sns.countplot(x=col, data=train_df, hue='target', palette='husl')


# # separating nominal Features

# As we see, we have a different range of nominal features and we know that is not good way to make OneHotencoder for variables taking more than 15 different values. so, we will separate them

# In[ ]:


low_cardinality_nom_cols = []
high_cardinality_nom_cols = []


for nom_col in range(10):
    nom_col_name = "nom_"+str(nom_col)
    if train_df[nom_col_name].nunique() < 10:
        low_cardinality_nom_cols.append(nom_col_name)
    else:
        high_cardinality_nom_cols.append(nom_col_name)

print("Nominal columns low cardinality (<=10):", low_cardinality_nom_cols)
print("Nominal columns with high cardinality (>10):", high_cardinality_nom_cols)


# In[ ]:


col_nom = train_df.columns[6:11]


# In[ ]:


col_nom


# # For (low) nominal features : using OneHotEencoder to encoding variables

# In[ ]:


from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_df[low_cardinality_nom_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(test_df[low_cardinality_nom_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = train_df.index
OH_cols_test.index = test_df.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = train_df.drop(low_cardinality_nom_cols, axis=1)
num_X_valid = test_df.drop(low_cardinality_nom_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_tset = pd.concat([num_X_valid, OH_cols_test], axis=1)


# In[ ]:


OH_X_train.head()


# In[ ]:


OH_X_tset.head()


# # Visualization and preprocessing of ordinal Features :

# In[ ]:


ord_col = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

for n, col in enumerate(train_df[ord_col]): 
    plt.figure(n)
    sns.countplot(x=col, data=train_df, hue='target', palette='husl')


# In[ ]:


from sklearn.preprocessing import LabelEncoder
train_n=pd.DataFrame()
label=LabelEncoder()
for col in  ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']:
   
        train_n[col]=label.fit_transform(OH_X_train[col])
    


# In[ ]:



data_t = OH_X_train.drop(['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5'], axis=1) 
train_dd = pd.concat([data_t,train_n], axis = 1)


# In[ ]:


train_dd.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
test_n=pd.DataFrame()
label=LabelEncoder()
for col in  ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']:
   
        test_n[col]=label.fit_transform(OH_X_tset[col])
    
    
data_t = OH_X_tset.drop(['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5'], axis=1) 
test_dd = pd.concat([data_t,test_n], axis = 1)


# In[ ]:


test_dd.head(4)


# # back to high nominal : 

# Trying some different ways to preprocessing high nominal like: hash, frequent, label encoder

# # Hash

# In[ ]:


for col in high_cardinality_nom_cols:
    train_dd[f'hash_{col}'] = train_dd[col].apply( lambda x: hash(str(x)) % 5000 )
    test_dd[f'hash_{col}'] = test_dd[col].apply( lambda x: hash(str(x)) % 5000 )


# # Frequent 

# In[ ]:


for col in high_cardinality_nom_cols:
    enc_nom_1 = (train_dd.groupby(col).size()) / len(train_dd)
    train_dd[f'freq_{col}'] = train_dd[col].apply(lambda x : enc_nom_1[x])
    #test_dd[f'enc_{col}'] = test_dd[col].apply(lambda x : enc_nom_1[x])


# # Label Encoder

# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Label Encoding
for f in ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']:
    if train_dd[f].dtype=='object' or test_dd[f].dtype=='object': 
        lbl = LabelEncoder()
        lbl.fit(list(train_dd[f].values) + list(test_dd[f].values))
        train_dd[f'le_{f}'] = lbl.transform(list(train_dd[f].values))
        test_dd[f'le_{f}'] = lbl.transform(list(test_dd[f].values))


# In[ ]:


new_feat = ['hash_nom_5', 'hash_nom_6', 'hash_nom_7', 'hash_nom_8',
            'hash_nom_9',  'freq_nom_5', 'freq_nom_6', 'freq_nom_7', 
            'freq_nom_8', 'freq_nom_9', 'le_nom_5', 'le_nom_6',
            'le_nom_7', 'le_nom_8', 'le_nom_9']

new_da = (train_dd[high_cardinality_nom_cols + new_feat])


# In[ ]:


new_da.describe()


# In[ ]:


train_dd[['nom_5', 'hash_nom_5', 'freq_nom_5', 'le_nom_5']].head()


# In[ ]:


train_dd.head(4)


# In[ ]:


test_dd.head(4)


# In[ ]:


train_dd.head(4)


# # choosing just one type and Dropping other:

# In[ ]:


train_dd.drop([ 
                #'hash_nom_6', 'hash_nom_7', 'hash_nom_8', 'hash_nom_9',
               'le_nom_5', 'le_nom_6', 'le_nom_7', 'le_nom_8', 'le_nom_9',
                'freq_nom_5','freq_nom_6', 'freq_nom_7', 'freq_nom_8', 'freq_nom_9',
              'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'
         ], axis=1, inplace=True)

#test_dd.drop([
              #'hash_nom_6', 'hash_nom_7', 'hash_nom_8', 'hash_nom_9', 
 #             'le_nom_5', 'le_nom_6', 'le_nom_7', 'le_nom_8', 'le_nom_9',
  #            'freq_nom_5', 'freq_nom_6', 'freq_nom_7', 'freq_nom_8', 'freq_nom_9',
   #           'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',
    #          ], axis=1, inplace=True)


# In[ ]:


train_dd.head(4)


# In[ ]:


test_dd.drop([
              #'hash_nom_6', 'hash_nom_7', 'hash_nom_8', 'hash_nom_9', 
            'le_nom_5', 'le_nom_6', 'le_nom_7', 'le_nom_8', 'le_nom_9',
  #          'freq_nom_5', 'freq_nom_6', 'freq_nom_7', 'freq_nom_8', 'freq_nom_9',
            'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'
            ], axis=1, inplace=True)


# In[ ]:


test_dd.head(4)


# NOW OUR DATA IS NUMERIC ^_^

# # Let's make some visualizations:

# In[ ]:


date_cols = ['day', 'month']

for n, col in enumerate(train_df[date_cols]): 
    plt.figure(n)
    sns.countplot(x=col, data=train_df, hue='target', palette='husl')


# In[ ]:


sns.scatterplot(x=train_df['bin_0'], y=train_df['ord_3'], hue=train_df['target'])


# In[ ]:


sns.swarmplot(x=train_dd.head(10)['hash_nom_5'],
              y=train_dd.head(10)['day'])


# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("ord 3 , by Month")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=train_dd.head(20)['month'], y=train_dd.head(20)['ord_3'])

# Add label for vertical axis
plt.ylabel("in ")


# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("hash_nom_6  , by day")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=train_dd.head(20)['day'], y=train_dd.head(20)['hash_nom_6'])

# Add label for vertical axis
plt.ylabel("in ")


# In[ ]:


# Set the width and height of the figure
plt.figure(figsize=(14,7))

# Add title
plt.title("Heatmap")

# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=train_dd.head(15), annot=False)

# Add label for horizontal axis
plt.xlabel('data')


# In[ ]:


cyclic_cols = ['day','month']

fig, axs = plt.subplots(1, len(cyclic_cols), figsize=(8, 4))

for i in range(len(cyclic_cols)):
    col = cyclic_cols[i]
    ax = axs[i]
    sns.barplot(x=col, y='target', data=train_dd, ax=ax)
    ax.set_title(col, fontsize=14, fontweight='bold')
    ax.legend(title="target", loc='upper center')


# # split data using : model_selection

# In[ ]:


train_dd.shape


# In[ ]:


test_dd.shape


# In[ ]:



train_dd = train_dd.drop(["id"],axis=1)


train_dd.shape


# # Separate data into training and validation sets

# In[ ]:


# Select  predictors
cols_to_use = [     'bin_0',      'bin_1',      'bin_2',      'bin_3',      'bin_4',
              'day',      'month',            0,            1,            2,
                  3,            4,            5,            6,            7,
                  8,            9,           10,           11,           12,
                 13,           14,           15,           16,           17,
                 18,           19,           20,           21,           22,
                 23,           24,      'ord_0',      'ord_1',      'ord_2',
            'ord_3',      'ord_4',      'ord_5', 'hash_nom_5', 'hash_nom_6',
       'hash_nom_7', 'hash_nom_8', 'hash_nom_9']

X = train_dd[cols_to_use]

# Select target
y = train_dd.target

# Separate data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y)


# In[ ]:


X_train.shape


# In[ ]:


X_valid.shape


# In[ ]:


test_dd.head()


# In[ ]:


#from sklearn.ensemble import RandomForestRegressor


# In[ ]:


#from xgboost import XGBRegressor

 
#my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
#my_model.fit(X_train, y_train, 
 #            early_stopping_rounds=5, 
  #           eval_set=[(X_valid, y_valid)], 
   #          verbose=False)


# In[ ]:


#from sklearn.metrics import mean_absolute_error

#predictions = my_model.predict(X_valid)

# Calculate MAE
#mae_1 = mean_absolute_error(y_valid, predictions) 


#print("Mean Absolute Error:" , mae_1)


# In[ ]:


#from sklearn.metrics import accuracy_score
#acc = accuracy_score(y_valid, predictions)

#print("accuracy_score:" , acc)


# In[ ]:



test_X = test_dd[cols_to_use]

# Use the model to make predictions
#predicted_target = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
#print(predicted_target)


# In[ ]:


#my_submission = pd.DataFrame({'Id': test_X.index, 'target': predicted_target})
# you could use any filename. We choose submission here
#my_submission.to_csv('submission.csv', index=False)


# In[ ]:


#from sklearn.linear_model import LogisticRegression

#lr_m = LogisticRegression( solver="lbfgs",max_iter=500,n_jobs=4)

#lr_m.fit(X_train, y_train)


# In[ ]:


#from sklearn.metrics import mean_absolute_error

#predictions = lr_m.predict(test_X)

# Calculate MAE
#mae_1 = mean_absolute_error(y_valid, predictions) 


#print("Mean Absolute Error:" , mae_1)


# In[ ]:


#from sklearn.metrics import accuracy_score
#acc = accuracy_score(y_valid, predictions)

#print("accuracy_score:" , acc)


# In[ ]:


#rf_model = RandomForestRegressor(n_estimators= 280,max_depth=40,max_features=11,max_leaf_nodes=350,random_state=1)
#rf_model.fit(X_train, y_train)


# In[ ]:


#rf_val_predictions = rf_model.predict(test_X)


# # Using XGBRegressor model with tuning parameter
# 
# 
# trying some models but, I found the XGBRegressor the best until now

# In[ ]:


from xgboost import XGBRegressor
my_model_2 = XGBRegressor(n_estimators=700, learning_rate=0.2, n_jobs=4)
my_model_2.fit(X_train,y_train)

test_preds = my_model_2.predict(test_X)


# In[ ]:


# generating one row  
#X_rows = X_train.sample(frac =.03) 
  
# checking if sample is 0.25 times data or not 
  
#if (0.03*(len(X_train))== len(X_rows)): 
 #   print( "Cool") 
  #  print(len(X_train))
   # print('\n')      
    #print(len(X_rows))       
 


# In[ ]:


# generating one row  
#y_rows = y_train.sample(frac =.03) 
  
# checking if sample is 0.25 times data or not 
  
#if (0.03*(len(y_train))== len(y_rows)): 
 #   print( "Cool") 
  #  print(len(y_train))
   # print('\n')      
   # print(len(y_rows))   


# In[ ]:


#parameters = [{'n_estimators': [ 800, 900, 1000], 
 #                    'learning_rate': [0.05, 0.1, 0.15, 0.2]
  #                  }]


# In[ ]:


#from sklearn.model_selection import GridSearchCV
#from xgboost import XGBRegressor
#gsearch = GridSearchCV(estimator=XGBRegressor(),
 #                      param_grid = parameters, 
  #                     scoring='neg_mean_absolute_error',
   #                    n_jobs=4,cv=3)


# In[ ]:


#gsearch.fit(X_rows,y_rows)


# In[ ]:


#gsearch.best_params_.get('n_estimators'), gsearch.best_params_.get('learning_rate')


# In[ ]:



#final_model = XGBRegressor(n_estimators=gsearch.best_params_.get('n_estimators'), 
                          # learning_rate=gsearch.best_params_.get('learning_rate'), 
                           #n_jobs=4)


# In[ ]:


#final_model.fit(X_rows,y_rows)


# In[ ]:


#test_preds = final_model.predict(test_X)


# # Preparing Submission File

# In[ ]:



#submission = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv', index_col='id')
samplesubmission = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv', index_col='id')

output = pd.DataFrame({'Id': samplesubmission.index, 'target': test_preds})
output.to_csv('submission.csv', index=False)


# In[ ]:


output.head()

