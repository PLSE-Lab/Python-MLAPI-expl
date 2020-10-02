#!/usr/bin/env python
# coding: utf-8

# This notebook contains work on a [kaggle competition](https://www.kaggle.com/c/titanic#description) in which we try to apply the tools of machine learning to predict which passengers survived the tragedy of sinking of the RMS Titanic.

# ### Imports and helpers

# In[ ]:


# General
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 100, "display.max_columns", 100): 
        display(df)


# ### Data preprocessing

# #### Read and look into the data

# ##### train set

# In[ ]:


# read the train data set and set the index to the PassengerId
train = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')

# bureau = pd.read_csv(INPUT_PATH / "bureau.csv")
# bureau_bal = pd.read_csv(INPUT_PATH / "bureau_balance.csv")
# credit_card_bal = pd.read_csv(INPUT_PATH / "credit_card_balance.csv")
# installment = pd.read_csv(INPUT_PATH / "installments_payments.csv")
# POS_bal = pd.read_csv(INPUT_PATH / "POS_CASH_balance.csv")
# previous_app = pd.read_csv(INPUT_PATH / "previous_application.csv")
submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


# look into the head of the data set
train.head()


# In[ ]:


# look into the head of the data set
#bureau.head()


# In[ ]:


# look into the head of the data set
#bureau_bal.head()


# In[ ]:


# look into the head of the data set
#credit_card_bal.head()


# In[ ]:


# look into the head of the data set
#installment.head()


# In[ ]:


# look into the head of the data set
#POS_bal.head()


# In[ ]:


# look into the head of the data set
#previous_app.head()


# In[ ]:


# get the shape of the data set
train.shape, test.shape, submission.shape


# In[ ]:


# get the column names and return them as a list
train.columns.tolist()


# In[ ]:


# get the statistics of the data 
train.describe()


# In[ ]:


# check the types of the data
#train.dtypes


# In[ ]:


# check the number of nulls in each feature
train.isnull().sum()


# ##### Sample submission

# In[ ]:


# read the Sample Submission and look into it
submission.head()


# ##### test set

# In[ ]:



# look into the data set
test.head()


# In[ ]:


# get the shape of the test set
test.shape


# In[ ]:


# get the columns names
test.columns.tolist()


# In[ ]:


# check the statistics of the test set
test.describe()


# In[ ]:


# check the data types 
test.dtypes


# In[ ]:


# check the missing values
test.isnull().sum()


# #### Prepare data for modeling

# In[ ]:


# combine the train and test sets
train["dataset"] = "train"
test["dataset"] = "test"
combined_df = train.append(test, sort=True)
combined_df.shape


# In[ ]:


# look into the combined data set
combined_df.head()


# In[ ]:


combined_df['TARGET'].value_counts()


# ##### Feature engineering

# In[ ]:


combined_df['TARGET'].astype(float).plot.hist()


# In[ ]:


combined_df.dtypes.value_counts()


# In[ ]:


combined_df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[ ]:


# look into the dataset
#train.ORGANIZATION_TYPE
#train.NAME_CONTRACT_TYPE
#combined_df.dataset


# In[ ]:


# drop the some features and get the shape 

lab_counter = 0
for c in [c for c in combined_df.columns if c!= "dataset"]:
    
    
    if combined_df[c].dtypes == 'object':
        
        if len(list(combined_df[c].unique())) <= 2:
            combined_df[c] = combined_df[c].astype('category')
            combined_df[c] = combined_df[c].cat.codes
            
        lab_counter = lab_counter + 1
        
print('%d columns.' % lab_counter)


# In[ ]:


# check the data types-
combined_df.head()


# In[ ]:


#combined_df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[ ]:


#combined_df['EMERGENCYSTATE_MODE'].isnull().sum()
#combined_df['EMERGENCYSTATE_MODE'].describe()
#combined_df.dtypes.value_counts()
dftrain = combined_df[combined_df.dataset == 'train'].drop('dataset', axis = 1)
dftest = combined_df[combined_df.dataset == 'test'].drop(['dataset', 'TARGET'], axis = 1)


# In[ ]:


train = pd.get_dummies(dftrain)
test = pd.get_dummies(dftest)
train.head()


# In[ ]:


#combined_df.dtypes.value_counts()
train.shape, test.shape
#train.dtypes.value_counts()


# In[ ]:


(train['DAYS_BIRTH'] / -365).describe()


# In[ ]:


train['DAYS_EMPLOYED'].describe()


# In[ ]:


train['DAYS_EMPLOYED'].plot.hist(title = 'Days of Employment');
plt.xlabel('Days')


# In[ ]:


irreg = train[train['DAYS_EMPLOYED'] == 365243]
reg = train[train['DAYS_EMPLOYED'] != 365243]
print('Regularities defaults on %0.2f%% of loans' % (100*reg['TARGET'].mean()))
print('Irregularities defaults on %0.2f%% of loans' % (100*irreg['TARGET'].mean()))
print('There are %d irregular days of employment' % len(irreg))


# In[ ]:


train['DAYS_EMPLOYED_IRREG'] = train['DAYS_EMPLOYED'] == 365243

train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

train['DAYS_EMPLOYED'].plot.hist(title = 'Days EMployment Hist');
plt.xlabel('Days')


# In[ ]:


test['DAYS_EMPLOYED_IRREG'] = test['DAYS_EMPLOYED'] == 365243

test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

test['DAYS_EMPLOYED'].plot.hist(title = 'Days EMployment Hist');
plt.xlabel('Days')

print('There are %d irregularities in the test out of %d entries' % (test['DAYS_EMPLOYED_IRREG'].sum(), len(test)))


# In[ ]:


correlations = train.corr()['TARGET'].sort_values()

print('Most Positive Correlations:\n', correlations.tail(15))

print('\nMost Negative Correlations:\n', correlations.head(15))


# In[ ]:


train['DAYS_BIRTH'] = abs(train['DAYS_BIRTH'])
train['DAYS_BIRTH'].corr(train['TARGET'])


# In[ ]:


plt.style.use('fivethirtyeight')

plt.hist(train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)

plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');


# In[ ]:


plt.figure(figsize = (10, 8))

sn.kdeplot(train.loc[train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

sn.kdeplot(train.loc[train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');


# In[ ]:


age = train[['TARGET', 'DAYS_BIRTH']]
age['YEARS_BIRTH'] = age['DAYS_BIRTH'] / 365

age['YEARS_BINNED'] = pd.cut(age['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age.head(10)


# In[ ]:


age_grp = age.groupby('YEARS_BINNED').mean()
age_grp


# In[ ]:


plt.figure(figsize = (8, 8))

plt.bar(age_grp.index.astype(str), 100 * age_grp['TARGET'])

plt.xticks(rotation = 75); plt.xlabel('Age Group (years)');
plt.ylabel('Failure to Repay (%)');
plt.title('Failure to Repay by Age Group');


# In[ ]:


ext_source_data = train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_source_corr = ext_source_data.corr() 
ext_source_corr


# In[ ]:


plt.figure(figsize = (8, 6))

sn.heatmap(ext_source_corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');


# In[ ]:


plt.figure(figsize = (10, 12))

for c, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    
    plt.subplot(3, 1, c + 1)
  
    sn.kdeplot(train.loc[train['TARGET'] == 0, source], label = 'target == 0')
    
    sn.kdeplot(train.loc[train['TARGET'] == 1, source], label = 'target == 1')
    
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)



# #### Save intermediate result

# In[ ]:


# save your intermediate results 


# #### Load preprocessed data

# In[ ]:


# load your preprocessed data sets 


# In[ ]:





# ### Modeling

# #### Split into train and validation

# In[ ]:


# split the train data set into train and validation enable shuffling 


# In[ ]:


# get the shapes of the data sets 


# In[ ]:


# impute the missing numerical valuese with median of the feature


# In[ ]:


# build the model, fit the train and get the score for train and 
# validation


# In[ ]:


# get the confusion matrix


# In[ ]:


# check the precision_recall_fscore_support


# ### Interpretation

# In[ ]:


# get the feature importance


# #### Visualize feature importance

# In[ ]:


# visualize the feature importance


# #### Use learnings to optimize model

# In[ ]:


# get rid of some feature


# In[ ]:


# do some hyperparameter tuning


# ### Final model

# #### Make predictions and save for submission

# In[ ]:


# impute the missing values in the test data set with the medians 
#from the train data set


# In[ ]:


# predict for the test data set


# In[ ]:


# look into your test data set


# In[ ]:


# creat the submission data frame and look into it


# In[ ]:


# change the data type of the prediction from float to integer


# In[ ]:


# check the submission data set 


# In[ ]:


# save the submission data set to a csv file


# In[ ]:


#,

