#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #visualization
import re #regular expressions, will be used when dealing with id_30 and id_31
import matplotlib.pyplot as plt #visualization
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder #encoding categorical features
from category_encoders import target_encoder #We'll use Target Encoder for the emails
from sklearn.preprocessing import StandardScaler #PCA, dimensionality reducion
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer #NaN imputation
from sklearn.impute import IterativeImputer #NaN imputation
from sklearn.impute import KNNImputer #NaN imputation

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


ss = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
train_t = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test_t = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
train_i = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
test_i = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')


# In[ ]:


train_t = train_t[train_t['ProductCD'] == 'W'].drop('ProductCD', axis = 1).copy()
train = pd.merge(train_t, train_i, how = 'inner', on = 'TransactionID')


# In[ ]:


test_t = test_t[test_t['ProductCD'] == 'W'].drop('ProductCD', axis = 1).copy()
test = pd.merge(test_t, test_i, how = 'inner', on = 'TransactionID')


# In[ ]:


train.head(5)
#The intersection between train_t and train_i is empty


# In[ ]:


test.head(5)
#The intersection between test_i and test_t is empty


# In[ ]:


train_t.describe()


# In[ ]:


train_t.head(5)


# In[ ]:


test_t.describe()


# In[ ]:


test_t.head(5)


# In[ ]:


#Identifying categorical features
#I have taken this code from: https://towardsdatascience.com/encoding-categorical-features-21a2651a065c
def identify_cat(dataframe):
    '''
    (pd.DataFrame) -> list
    This function identifies and returns a list with the names of all the categorical columns of a DataFrame.
    '''
    categorical_feature_mask = dataframe.dtypes==object
    categorical_cols = dataframe.columns[categorical_feature_mask].tolist()
    return categorical_cols
catego_t = identify_cat(train_t)


# In[ ]:


def convert_type(dataframe, catego_cols):
    '''
    (pd.DataFrame, list) -> None
    This is an optimization function. It converts the type of categorical columns in a DataFrame from 'object' to 'category',
    making operations faster.
    See the docs here: https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html
    '''
    for column in catego_cols:
        dataframe[column].astype('category')
convert_type(train_t, catego_t)
convert_type(test_t, catego_t)


# In[ ]:


#Checking for hidden NaN
for column in catego_t:
    print(column)
    print(train_t[column].unique())
#The binary columns are: M1, M2, M3, M5, M6, M7, M8, M9
#A special treatment must be given to the card4 column, because the 'credit or debt' value could be the mean of the
#'credit' and the 'debt' value
#No R_emaildomain


# In[ ]:


#Since we'll treat the binary columns separately, we'll make different lists to them:
binary_t = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']
for bina in binary_t:
    catego_t.remove(bina)


# ## Grouping the features by number of NaN ##
# The main reasoning of this section is described here: https://www.kaggle.com/carlosasdesouza/40-nan-classes-in-features

# In[ ]:


def group_by_nan_print(features, name):
    '''
    (pd.DataFrame, str) -> None
    This function groups the features by the number of NaN in 
    each feature and print the Categories in the screen'''
    
    nan_i_values = []
    for column in features.columns:
        nan_i_values.append(len(features[features[column].isna() == True][column]))
        #The command above counts the number of NaN in column and appends it to a list
    data_fr = pd.DataFrame(index = features.columns, data = nan_i_values, columns = ['NaN'])
    i = 1
    print("NaN Categories for the %s DataFrame : \n"%name)
    for unique_nan in data_fr['NaN'].unique():
        print("Category ", i)
        print("Number of NaN values: ", unique_nan)
        print("Features in category ", i, " : ")
        for column in data_fr[data_fr['NaN'] == unique_nan].index:
            print(column)
        print('\n')
        i+=1


# In[ ]:


group_by_nan_print(train_t, 'Transactions Train')
#Category 7 has all-nan values. The columns are: 
#['dist2','R_emaildomain' 'D6','D7','D8','D9' 'D12', 'D13','D14', 'V138-V278'. 'V322-V339']


# In[ ]:


binary_t = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
all_na = ['dist2','D6','D7','D8','D9','D12', 'D13','D14', 'R_emaildomain']
for i in range(138,279):
    all_na.append('V'+str(i))
for i in range(322,340):
    all_na.append('V'+str(i))
for col in all_na:
    train_t.drop(col, axis = 1, inplace = True)
    test_t.drop(col, axis = 1, inplace = True)
    if col in catego_t:
        catego_t.remove(col)


# ## Correlated Missing Groups ##
# Here, we'll make a list of lists, in which each element is a group of features with similar number of NaN. Later, we'll use this list to apply PCA in this small groups, for dimensionality reduction

# In[ ]:


def group_nan(features):
    '''
    (pd.DataFrame) -> list
    This function groups the features by the number of NaN in 
    each feature and returns a list of categories of NaN
    '''
    final_list = []
    nan_values = []
    for column in features.columns:
        nan_values.append(len(features[features[column].isna() == True][column]))
        #The command above counts the number of NaN in column and appends it to a list
    data_fr = pd.DataFrame(index = features.columns, data = nan_values, columns = ['NaN'])
    i = 1
    for unique_nan in data_fr['NaN'].unique():
        cat = []
        for column in data_fr[data_fr['NaN'] == unique_nan].index:
            cat.append(column)
        final_list.append(cat)
        i+=1
    return final_list


# In[ ]:


nan_groups = group_nan(train_t)
nan_groups.extend(group_nan(train_i))


# ## Organizing Data ##

# * *Part 1: Checking for Constant Columns*

# In[ ]:


all_dfs = [train_t, test_t]


# In[ ]:


for df in all_dfs:
    for col in df.columns:
        if df[col].dropna().nunique() <= 1:
            train_t.drop(col, axis = 1, inplace = True)
            test_t.drop(col, axis = 1, inplace = True)
            if col in catego_t:
                catego_t.remove(col)


# * *Part 2: Categorical columns*
# Let's build some intuition about the categorical columns with pandas_profiling

# In[ ]:


train_t[catego_t]


# In[ ]:


import pandas_profiling
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file

profile = ProfileReport(train_t[catego_t],
                        title='Fraud in Credit Cards - Categorical Features',
                        html={'style':{'full_width':True}},
                        minimal = True)

profile.to_notebook_iframe()


# * Brief Analysis on categorical features
# -> The condition Product CD == W changes drastically the dataset.
# 
# -> It doesn't have the addr2 feature,i.e.: it's some kind of local shopping.
# 
# -> card4 (mastercard, visa, american express or discover) doesn't have missing values.
# 
# -> card6 (credit or debt) has 1 mising value out of 439670 observations.
# 
# -> P_emaildomain has 5.1% missing values, which is still large, but no as much as the original sample.
# 
# -> R_emaildomain has all missing values.
# 
# The data from the original sample is this:
# 
# -> Missing: M4 (57.8% missing), id_23(96.4% missing),R_emaildomain (76.8% missing) are columns in which imputation is impossible
# 

# * *Personalized column: 'P_emaildomain'*
# 
# Let us take a closer look into the unique values for this column.

# In[ ]:


print('P_emaildomain')
print(train_t['P_emaildomain'].unique())   


# * P_emaildomain: we'll use Target Encoder

# ## A little bit of visualization ##

# In[ ]:


just_viz = train_t.copy().fillna('Unknown')


# * *Part 1: Target Distribution*

# In[ ]:


sns.distplot(train_t['isFraud'], kde = False)


# * *Part 2: Binary features*

# In[ ]:


sns.countplot('M1',data=just_viz, hue = 'isFraud') 


# In[ ]:


sns.countplot('M2',data=just_viz, hue = 'isFraud') 


# In[ ]:


sns.countplot('M3',data=just_viz, hue = 'isFraud') 


# In[ ]:


sns.countplot('M5',data=just_viz, hue = 'isFraud') 


# In[ ]:


sns.countplot('M6',data=just_viz, hue = 'isFraud') 


# In[ ]:


sns.countplot('M7',data=just_viz, hue = 'isFraud') 


# In[ ]:


sns.countplot('M8',data=just_viz, hue = 'isFraud') 


# In[ ]:


sns.countplot('M9',data=just_viz, hue = 'isFraud') 


# * *Part 3: Categorical features*

# In[ ]:


sns.countplot('card4',data=just_viz, hue = 'isFraud')


# In[ ]:


sns.countplot('card6',data=just_viz, hue = 'isFraud')


# In[ ]:


sns.countplot('card6',data=just_viz, hue = 'isFraud')


# In[ ]:


#TargetEncoder
te = target_encoder.TargetEncoder()
te.fit(X = train_t['P_emaildomain'], y = train_t['isFraud'])
train_t['P_emaildomain'] = te.transform(X = train_t['P_emaildomain'], y = train_t['isFraud'])
test_t['P_emaildomain'] = te.transform(X = test_t['P_emaildomain'], y = None)


# In[ ]:


#P_emaildomain
sns.pairplot(train_t[['isFraud', 'P_emaildomain']], hue = 'isFraud', height= 10)


# ## Feature Engineering ##
# -> In this section, we'll create new features, do NaN imputation, encode Categorical Features and apply dimensionality reduction methods 

# ## Creating new features ##

# * *Part 1: TransactionAMT column*
# 
# We'll split this column into two parts: one is the integer value and the other is the cents value

# In[ ]:


for df in train_t,test_t:
    df['dollars'] = df['TransactionAmt'].apply(lambda a: int(a))
    df['cents'] = df['TransactionAmt'].apply(lambda a: a - int(a))


# * *Part 2: Days of the week, hours of day*

# In[ ]:


for df in train_t, test_t:
    df['days_week'] = (((df['TransactionDT']//86400))%7)
    df['hours_day'] = (df['TransactionDT']%(3600*24)/3600//1)


# * *Part 3: Unique identification*

# ## Encoding Categorical Data ##

# In[ ]:


#Card4, card6
one_hot = ['card4','card6']
for cat in one_hot:
    test_t = pd.concat([test_t,pd.get_dummies(test_t[cat])], axis = 1)
    train_t = pd.concat([train_t,pd.get_dummies(train_t[cat])], axis = 1)
    train_t.drop(cat, axis = 1, inplace = True)
    test_t.drop(cat, axis = 1, inplace = True)


# In[ ]:


def equiv(element, dictionary):
    if element in list(dictionary.keys()):
        return dictionary[element]
    else:
        return element
#M4
keys = ['M2','M0','M1']
values = [2,0,1]
dicto = dict(zip(keys,values))
test_t['M4'] = test_t['M4'].map(lambda a: equiv(a, dicto))
train_t['M4'] = train_t['M4'].map(lambda a: equiv(a, dicto))


# * *Binary features*

# In[ ]:


def binarize(dataframe, column, pos_value):
    '''
    (pd.DataFrame, pd.Series) -> pd.Series
    Modifies a dataframe inplace, binarizing the Column into two (0\1) columns
    The pos_value is the positive value. It could be Yes, positive,etc...
    '''
    return dataframe[~ dataframe[column].isna()][column].map(lambda r: 1 if (r == pos_value) else 0 )
    #The strange indexation exists because we won't operate with NaN


# In[ ]:


for T in binary_t:
    train_t[T] = binarize(train_t, T, 'T')
    test_t[T] = binarize(test_t, T, 'T')


# ## NaN Imputation ##

# In[ ]:


#First of all, let's drop ALL THE COLUMNS that have a number of NaN greater than 50% of the length of the dataframe
def drop_nan(features):
    '''
    (pd.DataFrame) -> List
    This function receives a DataFrame and drops all of its columns that has a number of missing greater than 50%
    of the total values of the column. Returns the list of columns dropped.
    '''
    drop_columns = []
    threshold = .5*len(features.index) #50% of the total number of rows in the DataFrame
    for column in features.columns:
        nan_value = len(features[features[column].isna() == True][column]) #Number of NaN values in this particular column
        if nan_value > threshold:
            drop_columns.append(column)
            features.drop(column, axis = 1, inplace = True)
    return drop_columns
test_dropped = drop_nan(test_t)
train_dropped = drop_nan(train_t)
for column in train_dropped:
    if column in test_t.columns:
        test_t.drop(column, axis = 1, inplace = True)
for column in test_dropped:
    if column in train_t.columns:
        train_t.drop(column, axis = 1, inplace = True)


# In[ ]:


train_t.head(5)


# In[ ]:


test_t.head(5)


# In[ ]:


list_1 = list(train_t.drop('isFraud',axis=1).columns)
list_2 = list(test_t.columns)
for i in range(len(list_2)):
    if list_1[i] != list_2[i]:
        print(list_1[i-1])
        print(list_2[i-1])
        print(list_1[i])
        print(list_2[i])
        print(list_1[i+1])
        print(list_2[i+1])
        break


# In[ ]:


train_t.drop('debit or credit', axis = 1, inplace = True)


# In[ ]:


#We'll use IterativeImputer from sklearn. How to use cv to choose the best model?
features = train_t.drop('isFraud', axis = 1)
imp_1 = IterativeImputer(max_iter=10, n_nearest_features = 70)
nan_imp_train = pd.DataFrame(data = imp_1.fit_transform(features), columns = features.columns, index = features.index)


# In[ ]:


imp_2 = IterativeImputer(max_iter=10, n_nearest_features = 70)
nan_imp_test = pd.DataFrame(data = imp_2.fit_transform(test_t), columns = test_t.columns, index = test_t.index)


# In[ ]:


#Here, we'll use KNNImputer from sklearn
imp_3 = KNNImputer()
knn_imp_train = pd.DataFrame(data = imp_3.fit_transform(features), columns = features.columns, index = features.index)


# In[ ]:


imp_4 = KNNImputer()
knn_imp_test = pd.DataFrame(data = imp_4.fit_transform(test), columns = test.columns, index = test.index)


# ## Dimensionality reduction ##

# In[ ]:


#Apply this function to the final dataframes and the nan_groups list. It will return a list of columns for each DataFrame.
#Apply PCA to these columns and then LDA to all the remaining columns.
def correlated_columns(dataframe, lista):
    '''
    (pd.DataFrame, list) -> list
    This function receives a DataFrame and returns a list of highly correlated columns. 
    The analysis is done by groups of Nan.
    '''
    corr_cols = []
    for element in lista:
        if len(element) == 1:
            lista.remove(element) #One nan-group with just one column isn't useful
    for unique in lista:
        corr_df = dataframe[unique].corr() #This is a Dataframe with the correlation values
        for col in corr_df.columns:
            for row in corr_df.index:
                if col == row:
                    pass #Every column is 100% correlated with itself
                else:
                    if abs(corr_df.loc[row, col]) > .95: #This will be our 'high-correlated' threshold. 
                        if col not in corr_cols:
                            corr_cols.append(col)
    return corr_cols


# In[ ]:


nan_imp_train.head(5)


# In[ ]:


knn_imp_train.head(5)


# In[ ]:


nan_imp_test.head(5)


# In[ ]:


knn_imp_test.head(5)


# In[ ]:


print('Something')


# In[ ]:


print('Something')


# In[ ]:




