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


train_t = train_t[train_t['ProductCD'] == 'H'].drop('ProductCD', axis = 1).copy()


# In[ ]:


test_t = test_t[test_t['ProductCD'] == 'H'].drop('ProductCD', axis = 1).copy()


# In[ ]:


train = pd.merge(train_t, train_i, how = 'inner', on = 'TransactionID')
test = pd.merge(test_t, test_i, how = 'inner', on = 'TransactionID')


# ## Cij features ## 

# In[ ]:


c_features = []
for i in range(1,15):
    c_features.append('C'+str(i))
train[c_features]


# In[ ]:


# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 2.5, 2.5]) # left, bottom, width, height (range 0 to 1)

sns.heatmap(train[c_features].corr(), annot = True)


# In[ ]:


for c in c_features:
    print(c)
    print('Max = %f'%train[c].max())
    print('Min = %f'%train[c].min())
    print('Mode = %f'%train[c].value_counts().index[0])


# In[ ]:


train['C1'].value_counts()


# In[ ]:


train['C2'].value_counts()


# In[ ]:


train['C3'].value_counts()


# In[ ]:


train['C4'].value_counts()


# In[ ]:


train['C5'].value_counts()


# In[ ]:


train['C6'].value_counts()


# In[ ]:


train['C7'].value_counts()


# In[ ]:


train['C8'].value_counts()


# In[ ]:


train['C9'].value_counts()


# In[ ]:


train['C10'].value_counts()


# In[ ]:


train['C11'].value_counts()


# In[ ]:


train['C12'].value_counts()


# In[ ]:


train['C13'].value_counts()


# In[ ]:


train['C14'].value_counts()


# In[ ]:


test_i.describe()


# In[ ]:


test_i.head(5)


# In[ ]:


train_i.describe()


# In[ ]:


train_i.head(5)


# In[ ]:


train_t.describe()


# In[ ]:


train_t.head(5)


# In[ ]:


test_t.describe()


# In[ ]:


test_t.head(5)


# In[ ]:


#The id_xx are mislabeled in the test_i and test datasets. Let's correct it.
test_i = pd.DataFrame(data = test_i.values, columns = train_i.columns, index = test_i.index)
test_i.head(5)


# In[ ]:


#Identifying categorical features
#I have taken this code from: https://towardsdatascience.com/encoding-categorical-features-21a2651a065c
def identify_cat(dataframe):
    '''
    (pd.DataFrame) -> list
    This function identifies and returns a list with the names of all the categorical columns of a DataFrame.
    '''
    categorical_feature_mask = dataframe.dtypes==object #Here, t can be the entire dataframe or only the features
    categorical_cols = dataframe.columns[categorical_feature_mask].tolist()
    return categorical_cols
catego_i = identify_cat(train_i)
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
convert_type(train_i, catego_i)
convert_type(test_i, catego_i)
convert_type(train_t, catego_t)
convert_type(test_t, catego_t)


# In[ ]:


#Checking for hidden NaN
for column in catego_i:
    print(column)
    print(train_i[column].unique())
# id_15 -> Unknown
#Here we can see that there are a very interesting column: id_33, which is the size of the screen of the customer
#The binary columns are: id_12, id_15, id_16, id_27, id_28, id_29, id_35, id_36,id_37, id_38, 'DeviceType'


# In[ ]:


#we'll drop 'id_33' in the categorical lists
catego_i.remove('id_33')


# In[ ]:


#Checking for hidden NaN
for column in catego_t:
    print(column)
    print(train_t[column].unique())
#A special treatment must be given to the card4 column, because the 'credit or debt' value could be the mean of the
#'credit' and the 'debt' value


# In[ ]:


#Since we'll treat the binary columns separately, we'll make different lists to them:
binary_i = ['id_12', 'id_15', 'id_16', 'id_27', 'id_28', 'id_29', 'id_35', 'id_36','id_37', 'id_38', 'DeviceType']
binary_t = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']
for bina in binary_i:
    catego_i.remove(bina)
for bina in binary_t:
    catego_t.remove(bina)
train_t.drop(binary_t, axis = 1, inplace = True) #M1-M9 all nan
test_t.drop(binary_t, axis = 1, inplace = True)


# ## Grouping the features by number of NaN ##
# The main reasoning of this section is described here: https://www.kaggle.com/carlosasdesouza/40-nan-classes-in-features

# In[ ]:


#First, we'll map the 'Unknown' values in train_i['id_15'] and test_i['id_15'] by NaN
for index in train_i.index:
    if train_i.loc[index, 'id_15'] == 'Unknown': 
        train_i.loc[index, 'id_15'] = np.nan
for index in test_i.index:
    if test_i.loc[index, 'id_15'] == 'Unknown':
        test_i.loc[index, 'id_15'] = np.nan


# In[ ]:


train_i['id_15'].unique()


# In[ ]:


test_i['id_15'].unique()


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


just_for_nan_train = pd.merge(train_t, train_i, how = 'inner', on = 'TransactionID')
group_by_nan_print(just_for_nan_train, 'Merged Train')


# #Category 7 has all-nan values. The columns are: ['dist', 'D11', 'D12', 'M4', 'V1', 'V2', 'V3, 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11']

# In[ ]:


#Let's drop them
all_nans = ['dist1', 'D11', 'D12', 'M4', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11']
for col in all_nans:
    train_t.drop(col, axis = 1, inplace = True)
    test_t.drop(col, axis = 1, inplace = True)


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


all_dfs = [train_i, train_t, test_i, test_t]


# In[ ]:


#Let's drop them:
identity = [train_i, test_i]
transactions = [train_t, test_t]
for df in identity:
    for col in df.columns:
        if df[col].dropna().nunique() <= 1:
            train_i.drop(col, axis = 1, inplace = True)
            test_i.drop(col, axis = 1, inplace = True)
for df in transactions:
    for col in df.columns:
        if df[col].dropna().nunique() <= 1:
            train_t.drop(col, axis = 1, inplace = True)
            test_t.drop(col, axis = 1, inplace = True)


# * *Part 2: Categorical columns*
# Let's build some intuition about the categorical columns with pandas_profiling

# In[ ]:


catego_t.remove('M4')
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
# 
# -> The condition Product CD == H changes drastically the dataset.
# 
# -> card4 (mastercard, visa, american express or discover) has only 10 missing values out of 33024 observations
# 
# -> card6 (credit or debt) has 1 mising value out of 33024 observations
# 
# -> P_emaildomain has 8 missing values out of 33024 observations
# 
# -> R_emaildomain has 32.8% missing values, which is still huge, but no as much as the original sample.
# 
# -> The features of the identity dataset remains unchanged.
# 
# The data from the original sample is this:
# 
# -> Missing: M4 (57.8% missing), id_23(96.4% missing),R_emaildomain (76.8% missing) are columns in which imputation is impossible

# * *Personalized columns: 'P_emaildomain', 'R_emaildomain', 'id_30', 'id_31' , 'DeviceInfo'*
# 
# Let us take a closer look into the unique values for these columns.

# In[ ]:


for unique in ['id_30', 'id_31' , 'DeviceInfo']:
    print(unique)
    print(train_i[unique].unique())
for unique in ['P_emaildomain', 'R_emaildomain']:
    print(unique)
    print(train_t[unique].unique())   


# * ***We'll try to do the following (to train_i, train_t):***
# 
#     _> P_emaildomain, DeviceInfo and R_emaildomain: we'll apply Target_Encoder
#     
#     _>id_30: this is the operational system. We'll split this colum into two new columns - the first is the model and version: 'Windows', 'iOS', 'Mac OS X', 'Mac', 'Linux', 'Android' and 'func' or 'other' (motivated by https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575 and https://www.kaggle.com/yasagure/fraud-makers-may-be-earnest-people-about-browser). The version column will be the year and month of release, as a float number (see below). 
#     
#     
#     _>id_31: this is the browser. Since the number of different models is huge and for a lot of browsers the version isn't informed, we'll just group them into chrome, firefox, google, edge, ie, safari, opera, samsumg and other

# In[ ]:


#id_30

#An idea of automatically improving this code: use web-scraping to update the dictionaries of operational system versions
#The Windows Data came from here: https://www.webopedia.com/DidYouKnow/Hardware_Software/history_of_microsoft_windows_operating_system.html
#And here: https://pt.wikipedia.org/wiki/Windows_8.1
#For generality, we listed all versions of iOS 9, 10 and 11; The numbers came from
# https://en.wikipedia.org/wiki/IOS_9
# https://en.wikipedia.org/wiki/IOS_10
#https://en.wikipedia.org/wiki/IOS_11
#MAC_OS versions:
#https://en.wikipedia.org/wiki/Mac_OS_X_Snow_Leopard
#https://en.wikipedia.org/wiki/Mac_OS_X_Lion
#https://en.wikipedia.org/wiki/OS_X_Mountain_Lion
#https://en.wikipedia.org/wiki/OS_X_Mavericks
#https://en.wikipedia.org/wiki/OS_X_Yosemite
#https://en.wikipedia.org/wiki/OS_X_El_Capitan
#https://en.wikipedia.org/wiki/MacOS_Sierra
#https://en.wikipedia.org/wiki/MacOS_High_Sierra
#We use the versions as keys and the years of release as values, with the month of release as a fractional part
#Some operational systems are labeled as Windows, Mac, Android or iOS , without a version. We'll put NaNs in the version column
#For this OS and,later, we'll do a NaN imputation with the mode, since it's reasonable that, as time goes, more and more
#People update their operational systems.
#For the Linux, func and other operational systems, we'll put NaNs now and later we'll input the NaNs with the mean of all
#the values in the version column.
def split_os(os):
    '''
    (str) -> list
    This function receives a string containing the model of an OS and its version.
    It returns a list where the first element is the model and the second element is the version of the OS.
    If the version is not informed, the second element of the list is a np.nan
    Needs the re and numpy modules
    '''
    if type(os) == float: #If the OS isn't informed, i.e.: nan or NoneType
        return [np.nan, np.nan]
    else:
        op_system = re.compile(r'^Windows|^iOS|^Android|^Mac OS X|^func|^other|^Linux|^Mac$')
        if op_system.search(os):
            os_list = os.split(op_system.search(os).group()) #The second element is the version, if any
            os_list[0] = op_system.search(os).group() #This is the model
            if os_list[1] == '':
                os_list[1] = np.nan
        else:
            return [os, np.nan]
        return os_list
win_version = {' XP': 2001.83, ' Vista': 2006.917, ' 7': 2009.83, ' 8': 2012.67, ' 10': 2015.5833, ' Phone': 2010.917, ' 8.1': 2013.83}
ios_version = {' 9.0.1': 2015.75,
' 9.0.2': 2015.75,
' 9.1.0': 2015.83,
' 9.2.0': 2015.83,  
' 9.2.1': 2016.083, 
' 9.3.0': 2016.25, 
' 9.3.1': 2016.25, 
' 9.3.2': 2016.417,
' 9.3.3': 2016.583,
' 9.3.4': 2016.667,
' 9.3.5': 2016.667,
' 9.3.6': 2016.583,
' 10.0.1': 2016.75,
' 10.0.2': 2016.75,
' 10.0.3': 2016.83,
' 10.1.0': 2016.83,
' 10.1.1': 2016.83,
' 10.2.0': 2017,
' 10.2.1': 2017.083,
' 10.3.0': 2017.25,
' 10.3.1': 2017.33,
' 10.3.2': 2017.417,
' 10.3.3': 2017.583,
' 10.3.4': 2019.583,
' 11.0.0': 2017.50,
' 11.0.1': 2017.75,
' 11.0.2': 2017.83,
' 11.0.3': 2017.83,
' 11.1.0': 2017.83,
' 11.1.1': 2017.917,
' 11.1.2': 2017.917,
' 11.2.0': 2018,
' 11.2.1': 2018,
' 11.2.2': 2018.083,
' 11.2.5': 2018.083,
' 11.2.6': 2018.167,
' 11.3.0': 2018.25, 
' 11.3.1': 2018.33,
' 11.4.0': 2018.417,
' 11.4.1': 2018.583,
' 12.0.0': 2018.75,
' 12.0.1': 2018.83,
' 12.1.0': 2018.83,
' 12.1.1': 2019,
' 12.1.2': 2019}
mac_version = {' 10.6': 2009.67,
' 10_6_8': 2011.583,
' 10_7_5': 2012.83,
' 10_8_5': 2015.67,
' 10.9': 2013.83,
' 10_9_5': 2016.583,
' 10.10': 2014.83,
' 10_10_5': 2017.583,
' 10.11': 2015.83,
' 10_11_3': 2016.083,
' 10_11_4': 2016.25,
' 10_11_5': 2016.417,
' 10_11_6': 2018.583,
' 10.12': 2016.75,
' 10_12': 2016.75,
' 10_12_1': 2016.83,
' 10_12_2': 2017,
' 10_12_3': 2017.083,
' 10_12_4': 2017.25,
' 10_12_5': 2017.5,
' 10_12_6': 2017.583,
' 10.13': 2017.75,
' 10_13_1': 2017.83,
' 10_13_2': 2018,
' 10_13_3': 2018.083,
' 10_13_4': 2018.25,
' 10_13_5': 2018.5,
' 10_13_6': 2018.583,
' 10.14': 2018.75,
' 10_14': 2018.75,
' 10_14_0': 2018.75,
' 10_14_1': 2018.83,
' 10_14_2': 2019,
' 10_14_3': 2019.083,
' 10_14_4': 2019.25}
andr_version = {' 4.4.2': 2014,
                ' 5.0': 2015,
                ' 5.0.1': 2015,
                ' 5.0.2': 2015,
                ' 5.1': 2015.25,
                ' 5.1.1': 2015.33,
                ' 6.0': 2015.83,
                ' 6.0.1': 2016,
                ' 7.0': 2016.67,
                ' 7.1': 2016.83,
                ' 7.1.1': 2017,
                ' 7.1.2': 2017.33,
                ' 8.0.0': 2017.67,
                ' 8.1.0': 2018,
                ' 9': 2018.67}


# In[ ]:


#still id_30
model_test = [] 
version_test = []
for index in test_i.index:
    os_list = split_os(test_i.loc[index, 'id_30'])
    model_test.append(os_list[0])
    if type(os_list[1]) == float: #if it's a nan
        version_test.append(os_list[1])
    elif os_list[0] == 'Windows':
        version_test.append(win_version[os_list[1]]) #It will append the year of release, not the version'
    elif os_list[0] == 'iOS':
        version_test.append(ios_version[os_list[1]])
    elif os_list[0] == 'Mac OS X':
        version_test.append(mac_version[os_list[1]])
    elif os_list[0] == 'Android':
        version_test.append(andr_version[os_list[1]])
model_i = [] 
version_i = []
for index in train_i.index:
    os_list = split_os(train_i.loc[index, 'id_30'])
    model_i.append(os_list[0])
    if type(os_list[1]) == float: #if it's a nan
        version_i.append(os_list[1])
    elif os_list[0] == 'Windows':
        version_i.append(win_version[os_list[1]]) #It will append the year of release, not the version'
    elif os_list[0] == 'iOS':
        version_i.append(ios_version[os_list[1]])
    elif os_list[0] == 'Mac OS X':
        version_i.append(mac_version[os_list[1]])
    elif os_list[0] == 'Android':
        version_i.append(andr_version[os_list[1]])
test_i['os_model'] = model_test
test_i['os_version'] = version_test
train_i['os_model'] = model_i
train_i['os_version'] = version_i
train_i.drop('id_30', axis = 1, inplace = True)
test_i.drop('id_30', axis = 1, inplace = True)


# In[ ]:


#id_31
browsers = train_i['id_31']
browsers_test = test_i['id_31']
chrome = []
ch = re.compile(r'chrome')
firefox = []
fi = re.compile(r'firefox')
edge = []
ed = re.compile(r'edge')
safari = []
saf = re.compile(r'safari')
samsung_browser = []
sam = re.compile(r'samsung')
opera = []
op = re.compile(r'opera')
google = []
go = re.compile(r'google')
i_explorer = []
ie = re.compile(r'ie')
def encode_browsers(df, col = 'id_31'):
    grouped = []
    browsers = df[col]
    for browser in browsers:
        if type(browser) == str:
            if sam.findall(browser) == ['samsung']:
                grouped.append('samsung')
            elif ed.findall(browser) == ['edge']:
                grouped.append('edge')
            elif saf.findall(browser) == ['safari']:
                grouped.append('safari')
            elif ch.findall(browser) == ['chrome']:
                grouped.append('chrome')
            elif fi.findall(browser) == ['firefox']:
                grouped.append('firefox')
            elif op.findall(browser) == ['opera']:
                grouped.append('opera')
            elif go.findall(browser) == ['google']:
                grouped.append('google')
            elif ie.findall(browser) == ['ie']:
                grouped.append('internet explorer')
            else:
                grouped.append('other')
        else:
            grouped.append(browser)
    return grouped
grouped_browsers = encode_browsers(df = train_i)
grouped_browsers_test = encode_browsers(df = test_i)
train_i['g_browser'] = grouped_browsers
test_i['g_browser'] = grouped_browsers_test
train_i.drop('id_31', axis = 1, inplace = True)
test_i.drop('id_31', axis = 1, inplace = True)


# In[ ]:


#after all the cleaning, let us check the reduction of features in train_t
train_t.describe()


# ## A little bit of visualization ##

# In[ ]:


#JUST FOR VISUALIZATION. I'll merge the final train and test datasets when I finish the cleaning of the data
just_viz = pd.merge(train_t, train_i, how = 'inner', on = 'TransactionID')
train = just_viz.copy()


# In[ ]:


just_viz = just_viz.fillna('Unknown')


# * *Part 1: Target Distribution*

# In[ ]:


sns.distplot(just_viz['isFraud'], kde = False)


# * *Part 2: Binary features*

# In[ ]:


sns.countplot('id_12',data=just_viz, hue = 'isFraud')


# In[ ]:


sns.countplot('id_15',data=just_viz, hue = 'isFraud')


# In[ ]:


sns.countplot('id_16',data=just_viz, hue = 'isFraud')


# In[ ]:


sns.countplot('id_27',data=just_viz, hue = 'isFraud')


# In[ ]:


sns.countplot('id_28',data=just_viz, hue = 'isFraud')


# In[ ]:


sns.countplot('id_29',data=just_viz, hue = 'isFraud')


# In[ ]:


sns.countplot('id_35',data=just_viz, hue = 'isFraud')


# In[ ]:


sns.countplot('id_36',data=just_viz, hue = 'isFraud')


# In[ ]:


sns.countplot('id_37',data=just_viz, hue = 'isFraud')


# In[ ]:


sns.countplot('id_38',data=just_viz, hue = 'isFraud')


# In[ ]:


sns.countplot('DeviceType',data=just_viz, hue = 'isFraud')


# * *Part 3: Categorical features*

# In[ ]:


sns.countplot('id_23',data=just_viz, hue = 'isFraud')


# In[ ]:


#id_30 OPERATIONAL SYSTEM MODEL
# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 2.5, 1.2]) # left, bottom, width, height (range 0 to 1)

sns.countplot('os_model',data=just_viz, hue = 'isFraud')


# In[ ]:


#id_30 OS VERSION
#It seems like the version alone doesn't matter at all
sns.pairplot(train[['isFraud', 'os_version']], hue = 'isFraud', height= 10)


# In[ ]:


#id_31 GROUPED BROWSERS
# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 2.5, 1.2]) # left, bottom, width, height (range 0 to 1)
sns.countplot('g_browser',data=just_viz, hue = 'isFraud')


# In[ ]:


sns.countplot('id_34',data=just_viz, hue = 'isFraud')


# In[ ]:


sns.countplot('card4',data=just_viz, hue = 'isFraud')


# In[ ]:


sns.countplot('card6',data=just_viz, hue = 'isFraud')


# In[ ]:


test = pd.merge(test_t, test_i, how = 'inner', on = 'TransactionID')


# In[ ]:


#TargetEncoder
te_1 = target_encoder.TargetEncoder()
te_2 = target_encoder.TargetEncoder()
te_3 = target_encoder.TargetEncoder()
te_1.fit(X = train['DeviceInfo'], y = train['isFraud'])
te_2.fit(X = train['P_emaildomain'], y = train['isFraud'])
te_3.fit(X = train['R_emaildomain'], y = train['isFraud'])
train['P_emaildomain'] = te_2.transform(X = train['P_emaildomain'], y = train['isFraud'])
train['R_emaildomain'] = te_3.transform(X = train['R_emaildomain'], y = train['isFraud'])
test['P_emaildomain'] = te_2.transform(X = test['P_emaildomain'], y = None)
test['R_emaildomain'] = te_3.transform(X = test['R_emaildomain'], y = None)
train['DeviceInfo'] = te_1.transform(X = train['DeviceInfo'], y = train['isFraud'])
test['DeviceInfo'] = te_1.transform(X = test['DeviceInfo'], y = None)


# In[ ]:


test['P_emaildomain']


# In[ ]:


test['R_emaildomain']


# In[ ]:


test['DeviceInfo']


# In[ ]:


#R_emaildomain
sns.pairplot(train[['isFraud', 'R_emaildomain']], hue = 'isFraud', height= 10)


# In[ ]:


#P_emaildomain
sns.pairplot(train[['isFraud', 'R_emaildomain']], hue = 'isFraud', height= 10)


# In[ ]:


train_t.head(5)


# ## Feature Engineering ##
# -> In this section, we'll create new features, do NaN imputation, encode Categorical Features and apply dimensionality reduction methods 

# ## Creating new features ##

# * *Part 1: id_33 column*

# In[ ]:


#Mapping the screen area

def make_area(string):
    if type(string) == float: #Checking for NaN
        return string
    else:
        components = string.split('x')
        area = int(components[0])*(int(components[1]))
        return area
train['id_33_a'] = train['id_33'].apply(lambda a: make_area(a))
test['id_33_a'] = test['id_33'].apply(lambda a: make_area(a))


# * *Part 2: TransactionAMT column*
# 
# We'll split this column into two parts: one is the integer value and the other is the cents value

# In[ ]:


for df in train,test:
    df['dollars'] = df['TransactionAmt'].apply(lambda a: int(a))
    df['cents'] = df['TransactionAmt'].apply(lambda a: a - int(a))


# * *Part 3: Days of the week, hours of day*

# In[ ]:


for df in train, test:
    df['days_week'] = (((df['TransactionDT']//86400))%7)
    df['hours_day'] = (df['TransactionDT']%(3600*24)/3600//1)


# * *Part 4: Unique identification*

# ## Encoding Categorical Data ##

# In[ ]:


#id_23, card4, card6, g_browser, os_model
one_hot = ['card4','card6','g_browser','os_model']
for cat in one_hot:
    test = pd.concat([test,pd.get_dummies(test[cat])], axis = 1)
    train = pd.concat([train,pd.get_dummies(train[cat])], axis = 1)
    test.drop(cat, axis = 1, inplace = True)
    train.drop(cat, axis = 1, inplace = True)
train.drop('other', axis = 1, inplace = True)
test.drop('other', axis = 1, inplace = True)


# In[ ]:


def equiv(element, dictionary):
    if element in list(dictionary.keys()):
        return dictionary[element]
    else:
        return element
#id_34
keys = ['match_status:2','match_status:1','match_status:0','match_status:-1']
values = [2,1,0,-1]
dicto = dict(zip(keys,values))
test['id_34'] = test['id_34'].map(lambda a: equiv(a, dicto))
train['id_34'] = train['id_34'].map(lambda a: equiv(a, dicto))


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


for found in ['id_12', 'id_15', 'id_16', 'id_27', 'id_28', 'id_29']:
    if found in train.columns:
        train[found] = binarize(train, found, 'Found')
        test[found] = binarize(test, found, 'Found')
for t in ['id_35', 'id_36','id_37', 'id_38']:
    if found in train.columns:
        train[t] = binarize(train, t, 'T')
        test[t] = binarize(test, t, 'T')
train['DeviceType'] = binarize(train, 'DeviceType', 'mobile')
test['DeviceType'] = binarize(test, 'DeviceType', 'mobile')


# In[ ]:


#id_33, high cardinality
te = target_encoder.TargetEncoder()
te.fit(X = train['id_33'], y = train['isFraud'])
train['id_33'] = te.transform(X = train['id_33'], y = train['isFraud'])
test['id_33'] = te.transform(X = test['id_33'], y = None)


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
test_dropped = drop_nan(test)
train_dropped = drop_nan(train)
for column in train_dropped:
    if column in test.columns:
        test.drop(column, axis = 1, inplace = True)
for column in test_dropped:
    if column in train.columns:
        train.drop(column, axis = 1, inplace = True)


# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# In[ ]:


list_1 = list(train.columns)
list_1.remove('isFraud')
list_2 = list(test.columns)
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


test['charge card'].value_counts()


# In[ ]:


test.drop('charge card', axis = 1, inplace = True)


# In[ ]:


#We'll use IterativeImputer from sklearn. How to use cv to choose the best model?
features = train.drop('isFraud', axis = 1)
imp_1 = IterativeImputer(max_iter=10, n_nearest_features = 70)
nan_imp_train = pd.DataFrame(data = imp_1.fit_transform(features), columns = features.columns, index = features.index)


# In[ ]:


imp_2 = IterativeImputer(max_iter=10, n_nearest_features = 70)
nan_imp_test = pd.DataFrame(data = imp_2.fit_transform(test), columns = test.columns, index = test.index)


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





# In[ ]:




