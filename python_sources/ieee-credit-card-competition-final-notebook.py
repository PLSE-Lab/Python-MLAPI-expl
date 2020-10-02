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
from sklearn.preprocessing import normalizer #PCA, dimensionality reducion
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer #NaN imputation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Optimizing Memory usage ##

# In[ ]:


def PandasCompress(csv_location):
    '''This function is equivalent to pd.read_csv(csv_location), but it 
    optimizes the memory usage.'''

    dataset = pd.read_csv(csv_location, nrows = 5)
    numero_de_colunas = len(dataset.columns)
    del(dataset)

    vetor_quantidade_de_colunas = np.arange(numero_de_colunas)

    coluna_tipo = {}

    for i in range (0, (int(numero_de_colunas / 50) - 1)):

        pandas_df = pd.read_csv(csv_location, usecols = vetor_quantidade_de_colunas[i * 50: (((i+1) * 50) - 1)])
        pandas_describe = pandas_df.describe()

        for n in range (0, 49):  
            tipo_da_coluna = str(pandas_df.dtypes[n])
            if tipo_da_coluna == 'object':
                coluna_tipo[pandas_df.columns[n]] = 'category'
            elif tipo_da_coluna == 'int64':

                if (pandas_describe[pandas_df.columns[n]].loc['min'] >= -128 and
                pandas_describe[pandas_df.columns[n]].loc['max'] <= 128):
                    coluna_tipo[pandas_df.columns[n]] = 'int8'
                elif (pandas_describe[pandas_df.columns[n]].loc['min'] >= -32768 and
                pandas_describe[pandas_df.columns[n]].loc['max'] <= 32768):
                    coluna_tipo[pandas_df.columns[n]] = 'int16'
                elif (pandas_describe[pandas_df.columns[n]].loc['min'] >= -2147483648 and
                pandas_describe[pandas_df.columns[n]].loc['max'] <= 2147483648):
                    coluna_tipo[pandas_df.columns[n]] = 'int32'
                else:
                    coluna_tipo[pandas_df.columns[n]] = 'int64'

            elif tipo_da_coluna == 'float64':

                if (pandas_describe[pandas_df.columns[n]].loc['min'] >= -32768 and
                pandas_describe[pandas_df.columns[n]].loc['max'] <= 32768):
                    coluna_tipo[pandas_df.columns[n]] = 'float16'
                elif (pandas_describe[pandas_df.columns[n]].loc['min'] >= -2147483648 and
                pandas_describe[pandas_df.columns[n]].loc['max'] <= 2147483648):
                    coluna_tipo[pandas_df.columns[n]] = 'float32'
                else:
                    coluna_tipo[pandas_df.columns[n]] = 'float64'

        if numero_de_colunas % 50 != 0:
            pd.read_csv(csv_location, usecols = vetor_quantidade_de_colunas[int(numero_de_colunas / 50) : numero_de_colunas - 1])

        for n in range (0, ((numero_de_colunas % 50) -1)): 
            tipo_da_coluna = str(pandas_df.dtypes[n])
            if tipo_da_coluna == 'object':
                coluna_tipo[pandas_df.columns[n]] = 'category'
            elif tipo_da_coluna == 'int64':

                if (pandas_describe[pandas_df.columns[n]].loc['min'] >= -128 and
                pandas_describe[pandas_df.columns[n]].loc['max'] <= 128):
                    coluna_tipo[pandas_df.columns[n]] = 'int8'
                elif (pandas_describe[pandas_df.columns[n]].loc['min'] >= -32768 and
                pandas_describe[pandas_df.columns[n]].loc['max'] <= 32768):
                    coluna_tipo[pandas_df.columns[n]] = 'int16'
                elif (pandas_describe[pandas_df.columns[n]].loc['min'] >= -2147483648 and
                pandas_describe[pandas_df.columns[n]].loc['max'] <= 2147483648):
                    coluna_tipo[pandas_df.columns[n]] = 'int32'
                else:
                    coluna_tipo[pandas_df.columns[n]] = 'int64'

            elif tipo_da_coluna == 'float64':

                if (pandas_describe[pandas_df.columns[n]].loc['min'] >= -32768 and
                pandas_describe[pandas_df.columns[n]].loc['max'] <= 32768):
                    coluna_tipo[pandas_df.columns[n]] = 'float16'
                elif (pandas_describe[pandas_df.columns[n]].loc['min'] >= -2147483648 and
                pandas_describe[pandas_df.columns[n]].loc['max'] <= 2147483648):
                    coluna_tipo[pandas_df.columns[n]] = 'float32'
                else:
                    coluna_tipo[pandas_df.columns[n]] = 'float64'
    dataset = pd.read_csv(csv_location, dtype = coluna_tipo)                            
    return dataset


# In[ ]:


train_t = PandasCompress('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test_t = PandasCompress('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
train_i = PandasCompress('/kaggle/input/ieee-fraud-detection/train_identity.csv')
test_i = PandasCompress('/kaggle/input/ieee-fraud-detection/test_identity.csv')


# In[ ]:


train_t.head(5)


# In[ ]:


train_i.head(5)


# In[ ]:


test_t.head(5)


# In[ ]:


test_i.head(5)


# In[ ]:


#The id_xx are mislabeled in the test_i and test datasets. Let's correct it.
test_i = pd.DataFrame(data = test_i.values, columns = train_i.columns, index = test_i.index)


# In[ ]:


all_dfs = [train_t, test_t, train_i, test_i]


# ## Grouping the features by number of NaN ##
# 
# The main reasoning of this section is described here: https://www.kaggle.com/carlosasdesouza/40-nan-classes-in-features

# ## Cleaning Data ##
# 
# In this section we'll identify and clean the datasets from columns with 50% or higher number of NaN

# In[ ]:


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


#Dropping constant columns
for df in all_dfs:
    for col in df.columns:
        if df[col].dropna().nunique() <= 1:
            df.drop(col, axis = 1, inplace = True)


# ## Categorical Data: DeviceInfo, id_30, id_31, P_emaildomain and R_emaildomain ##

# * *Part 1: Identity Train and Test - DeviceInfo, id_30, id_31*

# -> DeviceInfo: too many unique values and there are a lot of unique values present in the test set that weren't in the training set. We'll drop this column in both the train and test sets. Try to group this data will be meaningless, as showed by some notebooks in the competition.

# In[ ]:


train_i.drop('DeviceInfo', axis = 1, inplace = True)
test_i.drop('DeviceInfo', axis = 1, inplace = True)


# ->id_30: this is the operational system. We'll split this colum into two new columns - the first is the model and version: 'Windows', 'iOS', 'Mac OS X', 'Mac', 'Linux', 'Android' and 'func' or 'other' (motivated by https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575 and https://www.kaggle.com/yasagure/fraud-makers-may-be-earnest-people-about-browser). The version column will be the year and month of release, as a float number (see below). 

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


# ->id_31: this is the browser. Since the number of different models is huge and for a lot of browsers the version isn't informed, we'll just group them into chrome, firefox, google, edge, ie, safari, opera, samsumg and other

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


# * *Part 2: Transaction Train and Test - P_emaildomain and R_emaildomain*
# 
# -> Here, we'll just encode this features with TargetEncoder, for visualization

# In[ ]:


#JUST FOR VISUALIZATION. I'll merge the final train and test datasets when I finish the feature engineering section
#The full plots are in the individual notebooks. Here,we'll just plot things that differ from one dataset to another
train = pd.merge(train_t, train_i, how = 'inner', on = 'TransactionID')
test = pd.merge(test_t, test_i, how = 'inner', on = 'TransactionID')


# In[ ]:


#Fitting the Target Encoder
te_1 = target_encoder.TargetEncoder()
te_2 = target_encoder.TargetEncoder()
te_1.fit(X = train['P_emaildomain'], y = train['isFraud'])
te_2.fit(X = train['R_emaildomain'], y = train['isFraud'])
for dataframe in train, test:
    dataframe['P_emaildomain'] = te_1.transform(X = dataframe['P_emaildomain'], y = dataframe['isFraud'])
    dataframe['R_emaildomain'] = te_2.transform(X = dataframe['R_emaildomain'], y = dataframe['isFraud'])


# ## Visualization with raw features ## 
# In this section, we'll just do some EDA with the raw features. After the feature engineering section (filling NaN, creating new features, applying Categorical Feature Encoding) we'll have another section on visualizing data.

# * *Part 1: Target Distribution*

# In[ ]:


sns.distplot(train['isFraud'], kde = False)


# * *Part 2: P_emaildomain, R_emaildomain, os_model, os_version and grouped_browsers*

# In[ ]:


#id_30 OPERATIONAL SYSTEM MODEL
# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 2.5, 1.2]) # left, bottom, width, height (range 0 to 1)

sns.countplot('os_model',data=train, hue = 'isFraud')


# In[ ]:


#id_30 OS VERSION
sns.pairplot(train[['isFraud', 'os_version']], hue = 'isFraud', height= 10)


# In[ ]:


#id_31 GROUPED BROWSERS
# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 2.5, 1.2]) # left, bottom, width, height (range 0 to 1)
sns.countplot('g_browser',data=train, hue = 'isFraud')


# In[ ]:


#R_emaildomain
sns.pairplot(train[['isFraud', 'R_emaildomain']], hue = 'isFraud', height= 10)


# In[ ]:


#P_emaildomain
sns.pairplot(train[['isFraud', 'P_emaildomain']], hue = 'isFraud', height= 10)


# ## Feature Engineering ##
# -> In this section, we'll create new features, do NaN imputation, encode Categorical Features and apply dimensional reduction methods 

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
train_i['id_33_a'] = train_i['id_33'].apply(lambda a: make_area(a))
test_i['id_33_a'] = test_i['id_33'].apply(lambda a: make_area(a))


# * *Part 2: TransactionAMT column*
# 
# We'll split this column into two parts: one is the integer value and the other is the cents value

# In[ ]:


for df in all_dfs:
    df['dollars'] = df['TransactionAmt'].apply(lambda a: int(a))
    df['cents'] = df['TransactionAmt'].apply(lambda a: a - int(a))


# * *Part 3: Date features*

# In[ ]:


for df in train_t, test_t:
    df['days'] = df['TransactionDT']//(86400)
    df['weeks'] = df['TransactionDT']//(7*86400) #Very long term 
    df['days_month'] = (((df['TransactionDT']//86400))%30) #Long term
    df['hours_day'] = (df['TransactionDT']%(3600*24)/3600//1) #Mid term
    df['minutes_hour'] = (df['TransactionDT']%(60*60)/60//1) #Short term


# In[ ]:





# In[ ]:




