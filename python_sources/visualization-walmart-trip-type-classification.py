#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
np.set_printoptions(suppress=True)


# In[ ]:


path = '../input/walmart-recruiting-trip-type-classification/'
data = pd.read_csv(path + 'train.csv.zip')


# In[ ]:


print('The number of samples {}'.format(data.shape))
data.head(5)


# In[ ]:


print('The unique value of the data {}'.format(data[['VisitNumber']].nunique()))
print('The number of the value each VisitNumber: \n{}'.format(data['VisitNumber'].value_counts().sort_values(ascending = False).head(5)))


# In[ ]:


data['TripType'].value_counts()


# In[ ]:


plt.figure(figsize = (12, 10))

sns.set_style('whitegrid')
ax = sns.countplot(x = 'TripType', data = data, palette = 'mako')
ax.set(title = 'The Frequent of Trip Type', ylabel = 'Counts', xlabel = 'Trip Type')


# In[ ]:


plt.figure(figsize = (12, 10))

sns.set_style('whitegrid')
ax1 = sns.countplot(x = 'Weekday', data = data, palette = 'mako')
ax.set(title = 'The Frequent of Weekday', ylabel = 'Counts', xlabel = 'Weekday')


# In[ ]:


plt.figure(figsize = (32, 10))

sns.set_style('whitegrid')
ax1 = sns.countplot(x = 'TripType', hue = 'Weekday', data = data, palette = 'mako')
ax.set(title = 'The Frequent of Weekday', ylabel = 'Counts', xlabel = 'Weekday')


# In[ ]:


data.groupby(['Weekday'])['ScanCount'].sum().plot.bar()


# In[ ]:


print('The types of goods are {}'.format(data['DepartmentDescription'].nunique()))
data['DepartmentDescription'].unique()


# In[ ]:


plt.figure(figsize = (30, 10))

sns.set_style('whitegrid')
ax1 = sns.countplot(x = 'DepartmentDescription', data = data, palette = 'mako')
plt.xticks(rotation = 90)
plt.xlabel('Department Description', fontsize = 15)
plt.ylabel('Counts', fontsize = 15)
plt.title('The Frequent of Department Description', fontsize = 15)


# In[ ]:


data['DepartmentDescription'].value_counts()


# In[ ]:


total = data.isnull().sum().sort_values(ascending = False)
print(total)
percentage = total / data.shape[0]
print('Percentage'.center(50, '-'))
print(percentage)
missingData = pd.concat([total, percentage], axis = 1, keys = ['Total', 'Percentage'])
missingData


# In[ ]:


data['Upc'].unique().tolist()[:10]


# In[ ]:


data.info()


# In[ ]:


data.select_dtypes(include = ["object"]).columns


# In[ ]:


def flot_to_str(obj):
    """
    Convert Upc code from float to string.
    Use this function by applying lambda
    Parameters: "Upc" column of DataFrame
    Return:string converted Upc removing dot
    """
    while obj != 'np.nan':
        obj = str(obj).split('.')[0]
        if len(obj) == 10:
            obj = obj + '0'
        elif len(obj) == 4:
            obj = obj + '0000000' 
        return obj


# In[ ]:


def company(upcData):
    """
    Return company code from given Upc code.
    Parameters:'Upc' column of DataFrame
    Return: company code
    """
    try:
        code = upcData[: 6]
        if code == '000000':
            return x[-5]
        return code
    except:
        return -9999


# In[ ]:


def prodct(upcData):
    """
    Return company code from given Upc code.
    Parameters:'Upc' column of DataFrame
    Return: company code
    """
    try:
        code = upcData[6 :]
        return code
    except:
        return -9999


# In[ ]:


data['handled_Upc'] = data['Upc'].apply(flot_to_str)


# In[ ]:


data['company_code'] = data['handled_Upc'].apply(company)


# In[ ]:


data['product_code'] = data['handled_Upc'].apply(prodct)


# In[ ]:


data['DepartmentDescription'].nunique()


# In[ ]:


data.drop(['Upc'], axis = 1, inplace = True)


# In[ ]:


data.drop(['handled_Upc'], axis = 1, inplace = True)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder

dummy_data = pd.get_dummies(data[['Weekday']])


# In[ ]:


data = pd.concat([data, dummy_data], axis = 1)


# In[ ]:


print('The number of ScanCount {}'.format(data['ScanCount'].nunique()))
data['ScanCount'].unique()


# In[ ]:


data['ScanCount'].value_counts().to_frame()


# In[ ]:


data['ScanCount_bool'] = 1
data.loc[data['ScanCount'] < 1, 'ScanCount_bool'] = 0
data['ScanCount_bool'].value_counts()


# In[ ]:


data['temp_ScanCount'] = data['ScanCount']
data.loc[data['ScanCount'] < 0, 'temp_ScanCount'] = 0
data['number_ScanCount'] = pd.cut(data['temp_ScanCount'], 3, labels = ['low', 'median', 'high'])
concatData = pd.get_dummies(data['number_ScanCount'])
data = pd.concat([data, concatData], axis = 1)


# In[ ]:


data.drop(['temp_ScanCount', 'ScanCount_bool'], axis = 1, inplace = True)


# In[ ]:


data['number_ScanCount'].value_counts().to_frame()


# In[ ]:


data['FinelineNumber'].value_counts().sort_values(ascending = False).to_frame()


# In[ ]:


plt.figure(figsize = (72, 10))

sns.set_style('whitegrid')
ax2 = sns.countplot(x = 'DepartmentDescription', data = data, palette = 'mako')

plt.show()


# In[ ]:


plt.figure(figsize = (12, 10))
sns.set_style('whitegrid')

ax3 = sns.stripplot(x = 'Weekday', y = 'TripType', data = data.loc[data['TripType'] < 999], palette = 'mako')
ax3.set(title = 'The Correlation with Weekday and TripTypes', xlabel = 'Weekday', ylabel = 'Trip Types')


# In[ ]:


plt.figure(figsize = (12, 10))
sns.set_style('dark')

ax4 = sns.stripplot(x = 'Weekday', y = 'FinelineNumber', data = data, palette = 'mako')
# ax4.set(title = 'The relationship between the Weekday and FinelineNumber', xlabel = 'Week Day', ylabel = 'FinelineNumber')
plt.title('The relationship between the Weekday and FinelineNumber', fontsize = 15)
plt.xticks(rotation = 45)
plt.xlabel('Week Day')
plt.ylabel('FinelineNumber')


# In[ ]:


print('The unique of VisitNumber is {}'.format(data['VisitNumber'].nunique()))
data.DepartmentDescription.nunique()


# In[ ]:


print('The missing data information'.center(50, '-'))
data.isnull().sum().sort_values(ascending = False)


# In[ ]:


data['DepartmentDescription'].fillna( 'None', inplace = True)


# In[ ]:


data['FinelineNumber'].fillna(data['FinelineNumber'].mean(), inplace = True)


# In[ ]:


data.isnull().sum().sort_values(ascending = False).to_frame()


# In[ ]:


tempData1 = pd.get_dummies(data[['DepartmentDescription']])
data = pd.concat([data, tempData1], axis = 1)


#  - The funciton to delete the 'nan'

# In[ ]:


def deleteNan(datas):
    """
    Delete the 'nan' value of columns
    Parameters: datas is the data to delete.
    Return: cleaned data
    """
    datas == 'nan'
    datas = np.nan
    return datas


# In[ ]:


# columns = data.columns.tolist()
indexList = []
# columns = ['company_code', 'product_code']
columns = ['company_code']
for column in columns:
#     for index in range(data.shape[0]):
    indexList = data.loc[range(data.shape[0]), column] == 'nan'


# In[ ]:


data.loc[indexList, column] = '000000'


# In[ ]:


indexList.value_counts()


# In[ ]:


indexList = []
columns = ['product_code']
for column in columns:
    indexList = data.loc[range(data.shape[0]), column] == 'nan'


# In[ ]:


data.loc[indexList, column] = '000000'


# In[ ]:


data['company_code'].value_counts().sort_values(ascending = False)


# In[ ]:


data[['product_code']].sample(10)


# In[ ]:


data.loc[data['product_code'] == '', 'product_code'] = '00000'


# In[ ]:


data['product_code'].value_counts().sort_values(ascending = False).head()


# In[ ]:


data.info()


# In[ ]:


objectData = data.select_dtypes(include = ['object', 'category']).head()


# In[ ]:


objectData.columns.tolist()


# In[ ]:


data.drop(['Weekday', 'DepartmentDescription','number_ScanCount'], axis = 1, inplace = True)


# In[ ]:


print('The data information'.center(50, '-') + '\n')
print(data.shape)
data.sample(3)

