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


# # Reading files

# In[ ]:


# First, we read the csv files.
data_train = pd.read_csv("/kaggle/input/titanic/train.csv")
data_gender = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
data_test = pd.read_csv("/kaggle/input/titanic/test.csv")


# # General outlook of the data

# In[ ]:


data_train


# In[ ]:


data_gender


# In[ ]:


data_test


# ## To see all data

# In[ ]:


# Here we change the display option of pandas. 
pd.options.display.max_rows = None
pd.options.display.max_columns = None
display(data_train)


# ## To reset display option

# In[ ]:


pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")
pd.get_option("display.max_rows")


# ## set_option

# In[ ]:


pd.set_option("display.max_rows",20)
display(data_test)


# ## We can see all three data sets as well.

# In[ ]:


data_train, data_gender, data_test


# In[ ]:


# We get general info about the our data set. 
data_train.info()
# We see that there are considerable missing values in cabin and age. 


# In[ ]:


data_gender.info()


# In[ ]:


data_test.info()
# We also see here that we have considerable missing values in age and cabin. 


# In[ ]:


# We check the features (columns)
data_train.columns, data_gender.columns, data_test.columns


# In[ ]:


# We check the thle length of features (columns)
len(data_train.columns), len(data_gender.columns), len(data_test.columns)


# In[ ]:


# We check the head and tail of the data set to see how the data look like. 
data_train.head()


# In[ ]:


data_train.tail()


# In[ ]:


data_gender.head()


# In[ ]:


data_gender.tail()


# In[ ]:


data_test.head()


# In[ ]:


data_test.tail()


# # Merging the data with merge and concat methods

# In[ ]:


# We see that the survived column is missing in the data_test and the data_gender completes this data. 
# Thus, we merge these two data. We make sure that they have the same column order as data_train.
data_merged = pd.merge(data_gender, data_test, on = "PassengerId")
data_merged


# In[ ]:


# We make sure that the columns of data_train and data_merged match.
data_train.columns == data_merged.columns


# In[ ]:


# When we look at the PassengerId of two data sets, we see that data_train ends with 891 and data_merged ends starts with 892.
# These means that we need to merge these two data sets vertically.
data = pd.concat([data_train,data_merged], ignore_index=True)
data


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.info()


# # Descriptive Statistics of the data

# In[ ]:


# We check the descriptive stats of our data. describe() method takes only the integer and float values. 
data.describe()


# In[ ]:


# We can transpose the table, if we feel that it is more readable.
# In this table, we have number of values, mean, std, min-max values, percentile values for each column.
data.describe().T


# In[ ]:


# We can also take get the statistical values for any single column. 
data.Age.mean()


# In[ ]:


data['Age'].mean()  # 2nd way of coding to call a column. 


# In[ ]:


data.Age.median()


# In[ ]:


data.Age.max()


# ## Changing the percentiles

# In[ ]:


perc = [.10,.20,.40,.60,.80]
include = ['float','integer']
desc = data.describe(percentiles = perc, include = include)
desc.T


# ## Correlation

# In[ ]:


correlation = data.corr()
correlation


# In[ ]:


# Getting the max values of correlation coefficient (as absolute values) for each variable.
correlation[correlation<1].abs().max()


# In[ ]:


# Getting the max values of correlation coefficient (as absolute values) for each variable (Another method)
a = correlation.abs()<1
correlation.abs()[a].max()
# We see relatively strong relation between fare and passenger class.


# In[ ]:


a = correlation.abs()<1
b = correlation.abs()>0.5
correlation.abs()[a&b]
# a method to see absolute values of correlation <1 and >0.5


# # NaN values
# - __isna()__

# In[ ]:


data.isna().sum()


# In[ ]:


# Getting the nan values in cabin column.
data[data.Cabin.isna()]
# We see that we have 1014 rows as nan values.


# In[ ]:


# Getting the data excluding nan values.
data[data.Cabin.isna()==False]


# # Repeating Values
# - __value_counts()__
# - Repeating values can be important to further look into our data and infer more information.
# - Also, it can be used to confirm if we have repeating data that was entered by mistake. 

# ## Cabin

# In[ ]:


data["Cabin"].value_counts()
# We can see repeating values, we can look into these data further to infer more information.


# ## Name

# In[ ]:


data["Name"].value_counts()
# We see two repeating names here. 


# In[ ]:


# We can check these names whether they are the same people or not.
data[(data.Name == "Connolly, Miss. Kate") | (data.Name == "Kelly, Mr. James")]
# We see that they have different information, so they are not the same persons.


# ## Ticket

# In[ ]:


data.Ticket.value_counts()
# We see also here some repeating values.


# In[ ]:


data[data.Ticket == 'CA. 2343']
# We can say for CA. 2343 that the people who have the same ticket number are from the same family.


# # Seeing different categories in categorical variables
# - __unique()__

# In[ ]:


data.Survived.unique(), data.Sex.unique(), data.Pclass.unique(), data.SibSp.unique(), data.Parch.unique(), data.Embarked.unique()


# ## Seeing passengers travelling alone

# In[ ]:


data[(data.Parch == 0) & (data.SibSp == 0)]
# We see that 790 people travelled alone.


# ## Comparing the mean of survived travellers

# In[ ]:


data[(data.Parch >= 1) | (data.SibSp >= 1)]['Survived'].mean(), data["Survived"].mean(), data["Survived"][(data.Parch == 0) & (data.SibSp == 0)].mean()
# We see that the mean of survival is higher in people who travel together, we can make an initial assessment that
# families, particularly the women and children, were given priority to get on rescue boats.


# In[ ]:


data.Survived[data.Sex=='female'].mean(), data["Survived"].mean(), data.Survived[data.Age<18].mean()
# We see that women and passengers younger than 18 have a higher survival average than the total average. 


# In[ ]:


data.Survived.mean(), data.Survived[data.Pclass==1].mean(),data.Survived[data.Pclass==2].mean(),data.Survived[data.Pclass==3].mean()
# We also see that passengers with higher class have survival rates above average.
# We can look into these kind of comparisons easier with pivot tables. 


# # Changing column names

# In[ ]:


data.columns


# In[ ]:


data1 = data.copy()
data1.rename(columns={'PassengerId':'YolcuId',
                    'Survived':'Yasam',
                    'Pclass':'Sinif',
                     'Name':'Ad',
                     'Sex':'Cinsiyet',
                     'Age':'Yas',
                     'SibSp':'Kardes_Es',
                     'Parch':'Eb_Cocuk',
                     'Ticket':'BiletId',
                     'Fare':'Ucret',
                     'Cabin':'Kabin',
                     'Embarked':'Liman'}, inplace = True)
data1


# > # Changing values in cells

# In[ ]:


data1.Yasam.replace(0,'yasamiyor',inplace=True)
data1.Yasam.replace(1,'yasiyor',inplace=True)
data1


# In[ ]:


# We can also define a list to make replacements. 
data1.replace(['S','C','Q'],['Southampton','Cherbourg','Queenstown'], inplace=True)
data1


# # Replacing NaN values
# - __fillna()__

# In[ ]:


# data1.Kabin.fillna('Belirsiz', inplace = True)
# data1


# In[ ]:


# We can replace NaN values with the mean so that it would not change the mean of overall data.
# For example, we replace the NaN values in cabin with mean of Age.
data1.

