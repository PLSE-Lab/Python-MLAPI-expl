#!/usr/bin/env python
# coding: utf-8

# # Pandas Quick overview
# Pandas is built over Numpy
# DataFrame is pretty much like spread sheet
# 

# In[ ]:


#import the packages
import numpy as np
import pandas as pd


# # Selecting and retrieving Data

# In[ ]:


my_exp = pd.Series([10,25,100,23])
my_exp[2]


# In[ ]:


my_exp.index =['day1','day2','day3','day4']
my_exp


# In[ ]:


my_exp[['day4','day2','day4']]


# In[ ]:


my_exp.index = ['day1','day2','day3','day4']
my_exp


# In[ ]:


my_exp['day2']


# In[ ]:


my_exp[['day1','day3']]


# In[ ]:


my_exp[['day3','day2']]


# In[ ]:


my_exp[[0,2]]


# In[ ]:


data = pd.Series([2,3,4,7,4],index=['row 1','row 2','row 3','row 4','row 5'])


# In[ ]:


data = pd.Series([2,3,4,7,4])
data.index=['row 1','row 2','row 3','row 4','row 5']


# In[ ]:


data


# In[ ]:


data['row 3']


# In[ ]:


s_data = pd.Series([2,3,4.2,'5'],index=['row 1','row 2','row 3','row 4'])
print(s_data[2]) # selecting in Index
print(s_data['row 4']) # selecting by label


# In[ ]:


s_data2 = pd.Series([3,9,6])
s_data2.index = ['row 1',"row 2",'row 3']
s_data2


# In[ ]:


series_data = pd.Series([4,7,10,'DataMites',4.2])
print(series_data)


# In[ ]:


series_data2 = pd.Series([34,23,20])
print(series_data2)


# In[ ]:


#row label assignment with index
series_data2.index = ['row 1','row 2', 'row 3']
series_data2


# In[ ]:


#retrieving value with label name
series_data2['row 2']


# In[ ]:


#label assignment during initiation
series1 = pd.Series([2,3,5,6],index=['row 1', 'row 2','row 3','row 4'])
series1


# In[ ]:


#retrieving multiple values with label as reference
series2= pd.Series(np.arange(8), index=['row 1', 'row 2','row 3', 'row 4','row 5', 'row 6','row 7', 'row 8'])
series2[['row 4','row 2']]


# In[ ]:


#retrieve records by label with slicing operator colon (:)
series2['row 2':'row 5']


# In[ ]:


#Retrieving with index through slicing
data


# In[ ]:


data[['row 2','row 5']]


# In[ ]:


data['row 2':'row 4']


# In[ ]:


np.random.randint(25)


# In[ ]:


df1 = pd.DataFrame(np.random.randint(10,100,25)
                .reshape(5,5))


# In[ ]:


get_ipython().run_line_magic('pwd', '')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


cars = pd.read_csv('../input/mtcars.csv')
print(cars.shape)
cars.head()


# In[ ]:


type(cars)


# In[ ]:


cars.head(3)


# In[ ]:


cars.iloc[2:5,0:4]


# In[ ]:


cars.loc[2:5,:]


# In[ ]:


cars[ (cars.hp>100) & (cars.mpg>30)]


# In[ ]:


cars.shape


# In[ ]:


cars.loc[[2,4],:'wt']


# In[ ]:


# Retrieve data with columns : Car_model, mpg, wt 
#for cars with HP > 150


# In[ ]:


cars.loc[(cars.hp>150),['car_model','mpg','wt']]


# In[ ]:


col_names = list(cars.columns)
type(col_names)
col_names[0] = 'model'
cars.columns = col_names


# In[ ]:


rvar = np.random.randint(0,31,5)
cars.loc[rvar, :]


# In[ ]:


cars.iloc[[4,16],0:11:2]


# In[ ]:





# In[ ]:


df1.iloc[2:4,[2,3]]


# In[ ]:


df1.columns = ['col1','col2','col3','col4','col5']
df1.index = ['row1','row2','row3','row4','row5']


# In[ ]:


df1


# In[ ]:


# 6,7,8, 16,17,18 , 21,22,23

df1.loc[['row2','row4','row5'],'col2':'col4']


# In[ ]:


df1.loc[['row2','row5'],['col1','col3']]


# In[ ]:





# In[ ]:


data[2:4]


# In[ ]:


#DataFrame object, Combining muliple Series as columns
data1 = pd.DataFrame([[2,3,4,5],[8,2,4,2],[12,23,9,3]])
#Assigning labels to columns
data1.columns = ['col 1','col 2','col 3','col 4']
#Assigning labels to rows
data1.index = ['row 1','row 2','row 3']
data1


# In[ ]:


#DataFrame from random numbers generated through numpy
df1 = pd.DataFrame(np.floor(np.random.rand(36).reshape(6,6)*100))
df1


# In[ ]:


get_ipython().run_line_magic('pwd', '')


# In[ ]:


cars = pd.read_csv('mtcars.csv')


# In[ ]:


cars.shape


# In[ ]:


cars.head()


# In[ ]:


cars.isnull().sum()


# In[ ]:


#Importing Data to DataFrame from CSV, comma separated value
cars = pd.read_csv("../input/mtcars.csv")
cars.head()  # head function displays first 5 rows


# In[ ]:


#reassigning column namesm
cars.columns = ['car_model','mpg','cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
cars.head()


# In[ ]:


#Statistical data analysis with pandas.DataFrame.describe()
cars.describe()


# # Data Slicing

# In[ ]:


#retrieving data from DataFrame with index using iloc function
cars.iloc[3:10,2:6]


# In[ ]:


# slicing of Series data with labels
s_data = pd.Series([2,3,4.2,'5',6,9],index=['row 1','row 2','row 3','row 4','row 5','row 6'])
s_data['row 2':'row 5']


# In[ ]:


#observe that start value is excluded unlike label indexer
s_data[2:4]


# In[ ]:


# retrieving data from cars DataFrame 


# In[ ]:


cars.head()


# In[ ]:


#slicing row and columns of dataframe with loc
cars.loc[2:6,'mpg':'hp']


# In[ ]:


#Retrieving multiple columns with list
cars.loc[2:6,['cyl','drat','wt']]


# In[ ]:


cars.loc[:,'hp':]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#leaving empty before slicing operator colon, assumes starting index
cars.loc[2:6,:'wt']


# In[ ]:


cars.loc[[6,5],['wt','hp','car_model']]


# In[ ]:


mpg = cars.mpg
hp = cars.hp
cars[(mpg>25) & (hp>100)]


# In[ ]:


cars[(cars.am==1) & (hp>150)]


# In[ ]:


#leaving empty after slicing operator colon, assumes end index
cars.loc[2:6,'cyl':]


# # Filtering with Scalars

# In[ ]:


cars.head(2)


# In[ ]:


cars.loc[:,'hp']
cars.head(2)


# In[ ]:


cars[(cars.hp>200) & (cars.wt<5)]


# In[ ]:


cars[(cars.mpg>30) & (cars.hp>100)]


# In[ ]:





# In[ ]:





# In[ ]:


manual_powerful_cars = cars[(cars.am==0) & (cars.hp>150)]


# In[ ]:


manual_powerful_cars.head()


# In[ ]:


#Using logical and comparision operators to retrieve required data from DataFrame
auto_cars = cars[(cars.am == 1) & (cars.hp<100) ]
auto_cars.head()


# In[ ]:


#Another example
cars[(cars.hp >110) & (cars.am == 1)]


# # Treating missing values in python

# In[ ]:


#real world data is normaly comes with some missing data. treating them is import part of data preparation
#missing value is represented by NAN. Using numpy.nan to create missing values in cars DataFrame
#cars.loc[2,'hp'] = '?'
#cars.loc[3,'hp'] = '?'
cars.loc[1,'hp'] = '?'
cars.loc[3,'mpg'] = np.nan
cars.loc[2,'mpg'] = np.nan
cars.loc[4,'hp'] = np.nan
cars.loc[4,'wt'] = np.nan

cars.head(5)


# In[ ]:


cars.info()


# In[ ]:


# \ is called as escape char, which means to consider char following
# as char not special one.
cars.hp.str.contains('\?').sum()


# In[ ]:


import numpy as np
cars.hp.replace('?',np.nan,inplace=True)


# In[ ]:


cars.head()


# In[ ]:


cars.loc[1,'wt']=np.nan


# In[ ]:


cars.head(6)


# In[ ]:


cars.isnull()


# In[ ]:


cars.dropna()


# ### Filling missing value with fillna() function

# In[ ]:


cars.isnull().sum()


# In[ ]:


cars.head()


# In[ ]:


# filling missing value with constant. 
# This is not recommended normally as it doesn't approximate nearest value of missing value, resulting poor analysis
cars.fillna(cars.mean())


# In[ ]:


#Treating missing value with forward filling. filling with value above the misssing value
cars.fillna(method='bfill')


# In[ ]:


cars.head()


# In[ ]:


#Treating missing value with back filling. filling with value below the misssing value
cars.fillna(cars.mean())


# In[ ]:


cars.loc[2,:]=np.nan


# In[ ]:


cars.head()


# In[ ]:





# In[ ]:





# In[ ]:


#Treating missing value with mean value.
cars.fillna(cars.mean()) # normally leads better results


# ### Removing rows with missing values with Dropna()

# In[ ]:


nonan_cars = cars.dropna()


# In[ ]:


nonan_cars.shape


# In[ ]:


cars.dropna(inplace=True)  # this drops missing row and replaces result to actual dataframe


# In[ ]:


cars.dropna(how="all") #Only drops row when all columns of that row is empty


# ## Removing Duplicates

# In[ ]:


import pandas as pd
data = pd.DataFrame([[1,2,2,3,3,3,4,4],
                  ['a','b','b','c','c','c','d','c'],
                  ['A','B','B','C','C','C','D','D']])
data = data.transpose()


# In[ ]:


data.columns = ['col 1','col 2', 'col 3']
data


# In[ ]:


data.drop_duplicates()


# In[ ]:


data.drop_duplicates(['col 2','col 3']) #drops rows with same values in all columns


# In[ ]:


data.drop_duplicates(['col 2','col 3']) #drops rows with same values in specified column, col 3 here.


# In[ ]:


#Use drop_duplicates()
data.transpose()


# ## Sorting Data in DataFrame

# In[ ]:


cars.head()


# In[ ]:


#use sort_values to sort dataframe
cars.sort_values(['hp','wt'], ascending=True)


# ## Transforming Data

# In[ ]:


#adding new records
new_data = pd.DataFrame(['Tata Nano',40,2,120,60,2.5,1.1,10,0,0,4,2])
new_data= new_data.transpose()
new_data


# In[ ]:


new_data.columns = cars.columns
new_data.head()


# In[ ]:


cars_updated = pd.concat([cars,new_data], ignore_index=True)
cars_updated


# In[ ]:


cars_updated.loc[0,:]


# ## Droping Data
# 

# In[ ]:


# dropping columns
cars_updated


# In[ ]:


cars.head()


# In[ ]:


np.nan==np.nan


# In[ ]:


cars[cars.hp==cars.isnull()]


# In[ ]:





# In[ ]:


cars.drop('mpg',axis=1)  # axis =1 implies to operate on columns, axis = 0 (default) applies row operation


# # End of Pandas Tutorial

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




