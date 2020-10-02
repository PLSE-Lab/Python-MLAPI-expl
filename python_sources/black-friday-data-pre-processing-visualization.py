#!/usr/bin/env python
# coding: utf-8

# ## Black Friday - Data Pre-processing & Visualization
# 
# ### Pre-Processing Data and Visualization 
# > My main activity using the Black Friday dataset has been *Data Pre-processing* and getting insight through *Visualization*.
# 
# 
# ### Description
# > The dataset here is a sample of the transactions made in a retail store. The store wants to know better the customer purchase behaviour against different products.<br>
# > __Black Friday__ is an informal name for the Friday following Thanksgiving Day in the United States, which is celebrated on the fourth Thursday of November. The day after Thanksgiving has been regarded as the beginning of America's Christmas shopping season since 1952, although the term "Black Friday" didn't become widely used until more recent decades. Black Friday date in 2018 fell on November 23. (Source : Wikipedia)
# 
# > __Additional Use of data__ <br>
# Additionally the data can also be considered for 'Classification', as several variables are categorical, like classifying the age group and category of goods purchased.
# 'Clustering' can also be incorporated from the dataset, providing different clusters based on product category, age, Gender.

# ### Contents
# [1.0 Importing required Python libraries and data](#1.0-Importing-required-libraries-and-data)<br>
#     [1.1 Exploring the data](#1.1-Exploring-the-data)<br>
# <br>
# [2.0 Data Pre-Processing](#2.0-Data-Pre-Processing)<br>
#     [2.1 Renaming columns](#2.1-Rename-columns)<br>
#     [2.2 Encoding categorical variables](#2.2-Encoding-categorical-varaibles-to-numerics)<br>
#     [2.3 Handling Missing Values](#2.3-Handling-Missing-Values)<br>
# <br>
# [3.0 Visualization](#3.0-Visualization)<br>
#    [3.1 Visualisation of correlation](#3.2-Visualization-of-correlation)<br>
# <br>
# [4.0 Conclusion](#4.0-Conclusion)
# 

# ### 1.0 Importing required libraries and data

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Importing data into python from the given csv file
dataset= pd.read_csv('../input/BlackFriday.csv')


# ***

# #### 1.1 Exploring the data 

# In[ ]:


print("****** Dataset - head *****", dataset.head(), sep="\n", end="\n\n\n\n")
print("****** Dataset - tail *****", dataset.tail(), sep="\n", end="\n\n")


# *Observation* <br>
# Data contains missing values (NaN) mainly in *'product category2 and 3'* columns

# In[ ]:


dataset.dtypes


# *Observation* <br>
# Most of the columns are *'numeric* except 5 columns 

# In[ ]:


dataset.info()


# *Observation* <br>
# Missing data in Product_Category_2 and Product_Category_3 columns

# ---

# ### 2.0 Data Pre-Processing

# __Activity__
# 
# After exploring the given data I could think of the following pre-processing activities <br> 
# that can be carried out on the data, to ensure that it is ready for Machine Learning
# 
# 1. Rename column names : Rename the column names to short names without space, so that it would be easy for coding
# 2. Encode category varaibles to numerics, so that the data can be used for any kind of Machine Learning
# 3. Handling missing values : Check whether Product_ID is correlated to Product Categories and fill in appropriate values

# #### 2.1 Rename columns

# In[ ]:


dataset.columns


# In[ ]:


dataset.columns = ['UserID', 'ProductID', 'Gender', 'Age', 'Occupation', 'CityCategory',
       'StayYearsCity', 'MaritalStatus', 'ProdCat1',
       'ProdCat2', 'ProdCat3', 'Purchase']


# In[ ]:


dataset.columns


# In[ ]:


# Replacing 0s and 1s in the Marital status column with the appropriate strings
dataset['MaritalStatus'] = dataset['MaritalStatus'].replace(0, 'Unmarried')
dataset['MaritalStatus'] = dataset['MaritalStatus'].replace(1, 'Married')


# In[ ]:


dataset['MaritalStatus'].unique()


# __Create copy of dataset 'before encoding'__
# > Creating a copy of the dataset before encoding the values <br>
# This is mainly done to use the values for 'visualization' purpose, with actual values

# In[ ]:


dataset_orig = dataset.copy()


# #### 2.2 Encoding categorical varaibles to numerics

# In[ ]:


# Importing required package
from sklearn.preprocessing import LabelEncoder
encode_x = LabelEncoder()


# In[ ]:


dataset.head()


# In[ ]:


# Encoding columns ProductID, 
dataset['ProductID'] = encode_x.fit_transform(dataset['ProductID'])
dataset['Gender'] = encode_x.fit_transform(dataset['Gender'])
dataset['Age'] = encode_x.fit_transform(dataset['Age'])
dataset['CityCategory'] = encode_x.fit_transform(dataset['CityCategory'])
dataset['MaritalStatus'] = encode_x.fit_transform(dataset['MaritalStatus'])


# In[ ]:


dataset.StayYearsCity.unique()


# In[ ]:


# Replacing '4+' years of with numerical number 4 
dataset['StayYearsCity'] = dataset['StayYearsCity'].replace('4+', 4)


# In[ ]:


dataset.info()


# In[ ]:


# Converting StayYearsCity from object to integer
dataset['StayYearsCity'] = dataset['StayYearsCity'].astype(str).astype(int)


# In[ ]:


dataset.info()


# #### 2.3 Handling Missing Values

# Examining __whether Product Category 2 and 3 are correlated__ to the __Product ID__

# In[ ]:


# Creating list of index 'without' null values on column ProdCat2
no_null_list1 = dataset[~dataset['ProdCat2'].isnull()].index.tolist()


# In[ ]:


dataset['ProductID'][no_null_list1].corr(dataset['ProdCat2'][no_null_list1])


# In[ ]:


# Creating list of index 'without' null values on column ProdCat2
no_null_list2 = dataset[~dataset['ProdCat3'].isnull()].index.tolist()


# In[ ]:


dataset['ProductID'][no_null_list2].corr(dataset['ProdCat3'][no_null_list2])


# __Observation__ : <br>
# From the correlation function, it can be concluded that the 'Product ID' is __NOT correlated__ to <br>
# either ProdCat2 or ProdCat3

# In[ ]:


print("Missing Product Category2 values :", len(dataset)-len(no_null_list1))
print("Missing Product Category3 values :", len(dataset)-len(no_null_list2))


# In[ ]:


# Checking values contained in ProdCat2 and ProdCat3
dataset['ProdCat2'].unique()


# In[ ]:


dataset['ProdCat3'].unique()


# __Conclusion__ : <br>
#  It appears that the values in ProdCat2 and ProdCat3 are numeric categories. <br>
#  Hence, replacing the 'NaN' with 0s and can be assumed as 'missing' category value.

# In[ ]:


dataset['ProdCat2'].fillna(value=0,inplace=True)
dataset['ProdCat3'].fillna(value=0,inplace=True)


# In[ ]:


# Recheck for missing values (NaN) in the dataset
dataset.isna().any()


# ### 3.0 Visualization

# Let us visualize the plots based on the categorical variables, 
# 1. Gender
# 2. Age
# 3. Occupation 
# 4. City Category

# In[ ]:


# Obtaining categorical data in terms of Percentage for each column 
group_1 = dataset_orig.groupby(['Gender'])
group_2 = dataset_orig.groupby(["Age"])
group_3 = dataset_orig.groupby(["CityCategory"])
group_4 = dataset_orig.groupby(["Occupation"])

print (group_1[['Purchase']].count()/len(dataset_orig)*100, end="\n\n\n\n")
print (group_2[['Purchase']].count()/len(dataset_orig)*100, end="\n\n\n\n")
print (group_3[['Purchase']].count()/len(dataset_orig)*100, end="\n\n\n\n")
print (group_4[['Purchase']].count()/len(dataset_orig)*100, end="\n\n\n\n")


# Visualization :  __By count__  

# In[ ]:


plt.figure(figsize=(15,10))

# Pie chart for gender distribution
plt.subplot(2,2,1)
gender_count = [dataset_orig.Gender[dataset_orig['Gender']=='F'].count(),
                dataset_orig.Gender[dataset_orig['Gender']=='M'].count()]
gender_lab = dataset_orig.Gender.unique()
expl = (0.1,0)
plt.pie(gender_count, labels=gender_lab, explode=expl, shadow=True , autopct='%1.1f%%');

# Bar chart for Age
plt.subplot(2,2,2)
ordr =dataset_orig.groupby(["Age"]).count().sort_values(by='Purchase',ascending=False).index
sns.countplot(dataset_orig['Age'], label=True, order=ordr)

# Bar chart for Occupation
plt.subplot(2,2,3)
ordr1 =dataset_orig.groupby(["Occupation"]).count().sort_values(by='Purchase',ascending=False).index
sns.countplot(y=dataset_orig['Occupation'], label=True, order=ordr1)

# Donut chart for City Category
plt.subplot(2,2,4)
city_count = group_3[['Purchase']].count().values.tolist()
city_lab = dataset_orig.groupby(["CityCategory"]).count().index.values
my_circle = plt.Circle( (0,0), 0.4, color='white')
expl1 = (0,0.1,0)
plt.pie(city_count, labels=city_lab,explode=expl1, shadow=True, autopct='%1.1f%%')
plt.gcf().gca().add_artist(my_circle)


plt.show()


# In[ ]:


plt.figure(figsize=(8,6))
ordr2 =dataset_orig.groupby(["StayYearsCity"]).count().sort_values(by='Purchase',ascending=False).index
sns.countplot(dataset_orig['StayYearsCity'], label=True, order=ordr2)
plt.show()


# __Observations__ : 
# 
# 1. There are __75.4% males__ and 24.6% female buyers from the store
# 2. The majority __(40%) of buyers__ are in between the __age group of 26-35__
# 3. Major buyers are from the __City category C (31%) and B (42%)__
# 4. __Occupation__ code __4 (13%), 0 (12.7%) and 7(10.8%)__ are the major buyers
# 5. It can be observed that, as people __who are new__ in the the current city, they tend to visit the mall more frequently.

# Visualization - __more insights__ <br>
# 
# 1. Combining __Age and Gender__  and also  __Gender and Marital Status__ 
# 2. Combining __Gender and Marital Status__ to create new column indicating both Gender and Marital status

# In[ ]:


#Creating new column in the dataset 
dataset_orig['Gender_MaritalStatus'] = dataset_orig.apply(lambda x:'%s_%s' % (x['Gender'],x['MaritalStatus']),axis=1)


# In[ ]:


dataset_orig.Gender_MaritalStatus.unique()


# In[ ]:


group_5 = dataset_orig.groupby(["Gender_MaritalStatus"])


# In[ ]:


plt.figure(figsize=(15,10))

plt.subplot(1,2,1)
count1 = group_5[['Purchase']].count().values.tolist()
lab1 = dataset_orig.groupby(["Gender_MaritalStatus"]).count().index.values
expl2 = (0,0,0.1,0.1)
plt.pie(count1, labels=lab1,explode=expl2, shadow=True, autopct='%1.1f%%')

plt.subplot(1,2,2)
sns.countplot(dataset_orig['Age'],hue=dataset_orig['Gender_MaritalStatus'])

plt.show()


# __Observations__ : 
# 1. Males (un-married and married) between the age group of 26-35 are major buyers
# 2. Un-married female buyers are more in number than married ones

# Visualization :  __By Average Purchase__  

# In[ ]:


# Bar chart for Age

sns.catplot(x='Gender_MaritalStatus', y='Purchase', data=dataset_orig, kind='boxen')

ordr_occ =dataset_orig.groupby(["Age"]).mean().sort_values(by='Purchase',ascending=False).index
sns.catplot(x='Age', y='Purchase', order=ordr_occ, data=dataset_orig, kind='bar')

ordr_occ =dataset_orig.groupby(["Occupation"]).mean().sort_values(by='Purchase',ascending=False).index
sns.catplot(x='Occupation', y='Purchase', order=ordr_occ, data=dataset_orig, kind='bar')

sns.catplot(x='CityCategory', y='Purchase', data=dataset_orig, kind='boxen')


plt.show()


# __Observations__ : 
# 1. __Males__ spend more than females
# 2. Age factor : People who have earned more (*spend more time working*) and have saved enough money, tend to spend more
# 3. __Occupation codes 17, 12 and 15__ appears to be earning more and accordingly spend more 
# 4. People living in __Category C__ city appears to spend more in the mall
# 

# #### 3.2 Visualization of correlation
# 

# In[ ]:


corrmat = dataset.corr()


# In[ ]:


f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, cmap="YlGnBu", square=True,linewidths=.5, annot=True)
plt.show()


# In[ ]:


dataset[dataset.columns[0:]].corr()['Purchase'].sort_values(ascending=False)


# In[ ]:


# Obtaining top K columns which affects the Purchase the most
k= 8
corrmat.nlargest(k, 'Purchase')


# In[ ]:


# Replotting the heatmap with the above data
cols = corrmat.nlargest(k, 'Purchase')['Purchase'].index
cm = np.corrcoef(dataset[cols].values.T)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(cm, cmap="YlGnBu", cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# __Observation__ :
# From the above correlation matrix and plot we understand that the __Gender, CityCategory and Product Category 3__ are the three (3) main parameters relating to the Purchase 

# ### 4.0 Conclusion

# 1. Based on the above visualizations and correlation matrix, it appears that the __customer purchase behaviour__ mainly depends on __4 major factors__ ; <br>
#     a. __City Category__ (Category C and B are more in number and spend more on items)<br> 
#     b. __Gender__ (Males (esp. unmarried) buy more than females) <br>
#     c. __Occupation__ (codes 4,0,7 are regular buyers, codes 17,12,15 spend more on items) <br>
#     d. People __Aged__ between 26-35 visit more and also spend more more on items <br>
# 2. By observing the dataset, it appears that column Product Category 1 is the 'main' category type and other 2 columns indicating  Product categories (2 & 3) are sub categories of the product. As there were quite a few missing values in Product category2 and Product Category3 (replaced by 0s) no particular insight can be provided.

# In[ ]:




