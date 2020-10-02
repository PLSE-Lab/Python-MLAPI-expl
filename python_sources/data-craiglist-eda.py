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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('ggplot') # theme for plot's result


# ### Importing data craiglist

# In[ ]:


data=pd.read_csv("/kaggle/input/craigslist-carstrucks-data/craigslistVehiclesFull.csv")
# Data2 used for comparison when dealing with missing values
data2=pd.read_csv("/kaggle/input/craigslist-carstrucks-data/craigslistVehiclesFull.csv")
data.head(5)


# Based on above, we can see that there are many <i>NaN</i> values. Even if Exploratory analysis can be done with missig values data, it is best to deal with them when further analyze the data

# In[ ]:


data.info()


# In[ ]:


sum_null=data.isnull().sum()
sum_null=pd.DataFrame(sum_null,columns=['null'])
j=1
sum_tot=len(data)
sum_null['percent']=sum_null['null']/sum_tot
round(sum_null,3).sort_values('percent',ascending=False)


# Above, we can see the percentage of missing values in each columns. Because there are more than 60% missing values for columns "size" and "vin", we drop this columns. <br>
# 

# In[ ]:


data=data.drop(columns=['size',"vin"],axis=1)


# <br>
# For more than 30% missing values, we replace them with mode() for categorical data and with median for numeric data as the data mostlikely to skewed
# <br>
# First we split categorical data (object type) and numerical data

# In[ ]:


numer=data.columns[[2,3,9,16,17,18,20,23]]
data_numer=data[numer]
data_categ=data.drop(columns=numer,axis=1)
categ=data_categ.columns
sum_null.loc[numer,:].sort_values(by='percent',ascending=False)


# Replace missing values in numeric using median. Noted that for 'odometer' with more than 30% missing values, it is not recomended to replace missing values with median (we analyze further below) but for right now we replace it just like other numerical columns

# In[ ]:


null_numer=sum_null.loc[numer,:][sum_null['percent']>0].index
data[null_numer]=data[null_numer].fillna(data[null_numer].median())
data[null_numer].isnull().sum()


# In[ ]:


data[numer].head(20)


# After we replace missing values, we realize something. there are value "0" in column 'odomotor'. Because compared to other element in 'odometer' there are not any zero values, and it is ilegal to reset odometer and even new car does not has zero odometer, we suspect this was caused by input error (or some fraud) in the data, so we also replace it with median . We see below that there are 5460 zero values in 'odometer'

# In[ ]:


data_numer=data[null_numer]
data_numer['state_fips'].value_counts()
data_numer=data_numer.astype('int64')
for i in data_numer.columns:
    if data_numer[i][data_numer[i]==0].count()>0:
        print('There are %d zero values in %s' %(data_numer[i][data_numer[i]==0].count(),i))
data_numer['odometer']=data_numer['odometer'].replace(0,data_numer['odometer'].median())
data['odometer']=data_numer['odometer']


# In[ ]:


data['odometer'][data['odometer']==0]


# Now we create corelation between them

# In[ ]:


#create correlation with hitmap

#create correlation
corr = data[numer].corr(method = 'pearson')

#convert correlation to numpy array
mask = np.array(corr)

#to mask the repetitive value for each pair
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots(figsize = (20,5))
fig.set_size_inches(20,10)
sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True)
plt.show()


# We see that 'lat' and 'weather' has strong corelation. We can also see that couty-fips and state_fips has corelation one. This was caused by similiar variable or maybe country_fips is just another descriptive of state_fips. So we drop one of this variable. 
# 

# In[ ]:


data=data.drop(columns='county_fips',axis=1)
data2=data2.drop(columns='county_fips',axis=1)
numer=numer.drop('county_fips')
null_numer=null_numer.drop('county_fips')


# 
# Below, we see the distribution of odometer of replaced missing values vs original data. there are no major difference between them

# In[ ]:


f=plt.figure(figsize=(20,5))
f.add_subplot(1,2,1)
sns.distplot(data['odometer'], kde = True, color = 'darkblue', label = 'odometer').set_title('Distribution Plot of odometer')
f.add_subplot(1,2,2)
sns.distplot(data2['odometer'][data2['odometer'].notnull()], kde = True, color = 'darkblue', label = 'odometer').set_title('Distribution Plot of odometer')

plt.show()


# We plot weather and odometer below, and compared it with original data. because there are more than 30% missing values replaced by median, we see that there are anomalies with replaced missing values. We also see that in original data, the pattern between 'weather' and 'odometer' is visible (and this was not the case in replaced missing values data). This is the reason why sometimes, replacing missing values is not a good idea

# In[ ]:


f=plt.figure(figsize=(10,10))
f.add_subplot(1,2,1)
plt.scatter(data['lat'],data['weather'])
f.add_subplot(1,2,2)
plt.scatter(data2['lat'],data2['weather'])
            
plt.show()


# In[ ]:


data[null_numer]=data[null_numer].astype('int64')


# In[ ]:


f=plt.figure(figsize=(25,25))
j=1
for i in numer[[5,6,1]]:
    f.add_subplot(3,1,j)
    sns.countplot(data[i],order=data[i].value_counts().index)
    if i=='year':  
        plt.xticks(rotation=90)
    plt.xticks(fontsize=10)
    j+=1
    
plt.show()


# Above, we can see the comparisson of each state_fips, weather, or year on each othet, For example we can see year 2007 has the most frequent year in this data.

# In[ ]:


f=plt.figure(figsize=(25,25))
j=1
for i in numer[[0,5,6,1]]:
    f.add_subplot(4,1,j)
    sns.distplot(data[i])
    if i=='year':  
        plt.xticks(rotation=90)
    plt.xticks(fontsize=10)
    j+=1
    
plt.show()


# Above is distribution plot of some of the columns. there are many outliers especialy in year and price which make it harder for us to analyze this distribution plot. So, we tried to slice the data to remove outlier or extreme values

# In[ ]:


round(data['price'].describe(),2)


# In[ ]:


data['price'][data['price']> 1.499900e+04].count()


# There are more than 428000 elements with more than 3rd quartil in price. We want to see belom 3rd quartil distribution of price

# In[ ]:


price=data['price'][data['price']<=round(data['price'].describe(),2)[-2]]
price.describe()


# In[ ]:


sns.distplot(price)
plt.show()


# Under 3rd quartil, we see that the price has decrease pattern

# In[ ]:


sns.distplot(data['price'][data['price']>150000])
plt.show()


# While above 3rd quartil, the patter sem hard to see. For us to make it easier to understand, we use belom 3rd quartil for further analysis

# For 'year', noted below result

# In[ ]:


data['year'][data['year']<1900].sort_values(ascending=False)


# In[ ]:


data['year'].describe()


# Same as price we use above 1st quartil. Noted that there are few data below 1900, so we use above 1900 for our futher analysis after this

# In[ ]:


year=data['year'][data['year']>=data['year'].describe()[-4]]
year.describe().astype('int64')


# In[ ]:


f=plt.figure(figsize=(20,10))
f.add_subplot(1,2,1)
sns.distplot(year.astype('int'))
f.add_subplot(1,2,2)
sns.countplot(year.astype('int'),order=year.value_counts().index)
plt.show()


# Above wee see that in year above 2002, the data is kinda has middle distribution

# In[ ]:


f=plt.figure(figsize=(20,10))
price_year=data[['price','year']]
price_year=price_year[price_year['year']>=1900]
price_year=price_year[price_year['price']<150000]
p_y=price_year.groupby('year').mean()
p_y.reset_index(level=0,inplace=True)
f.add_subplot(2,1,1)
plt.bar(p_y['year'].astype('int'),p_y['price'],color='green')
plt.xticks(rotation=90,fontsize=10)
f.add_subplot(2,1,2)
plt.plot(p_y['year'].astype('int'),p_y['price'],color='green')
plt.xticks(rotation=90,fontsize=10)
plt.show()


# Above, we see timeseries of price on each year above 1900. Old car seem has higher price 

# In[ ]:


p_y.sort_values('price',ascending=False).head(20)


# Above is the most expensive year in this data

# # For Category

# In[ ]:


data_categ.head(10)


# In[ ]:


data_categ=data[categ]
data_categ=data_categ.fillna(data_categ.mode().loc[0,:])
data[categ]=data_categ


# In[ ]:


data.isnull().sum()


# In[ ]:


categ


# In[ ]:


city=data['city'].value_counts()
city
x=range(470)
plt.bar(x,city)

plt.show()



# In[ ]:


data[categ].describe()


# In[ ]:


manufac=data['manufacturer'].value_counts().head(15)
plt.figure(figsize=(10,5))
plt.bar(manufac.index,manufac)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)



# In[ ]:


make=data['make'].value_counts().head(30)
plt.figure(figsize=(10,5))
sns.barplot(make.index,make)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)


# In[ ]:


make=data['state_code'].value_counts()
plt.figure(figsize=(20,5))
sns.barplot(make.index,make)
plt.xticks(rotation=90,fontsize=17)
plt.yticks(fontsize=15)


# In[ ]:


make=data['state_name'].value_counts()
plt.figure(figsize=(20,5))
sns.barplot(make.index,make)
plt.xticks(rotation=90,fontsize=17)
plt.yticks(fontsize=15)


# In[ ]:


f=plt.figure(figsize=(10,30))
j=1
for i in categ[4:11]:
    f.add_subplot(7,1,j)
    sns.countplot(data[i],order=data[i].value_counts().index)
    j+=1
plt.show()


# In[ ]:


box=data[categ[4:11]]
box['price']=data['price']
box=box[box['price']<150000]
j=1    
f=plt.figure(figsize=(20,30))
for i in categ[4:8]:
    
    f.add_subplot(4,1,j)
    sns.boxplot(box[i],box['price'])
    j+=1
plt.show()


# In[ ]:


j=1    
f=plt.figure(figsize=(20,25))
for i in categ[8:11]:
    
    f.add_subplot(3,1,j)
    sns.violinplot(box[i],box['price'])
    j+=1
plt.show()


# In[ ]:


citman=data[['state_name','manufacturer','price']]
citman=citman[citman['price']<=150000]

f=plt.figure(figsize=(20,20))
f.add_subplot(2,1,1)
manu=citman[['manufacturer','price']].groupby('manufacturer').mean().sort_values('price',ascending=False).head(30)
manu.reset_index(level=0,inplace=True)
plt.bar(manu['manufacturer'],manu['price'])
plt.xticks(rotation=90,fontsize=15)

f.add_subplot(2,1,2)
state=citman[['state_name','price']].groupby('state_name').mean().sort_values('price',ascending=False).head(30)
state.reset_index(level=0,inplace=True)
plt.bar(state['state_name'],state['price'])
plt.xticks(rotation=90,fontsize=15)
plt.subplots_adjust(hspace = 0.5)

plt.show()


# In[ ]:



try2=data[['type','fuel','price']][data['price']<=150000].groupby(['type','fuel']).mean()
try2.reset_index(level=0,inplace=True)
try2.reset_index(level=0,inplace=True)
try2=try2.sort_values(['type','fuel'])
plt.figure(figsize=(15,10))
sns.barplot(x='type', y='price', hue='fuel', data=try2)
plt.xticks(rotation=90,fontsize=20)
plt.ylabel('Returns',fontsize=20)
plt.legend(fontsize=20)
plt.title('Price of type of cars for each type of fuel');

