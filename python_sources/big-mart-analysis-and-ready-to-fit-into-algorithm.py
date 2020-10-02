#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing basic libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


train= pd.read_csv('/kaggle/input/bigmart-sales-data/Train.csv')
test = pd.read_csv('/kaggle/input/bigmart-sales-data/Test.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


#most of the items has non - zero values or non null
train.describe()


# In[ ]:


train.columns


# In[ ]:


Uniqueid =len(set(train.Item_Identifier))  # set removes dupplicate values


# In[ ]:


Totalid =train.shape[0]
Duplicate = Totalid - Uniqueid
print("no . of duplicate id is ",Duplicate)


# In[ ]:


#histogram of Item_outlet_sales vs Number of sales
plt.figure(figsize= (13,6))
sns.distplot(train.Item_Outlet_Sales,bins = 25)
plt.ticklabel_format(style = 'plain',axis = 'x',scilimits=(0,1))
plt.xlabel('Item_Outletsales')
plt.ylabel('number of sales')
plt.show()


# In[ ]:


# so more the item outlests the sale is decreasing
#skewness - > distortion in data
print('the skewness is',train.Item_Outlet_Sales.skew())


# so our target variable is skewwed to the right
# 

# after this lets see which of pur features are numeric
# 

# In[ ]:


nm_features =train.select_dtypes(include=[np.number])
nm_features.dtypes


# In[ ]:


#relation between numerical predictors and target variable
correlation = nm_features.corr()
correlation


# In[ ]:


correlation['Item_Outlet_Sales'].sort_values


# In[ ]:


sns.countplot(train.Item_Fat_Content)


# In[ ]:


#for Item_Type
sns.countplot(train.Item_Type)
plt.xticks(rotation=90)


# In[ ]:


#for Outlet_Size
sns.countplot(train.Outlet_Size)


# In[ ]:


sns.countplot(train.Outlet_Location_Type)


# In[ ]:


sns.countplot(train.Outlet_Type)
plt.xticks(rotation=90)


# In[ ]:


#Item_Weight vs Item_Sales
plt.figure(figsize=(12,8))
plt.xlabel("Item weight")
plt.ylabel('Item sales')
plt.title('Item weight vs Item sales')
plt.plot(train.Item_Weight,train.Item_Outlet_Sales,'.',alpha = 0.4)#alpha = size of dots


# In[ ]:


plt.xlabel('Item_Visibility')
plt.ylabel('Item outlet sales')
plt.title('Item visibility vs Item outlet sales')
plt.plot(train.Item_Visibility,train.Item_Outlet_Sales,'.',alpha= 0.4)


# In[ ]:


#Item_Fat_Content vs Item_Outlet_Sales
#Create a spreadsheet-style pivot table as a DataFrame. pivot_table

Item_Fat_Content_pivot=train.pivot_table(index ='Item_Fat_Content',values ='Item_Outlet_Sales',aggfunc=np.median)
Item_Fat_Content_pivot.plot(kind = 'bar',color = 'blue',figsize=(12,7))
plt.xlabel('Item_Fat_Content')
plt.ylabel('Item_Outlet_Sales')
plt.show()


# In[ ]:


#impact of Outlet_identifier on Item_outlet_sales
Outlet_Identifier_pivot = train.pivot_table(index ='Outlet_Identifier',values = 'Item_Outlet_Sales',aggfunc=np.median)
Outlet_Identifier_pivot.plot(kind ='bar',color = 'red')
plt.xlabel('Outlet_identifier')
plt.ylabel('Item_outlet_sales')
plt.show()


# we can see thet outlet 27  had the highest sales  except 2 all others had median sales
# 

# In[ ]:


train.pivot_table(values='Outlet_Type',columns = 'Outlet_Identifier',aggfunc=lambda x:x.mode())


# In[ ]:


train.pivot_table(values='Outlet_Type',columns = 'Outlet_Size',aggfunc=lambda x:x.mode())


# impact of outlet size on item outlet sales
# 

# In[ ]:


Outlet_size_pivot = train.pivot_table(index='Outlet_Size',values='Item_Outlet_Sales',aggfunc=np.median)
Outlet_size_pivot.plot(kind = 'bar',color = 'red', )
plt.xlabel('Outlet size')
plt.ylabel('Item outlet sales')
plt.show()


# so medium size supermarkets had the highest sales compared to high ones and the small ones
# 

# In[ ]:


#outlet_type vs item outlet sales


# In[ ]:


Outlet_type_pivot= train.pivot_table(index ='Outlet_Type',values='Item_Outlet_Sales',aggfunc = np.median)
Outlet_type_pivot.plot(kind = 'bar', color = 'red')
plt.xlabel('Outlet_type')
plt.ylabel('Item_Outlet_Sales')
plt.show()


# it shows that supermarket type 3 has highest outlety types

# In[ ]:


#outlet location type vs item outlet saled
outlet_location_type_pivot = train.pivot_table(index='Outlet_Location_Type',values = 'Item_Outlet_Sales',aggfunc=np.median)
outlet_location_type_pivot.plot(kind= 'bar',color = 'red')
plt.xlabel('Outlet location')
plt.ylabel('item outlet sales')
plt.show()


# supermarket of type 2 location had the highest outlet sales

# for cleaning the data we have to merge the data clean it then again we can split the data 
# instead of doing everything twice

# In[ ]:


#joining the data
train['source']= 'train'
test['source'] ='test'
data = pd.concat([train,test],ignore_index=True,sort=True)


# In[ ]:


data.shape


# In[ ]:


data.info


# In[ ]:


data.info()


# In[ ]:


#checking the percentage of non null values
data.isnull().sum()


# In[ ]:


#calculating mean value for replacing null values in null value
avg_mean=data.pivot_table(values = 'Item_Weight',index ='Item_Identifier')
print(avg_mean)


# In[ ]:


def replace_null(cols):
    weight = cols[0]
    identifier =cols[1]
    if pd.isnull(weight):
        return  avg_mean['Item_Weight'][avg_mean.index==identifier]
    else:
        return weight
print("original number of null values",sum(data['Item_Weight'].isnull()))

data['Item_Weight'] =data[['Item_Weight','Item_Identifier']].apply(replace_null,axis=1).astype(float)
print('final number of null values',sum(data['Item_Identifier'].isnull()))


# replacing null values of Outlet_size with mode  #mode used when there is more frequent values

# In[ ]:


from scipy.stats import mode  # scipy = more mathematical functions than numpy
outlet_size_mode  = data.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc =lambda x:x.mode())
outlet_size_mode


# In[ ]:


#to calculate the average value
visibility_item_avg = data.pivot_table(values='Item_Visibility', index='Item_Identifier')


# In[ ]:


#replacing null values in outlet_size
def replace_outlet(cols):
    size=cols[0]
    Type = cols[1]
    if pd.isnull(size):
        return outlet_size_mode.loc['Outlet_Size'][outlet_size_mode.columns==Type][0]
    else:
        return size
print("original number of null values",sum(data['Outlet_Size'].isnull()))
data['Outlet_Size']=data[['Outlet_Size','Outlet_Type']].apply(replace_outlet,axis =1)
print('final data with null vakues',sum(data['Outlet_Size'].isnull()))


# previously we saw that item_visibility has minimum value of 0 which implies that is hidden which doesn't make sense replacing those values with mean values
# 

# In[ ]:


def replace_visibility(cols):
    visibility = cols[0]
    item = cols[1]
    if visibility ==0:
        return visibility_item_avg['Item_Visibility'][visibility_item_avg.index==item]
    else:
        return visibility
print('original numbers with 0 as values',sum(data['Item_Visibility']==0))
data['Item_Visibility']=data[['Item_Visibility','Item_Identifier']].apply(replace_visibility,axis=1).astype(float)
print('final number with 0 as its value',sum(data['Item_Visibility']==0))


# the data is very old from 2013 

# In[ ]:


data['Outlet_years']=2013-data['Outlet_Establishment_Year']
data['Outlet_years'].describe()


# item_type has 14 categories which is a lot to process individually 
# so combining it

# In[ ]:


data['ItemType_Combined']= data['Item_Identifier'].apply(lambda x:x[0:2])
data['ItemType_Combined']= data['ItemType_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
data['ItemType_Combined'].value_counts()


# In[ ]:


#there was some typos in item_fat_content fixing it
data['Item_Fat_Content']=data['Item_Fat_Content'].replace({'LF':'Low fat','reg':'Regular','low fat':'Low fat'})


# but again there are some non consumable product also so creating a new column 

# In[ ]:


data.loc[data['ItemType_Combined']=="Non-Consumable",'Item_Fat_Content']='Non-Edible'


# creating a variable for visibility

# In[ ]:


func = lambda x:x['Item_Visibility']/visibility_item_avg['Item_Visibility'][visibility_item_avg.index==x['Item_Identifier']][0]
data['Item_Visibility_MeanRatio']= data.apply(func,axis=1).astype(float)
data['Item_Visibility_MeanRatio'].describe()


# we all know that scikit only works with numerical values so converting using
# LabelEncoder()

# In[ ]:


from sklearn.preprocessing  import LabelEncoder
Le = LabelEncoder()
#for outlet
data['outlet'] =Le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','ItemType_Combined','Outlet_Type','outlet']
for i in var_mod:
    data[i]=Le.fit_transform(data[i])


# we have done something called as One-Hot-Coding -> creating a dummy variables one for each type of type variable

# In[ ]:


data = pd.get_dummies(data,columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','ItemType_Combined','Outlet_Type','outlet'])
data.dtypes


# In[ ]:


data.drop(['Item_Type','Outlet_Establishment_Year'],axis =1,inplace = True)
#dividing into trains and test
train= data.loc[data['source']=='train']
test = data.loc[data['source']=='test']
train.drop(['source'],axis =1,inplace=True)
test.drop(['source'],axis =1,inplace = True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




