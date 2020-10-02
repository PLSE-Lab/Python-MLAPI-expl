#!/usr/bin/env python
# coding: utf-8

# Importing the libraries

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# Importing the dataset

# In[ ]:


df = pd.read_csv("../input/blore_apartment_data.csv")


# Have a look at dataset

# In[ ]:


df.head()


# Checking for null values

# In[ ]:


df.isnull().values.any()


#  Checking how many null are in the columns

# In[ ]:


df.isnull().sum()


# Removing nulls from all columns

# In[ ]:


df = df.dropna(how='any',axis=0)


# Check Price Columns

# In[ ]:


df['Price'].value_counts()


# To remove units, and further manipulation,converting column into list

# In[ ]:


PriceList = list(df['Price'])


# User defined function for removing the '-' between the values and save it into two lists.

# In[ ]:


def sep(li):
    newli1=[]
    newli2=[]
    for i in range(len(li)):
        text = li[i]
        head, sep, tail = text.partition('-')
        newli1.append(head)
        newli2.append(tail)
    return newli1,newli2


# In[ ]:


Min,Max= sep(PriceList)


# User defined function to remove L,Cr & K, and add respective numbers so further calculation can be done

# In[ ]:


def Converter(li):
    newli=[]
    for i in range(len(li)):
        if 'L' in li[i]:
            text = li[i]
            li[i] = li[i].replace('L',' ')
            li[i] = float(li[i])
            li[i] = li[i]*100000
            li[i] = int(li[i])
            newli.append(li[i])
            li[i] = str(li[i])
        elif 'K' in li[i]:
            text = li[i]
            li[i] = li[i].replace('K',' ')
            li[i] = float(li[i])
            li[i] = li[i]*1000
            li[i] = int(li[i])
            newli.append(li[i])
            li[i] = str(li[i])
        elif 'Cr' in li[i]:
            text = li[i]
            li[i] = li[i].replace('Cr',' ')
            li[i] = float(li[i])
            li[i] = li[i]*10000000
            li[i] = int(li[i])
            newli.append(li[i])
            li[i] = str(li[i])
        else:
            newli.append(li[i])
    return newli


# In[ ]:


MinRange = Converter(Min)
MaxRange = Converter(Max)     


# Adding these two list to the dataframe

# In[ ]:


df['MinRange'] = MinRange
df['MaxRange'] = MaxRange


# Convert them into numeric type 

# In[ ]:


df[["MinRange", "MaxRange"]] = df[["MinRange", "MaxRange"]].apply(pd.to_numeric)


# Taking the mean of Minimum & Maximum Range and adding new column to the dataframe named Average Price

# In[ ]:


col = df.loc[: , "MinRange":"MaxRange"]
df['AveragePrice'] = col.mean(axis=1)


# Removing the Minimum Range, Maximum Range and Price column from the dataframe

# In[ ]:


df = df.drop(['MaxRange','MinRange','Price'], axis=1)


# Now have a look at dataframe

# In[ ]:


df.head()


# Now, Removing Area Column's Unit sq.ft for calculations, for this a function is defined.

# In[ ]:


def AreaConverter(li):
    newli=[]
    for i in range(len(li)):
        if 'sq.ft' in li[i]:
            text = li[i]
            li[i] = li[i].replace('sq.ft','')
            newli.append(li[i])
        else:
            newli.append(li[i])
    return newli


# Converting Area column into list, and list is the parameter in the defined function

# In[ ]:


AreaList = list(df['Area'])
AreaWithoutUnit = AreaConverter(AreaList)


# Have a look at AreaWithoutUnit 

# In[ ]:


AreaWithoutUnit 


# Extracting minimum & maximum range from Area list using function 'sep'

# In[ ]:


Min,Max= sep(AreaWithoutUnit)


# Adding these columns in the dataset

# In[ ]:


df['MinArea'] = Min
df['MaxArea'] = Max


# Taking the mean of these ranges and assign it to a new column named AverageArea

# In[ ]:


df['AverageArea'] = df[['MinArea','MaxArea']].mean(axis=1)


# Dropping the columns

# In[ ]:


df = df.drop(['MinArea','MaxArea','Area'], axis=1)


# Now have a look at the dataset

# In[ ]:


df.head()


# Converting the UnitType into List 

# In[ ]:


UnitTypeList= list(df['Unit Type'])


# User defined function for adding not bhk/Rk, if its value is "plot"

# In[ ]:


def BHK(li):
    newli = []
    for i in range(len(li)):
        if 'Plot' in li[i]:
            li[i] = str("0 Not:BHK/RK ") + li[i]
            newli.append(li)
        else:
            newli.append(li)
    return newli


# In[ ]:


BHK1 = BHK(UnitTypeList)


# In[ ]:


def Unitsep(li):
    newli1=[]
    newli2=[]
    for i in range(len(li)):
        text = li[i]
        head, sep, tail = text.partition(' ')
        newli1.append(head+sep)
        newli2.append(tail)
    return newli1,newli2


# Extracting the values from UnitType list

# In[ ]:


NoUnitType,Uni2 = Unitsep(UnitTypeList)
BHKorRK,UnitType1 = Unitsep(Uni2)


# Adding the columns to the dataframe

# In[ ]:


df['UnitNo'] = NoUnitType
df['BHKorRK'] = BHKorRK
df['UnitType']= UnitType1


# Removing addition values to the columns

# In[ ]:


df = df[df.UnitNo != 'Studio ']
df= df[df.UnitNo != 'Apartment']
df= df[df.UnitType != 'BHK Apartment']


# Removing Unit Type Column

# In[ ]:


df = df.drop(['Unit Type'], axis=1)


# Have a look at dataset

# In[ ]:


df.head()


# In[ ]:


df['UnitNo'].value_counts()


# Changing record '4+' into '4' for calculation

# In[ ]:


df = df.replace('4+ ','4.5')
df.UnitNo = df.UnitNo.astype(float)
df['UnitNo'].value_counts()


# In[ ]:


df.head()


# Linear Regression

# In[ ]:


dataset = df[['names','UnitNo','BHKorRK','UnitType','AverageArea','AveragePrice']]


# In[ ]:


dataset.head()


# Using the variables AverageArea and AveragePrice for simple linear regression

# In[ ]:


X = dataset.iloc[:, 4:5].values
y = dataset.iloc[:, 5:].values


# Splitting the dataset into the Training set and Test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# 
# Fitting Simple Linear Regression to the Training set

# X_train = X_train.reshape(1,-1)
# y_train = y_train.reshape(1,-1)
# 

# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results

# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price vs Area (Training set)')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()


# In[ ]:


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price vs Area (Test set)')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2_score(y_test, y_pred)

