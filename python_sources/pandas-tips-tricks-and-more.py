#!/usr/bin/env python
# coding: utf-8

# # Pandas Tips & Tricks & More
# 
# ### Hello Kaggler!
# ### <span style="color:PURPLE">Objective of this kernal is to demonstrate most commonly used</span> <span style="color:red">Pandas Tips & Tricks and More</span> .

# # Contents
# 
# 1. [Check Package Version](#CheckPackageVersion)
# 1. [Ignore Warnings](#IgnoreWarnings)
# 1. [Read data](#Readdata)
# 1. [Peek data](#Peekdata)
# 1. [Query Data Type](#QueryDataType)
# 1. [Columns With Missing Values as a List](#ColumnsWithMissingValues)
# 1. [Columns of object Data Type (Categorical Columns) as a List](#CatColumns)
# 1. [Columns that Contain Numeric Values as a List](#NumColumns)
# 1. [Categorical Columns with Cardinality less than N](#CatColsCar)
# 1. [Count of Unique Values in a Column](#UniqueCount)
# 1. [OneHot Encode the Dataframe](#OneHotEncode)
# 1. [Select Columns Based on Data Types](#DTypeColSelect)
# 1. [Get Missing Values Info](#GetMissingValuesInfo)
# 1. [Missing Value Handling](#MissingValueHandling)
# 1. [Logistic Regression](#LogisticRegression)
# 1. [pandas series to pandas dataframe](#series2df)
# 1. [Convert categorical columns in numerical dtype to object type](#Convertnumericalcategoricalobject)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# #### Check Package Version[^](#CheckPackageVersion)<a id="CheckPackageVersion" ></a><br>

# In[ ]:


print(pd.__version__)
print(np.__version__)


# #### Ignore Warnings[^](#IgnoreWarnings)<a id="IgnoreWarnings" ></a><br>

# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# #### Read data[^](#Readdata)<a id="Readdata" ></a><br>

# In[ ]:


data = pd.read_csv("../input/titanic/train.csv")


# #### Peek data[^](#Readdata)<a id="Readdata" ></a><br>

# In[ ]:


#peek top
data.head()


# In[ ]:


#peek tail
data.tail()


# #### Query Data Type[^](#QueryDataType)<a id="QueryDataType" ></a><br>

# In[ ]:


values = {}
arr = []
print('values is a ' ,type(values))
type(arr)


# #### Columns With Missing Values as a List[^](#ColumnsWithMissingValues)<a id="ColumnsWithMissingValues" ></a><br>

# In[ ]:


def getColumnsWithMissingValuesList(df):
    return [col for col in df.columns if df[col].isnull().any()] 

getColumnsWithMissingValuesList(data)


# #### Columns of object Data Type (Categorical Columns) as a List[^](#CatColumns)<a id="CatColumns" ></a><br>

# In[ ]:


def getObjectColumnsList(df):
    return [cname for cname in df.columns if df[cname].dtype == "object"]

cat_cols = getObjectColumnsList(data)
cat_cols


# #### Columns that Contain Numeric Values as a List[^](#NumColumns)<a id="NumColumns" ></a><br>

# In[ ]:


def getNumericColumnsList(df):
    return [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]

num_cols = getNumericColumnsList(data)
num_cols


# #### Categorical Columns with Cardinality less than N[^](#CatColsCar)<a id="CatColsCar" ></a><br>

# In[ ]:


def getLowCardinalityColumnsList(df,cardinality):
    return [cname for cname in df.columns if df[cname].nunique() < cardinality and df[cname].dtype == "object"]

LowCardinalityColumns = getLowCardinalityColumnsList(data,10)
LowCardinalityColumns


# #### Count of Unique Values in a Column[^](#UniqueCount)<a id="UniqueCount" ></a><br>

# In[ ]:


data['Embarked'].nunique()


# #### OneHot Encode the Dataframe[^](#OneHotEncode)<a id="OneHotEncode" ></a><br>

# In[ ]:


def PerformOneHotEncoding(df,columnsToEncode):
    return pd.get_dummies(df,columns = columnsToEncode)

oneHotEncoded_df = PerformOneHotEncoding(data,getLowCardinalityColumnsList(data,10))
oneHotEncoded_df.head()


# #### Select Columns Based on Data Types[^](#DTypeColSelect)<a id="DTypeColSelect" ></a><br>

# In[ ]:


# select only int64 & float64 columns
numeric_data = data.select_dtypes(include=['int64','float64'])

# select only object columns
categorical_data = data.select_dtypes(include='object')


# In[ ]:


numeric_data.head()


# In[ ]:


categorical_data.head()


# #### Get Missing Values Info[^](#GetMissingValuesInfo)<a id="GetMissingValuesInfo" ></a><br>

# In[ ]:


def missingValuesInfo(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100, 2)
    temp = pd.concat([total, percent], axis = 1,keys= ['Total', 'Percentage'])
    return temp.loc[(temp['Total'] > 0)]

missingValuesInfo(data)


# #### Missing Value Handling[^](#MissingValueHandling)<a id="MissingValueHandling" ></a><br>

# In[ ]:


# for Object columns fill using 'UNKOWN'
# for Numeric columns fill using median
def HandleMissingValues(df):
    num_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]
    cat_cols = [cname for cname in df.columns if df[cname].dtype == "object"]
    values = {}
    for a in cat_cols:
        values[a] = 'UNKOWN'

    for a in num_cols:
        values[a] = df[a].median()
        
    df.fillna(value=values,inplace=True)
    
    
HandleMissingValues(data)
data.head()


# In[ ]:


#check for NaN values
data.isnull().sum().sum()


# Logistic Regression[^](#LogisticRegression)<a id="LogisticRegression" ></a><br>

# In[ ]:


def logistic(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
    lr=LogisticRegression()
    lr.fit(X_train,y_train)
    y_pre=lr.predict(X_test)
    print('Accuracy : ',accuracy_score(y_test,y_pre))


# #### pandas series to pandas dataframe[^](#series2df)<a id="series2df" ></a><br>

# In[ ]:


series = data['Fare']
d = {series.name : series}
df = pd.DataFrame(d) 
df.head()


# #### Convert categorical columns in numerical dtype to object type[^](#Convertnumericalcategoricalobject)<a id="Convertnumericalcategoricalobject" ></a><br>
# 
# Sometimes categorical columns comes in numerical data types. This is the case for all most all ordinal columns. If not converted to 'category' descriptive statistic summary does not makes sense.

# In[ ]:


PassengerClass = data['Pclass'].astype('category')
PassengerClass.describe()

