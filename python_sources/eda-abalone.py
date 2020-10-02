#!/usr/bin/env python
# coding: utf-8

# # **EDA : Abalone Snails**

# **The Abalone DataSet consist of Following Attributes**
# 
# 1. Sex
# 2. Length                            ** Longest shell measurement**
# 3. Diameter                       ** perpendicular to length **
# 4. Height                             **with meat in shell **
# 5. Whole weight                ** whole abalone **
# 6. Shucked weight           **   weight of meat **
# 7. Viscera weight             **   gut weight (after bleeding) **
# 8. Shell weight                **    after being dried **
# 9. Rings

# > **The Aim of this Analysis to determine the Sex of an Abalone Snail based on its Physical Measurements.**

# ## Step -1 Load Dependencies

# In[ ]:


#Basic Dependencies
import numpy as np;
import pandas as pd;
from os.path import exists,basename,dirname,join;
import os;
import matplotlib.pyplot as plt;
from time import time


# ## Step - 2 Load Data

# In[ ]:


#Loading Dataset
csv_path = "../input/abalone.csv"
def read_csv_f(*path):
    new_path = join(*path)
    return pd.read_csv(new_path)

df = read_csv_f(csv_path)
df.head(3) # Preview of Data


# In[ ]:


df_attributes = list(df.columns)
num = df.shape[0]
print(df_attributes, num ,sep= "\n", end= "")


# # Step - 3 Preprocessing

# **`Data Cleaning - `**
# It refers to removal ,deletions of unwanted noise and missing data.

# **`Handling Categorical and Text Attribute - `**
# A computer doesn't understand text language it only understand numeric value thus it is required to convert Cateorical features into numeric ones
# 

# In[ ]:


#Encode Categoorical data into numeric data
from sklearn.preprocessing import LabelEncoder

def encode_category(data,columns,copy=False):
    meta=dict()
    X=data
    if copy is True:
        X=data.copy()
    lb_make = LabelEncoder()
    for col in columns:
        X[col] = lb_make.fit_transform(X[col])
        meta[col]=list(lb_make.classes_)
    return X,meta

df,encode_info = encode_category(df, ["Sex"])
print(encode_info)
df.head(2)


# `In above operation the Sex Features is encoded`
# 
# Female  F    = 0 
# 
# Infant  I    = 1
# 
# Man     M    = 2
# 

# **`Feature Scaling -`** It refers to normalize or making features value within a range to have better understanding as well as
# to  find coorelation b/w data
# 
# There is no need of it as the data seems to be normalize enough
# 

#  Correlation and Description of Features

# In[ ]:


df.info()


# In[ ]:


#The Coorelation b/w various features are indicating the degree of association b/w them
#In Abalone Dataset , there is a strong association b/w Weight, Diameter and Length
attributes = ["Length", "Diameter", "Height", "Rings", "Whole weight"]
print(df[attributes].corr())
df[["Length", "Diameter", "Height", "Rings", "Whole weight","Viscera weight", "Shucked weight", "Shell weight"]].corr()


# **In  Order to Determine the Sex of a Abalone we require only specific Attributes**
# 1.  Length
# 2. Diameter
# 3. Height
# 4. Whole Weight
# 5. Rings
# This the New Dataset will be
# 

# In[ ]:


attributes = ["Length", "Diameter", "Height", "Rings", "Whole weight"]
X = df[attributes]
y = df["Sex"]
print(X.head(2))
X.corr()


# # Step - 4 Visualization

# In[ ]:


#Features Distribution for Abalone Snails
figures = [221,222,223,224,111]

for attr,fig in zip(attributes,figures):
    plt.figure( figsize=(7,7))
    plt.subplot(fig)
    plt.hist(df[attr])
    plt.legend()
    plt.title(attr)
    plt.grid()
    plt.show()


# In[ ]:


female, infant, male = df[ df["Sex"] == 0], df[ df["Sex"] == 1], df[ df["Sex"] == 2]
temp = pd.DataFrame( {"infant": infant.max(), "male": male.max(), "female": female.max() })
temp


#  # ***Any Suggestion are always accepted.***
