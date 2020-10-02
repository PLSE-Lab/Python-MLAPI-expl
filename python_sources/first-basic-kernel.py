#!/usr/bin/env python
# coding: utf-8

# **First of all i would like to Thanks to kaggle team for providing such an amazing platform.
# This is my first notebook on kaggle, i am a beginner i just wanted to try to make an exploratory data analysis.
# I used the most famous chennai_house price_prediction dataset.
# If you have any suggestions/critics/recommendations for me to improve my skills and knowledge please go ahead and post it in comments.
# Thanks!**

# # importing libraries

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


data=pd.read_csv("../input/first-model-chennai-house/chennai_house_price_prediction.csv")


# In[ ]:


data.head()  


# # Exploring Data and its variables, "Sales_Price" is our target variable 

# In[ ]:


data.describe()    # gives a description of continous variables


# 1. By analysing the count variable we get to know about the presence of missing values in particular columns.
# 2. 75%(third quartile) and max by comparing these we get to know about outliers.

# In[ ]:


data.describe(include="all")   # by including all we get details about categorical values also


# 1. 'unique' gives the value count for unique categories given in a column

# # DATA MANIPULATION
# 
# 1. Removing duplicate rows.
# 2. Filling missing values.
# 3. Fixing spelling errors if any.
# 4. Correcting data types.

# Drop duplicates ,if any.

# In[ ]:


data.drop_duplicates()


# In[ ]:


data.drop_duplicates().shape


# In[ ]:


data.shape


# Since there is no change in shape it means we have no duplicate rows, ( rows which have each column value same )

# In[ ]:


data.isnull().sum()      # it gives the count of missing values present in column


# In[ ]:


data.dropna(axis=0)


# # Dealing with missing values
# 1. if we drop the columns having missing values then we loss data of 3 columns. ( QS_OVERALL , N_BATHROOM , N_BEDROOM )
# 2. if we drop rows then we loss 53 rows of data ( 7109-7056=53)
# 
# So to avoid data loss we need to interpret these missing values by using:
# 1. mean/median for continous variables.
# 2. mode for categorical variable.

# In[ ]:


data["N_BEDROOM"].fillna( value=(data["N_BEDROOM"].mode()[0]) , inplace=True)   
#inplace=True made the changes in orignal data


# In[ ]:


data["N_BEDROOM"].value_counts()/len(data)


# In[ ]:


data["N_BATHROOM"].value_counts()/len(data)


# We need to impute the N_Bathroom value by using the common assumption that house which have no of bedrooms > 2 generally have higher chances of having 2 bathrooms. And house which have N_bedroom<=2 will have 1 bathroom.
# 

# In[ ]:


for i in range(0,len(data)):
    if pd.isnull(data["N_BATHROOM"][i])==True:
        if data["N_BEDROOM"][i]==1.0 or 2.0:
            data["N_BATHROOM"][i]=1.0
        else:
            data["N_BATHROOM"][i]=2.0
            
    


# In[ ]:


for i in range(0,len(data)):
    if pd.isnull(data["QS_OVERALL"][i])==True:
        x = (data["QS_ROOMS"][i]+data["QS_BEDROOM"][i]+data["QS_BATHROOM"][i])/3
        data["QS_OVERALL"][i]=x


# As overall quality score can be filled by mean value of quality score of rooms,bedrooms,bathrooms.

# In[ ]:


data.isnull().sum()   # all missing values filled, now we have complete data


# # DATA TYPES

# In[ ]:


data.dtypes


# In[ ]:


# N_BEDROOM , N_BATHROOM must be in integers , as they can not have float values
data=data.astype({"N_BEDROOM": int, "N_BATHROOM": int })


# In[ ]:


data.dtypes


# Now we need a value count for each **unique category** in a **CATEGORICAL VARIABLE**  so we can easily analyse about spelling errors.

# In[ ]:


temp = ["AREA","N_BEDROOM","N_BATHROOM","N_ROOM","SALE_COND","PARK_FACIL","BUILDTYPE","UTILITY_AVAIL","STREET","MZZONE"]
for i in temp:
    print("********** Value count for category",i,"********** ")
    print(data[i].value_counts())
    print(" ")


# # Analysis on above output.
# 
# 1. There are only 7 unique areas rest all have wrong spellings.
# 2. There are spelling errors in SALE_COND , UTILITY_AVAIL , STREET , PARK_FACIL , BUILD_TYPE too which needs to be resolved.
# 3. There are very less number of houses having >=3 bedrooms.

# In[ ]:


data["PARK_FACIL"].replace({"Noo":"No"}, inplace=True )


# In[ ]:


data["AREA"].replace({"Chrmpet":"Chrompet","Chormpet":"Chrompet","Chrompt":"Chrompet","Karapakam" : "Karapakkam",
                      "KKNagar":"KK Nagar", "TNagar": "T Nagar", "Adyr": "Adyar","Ana Nagar":"Anna Nagar",
                      "Ann Nagar":"Anna Nagar", "Velchery":"Velachery",}, inplace=True )


# In[ ]:


data["AREA"].value_counts()


# In[ ]:


data["UTILITY_AVAIL"].replace({"All Pub":"AllPub"}, inplace=True )
data["STREET"].replace({"Pavd":"Paved", "No Access": "No Access"}, inplace=True )
data["SALE_COND"].replace({"Partiall":"Partial","PartiaLl":"Partial","Ab Normal":"AbNormal","Adj Land":"AdjLand"}, inplace=True )
data["BUILDTYPE"].replace({"Comercial":"Commercial", "Other": "Others"}, inplace=True )


# In[ ]:


temp = ["SALE_COND","PARK_FACIL","UTILITY_AVAIL","STREET","BUILDTYPE"]
for i in temp:
    print("********** Value count for category",i,"********** ")
    print(data[i].value_counts())
    print(" ")


# In[ ]:




