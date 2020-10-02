#!/usr/bin/env python
# coding: utf-8

# We import the required libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
import re
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Loading the data
df= pd.read_csv('../input/planecrashinfo_20181121001952.csv')
df.head(2)


# We see that we have "?" instead of NA's, let's remove them first

# In[ ]:


#we have to replace '?' with NaN
df.replace("?", np.nan, inplace = True)
df.head(2)


# In[ ]:


df.shape


# In[ ]:


#check for where all there are missing values
df.isnull().nunique()


# We will extract the non-missing values and see which columns have the most missing values.

# In[ ]:


#plot to see which columns have most missing values
n = np.arange(13)
# x stores the data returned by isnull().value_counts() on each column
x = [[df.iloc[:,i].isnull().value_counts()]for i in range(13)]

#y stores only the "false" of the isnull().value_counts()
y = [x[i][0][0] for i in range(len(x))]
y.sort()

y


# In[ ]:


plt.bar(n, y, 0.45)
plt.title("Count of non-missing values in our dataset")
plt.xticks(n, df.columns, rotation =90)
plt.xlabel("columns")
plt.ylabel("number of non-missing values")
plt.show()


# If we check the "aboard" columns, it still has "?" values, we shall replace the "?" using regex

# In[ ]:


df["aboard_numbers"] = df["aboard"].str[0:4].str.strip()
df["aboard_numbers"].str.replace("^\?", "?")
df["aboard_numbers"].replace("?", np.nan, inplace = True)
df["aboard_numbers"].isnull().value_counts()


# Now we will replace those remaining '?' with a NaN

# In[ ]:


df["aboard_numbers"].replace("?", np.nan, inplace = True)
df["aboard_numbers"] = pd.to_numeric(df["aboard_numbers"] )
df["aboard_numbers"].isnull().value_counts()


# In[ ]:


df["fatalities"].head(5)


# We have to do the same thing for fatalities

# In[ ]:


# pd.Series(df.fatalities == '?').value_counts()#no '?'
df["fatalities"] = df["fatalities"].str[0:4].str.strip()
df["fatalities"].str.replace("^\?", "?")
df["fatalities"].replace("?", np.nan, inplace = True)
df["fatalities"] = pd.to_numeric(df["fatalities"] )


# In[ ]:


df["fatalities"].isnull().value_counts()


# we can drop the NaN values as they are very few in number

# In[ ]:


df_operator = df[["aboard_numbers", "fatalities", "operator"]]


# Let us see which operator are the most frequent to crash

# In[ ]:


df_operator.operator.value_counts()[0:10].plot(kind='bar',figsize=(16,8),title='Frequency of crash based on operator')


# Let us see the top 10 operators with highest fatilities 

# In[ ]:


df_operator.groupby("operator")[["fatalities"]].sum().sort_values(by = "fatalities", ascending = False)[0:10].plot(kind='bar',figsize=(16,8),title='Fatalities Based on Operator')


# Aeroflot has large numbers of crashes and fatalities, let us look at them

# In[ ]:


df_op_act = df[["operator", "ac_type", "fatalities", "aboard_numbers"]]
df_aeroflot_actype = df_op_act[df_op_act["operator"] == "Aeroflot"].ac_type.value_counts()[:13].plot(kind='bar',figsize=(16,8),title='Frequency of crash based on ac_type')


# In[ ]:



df_aeroflot_ac_vs_fatalities = df_op_act[df_op_act["operator"] == "Aeroflot"].groupby(["ac_type"])[["fatalities"]].sum().sort_values(by = "fatalities", ascending = False)[:13]
df_aeroflot_ac_vs_fatalities.reset_index(inplace = True)
df_aeroflot_ac_vs_fatalities.plot.bar(x = "ac_type", y= "fatalities")


# In[ ]:




