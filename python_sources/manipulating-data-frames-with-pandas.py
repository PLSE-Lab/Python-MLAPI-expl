#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt


# Importing Data

# In[ ]:


mydata = pd.read_csv("../input/Pokemon.csv")
mydata = mydata.set_index("#")
mydata.head()


# In[ ]:


#we can see every column and raw
mydata["HP"][1]


# In[ ]:


#we can see the same result 
mydata.HP[1]


# In[ ]:


# using loc accessor (location of ...)
mydata.loc[1,["Speed"]]


# In[ ]:


# Selecting only some columns
mydata[["Speed","Defense"]]


# In[ ]:


# if we make one square braket, data will be series and 
#if we make 2 , data will be dataframes
print(type(mydata["Sp. Def"]))     # series
print(type(mydata[["Sp. Def"]]))   # data frames


# In[ ]:


# Slicing and indexing series
mydata.loc[1:7,"Sp. Def":"Speed"]   # 7 columns and "Sp.Def", "Speed"


# In[ ]:


# Reverse slicing 
mydata.loc[7:1:-1,"Sp. Def":"Speed"] 


# In[ ]:


# From specific point to the end of data's value
mydata.loc[3:7,"Sp. Def":] 


# FILTERING DATA FRAMES

# In[ ]:


bl=mydata.Speed>150
mydata[bl]


# In[ ]:


# Combining filters
filter1 = mydata.Speed > 130
filter2 = mydata.Generation > 2
mydata[filter1 & filter2]


# In[ ]:


# Nested Filters
mydata
mydata.Speed[mydata.Generation>2]


# TRANSFORMING DATA

# In[ ]:


# Plain python functions
def divition(n):
    return n/2
mydata.Speed.apply(divition)


# In[ ]:


# Or we can use lambda function (it is the same with last divition function)
mydata.Speed.apply(lambda n : n/2)


# In[ ]:


# Defining new column using other columns
mydata["new"] = (mydata.Speed + mydata.Defense)/2
mydata.head()


# PIVOTING DATA FRAMES

# In[ ]:


dictionary = {"Group":["A","A","B","B"],"gender":["Female","Male","Female","Male"],"count":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dictionary)
df


# In[ ]:


# pivoting
df.pivot(index="Group",columns = "gender",values="count")


# STACKING and UNSTACKING DATAFRAME

# In[ ]:


df1 = df.set_index(["Group","gender"])
df1


# In[ ]:


#unstack
# level determines indexes
df1.unstack(level=0)


# In[ ]:


df1.unstack(level=1) # level number change the place of Group and gender. we select in df1 function


# In[ ]:


# we can change inner and outer level of the index positions
df2 = df1.swaplevel(0,1)
df2


# MELTING DATA FRAMES
# * Reverse of pivoting

# In[ ]:


df
# df.pivot(index="treatment",columns = "gender",values="response")
pd.melt(df,id_vars="Group",value_vars=["age","count"])


# CATEGORICALS AND GROUPBY

# In[ ]:


# according to treatment take means of other features
df.groupby("count").mean()   # mean is aggregation / reduction method
# there are other methods like sum, std,max or min


# In[ ]:


# we can only choose one of the feature
df.groupby("count").age.max() 


# In[ ]:


# Or we can choose multiple features
df.groupby("Group")[["age","count"]].min() 


# In[ ]:


df.info()


# thank you for look, your comment and your vote
# thanks to DATAI Team
