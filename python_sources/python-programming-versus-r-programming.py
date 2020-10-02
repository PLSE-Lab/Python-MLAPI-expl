#!/usr/bin/env python
# coding: utf-8

# [R Programming versus Python Programming](https://www.kaggle.com/ekrembayar/r-programming-versus-pyhton-programming)

# # 1. Packages

# In[ ]:


import numpy as np 
import pandas as pd 


# # 2. Read Data

# In[ ]:


df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")


# # 3. Data Structure

# ## 3.1. Data Frame Head & Tail 

# In[ ]:


df.head()


# In[ ]:


df.tail()


# ## 3.2. Data Information

# In[ ]:


df.info()


# In[ ]:


df.describe()


# ## 3.3. Data Types

# In[ ]:


type(df)


# In[ ]:


type(df.name)


# In[ ]:


df["price"].dtype


# In[ ]:


df["name"].dtype


# ## 3.4. Variable Transformation

# In[ ]:


df.name.dtype


# In[ ]:


df["name"] = df["name"].astype("category")
df.name.dtype


# # 4. Generate Random Numbers

# In[ ]:


values = np.random.randn(50)
values


# In[ ]:


normal_distribution = np.random.normal(0,1, 50)
normal_distribution


# # 5. Data Frame

# ## 5.1. Create Data Frame

# In[ ]:


data = pd.DataFrame({"A" : [1,2,3,4], 
                     "B" : [1.0, 2.2, 3.5, 5.4]
                    })
data


# ## 5.2. Variable Names

# In[ ]:


data.columns


# ## 5.3. Variable Selection

# In[ ]:


data.A


# In[ ]:


data["A"]


# In[ ]:


data.iloc[ : , :1 ]


# In[ ]:


data.iloc[ : , 1 ]


# ## 5.4. Add Rows

# In[ ]:


data.loc[5] = [1, 6.1]
data


# # 6. Filter Data

# Filtering one category

# In[ ]:


private = df["room_type"] == "Private room"
df[private].head()


# In[ ]:


df.query('room_type == "Private room"').head()


# Filtering many category

# In[ ]:


df[df["room_type"].isin(["Private room", "Entire home/apt"])].head()


# In[ ]:


rt = ["Private room", "Entire home/apt"]
df.query('room_type == @rt').head()


# # 7. Group Data

# In[ ]:


df.groupby("neighbourhood_group")["price"].mean()


# In[ ]:


df.groupby(["neighbourhood_group", "room_type"])["price"].mean()


# In[ ]:


df.groupby(["neighbourhood_group", "room_type"])["price"].agg(['mean', "sum"])


# If you want a data frame in R, you should add reset_index() 

# In[ ]:


df.groupby(["neighbourhood_group", "room_type"])["price"].agg(['mean', "sum"]).reset_index()


# # 8. Select Data 

# In[ ]:


df.columns


# In[ ]:


df.filter(["price", "host_name", "number_of_reviews"]).head()


# In[ ]:


df.filter(like = "neighbourhood").head()

