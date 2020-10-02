#!/usr/bin/env python
# coding: utf-8

# Automobile dataset, Data Manipulatin using Pandas
# 
# We shall now test your skills in using Pandas package. We will be using the [automobiles Dataset](https://www.kaggle.com/nisargpatel/automobiles/data) from Kaggle. 
# 
# Answer each question asked below wrt the automobiles dataset. Load pandas as pd and upload the Automobile.csv file as auto

# In[ ]:


import pandas as pd


# **Load the Automobile dataset into variable "auto"**

# In[ ]:


auto = pd.read_csv("../input/automobiles/Automobile.csv")


# **Check the head of the DataFrame.**

# In[ ]:


auto.head()


# **How many rows and columns are there?**

# In[ ]:


auto.info()


# **What is the average Price of all cars in the dataset?**

# In[ ]:


auto["price"].mean()


# **Which is the cheapest make and costliest make of car in the lot?**

# In[ ]:


auto[auto["price"] == auto["price"].max()]


# In[ ]:


auto[auto["price"] == auto["price"].min()]


# **How many cars have horsepower greater than 100?**

# In[ ]:


auto[auto["horsepower"]>100].count()


# **How many hatchback cars are in the dataset ?**

# In[ ]:


auto[auto["body_style"] == "hatchback"].info()


# **What are the 3 most commonly found cars in the dataset?**

# In[ ]:


auto["make"].value_counts().head(3)


# **Someone purchased a car for 7099, what is the make of the car?**

# In[ ]:


auto[auto["price"]== 7099]["make"]


# **Which cars are priced greater than 40000?**

# In[ ]:


auto[auto["price"]>40000]


# **Which are the cars that are both a sedan and priced less than 7000?**

# In[ ]:


auto[(auto["body_style"]=="sedan") & (auto["price"]<7000)]

