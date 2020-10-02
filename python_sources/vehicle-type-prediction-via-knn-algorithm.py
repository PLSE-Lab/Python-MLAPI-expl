#!/usr/bin/env python
# coding: utf-8

# The dataset includes some technical information about vehicles. I collected from various sources from the internet. 
# I can't guarantee its truth, but the dataset can be said useful.
# 

# Let's start with reading dataset,

# In[ ]:


import pandas as pd
import numpy as np
df= pd.read_csv("../input/VehicleInformation.csv",sep=";")
#here 'sep' means seperator, it is ',' by default.
# to learn more about it run pd.read_csv? on your notebook 


# Have a quick look at the dataset, via head(),info() or describe() methods.

# In[ ]:


df.head()
#head() method,by default, gives us the first 5 rows of dataframe or series.
#and also we have tail(), which gives the last 5 rows by default. we can write any negative or positive number in ().
#try df.head(100000) or df.head(-50000)


# We have duplicates, we must remove them.

# In[ ]:


df.drop_duplicates(inplace=True)
len(df)


# inplace=True..it is the short way of writing df=df.drop_duplicates(). 

# In[ ]:


df.info()


# In[ ]:


print(list(df["FUEL_TYPE"].unique()))
print(np.min(df["MAX_SPEED"]))
print(np.max(df["MAX_SPEED"]))


# All values of FUEL_TYPE is "null", and MAX_SPEED is 0, we can drop whole these columns. Also BRAND_CODE,VEHICLE_CODE are useless for our prediction model.

# In[ ]:


df.drop(["FUEL_TYPE","MAX_SPEED","BRAND_CODE","VEHICLE_CODE"],inplace=True,axis=1)  #axis=1 means columns. it is axis=0 by default, which is rows.


# In[ ]:


#lets check again our DataFrame
print(len(df))
df.head()


# The VEHICLE_CLASS is our target feature. We want to find our whether other features are good enough to predict the class. 
# We have to create x and y for our model, and y will be the target variable which is VEHICLE_CLASS, and x will be the rest.

# In[ ]:


y=df["VEHICLE_CLASS"]
x=df.drop(["VEHICLE_CLASS"],axis=1)
#Dont run the drop method first, you must either run these two together, or y=df["VEHICLE_CLASS"] first. 
#Otherwise, you may lose "VEHICLE_CLASS" by dropping it.


# In[ ]:


print(x.head())
print("\n")
print(y.head())


# Now, split x and y sets as train and test sets. We use sklearn library for this aim.

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test =train_test_split(x,test_size=0.2,random_state=42)
y_train,y_test=train_test_split(y,test_size=0.2,random_state=42)


# Generally, test_size is chosen between 0.30, or 0.25. It depends on the data you have and your way to model.
# We first use test_size=0.2, and check the accuracy.

# As far as I know, random_state has almost no effect on the model. It simply does split data to the same sets for each run of train_test_split. If we don't specify random_state, each run of the model gives us different results.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=len(y.unique()))
knn.fit(x_train,y_train)


# In[ ]:


print("score with test_size=0.2 : ",knn.score(x_test,y_test))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
x_train_new,x_test_new =train_test_split(x,test_size=0.3,random_state=42)
y_train_new,y_test_new=train_test_split(y,test_size=0.3,random_state=42)
knn=KNeighborsClassifier(n_neighbors=len(y.unique()))
knn.fit(x_train_new,y_train_new)
print("score with test_size=0.3 : ",knn.score(x_test_new,y_test_new))


# If we use test_size=0.3, the accuracy gets lower.

# So, we may conclude that either model is not good or or classes are not enough to differentiate.
# But in either way, this is how KNN Model is used.
