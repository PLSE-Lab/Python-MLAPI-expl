#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data= pd.read_csv("../input/abalone-dataset/abalone.csv")


# In[ ]:


data.head()


# In[ ]:


data.info()


# > #### features with null values

# In[ ]:


feature_with_null= [feature for feature in data.columns if data[feature].isnull().sum() >1 ]
feature_with_null


# In[ ]:


data.describe()


# In[ ]:





# #### Different types of gender or sex in the given data

# In[ ]:


data["Sex"].unique()


# > #### Must be "Male", "Female" or "Infant".

# #### Adding 1.5 to Rings to get the Age

# In[ ]:


data.insert(9, "Age",value= data["Rings"] +1.5)
data.head()


# > ### univariate analysis

# In[ ]:


# ACROSS RINGS

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
sns.distplot(data["Rings"],kde=False,bins=range(0,31,2))

plt.subplot(2,2,2)
sns.distplot(data["Rings"])

plt.subplot(2,2,3)
sns.boxplot(data["Rings"])


# In[ ]:


# HEIGHT, LENGTH, DIAMETER

plt.figure(figsize=(12,12))
color= sns.color_palette()

plt.subplot(3,3,3)
sns.distplot(data["Height"])

plt.subplot(3,3,1)
sns.distplot(data["Length"])

plt.subplot(3,3,2)
sns.distplot(data["Diameter"])

plt.subplot(3,3,6)
sns.distplot(data["Height"], kde=False, bins=10)

plt.subplot(3,3,4)
sns.distplot(data["Length"], kde=False,bins=10)

plt.subplot(3,3,5)
sns.distplot(data["Diameter"], kde=False, bins=10)

plt.subplot(3,3,9)
sns.boxplot(data["Height"])


plt.subplot(3,3,7)
sns.boxplot(data["Length"])


plt.subplot(3,3,8)
sns.boxplot(data["Diameter"])


# #### There are 2 outliers in the Height, so removing them will give more accurate data.

# In[ ]:


data= data[data["Height"] < 0.4]


# In[ ]:


plt.scatter(data["Height"], data["Age"])


# In[ ]:


# HEIGHT, LENGTH, DIAMETER

plt.figure(figsize=(12,12))
color= sns.color_palette()

plt.subplot(3,3,3)
sns.distplot(data["Height"])

plt.subplot(3,3,1)
sns.distplot(data["Length"])

plt.subplot(3,3,2)
sns.distplot(data["Diameter"])

plt.subplot(3,3,6)
sns.distplot(data["Height"], kde=False, bins=10)

plt.subplot(3,3,4)
sns.distplot(data["Length"], kde=False,bins=10)

plt.subplot(3,3,5)
sns.distplot(data["Diameter"], kde=False, bins=10)

plt.subplot(3,3,9)
sns.boxplot(data["Height"])


plt.subplot(3,3,7)
sns.boxplot(data["Length"])


plt.subplot(3,3,8)
sns.boxplot(data["Diameter"])


# In[ ]:


data.head()


# > #### Plotting the weight attribtes 

# In[ ]:


plt.figure(figsize=(12,12))

plt.subplot(3,4,1)
sns.distplot(data["Whole weight"])
plt.subplot(3,4,2)
sns.distplot(data["Shucked weight"])
plt.subplot(3,4,3)
sns.distplot(data["Viscera weight"])
plt.subplot(3,4,4)
sns.distplot(data["Shell weight"])

plt.subplot(3,4,5)
sns.distplot(data["Whole weight"],kde=False,bins=14)
plt.subplot(3,4,6)
sns.distplot(data["Shucked weight"], kde= False, bins=14)
plt.subplot(3,4,7)
sns.distplot(data["Viscera weight"], kde= False, bins=14)
plt.subplot(3,4,8)
sns.distplot(data["Shell weight"], kde= False, bins=14)

plt.subplot(3,4,9)
sns.boxplot(data["Whole weight"])
plt.subplot(3,4,10)
sns.boxplot(data["Shucked weight"])
plt.subplot(3,4,11)
sns.boxplot(data["Viscera weight"])
plt.subplot(3,4,12)
sns.boxplot(data["Shell weight"])


# ### multivariate

# In[ ]:


#corr= correlation

corr= data.corr()
corr


# #### Generating the Heatmap of the correlations

# In[ ]:


plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True)


# In[ ]:


sns.countplot(data["Sex"])
plt.title("Count of each sex of abalone")


# In[ ]:


sns.jointplot(data=data, x="Age",y="Shell weight",kind= "reg")

sns.jointplot(data=data, x="Age", y="Height",kind="reg")


# In[ ]:





# In[ ]:


plt.figure(figsize=(16,16))

plt.subplot(3,3,1)
plt.scatter(data["Length"],data["Age"])
plt.xlabel("Length")
plt.ylabel("Age")

plt.subplot(3,3,2)
plt.scatter(data["Height"],data["Age"])
plt.xlabel("Height")
plt.ylabel("Age")

plt.subplot(3,3,3)
plt.scatter(data["Diameter"],data["Age"])
plt.xlabel("Diameter")
plt.ylabel("Age")


# #### features with most correlation with rings

# In[ ]:


features=["Length","Height","Diameter","Shell weight","Rings"]
sns.pairplot(data[features],size=2,kind="scatter")


# In[ ]:




