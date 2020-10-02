#!/usr/bin/env python
# coding: utf-8

# **The main purpose of this tutorial is to highlight the importance of pre-processing steps we need to follow for any data analysis. Sometimes we tend to forget the syntax of python program for pre-processing. The dataset is used as per my convenience in showing the desired output. I am hopeful this tutorial will be a quick reference for everyone**

# In[1]:


from pandas import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
from scipy.stats import mode

import os
print(os.listdir("../input"))


# In[2]:


diabetes = pd.read_csv('../input/pima-diabetes/diabetes.csv')


# In[3]:


diabetes.head(5)


# ### Data Exploration

# #### Check if the data types are as expected

# In[4]:


diabetes.dtypes


# ### Memory Optimzations

# In[5]:


memory = diabetes.memory_usage()
print(memory)
print("Total Memory Usage = ",sum(memory))


# In[6]:


diabetes.iloc[:,0:9] = diabetes.iloc[:,0:9].astype('float16')


# In[7]:


memory = diabetes.memory_usage()
print(memory)
print("Total Memory Usage = ",sum(memory))


# ## Check for Summary Statistics

# In[8]:


diabetes.describe()


# In[9]:


diabetes["Outcome"].value_counts()


# ## Check for outliers

# In[10]:


fig, axs = plt.subplots()
sns.boxplot(data=diabetes,orient='h',palette="Set2")
plt.show()


# ### Dealing with outliers

# In[11]:


q75, q25 = np.percentile(diabetes["Insulin"], [75 ,25])
iqr = q75-q25
print("IQR",iqr)
whisker = q75 + (1.5*iqr)
print("Upper whisker",whisker)


# In[12]:


diabetes["Insulin"] = diabetes["Insulin"].clip(upper=whisker)


# In[13]:


fig, axs = plt.subplots()
sns.boxplot(data=diabetes,orient='h',palette="Set2")
plt.show()


# ### Check missing values

# In[14]:


diabetes.head()


# In[15]:


diabetes.columns


# In[16]:



print((diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']] == 0).sum())


# In[17]:


diabetes.loc[:,['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI']] = diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI']].replace(0, np.NaN)
diabetes.head()


# In[18]:


diabetes.isnull().sum()


# ## Dealing with missing values

# ## A. Drop rows having NaN

# In[19]:


print("Size before dropping NaN rows",diabetes.shape,"\n")

nan_dropped = diabetes.dropna()

print(nan_dropped.isnull().sum())
print("\nSize after dropping NaN rows",nan_dropped.shape)


# ## Drop row/columns having more than certain percentage of NaNs

# In[20]:


diabetes.isnull().mean()


# In[21]:


print("Size before dropping NaN rows",diabetes.shape,"\n")

col_dropped = diabetes.loc[:, diabetes.isnull().mean() < .4]
row_dropped = diabetes.loc[diabetes.isnull().mean(axis=1) < .4, :]

print(nan_dropped.isnull().sum())
print("\nSize after dropping Columns with rows",col_dropped.shape)
print("Size after dropping Columns with rows",row_dropped.shape)


# ### Imputing missing values

# #### We can impute the missing values by many ways. Here are the 3 most common ways we can do

# #### 1. Some constant value that is considered "normal" in the domain
# #### 2. Summary statistic like Mean, Median, Mode
# #### 3. A value estimated by algorithm or predictive model

# ### Binning

# In[22]:


bins = [0,25,30,35,40,100]

group_names = ['malnutrition', 'Under-Weight', 'Healthy', 'Over-Wight',"Obese"]
diabetes['BMI_Class'] = pd.cut(diabetes['BMI'], bins, labels=group_names)
diabetes.head(10)


# In[23]:


diabetes.dtypes


# In[24]:


#Impute the values:
diabetes['BMI_Class'].fillna((diabetes['BMI_Class']).mode()[0], inplace=True)
diabetes['Insulin'].fillna((diabetes['Insulin']).mean(), inplace=True)
diabetes['Pregnancies'].fillna((diabetes['Pregnancies']).median(), inplace=True)

# #Now check the #missing values again to confirm:
print(diabetes.isnull().sum())


# ### **Scaling**

# In[25]:


vector = np.random.chisquare(1,500)
print("Mean",np.mean(vector))
print("SD",np.std(vector))
print("Range",max(vector)-min(vector))


# In[26]:


from sklearn.preprocessing import MinMaxScaler
range_scaler = MinMaxScaler()
range_scaler.fit(vector.reshape(-1,1))
range_scaled_vector = range_scaler.transform(vector.reshape(-1,1))
print("Mean",np.mean(range_scaled_vector))
print("SD",np.std(range_scaled_vector))
print("Range",max(range_scaled_vector)-min(range_scaled_vector))


# ### Standardization
# 

# In[27]:


from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
standardizer.fit(vector.reshape(-1,1))
std_scaled_vector = standardizer.transform(vector.reshape(-1,1))
print("Mean",int(np.mean(std_scaled_vector)))
print("SD",int(np.std(std_scaled_vector)))
print("Range",max(std_scaled_vector)-min(std_scaled_vector))


# ### Dummification

# In[28]:


dummified_data = pd.concat([diabetes.iloc[:,:-1],pd.get_dummies(diabetes['BMI_Class'])],axis=1)
dummified_data.head()


# ### Reshape

# In[29]:


vector.shape


# In[30]:


row_vector = vector.reshape(-1,1)
row_vector.shape


# In[31]:


col_vector = vector.reshape(1,-1)
col_vector.shape


# In[32]:


matrix = vector.reshape(10,50)
matrix.shape


# ### Pivot Table

# In[33]:


#Determine pivot table
df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                       "bar", "bar", "bar", "bar"],
                    "B": ["one", "one", "one", "two", "two",
                          "one", "one", "two", "two"],
                    "C": ["small", "large", "large", "small",
                          "small", "large", "small", "small",
                          "large"],
                    "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                    "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})


# In[34]:


df.head(3)


# In[35]:


table = pivot_table(df, values='D', index=['A', 'B'],
                  columns=['C'], aggfunc=np.sum)
table


# http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pivot_table.html
# 

# ### Crosstab

# In[36]:


pd.crosstab(df["D"],df["B"],margins=True)


# ### Merging dataframes

# In[37]:


df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                       index=[0, 1, 2, 3])
    

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                       index=[4, 5, 6, 7])
    

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                       'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                       index=[8, 9, 10, 11])
    

frames = [df1, df2, df3]

result = pd.concat(frames)


# https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

# ### Plotting

# In[38]:


grade = pd.read_csv("../input/grades/Grade.csv")
grade.head()


# In[39]:


grade.describe()


# In[40]:


fig, axs = plt.subplots()
sns.regplot(x="StudentId",y="OverallPct1",data=grade,scatter=True,fit_reg=False)
plt.show()


# In[41]:


fig, axs = plt.subplots()
sns.boxplot(data=grade.iloc[:,1:-1],orient='h',palette="Set2")
plt.show()


# In[42]:


# Basic Plots
fig, axs = plt.subplots(ncols=3)
sns.distplot(grade['English1'],ax=axs[0])
sns.distplot(grade['Math1'],ax=axs[1])
sns.distplot(grade['Science1'],ax=axs[2])
plt.show()


# In[43]:


# Axis range and granularity are very important
fig, axs = plt.subplots(ncols=3)
sns.distplot(grade['English1'],ax=axs[0],bins=20)
sns.distplot(grade['Math1'],ax=axs[1],bins=20)
sns.distplot(grade['Science1'],ax=axs[2],bins=20)
axs[0].set_xlim(0,100)
axs[1].set_xlim(0,100)
axs[2].set_xlim(0,100)
axs[0].set_ylim(0,0.12)
axs[1].set_ylim(0,0.12)
axs[2].set_ylim(0,0.12)
plt.show()


# In[44]:


#Asthetics
fig, axs = plt.subplots()
sns.distplot(grade['English1'], bins=10, rug=True, rug_kws={"color": "red"},
             kde_kws={"color": "black", "lw": 2, "label": "KDE"},
             hist_kws={"histtype": "step", "lw": 3,"color": "green"})
plt.show()


# In[45]:


fig, axs = plt.subplots()
normal_dist = np.random.randn(1,1000)
normal_plot = sns.distplot(normal_dist)
normal_plot.set(xlabel='Value', ylabel='Frequency')
plt.show()


# In[46]:


fig, axs = plt.subplots()
vector = np.random.chisquare(1,500)
vector_plot = sns.distplot(vector)
vector_plot.set(xlabel='Value', ylabel='Probability')
plt.show()


# In[47]:


iris = pd.read_csv("../input/iris-dataset/Iris.csv")
iris.head()


# In[48]:


fig, axs = plt.subplots(nrows=2)
sns.swarmplot(x="Species", y="SepalLengthCm",ax=axs[0], data=iris)
sns.swarmplot(x="Species", y="PetalLengthCm",ax=axs[1], data=iris)
plt.show();


# In[49]:


plt.figure(1)
plt.figure(figsize = (12,8))
plt.scatter(iris['PetalLengthCm'], iris['PetalWidthCm'], s=np.array(iris.Species == 'Iris-setosa'), marker='^', c='green', linewidths=5)
plt.scatter(iris['PetalLengthCm'], iris['PetalWidthCm'], s=np.array(iris.Species == 'Iris-versicolor'), marker='^', c='orange', linewidths=5)
plt.scatter(iris['PetalLengthCm'], iris['PetalWidthCm'], s=np.array(iris.Species == 'Iris-virginica'), marker='o', c='blue', linewidths=5)
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.legend(loc = 'upper left', labels = ['Setosa', 'versicolor', 'virginica'])
plt.show();


# **For other plots and exploratory data analysis, please refer to my kernel here.**
# 
# https://www.kaggle.com/shravankoninti/starter-code-and-eda-on-iris-species
# ![](http://)
