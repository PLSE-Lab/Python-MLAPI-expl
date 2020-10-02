#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing all the required package 

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np


# In[ ]:


# Importing Haberman Dataset
# Renaming the columns

# 30 = Age              (Age of Patient at the time of operation)
# 64 = Operation_Year   (Year at which Operation Takes place)
# 1 = Active_Lymph      (Number of Active Lymph node)
# 1.1 = Survival_Status (Survival Status where 1 = Patient survive 5 years or more , 
#                                               2 = Patient died within 5 years)


df=pd.read_csv("../input/haberman.csv")
df.rename(columns={"30":"Age","64":"Operation_Year","1":"Active_Lymph","1.1":"Survival_Status"},inplace=True)


# In[ ]:


# Printing only some value in the dataset
print(df.head())


# In[ ]:


# Shape of data
# It tells us about number of rows and columns present in the dataset respectively.

df.shape


# In[ ]:


# To get Information About dataset

df.info()


# **Observation :**
# 
# *1 . It contains total 4 columns.*
# 
# *2 . Each rows contains 305 data points of integer type.*
# 
# *3 . There is no any missing or null value in the dataset.*
# 
# *4 . Survival Status has integer data so we have to convert it into categorical data to provide meaning.*

# In[ ]:


# Converting Survival_Status into categorical data

df["Survival_Status"]=df["Survival_Status"].map({1:"Yes",2:"No"})


# In[ ]:


df.head()


# **High Level Statistics**

# In[ ]:


# Description about Dataset

print(df.describe())


# In[ ]:


# Target Variable Distribution
print("\nTarget Variable Distribution")
print(df["Survival_Status"].value_counts())

# Normalize
print("\nTarget Variable Distribution After Normalization")
print(df["Survival_Status"].value_counts(normalize=True))


# **Observation :**
# 
# *1 . The minimum and maximum age of patient is 30 and 83 respectively.*
# 
# *2 . Maximum number of Active Lymph is 52.*
# 
# *3 . 25% of patients have 0 Active Lymph and 75% of patients have 4 Lymph.*
# 
# *4 . Since Yes = 73% and No ~ 27% it is imbalanced.*
# 
# 

# **Objective : **
# 
# *To predict whether the patient will survive after 5 years or not based upon the patient's age, year of treatment and the number of active lymph nodes.*

# **Uni-variate Analysis**

# **1 . Analysis Using PDF**

# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(df,hue="Survival_Status",size=5).map(sns.distplot,"Age").add_legend()
plt.show()


# *Using Age as a parameter and plotting PDF.*
# 
# *It is difficult to predict survival status .*

# In[ ]:


sns.set_style("whitegrid");
sns.FacetGrid(df,hue="Survival_Status",size=5).map(sns.distplot,"Operation_Year").add_legend();
plt.show();


# *Using Operation Year as a parameter and plotting PDF.*
# 
# *It is difficult to predict survival status .*

# In[ ]:


sns.set_style("whitegrid")
sns.FacetGrid(df,hue="Survival_Status",size=5).map(sns.distplot,"Active_Lymph").add_legend()
plt.show()


# *Using Active Lymph as a parameter and plotting PDF .
# *
# 
# *It is observed that with ZERO Active Lymph are survived more than 5 years.*

# **2. Analysis Using CDF**

# In[ ]:


survival_less_5yrs = df[df["Survival_Status"]=="No"]
survival_more_5yrs=df[df["Survival_Status"]=="Yes"]


# In[ ]:


# Taking Age As A Parameter

counts ,bin_edges =np.histogram(survival_less_5yrs['Age'],bins=10,density=True)
pdf=counts/sum(counts)
print("Pdf And Bin_Edges For Survival Less Than 5 Years On basis of Age\n")
print(pdf)
print(bin_edges)
cdf=np.cumsum(pdf)
plt.subplot(121)
plt.grid()
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend(['Pdf Survived < 5Yrs','Cdf Survived < 5Yrs'])



counts,bin_edges = np.histogram(survival_more_5yrs['Age'],bins=10,density=True)
pdf=counts/sum(counts)
print("\n Pdf And Bin_Edges For Survival More Than 5 Years On basis of Age\n")
print(pdf)
print(bin_edges)
cdf =np.cumsum(pdf)
plt.subplot(122)
plt.grid()
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend(['Pdf Survived > 5Yrs','Cdf Survived > 5Yrs'])
plt.show()


# **Observation :**
# 
# *Taking **Age** As Parameter and plotting PDF and CDF Curve. 
# It can be observed that due to overlapping we can not say that at this age people survived more than 5 years or less than 5 years. So Age Can not be taken as Parameter.*

# In[ ]:


# Taking Active Lymph As A Parameter

counts,bin_edges=np.histogram(survival_less_5yrs["Active_Lymph"],bins=10,density=True)
pdf=counts/sum(counts)
print("Pdf And Bin_Edges For Survival Less Than 5 Years On basis of Active Lymph\n")
print(pdf)
print(bin_edges)
plt.subplot(121)
plt.grid()
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend(['Pdf Survived < 5Yrs','Cdf Survived < 5yrs'])
plt.xlabel("No Of Active Lymph")
plt.ylabel("Probability")



counts,bin_edges=np.histogram(survival_more_5yrs["Active_Lymph"],bins=10,density=True)
pdf=counts/sum(counts)
print("Pdf And Bin_Edges For Survival More Than 5 Years On basis of Active Lymph\n")
print(pdf)
print(bin_edges)
plt.subplot(122)
plt.grid()
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend(['Pdf Survived > 5Yrs','Cdf Survived > 5yrs'])
plt.xlabel("No Of Active Lymph")
plt.show()


# **Observation :**
# 
# *Taking **Active Lymph** as Parameter and Plotting PDF and CDF curve . It can be observed that approximately 83% of the people survived more than 5 years having number of active lymph less than 5.*

# In[ ]:


# Taking Operation Year As A parameter 

counts,bin_edges=np.histogram(survival_less_5yrs["Operation_Year"],bins=10,density=True)
pdf=counts/sum(counts)
print("Pdf And Bin_Edges For Survival Less Than 5 Years On basis of Operation Year\n")
print(pdf)
print(bin_edges)
cdf=np.cumsum(pdf)
plt.subplot(121)
plt.grid()
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend(['Pdf Survived < 5Yrs','Cdf Survived < 5yrs'])
plt.xlabel("Operation Year")
plt.ylabel("Probability")



counts,bin_edges=np.histogram(survival_more_5yrs["Operation_Year"])
pdf=counts/sum(counts)
print("Pdf And Bin_Edges For Survival More Than 5 Years On basis of Operation Year\n")
print(pdf)
print(bin_edges)
cdf=np.cumsum(pdf)
plt.subplot(122)
plt.grid()
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend(['Pdf Survived > 5Yrs','Cdf Survived > 5yrs'])
plt.xlabel("Operation Year")
plt.show()


# **Observation :**
# 
# *Taking **Operation Year** as Parameter and Plotting Pdf and Cdf. It can be observed due to overlapping we can not say that in this year people who got treatment survived for more than or less than 5 years. So we can not take Operation year as parameter.*

# **3. Analysis Using Box Plot**

# In[ ]:


# Taking Age As A Parameter

sns.boxplot(x="Survival_Status",y="Age",data=df)
plt.show()


# **Observation :**
# 
# *Taking **Age** as a parameter we can not distinguish between the survival status due to overlapping.*

# In[ ]:


# Taking Active Lymph as a Parameter
sns.boxplot(x="Survival_Status",y="Active_Lymph",data=df)
plt.show()


# **Observation :**
# 
# *Taking **Active Lymph** as a parameter we can distinguish since median line of both are not overlapping.*

# In[ ]:


# Taking Operation Year as a Parameter
sns.boxplot(x="Survival_Status",y="Operation_Year",data=df)
plt.show()


# **Observation :**
# 
# *Taking **Operation Year** as a parameter we can not distinguish survival status due to overlapping. Median line of both the box is overlapped.*
# 
# 

# **4. Analysis Using Violin Plot**

# In[ ]:


# Taking Age as a parameter
sns.violinplot(x="Survival_Status",y="Age",data=df)
plt.show()


# In[ ]:


# Taking Active Lymph as a parameter

sns.violinplot(x="Survival_Status",y="Active_Lymph",data=df)
plt.show()


# In[ ]:


# Taking Operation Year as a Parameter

sns.violinplot(x="Survival_Status",y="Operation_Year",data=df)
plt.plot()


# **Conclusion :**
# 1. *From Uni-Variate Analysis we are able to find which parameter can help in order to determine the survival_status. *
# 
# 2. *Active Lymph seems to be good parameter from other parameter for uni-variate analysis. *

# **Analysis Using Bi-Variate**

# **1. Pair-Plot**

# In[ ]:


sns.set_style("whitegrid")
sns.pairplot(df,hue="Survival_Status",diag_kind="hist",size=3)
plt.show()


# **2. 2-D Scatter Plot**

# In[ ]:


# Taking operation year and Active Lymph as Parameter

sns.set_style("whitegrid")
sns.FacetGrid(df,hue="Survival_Status",size=4).map(plt.scatter,"Operation_Year","Active_Lymph").add_legend()
plt.show()


# In[ ]:


# Taking Age and Active Lymph as Parameter
sns.set_style("whitegrid")
sns.FacetGrid(df,hue="Survival_Status",size=4).map(plt.scatter,"Age","Active_Lymph").add_legend()
plt.show()


# **Conclusion :**
# 
# *Combination of parameters can not be used in order to distinguish survival status because it is not linearly separable.*

# In[ ]:




