#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.pyplot as plote
from sklearn import preprocessing
from sklearn import cross_validation, metrics
from sklearn.model_selection import train_test_split
from pylab import plot, show, subplot, specgram, imshow, savefig


# **loading data**

# In[2]:


# Reading the dataset
train = pd.ExcelFile("../input/research_student (1) (1).xlsx")
dataset = train.parse('Sheet1', header=0)


# In[3]:



dataset.head()


# ***Removing outliers     ***

# In[4]:


dataset = dataset.drop(dataset[dataset['GPA 2']>10].index)


# # dropping the NaN values 
# 

# In[5]:


dataset = dataset.dropna()


# In[6]:


dataset.head()


# # separating the dependent and independent variables

# In[7]:


target = dataset.CGPA
features = dataset.drop(['CGPA'], axis = 1)


# # Plotting Heat map 

# In[8]:


f,ax = plote.subplots(figsize=(18, 18))
sns.heatmap(features.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# # dropping 'Rank' features as the dataset has already Normalised Rank feature(dropping not usefull data
# 

# In[9]:


features = features.drop(['Rank'], axis=1)


# In[10]:


features.describe()


# In[11]:


features.iloc[:,0:14].describe()


# In[12]:


features.iloc[:,14:].describe()


# # Making list of columns

# In[13]:


list1 = []
for item in features.columns:
    item = item.replace("[","")
    item = item.replace("]","")
    list1.append(item)
    
features.columns = list1


# In[14]:


list1


# # Visualization

# **vis_dataset is the combined dataset of features and the target(CGPA)**

# In[15]:


#Combining features and CGPA(Target)
vis_dataset = pd.concat([features, target], axis = 1)


# # Distplot
#  plot shows the distribution of CGPA (target) in the dataset.

# In[16]:


sns.distplot(target)


# # Branchwise CGPA Distribution plot(between 0 and 1)

# In[17]:


f, axes = plote.subplots(2, 4, figsize=(35, 15), sharex=True)
# CSE
d = vis_dataset.drop(vis_dataset[vis_dataset["Branch"]!="CSE"].index)
d = pd.Series(d.CGPA, name="CSE")
sns.distplot(d, color="r", ax=axes[0,0])
# IT
d = vis_dataset.drop(vis_dataset[vis_dataset["Branch"]!="IT"].index)
d = pd.Series(d.CGPA, name = "IT")
sns.distplot(d, color="b", ax=axes[0,1])
# ECE
d = vis_dataset.drop(vis_dataset[vis_dataset["Branch"]!="ECE"].index)
d = pd.Series(d.CGPA, name = "ECE")
sns.distplot(d, color="g", ax=axes[0,2])
# EEE
d = vis_dataset.drop(vis_dataset[vis_dataset["Branch"]!="EEE"].index)
d = pd.Series(d.CGPA, name = "EEE")
sns.distplot(d, color="y", ax=axes[0,3])
# CIVIL
d = vis_dataset.drop(vis_dataset[vis_dataset["Branch"]!="CIVIL"].index)
d = pd.Series(d.CGPA, name = "CIVIL")
sns.distplot(d, color="m", ax=axes[1,0])
# Mech
d = vis_dataset.drop(vis_dataset[vis_dataset["Branch"]!="MECH"].index)
d = pd.Series(d.CGPA, name = "MECH")
sns.distplot(d, color="r", ax=axes[1,1])
# Prod
d = vis_dataset.drop(vis_dataset[vis_dataset["Branch"]!="PROD"].index)
d = pd.Series(d.CGPA, name = "PROD")
sns.distplot(d, color="g", ax=axes[1,2])


# # Conclusion
# * Mech and CSE have widest range of distribution.
# * CIVIL CGPA distribution is skewed leftwards.
# * ECE and EEE have moderate and similar distribution.

# # Gender wise CGPA Distribution plot

# In[18]:


f, axes = plote.subplots(1, 2, figsize=(15, 7), sharex=True)
# Male
d = vis_dataset.drop(vis_dataset[vis_dataset["Gender"]!="Male"].index)
d = pd.Series(d.CGPA, name="Male")
sns.distplot(d, color="b", ax=axes[0])
# Fe
d = vis_dataset.drop(vis_dataset[vis_dataset["Gender"]!="Female"].index)
d = pd.Series(d.CGPA, name = "Female")
sns.distplot(d, color="y", ax=axes[1])


# # Conclusion 
# Girls are performing very well in semester exams -)
# 
# The Distribution of Male CGPA is leftwards skewed whereas the distribution of Female CGPA is rightwards skewed. CGPA distribution near 8-9 is greater for Female whereas the distribution near 7 is very large in Male.

# 
# # Category wise CGPA Distribution plot

# In[19]:


f, axes = plote.subplots(2, 2, figsize=(15, 10), sharex=True)
#General
d = vis_dataset.drop(vis_dataset[vis_dataset["Category"]!="GEN"].index)
d = pd.Series(d.CGPA, name="GEN")
sns.distplot(d, color="r", ax=axes[0,0])
# obc
d = vis_dataset.drop(vis_dataset[vis_dataset["Category"]!="OBC"].index)
d = pd.Series(d.CGPA, name = "OBC")
sns.distplot(d, color="b", ax=axes[0,1])
# sc
d = vis_dataset.drop(vis_dataset[vis_dataset["Category"]!="SC"].index)
d = pd.Series(d.CGPA, name = "SC")
sns.distplot(d, color="g", ax=axes[1,0])
# st
d = vis_dataset.drop(vis_dataset[vis_dataset["Category"]!="ST"].index)
d = pd.Series(d.CGPA, name = "ST")
sns.distplot(d, color="y", ax=axes[1,1])


# # Pair Plot
# The following plot pairs each and every attribute using scatter plot showing "Gender" distinction using different colors. The graphs in the diagonal represents distribution plot of each pair of features. These plots infer the pairs of attributes which can be used to distinguish Gender.
# 

# In[20]:


sns.reset_orig()
sns.pairplot(features[['GPA 1','GPA 5','GPA 6','Normalized Rank','GPA 4','Gender']],hue='Gender',palette='inferno')


# In[21]:


sns.pairplot(features[['GPA 2','Marks10th','GPA 3','Marks12th','Gender']],hue='Gender',palette='inferno')


# # Violin Plot
# Violin plot combines boxplot and kernel density estimate. It plays the same role as Box plot and whisker plot. The three quartile values are represented by dashed line in the both halves. It shows the distribution of quantitative data across several levels of one (or more) categorical variables such that those distributions can be compared. Unlike a box plot, in which all of the plot components correspond to actual datapoints, the violin plot features a kernel density estimation of the underlying distribution.

# 
# Both halves are separated easing out the comparision of the kernel density distribution and quartile values. They can also estimate outliers.

# # Weak vs Average Students
# 

# In[24]:


vis_dataset.CGPA = vis_dataset.CGPA.apply(lambda x: int(x))
vis_dataset_WM = vis_dataset.drop(vis_dataset[vis_dataset['CGPA']>7].index)
vis_dataset_WM = vis_dataset_WM.drop(vis_dataset_WM[vis_dataset_WM['CGPA']<6].index)
optimal_features1 = ['GPA 1','GPA 2','GPA 3','GPA 4','GPA 5','GPA 6','CGPA']
optimal_features2 = ['Normalized Rank','Marks10th','Marks12th','CGPA']


# In[52]:


optimal_features2 = ['Normalized Rank','Marks10th','Marks12th','CGPA']
data_wa = pd.melt(vis_dataset_WM[optimal_features1],id_vars="CGPA",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="CGPA", data=data_wa,split=True, inner="quart")
plt.xticks(rotation=45)
plt.title("Values of diff semester exams  ")


# In[53]:


data_wa2 = pd.melt(vis_dataset_WM[optimal_features2],id_vars="CGPA",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="CGPA", data=data_wa2,split=True, inner="quart")
plt.xticks(rotation=90)
plt.title("Values of diff exams before coming to college  ")


# # Average vs Good Students
# 
# 

# In[54]:


vis_dataset_WM = vis_dataset.drop(vis_dataset[vis_dataset['CGPA']>8].index)
vis_dataset_WM = vis_dataset_WM.drop(vis_dataset_WM[vis_dataset_WM['CGPA']<7].index)


# In[55]:


sns.reset_orig()
data_ag1 = pd.melt(vis_dataset_WM[optimal_features1],id_vars="CGPA",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="CGPA", data=data_ag1,split=True, inner="quart")
plt.xticks(rotation=90)
plt.title("Values of diff exams")


# In[56]:


sns.reset_orig()
data_ag1 = pd.melt(vis_dataset_WM[optimal_features2],id_vars="CGPA",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="CGPA", data=data_ag1,split=True, inner="quart")
plt.xticks(rotation=90)
plt.title("Values of diff exams before coming to college  ")


# # Branch and Gender Violin Plot
# 

# In[64]:


plt.figure(figsize=(10,10))
sns.violinplot(x="Branch", y="CGPA", hue="Gender", data=vis_dataset,split=True, inner="quart", palette={"Male": "b", "Female": "y"})
plt.xticks(rotation=90)


# # Conclusion
# The range of CGPA distribution is higher for Female in ECE and IT whereas the range is almost the same for CSE and EEE. The skewed nature of distribution as shown in Distribution plot is highlighted in CSE plot. For some Branches the upper or lower quartile values merges with the boundary. This happens due to skewed nature of kernel density distribution. For CIVIL Kernel density distribution is highly skewed merging the lower quartile value with the lower boundary. The same can be verified by the distribution plot.
# 
# 

# # Swarm Plot
# A swarm plot can be drawn on its own, but it is also a good complement to a box or violin plot in cases where you want to show all observations along with some representation of the underlying distribution. The distribution contrasts level of students using different colors. Outliers can be monitored by spotting values lying at distance with main distribution.
# 
# 

# **Weak vs Average Students**

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




