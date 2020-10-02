#!/usr/bin/env python
# coding: utf-8

# In[5]:


#to enable visualizations 
get_ipython().run_line_magic('matplotlib', 'inline')

# First, import pandas, a useful data analysis tool especially when working with labeled data
import pandas as pd

# import seaborn, visualization library in python 
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Next, we'll load the cereal dataset, which is in the specified directory below
cereal = pd.read_csv("C:\\Users\\oddin\\Documents\\projects\\project2\\cereal.csv")

# Next, display the first 20 rows and all columns of the iris dataframe, good way to see the colum headings for the dataset
cereal.head(20)


# In[6]:


# to count the frequency of values for each cereal brand manufacturer in the dataset
cereal["manufacturer"].value_counts()


# In[7]:


#To create a bar chart showing the various cereal brands on the x-axis and calories on the y-axis
fig,ax = plt.subplots(figsize=(10,10))
cereal['manufacturer'].value_counts(sort=False).plot(kind='bar',color = 'blue')
plt.title('Brand Manufacturers and Calorie Counts',fontsize=20)
plt.xlabel('manufacturer',fontsize=20)
plt.ylabel('calories',fontsize=20)


# In[8]:


#To create a bar chart showing the various cereal brands on the x-axis and sodium on the y-axis
fig,ax = plt.subplots(figsize=(10,10))
cereal['manufacturer'].value_counts(sort=False).plot(kind='bar',color = 'blue')
plt.title('Brand Manufacturers and Sodium Quantity',fontsize=20)
plt.xlabel('manufacturer',fontsize=20)
plt.ylabel('sodium',fontsize=20)


# In[6]:


#To create a pie chart showing cereal manufacturers and ratings
fig,ax = plt.subplots(figsize=(12,12))
cereal['manufacturer'].value_counts(sort=False).plot(kind='pie')
plt.title('Manufacturer and Rating',fontsize=20)
plt.xlabel('manufacturer',fontsize=16)
plt.ylabel('rating',fontsize=16)


# In[7]:


# A good way to complement the boxplot is by using the Seaborn's striplot
# Use jitter=True so that all the points are not represented(clustered) on the same axis,
#this allows the data to be properly represented
# Saving the resulting axes as ax each time causes the resulting plot to be shown
# on top of the previous axes
# added size to make the change the size of the dots i.e. bigger or smaller
# changed edge color
ax = sns.boxplot(x="manufacturer", y="calories", data=cereal)
ax = sns.stripplot(x="manufacturer", y="calories", data=cereal, jitter=True, size = 12, edgecolor="black")


# In[8]:


# To create a kdeplot which is a seaborn plot useful for looking at univariate relations 
# Creates and visualizes a kernel density estimate of the underlying feature

sns.FacetGrid(cereal, hue="manufacturer", size=6)    .map(sns.kdeplot, "rating")   .add_legend()


# In[9]:


# Violin plot, unlike box plots, depict the density of the data
# Denser regions of the data are fatter, and sparser thiner in a violin plot
# further showing the distributions of the features 
sns.violinplot(x="manufacturer", y="sodium", data=cereal, size=10)

