#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/10_Property_stolen_and_recovered.csv")


# After loading the file, We must first look for the datatypes of attributes to ensure we plot them properly.

# In[ ]:


data.dtypes


# Looks like the data format is pretty tidy! Let's jump right into what the data has to reveal! 
# We will first start with the most basic vizzes.

# Matplotlib.pyplot library stores methods that look like Matlab. In matplotlib.pyplot, certain states are preserved about the plot figure and the plotting area. Whatever, further methods might be used are assumed to be for this particular plot itself.
# Let's build a simple line chart first!

# In[ ]:


plt.plot([1,4,5,6], [1,8,9,16])
plt.axis([0, 7, 0, 18])
plt.show()


# Sweet! Here the plot() function simply plots the two vectors onto x and y axes respectively. The axis() method specifies the lower and upper bound of the both axes. show() function displays the output and then returns back to the IPython prompt. 

# Let's work on our dataset now for a line chart. 
# Here's a pivot table of data 

# In[ ]:


line = data[data['Area_Name']=='Delhi'].pivot_table(index='Year',values='Cases_Property_Stolen')


# In[ ]:


plt.plot(line)
plt.xlabel('Year')
plt.ylabel('Number of Property Stolen Cases in Delhi')
plt.show()


# The bar() method is used to build a bar chart. It needs two mandatory parameters i.e the labels to be shown on x-axis and their value count as the height on the bar.

# In[ ]:


x = data[data.loc[:,'Sub_Group_Name']=='1. Dacoity']


# In[ ]:


bar = x.pivot_table(index='Area_Name',values='Cases_Property_Stolen',aggfunc=np.sum)


# In[ ]:


index=bar[5:10].index


# In[ ]:


plt.figure(figsize=(9,5))
plt.bar(index,bar.Cases_Property_Stolen[5:10],width=0.5)
plt.ylabel('Number of property stolen')
plt.xlabel('States in India')
plt.title('Dacoity from 2001- 2010')
plt.show()


# Pie charts on the other hand provide a comparative study of one label with every other label. 

# In[ ]:


plt.figure(figsize=(10,6))
plt.pie(bar.Cases_Property_Stolen[5:10],labels=index,autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')
plt.title("DACOITY CASES RECORDED(2000 - 2010)")
plt.show()


# In[ ]:


scatter


# Scatterplots are used to depict the relationship of two variables or their distribution over the axes. Scatter plots give you a sense of how the two data points are binded to each other.

# In[ ]:


plt.figure(figsize=(10,5))
colors = np.random.rand(35)
scatter = data.pivot_table(index='Area_Name',values=['Cases_Property_Stolen','Value_of_Property_Stolen','Value_of_Property_Recovered'],aggfunc=np.sum)
plt.scatter(scatter.Cases_Property_Stolen,scatter.Value_of_Property_Stolen,c=colors)
plt.xlabel('Number of Property Stolen cases')
plt.ylabel('Value of Property Stolen')
plt.show()


# In[ ]:


plt.figure(figsize=(16,9))
plt.hist(data.Cases_Property_Stolen,bins=35)
plt.xlabel("Bins bases on quantity of property stealth")
plt.ylabel("Count")
plt.axis([0,50000,0,3000])
plt.show()


# In[ ]:


plt.figure(figsize=(8,10))
plt.boxplot(data.Cases_Property_Recovered)
plt.ylabel("Count")
plt.show()

