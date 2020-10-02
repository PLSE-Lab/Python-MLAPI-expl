#!/usr/bin/env python
# coding: utf-8

# # Casey's DonorChoose EDA: Resources!
# 
# ## Contents
# 
# 1.) Introduction
# 
# 2.) Packages
# 
# 3.) Original Set of Features
# 
# 4.) Distribution of Resource Value by Project
# 
# 5.) Spread of Resource Values Across Vendors
# 
# 6.) Resource Value Heatmap
# 
# 7.) Conclusion

# ### Introduction
# 
# This kernel is to do some exploration on the resources.csv dataframe and answer some questions about the resources being purchased for projects.  This will influence further exploration and eventually more advanced analytics.

# ### Packages

# In[ ]:


# Import packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualizations, classic
get_ipython().run_line_magic('matplotlib', 'inline')
import os 
import seaborn as sns # more data visualizations


# ### Original Set of Features

# In[ ]:


# Bring in the resources.csv dataset

resources = pd.read_csv('../input/Resources.csv')
resources.head()


# _Well, it looks like each observation corresponds to a resource being purchased for a project by a donor and the name, quantity, and price per unit is given.  The only thing missing is the value of each resource, i.e the **Quantity*Price**._

# In[ ]:


# Create an empty column in the pandas data frame for the Quantity*price

# Define a function for filling that column

# Apply that function across axis 1 (columns)

resources['Resource Value'] = ""

def multiplier(row):
    prod = row['Resource Quantity'] * row['Resource Unit Price']
    return prod

resources['Resource Value'] = resources.apply(multiplier,axis=1)


# _Now I have a column in the 'resources' dataframe that corresponds to the resource value, but I want to see these metrics by each unique project._
# 
# The idea here is to see what type of resources are purchased for projects and from where.  The name of the resources might be interesting later.

# In[ ]:


# Group-by project id to look at the aggregate resource value of a project

# Rename column names

# Reset index and display first 10 rows

projectmeans = resources.groupby(['Project ID']).agg([np.sum,np.mean])
projectmeans.columns = list(['Resource Quanity Sum','Resource Quantity Mean',
                             'Resource Unit Price Sum','Resource Unit Price Mean',
                             'Resource Value Sum','Resource Value Mean'])
projectmeans=projectmeans.reset_index(level=0)
projectmeans.head()


# In[ ]:


# Looking for outliers with the project's prices, quantity, and value

projectmeans.describe()


# ### Distribution of Resource Value by Project

# In[ ]:


# Histogram of minimum through the IQR to see the majority of the total Resource Values by project

sns.distplot(projectmeans['Resource Value Sum'][projectmeans['Resource Value Sum']<6.56*100])
plt.show()


# Here we can see how often certain total resource values of projects have happened in the past.  There is a beginning 'spike' in the value of projects about 100 dollar, and it generally tapers off at around 650 dollars.  The next thing to do would be to see who these resources are purchased from for these projects.

# ### Spread of Resource Values Across Vendors

# In[ ]:


# Pre-format DataFrame
resources_df = resources[['Resource Vendor Name','Resource Value']]

# Boxplot
plt.figure(figsize=(20,10))
sns.boxplot(y="Resource Vendor Name",x="Resource Value",data=resources_df)
plt.show()


# There are a few vendors that have a large spread of resource values and some that do not.  I'm taking note of **CDW-G**, **Woodwind & Brasswind**, **Best Buy**, and **School Specialty**.

# ### Resource Value Heatmap

# In[ ]:


resources['Resource Quantity'].describe()


# A basic description (5 number summary) tells us that resources are usually bought in quantities of one or two, so are these expensive items or inexpensive items?
# 
# _That is the question I'm looking to answer with a heatmap_

# In[ ]:


# Defining a function to fill a blank pandas column value based on a condition.  

# This is the only way I know to do this right now, any tips or tricks would be great.

def f(row):
    if row['Resource Quantity'] == 0:
        val = "Zero"
    elif row['Resource Quantity']  ==1:
        val = "One"
    elif row['Resource Quantity'] ==2:
        val = "Two"
    else:
        val = "Multiple"
    return val


# In[ ]:


# More data manipulations

resources['Quantity Label']=""
resources['Quantity Label'] = resources.apply(f,axis=1)
resourceheat=pd.pivot_table(resources,values="Resource Value", columns=['Quantity Label'],index='Resource Vendor Name',aggfunc=np.mean)
resourceheat=resourceheat.drop('Zero',axis=1)


# In[ ]:


#Create the heatmap

plt.figure(figsize=(20,10))
sns.heatmap(resourceheat.drop('MakerBot'),annot=True,linewidths=1,linecolor='black',cmap='YlGn',fmt='f')
plt.title('Heatmap: Resource Value for Vendor and Quantity of Resource')
plt.show()


# Well, Lego has high value resources when purchased in multiple increments, so these projects might be large quantity purchases of many LEGO products from LEGO Education.  CDW-G has high value resource values across all quantities, meaning that this company is likely supplying high priced products.  Similarly for Best Buy.  But, Woodwind and Brasswind has a consistent value across quantities for projects.

# ### Conclusion
# 
# The resources table is conveniently presented and easy to manipulate.  Looking at resources for projects, there are numerous vendors and even more individual resources that have been purchased for these projects.  To make looking at the resources easier, I decided to aggregate by project and resource vendor and we can see that certain vendors are high price vendors and high value resource vendors.  Projects start at around 100 dollars value and mostly taper off and end around 650 dollars, although there are some outliers.

# In[ ]:




