#!/usr/bin/env python
# coding: utf-8

# In this notebook I have performed Exploratory Data Analysis on the **Child Labour in India** dataset and tried to identify various trends and features from the given dataset.

# I hope you find this kernel helpful and some **<font color='red'>UPVOTES</font>** would be very much appreciated

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Importing required libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Describing the Dataset
# 

# #### Loading the data

# In[ ]:


df = pd.read_csv('/kaggle/input/child-labour-in-inida/Child Labour in India.csv')


# Let's take a quick a look at the data

# In[ ]:


df.head(3)


# In[ ]:


df.tail(3)


# #### Dimensions of the dataset

# In[ ]:


print('Number of rows in the dataset: ',df.shape[0])
print('Number of columns in the dataset: ',df.shape[1])


# #### Features of the data set

# In[ ]:


df.info()


# **The features present in the above dataset are:**
# 
# **1. Category of States:** The states are classified as special category or non special category states. A Special Category Status (SCS) is a classification given by Central Government to assist in the development of those states that face geographical and socio-economic disadvantages like hilly terrains, strategic international borders, economic and infrastructural backwardness, and non-viable state finances.
# 
# **2. States:** Name of the state.
# 
# **Rest of the columns contain the percentage of children employed as child labour in various professional categories**
# 
# **The total column is the summation of percentages of child labor employed in each state in each professional category and is always equal to 100**

# **Also the last row in the dataset contains the all india average of child labour across various categories**

# Renaming the **Category of States** as **Category** and also replacing the values present in the column from **Special Category States** to **Special** and **Non Special Category states** to **Non Special**

# In[ ]:


df.rename(columns={
    'Category of States':'Category'
}, inplace=True)


# In[ ]:


# Renaming the values present in the category column
df['Category'] = df['Category'].replace(['Non Special Category states', 'Special Category States'],
                                      ['Non Special', 'Special'])


# Also the type of values present in the column Manufacturing is object.Changing manufacturing type from object to float

# In[ ]:


df['Manufacturing'] =df['Manufacturing'].replace('9. 9', '9.9')
df['Manufacturing'] = df['Manufacturing'].astype('float')


# Also dropping the **Total** column present in the dataset.

# In[ ]:


df.drop('Total', axis=1, inplace=True)


# Let's take the all india category as a seperate dataset and remove it from the rest of the dataset.

# In[ ]:


india = df.loc[df['Category'] == 'All India']
india


# Now lets remove the **'All India'** value from the rest of the dataset.

# In[ ]:


df =df[df['Category'] != 'All India']


# ### Exploratory Data Analysis

# #### Number of States in the dataset

# In[ ]:


df['States'].nunique()


# #### Percentage of Special and Non-Special Category of States

# In[ ]:


special =len(df[df['Category'] == 'Special'])
non_special = len(df[df['Category']== 'Non Special'])

plt.figure(figsize=(8,6))

# Data to plot
labels = 'Special','Non Special'
sizes = [special, non_special]
colors = ['skyblue', 'yellowgreen']
explode = (0, 0.2)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True)
plt.title('Percentage of Special and Non Special States in the Dataset')
plt.axis('equal')
plt.show()


# #### Distribution of child labour across various profession categories

# In[ ]:


sns.set_style('darkgrid')
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16,8), sharey=True)
sns.distplot(df['Agriculture'], kde=False, bins=15,color='red', hist_kws=dict(edgecolor="black", linewidth=2),ax=ax[0][0])
sns.distplot(df['Manufacturing'], kde=False, bins=15,color='blue', hist_kws=dict(edgecolor="black", linewidth=2),ax=ax[0][1])
sns.distplot(df['Construction'], kde=False, bins=15,color='green', hist_kws=dict(edgecolor="black", linewidth=2),ax=ax[0][2])
sns.distplot(df['Trade Hotels & Restaurants'], kde=False, bins=15,color='purple', hist_kws=dict(edgecolor="black", linewidth=2),ax=ax[1][0])
sns.distplot(df['Community, Social and Personal Services'], kde=False, bins=15,color='aqua', hist_kws=dict(edgecolor="black", linewidth=2),ax=ax[1][1])
sns.distplot(df['Others'], kde=False, bins=15,color='gold', hist_kws=dict(edgecolor="black", linewidth=2),ax=ax[1][2])
plt.tight_layout()


# The distribution plots show that most of the children employed in child labour work in agriculture related work. Most of the employement percentage in agriculture ranges from 60 to 80 percent whereas in other categories it lies between 0 to 10 percent.

# In[ ]:


# Function to Draw pie chart in terms of Special and Non Special Categories
def draw_piechart(feature):
    special =df[df['Category'] == 'Special'][feature].mean()
    non_special =df[df['Category']== 'Non Special'][feature].mean()

    plt.figure(figsize=(8,6))

    # Data to plot
    labels = 'Special','Non Special'
    sizes = [special, non_special]
    colors = ['skyblue', 'yellowgreen']
    explode = (0, 0.1)  # explode 1st slice

    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=True)
    plt.title('Percentage of children employed in ' +feature+' category in Special and Non Special category States')
    plt.axis('equal')
    plt.show()


# In[ ]:


#Function to draw a barchart in terms of states
def draw_barchart(feature):
    plt.figure(figsize=(15, 6))
    sns.barplot(x='States', y=feature, data=df, edgecolor='black', order=list(df.sort_values(by=feature,ascending=False)['States']))
    plt.title('Percentage of Children employed in ' + feature + ' category in various states')
    plt.xlabel('States')
    plt.ylabel('Percentage of children working')
    plt.xticks(rotation=90)
    plt.show()


# Let's analyze each of the given profession categories one by one in terms of Category of States and States.

# #### 1. Agriculture

# #### i. In terms of Categories of States

# In[ ]:


draw_piechart('Agriculture')


# More than half of the children working in agriculture category belong to states which have been provided with special status by the government.

# #### ii. In terms of States

# In[ ]:


draw_barchart('Agriculture')


# #### 2. Manufacturing

# #### i. In terms of Categories of States

# In[ ]:


draw_piechart('Manufacturing')


# #### ii. In terms of States

# In[ ]:


draw_barchart('Manufacturing')


# **Tamil Nadu** and **West Bengal** have the most number of child labourers working in the Manufacturing Industry. Comparing this with the Agricultural sector where Tamil Nadu and West Bengal are one of the lowest child labour employers.

# #### 3. Construction

# #### i. In terms of Categories of States

# In[ ]:


draw_piechart('Construction')


# #### ii. In terms of States

# In[ ]:


draw_barchart('Construction')


# **Haryana** has the highest number of child labourers in the Construction category in India.
# 
# It's also interesting to see that in construction category the percentage of children working ranges from 0 to 10 % where as in agriculture and manufacturing it ranges upto 80 and 40% respectively.

# #### 4. Trade Hotels & Restaurants

# #### i. In terms of Categories of States

# In[ ]:


draw_piechart('Trade Hotels & Restaurants')


# #### ii. In terms of States

# In[ ]:


draw_barchart('Trade Hotels & Restaurants')


# **Delhi** has a staggering almost 60% child labourers working in Hotels and Resturants. All the other states except Kerela have less than 20% child labourers working in Hotels and Resturants.

# #### 5. Community, Social and Personal Services

# #### i. In terms of Categories of States

# In[ ]:


draw_piechart('Community, Social and Personal Services')


# #### ii. In terms of States

# In[ ]:


draw_barchart('Community, Social and Personal Services')


# In this category also **Delhi** ranks on top for having the most child labourers. Rest all the states have percentages around 15% or less.

# #### 6. Others

# #### i. In terms of Categories of States

# In[ ]:


draw_piechart('Others')


# #### ii. In terms of States

# In[ ]:


draw_barchart('Others')


# #### Stacked Bar Chart of Child labourers employed across various states.

# In[ ]:


fig,ax = plt.subplots(figsize=(17,8))
ax.bar(df['States'],df['Agriculture'],color='#70C1B3',label='Agriculture')
ax.bar(df['States'], df['Manufacturing'], bottom=df['Agriculture'], color='#247BA0', label='Manufacturing')
ax.bar(df['States'], df['Construction'], bottom=df['Agriculture']+df['Manufacturing'], color='#FFE066',label='Construction')
ax.bar(df['States'], df['Trade Hotels & Restaurants'], bottom=df['Agriculture']+df['Manufacturing']+df['Construction'], color='#F25F5C', label='Trade Hotels & Restaurants')
ax.bar(df['States'], df['Community, Social and Personal Services'], 
       bottom=df['Agriculture']+df['Manufacturing']+df['Construction'] + df['Trade Hotels & Restaurants'], color='#50514F', label='Community, Social and Personal Services')

ax.bar(df['States'], df['Others'], 
       bottom=df['Agriculture']+df['Manufacturing']+df['Construction'] + df['Trade Hotels & Restaurants']+df['Community, Social and Personal Services'],
       color='#A1CF6B', label='Others')

# ax.bar(df['States'],df['Other'])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=3, mode="expand", borderaxespad=0.)
plt.xticks(rotation=90)
plt.ylim((0, 110))
plt.show()


# I will update the notebook with more analysis and trends in the future. If you found this notebook useful then please <font color='red'>UPVOTE</font> the notebook.
# 
# Suggestions are welcome
