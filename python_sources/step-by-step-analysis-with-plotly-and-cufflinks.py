#!/usr/bin/env python
# coding: utf-8

# ## Step by step Data Analysis using pandas,numpy,seaborn,maplotlib, Plotly and cufflinks
# 
# This is the step by step tutorial(sort of) to get started with data analysis.
# 
# It wont be too basic and dont expect it to be a pro notebook also.
# 
# The aim is to get you started, if you havent yet. atleast to provide a introduction to some libraries.
# 
# 
# #### Thank you for checking out.

# Kaggle script runner doesn't have the plotly and cufflinks module, so i have commented that part of the code.
# You can download this notebook and then uncomment or can follow along or just simply copy paste the code in your note book.
# 
# you can find that code under this #***** tag (Hash followed by 5 stars)
# 
# Remember to install plotly and cufflinks libraries first.
# 
# conda install plotly
# conda install cufflinks
# or
# pip install plotly
# pip install cufflinks
# 

# ### Import the all important libraries
# 

# In[ ]:


import pandas as pd
import numpy as np 

import seaborn as sns 
import matplotlib.pyplot as plt
#*****
'''
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf
# For Notebooks
init_notebook_mode(connected=True)
# For offline use
cf.go_offline()
'''
#*****

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Read the dataset into the variable(DataFrame)
pokemon = pd.read_csv('../input/Pokemon.csv')


# It is good habbit to have a good look at the data in the beginning itself so that we can plan our process

# In[ ]:


#Just have the idea about the dataset 
pokemon.head()

# we can see all the different coluimns and we can also see that one pokemon has type 2 value missing
#lets investigate further


# In[ ]:


#Check the datatype and number of columns
pokemon.info()


# We can see here that type 2 column has only 414 values , and thats okay because we know that not every pokemon can have more than one types .

# In[ ]:


pokemon.describe()


# In[ ]:


#we can see the maximum , minimum, mean, std.... all the value and can have some idea about the range of the all numerical fields


# ## EDA
# ### Lets start plotting some plots 
# #### But let me first show you the power of plotly

# In[ ]:


#*****
#pokemon[['Speed','Total']].iplot(kind='spread')
#*****


# ### Move your cursor over the plot or select some part on the plot to zoom in and then double click to zoomout
# Such interactive plots can be achieved with the help of plotly 

# ### Okay now lets start the analysis

# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


sns.pairplot(data=pokemon.drop(['Name','Type 1','Type 2','Legendary','Generation','#'],axis=1),palette='rainbow')


# ### Here we can see the relation between all the numerical fields
# but this is does not give a clear picture
# lets try with adding the hue parameter

# In[ ]:


sns.pairplot(data=pokemon.drop(['Name','Type 2','Legendary','Generation','#'],axis=1),hue='Type 1')


# Okay this is fine lets go and check someother method to get more clearinformation

# ### Groupby
# we can use groupby function to look for the aggregate values for each type 1

# In[ ]:


#Here i am dropping some of the columns which i might not need now
pkmn_num = data=pokemon.drop(['Name','Type 2','Legendary','Generation','#'],axis=1)
p_mean = pkmn_num.groupby('Type 1').mean()
p_mean


# In[ ]:


plt.figure(figsize=(10,8))
sns.barplot(y='Type 1',x='Attack',data=pkmn_num)


# Here we can see that dragon pokemon has high mean value for attack and Fairy has the least

# Lets check violin and swarmplot

# In[ ]:


plt.figure(figsize=(10,8))# this lets us set the size of the figure(atleast for this notebook)
sns.violinplot(x='Type 1',y='Total',data =pkmn_num)


# Well this looks nice but x axis values(type 1)doesnot look good, there are values which are overlapping each other.
# we can use tight layout method to fix this problem

# In[ ]:



plt.figure(figsize=(10,8))
sns.violinplot(x='Type 1',y='Total',data =pkmn_num)
plt.tight_layout()


# Yes now it looks good

# In[ ]:


plt.figure(figsize=(14,8))
sns.swarmplot(x='Type 1',y='Total',data =pkmn_num,palette='viridis',size=4)
plt.tight_layout()


# now lets try putting these two plot together

# In[ ]:


plt.figure(figsize=(12,8))
sns.violinplot(x='Type 1',y='Total',data =pkmn_num)
sns.swarmplot(x='Type 1',y='Total',data =pkmn_num,color='white',size=4)
plt.tight_layout()


# ### lets categorize pokemon on the basis of the mean of total column
# Lets make a column to check whether the given pokemon has the 'total' column value above the mean value of that column or not
# 
# we will use apply function for that.
# 
# Apply function allows us to apply function on the columns(sounds confusing as well as interesting) . they are very helpful in many scenarios and we can write our own functions, and implement complex logics also
# 
# Below example might clear things out.

# In[ ]:


# creating a new column as 'Abv_avg' and assigning it values 
pkmn_num['Abv_avg'] = pkmn_num['Total'].apply(lambda x : x>pkmn_num['Total'].mean() )


# In[ ]:


#Apply example
#The above logic could be implemented this way also
#************************************************
'''
avg = pkmn_num['Total'].mean()
def abv(col):
    if col> avg:
        return True
    return False
pkmn_num['Abv_avg'] = pkmn_num['Total'].apply(abv)
'''
#**************************************************

#But right now lambda will be just fine, since we dont need to implement any complex logic


# In[ ]:


pkmn_num.head()


# #### Now lets use this column for plotting 

# In[ ]:


plt.figure(figsize=(15,8))
sns.boxplot(x='Type 1',y='Total',data=pkmn_num,hue='Abv_avg')


# ### Here we can see that mostly flying pokemon are above average 
# Earlier we saw that dragon pokemon are more attacking than anyone and now we can see that flying pokemon are generally above average.
# 
# So next time if you go catching pokemon, try looking for flying dragon pokemon(atleast thats what i would do).
# but maybe you like less attacking pokemon and more defensive pokemon
# 
# I would recommend you to go and try something different , maybe you will find something more interesting

# ## Now lets chevk out some cool and interactive plots with plotly and cufflinks
# #### They are very easy to plot , you have to just take your dataframe and use 'iplot' on it and then provide the kind of plot

# In[ ]:


#*****
#pkmn_num.drop(['Type 1','Total'],axis=1).iplot(kind='box')
#*****


# In[ ]:


#lets look at one more interesting plot
#*****
'''
pk3d = pd.DataFrame({'x':pkmn_num['Attack'][0:5],'y':pkmn_num['Defense'][0:5],'z':pkmn_num['Speed'][0:5]})
pk3d.iplot(kind='surface')
'''
#*****


# You can drag this 3d figure and play with it.

# ### Thank you for following, I hope this helped you

# In[ ]:




