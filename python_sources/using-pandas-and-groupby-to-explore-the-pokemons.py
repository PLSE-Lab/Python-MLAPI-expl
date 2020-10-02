#!/usr/bin/env python
# coding: utf-8

# This is a short tutorial on exploratory data analysis in python using the groupby function of pandas, and using styles to create a nice table. In this notebook I will calculate the average statistics for each Type 1 category for the Pokemons
# ------------------------------------------------------------------------

# **Lets start by importing the 2 packages we need.**

# In[ ]:


import seaborn as sns
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# 
# 
# **Import the data, and view the first few rows**

# In[ ]:


pokemonData = pd.read_csv('../input/Pokemon.csv')
pokemonData.head()


# **print out the column names. This can be copy/pasted to create the 'Columns_to_include' vector further below**

# In[ ]:


pokemonData.columns


# 
# 
# 
# 
# **Now perform the grouping, and calculate the mean value , and number of rows for each value in the 'Type 1' column**

# In[ ]:


Column_to_group_by='Type 1'
Columns_to_include=['Total', 'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed','Legendary']
groups=pokemonData.groupby(Column_to_group_by)[Columns_to_include]
meanValues=groups.mean()
levelCount=groups.count()
print(meanValues)
print('--------------------------------------------------------------------------------------------')
print(levelCount)


# 
#  
#  
# 
# 
# **Concatenate the two tables. We only need one column from the levelCount table**

# In[ ]:


merged_frame = pd.concat([levelCount['Total'], meanValues], axis=1)

# add new column names to the merged dataframe
new_col_names=['NumRows','TotalPoints', 'HP', 'Attack', 'Defense','SpecialAttack', 'SpecialDefence', 'Speed','%Legendary']
merged_frame.columns=new_col_names

# sort the results by the Total_points
merged_frame.sort_values("TotalPoints",ascending=False,inplace=True)
merged_frame


# **now lets add some style to the table**

# In[ ]:


#Start with rounding to 2 decimals
merged_frame.style.format("{:.2f}")


# In[ ]:



#Add that the %Legend colums should be printed as percentages with one decimal
merged_frame.style.format("{:.2f}").format({'%Legendary': '{:.1%}'})


# In[ ]:


#And now add some colour coding to make the table easier to read.
cmap=sns.diverging_palette(250, 5, as_cmap=True)

merged_frame.style.format("{:.2f}").format({'%Legendary': '{:.1%}'}).background_gradient(cmap, axis=0)


# **I hope this has been useful if you have not used the groupby function in pandas before...** 
