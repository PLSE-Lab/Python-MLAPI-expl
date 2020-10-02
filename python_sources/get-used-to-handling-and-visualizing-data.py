#!/usr/bin/env python
# coding: utf-8

# ## Getting used to pandas and seaborn - using superbowl data!
# This is a notebook for practicing **data handling with pandas and visualizing with seaborn library**.
# In this notebook, we are going to practice things below.
# 
#     1. Dropping, renaming, and creating columns in pandas dataframe
#     2. Counting elements' frequency in specific column
#     3. Using various plotting functions in seaborn for visualizing    
# After following notebook codes, you will be able to perform basic operations to prepare to do 'something' with given data, or getting an insight from data by visualizing it. 

# First, we are going to import essential packages for data handling and visualization.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Then, we are going to load data csv file using **pd.read_csv()** method to a dataframe called **superbowl_df**, and take a look at given data.

# In[ ]:


superbowl_df = pd.read_csv('/kaggle/input/superbowl-history-1967-2020/superbowl.csv')    #Reading superbowl dataset
superbowl_df.head(5)    #Show first five elements


# It is possible to get detalied information about each column using **info()** method.

# In[ ]:


print('DataFrame Size : ', superbowl_df.shape)    #checking dataset size
superbowl_df.info()  #getting informations about the dataset


# It seems that 'SB' column is quite meaningless, since it is just an unique number for each superbowl event. 
# 
# We can drop this column by using **drop()** method.

# In[ ]:


superbowl_df.drop(['SB'], axis=1, inplace=True)    #dropping column 'SB'
superbowl_df.head(5)


# Uselesss column has been successfully dropped!
# Let's find out the best football team - the team which won superbowl the most!
# 
# We can create new dataframe using **Winner** column in the original dataframe as index by **groupby()** method, and count the frequency of each team by using **count()** method as follows.

# In[ ]:


superbowl_groupby_winner = superbowl_df.groupby(by='Winner').count()    ##grouping data by 'Winner' coulmn value
superbowl_groupby_winner


# In order to use seaborn method for to visualize data, we use **reset_index()** method to re-initialize index.
# Then we can rename **Date** column (in fact, we can rename any column, not only **Date**) to **Count** using **rename()** method.
# Here's a modified dataframe with new index and **Count** column!

# In[ ]:


superbowl_groupby_winner.reset_index(inplace=True)    #reset index
superbowl_groupby_winner.rename(columns = {"Date": "Count"}, inplace=True)    ##remand 'Date' coulmn
superbowl_groupby_winner


# It seems **barplot()** method would fit into such type of data. 
# 
# Let's figure out which the best football team is!

# In[ ]:


plot = sns.barplot(x = 'Count', y = 'Winner', data = superbowl_groupby_winner, orient = "h").set_title('Superbowl winning teams!')


# Well, I'd also like to look for the best football player - the person nominated as MVP the most!
# 
# We can do so by repeating methods we just used.

# In[ ]:


superbowl_groupby_mvp = superbowl_df.groupby(by='MVP').count()
superbowl_groupby_mvp.reset_index(inplace=True)
superbowl_groupby_mvp.rename(columns = {"Date": "Count"}, inplace=True)


# In[ ]:


plot = sns.barplot(x = 'Count', y = 'MVP', data = superbowl_groupby_mvp, orient = "h").set_title('Superbowl MVP!')


# Oops! The plot isn't quite pretty this time. Players' names are overlapping each other.
# 
# Maybe we could resize the figure and make it little bit larger to avoid overlapping problem by **plt.figure()**.

# In[ ]:


plt.figure(figsize=(20,10))    #modify figure size
plot = sns.barplot(x = 'Count', y = 'MVP', data = superbowl_groupby_mvp, orient = "h").set_title('Superbowl MVP!')


# It seems much better!
# 
# Now, let's look at point differences between winning team losing team.
# 
# We can create new column named **point difference** easily as below.

# In[ ]:


superbowl_df['point difference'] = superbowl_df['Winner Pts'] - superbowl_df['Loser Pts']    #create new column
superbowl_df.head(3)


# For a column with numerical values, **describe()** method can be used to get some statistics of the column.

# In[ ]:


superbowl_df['point difference'].describe()    #get stats for numerical values


# For more interpretable statistics, we can draw a box plot using **boxplot()** method.

# In[ ]:


superbowl_df.boxplot(column = 'point difference')


# Congratulations! We've gone through all codes in this notebook.
# 
# Now you could probably handle new data using pandas methods, and visualize it using seaborn methods by yourselves!
