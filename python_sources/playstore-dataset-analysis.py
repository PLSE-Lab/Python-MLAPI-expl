#!/usr/bin/env python
# coding: utf-8

# # Data Preparation

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/appstore_games.csv')


# In[ ]:


data.head(20)


# In[ ]:


data.isnull()


# In[ ]:


data.dtypes


# In[ ]:


df = pd.DataFrame(data)


# # Interference
# With the dtypes method we could draw the interference that there some 5 columns that have numerical values.
# We have columns ID,Average User Rating, User Rating Count, Price and Size for futher processing.

# In[ ]:


working_data = df.select_dtypes(include='float64')


# In[ ]:


working_data


# In[ ]:


df.info()


# In[ ]:


working_data.info()


# # Interference
# We could see that in the average user rating and user rating count we have maximum null values as compared to the other numerical valued columns hence we could only have these values(7561) for our further analysis.
# Note: We could also replace the null values with the mean or median or mode values but it will interfere with the best results.

# In[ ]:


working_data2 = working_data.dropna()


# Now we have the working_data2 as the new data set values that don't have any null values.

# In[ ]:


working_data2.info()


# In[ ]:


working_data2.isnull()


# In[ ]:


working_data2.describe()


# Displaying the statistics

# # Analysis

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


working_data2.head(15)


# In[ ]:


sns.heatmap(working_data2.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# we can see through the heatmap that we don't have any null values

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Average User Rating',data=working_data2);


# So this tells that max average user rating is 4.5 or above 4.5

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Average User Rating',hue = 'Price',data=working_data2);


# In[ ]:


ax=sns.distplot(working_data2['Average User Rating'],rug=True,rug_kws={"color":"g"},kde_kws={"color":"k","lw":3,"label":"KDE"},hist_kws={"histtype":"step","linewidth":3,"alpha":1,"color":"y"})


# In[ ]:


g=sns.relplot(x="Price",y="Average User Rating",data=working_data2)


# # Interference
# Here the visualization tells us that:

# a) We have all kinds of rating by users for a Rs0 or free games

# b) For games having a cost of Rs 20 or higher we have a minimum rating of 3.5 or above that.

# In[ ]:


sns.relplot(x="Price",y="User Rating Count",hue="Average User Rating",data=working_data2);


# This Scatter Visualization tells us that there are approximately 3M users that prefer to use free apps against handful of the users that use paid apps

# In[ ]:


g=sns.relplot(x="Price",y="Average User Rating",data=working_data2,kind="line");


# Majority of games having 4+ rating are having the cost less than Rs20

# In[ ]:


sns.catplot(x="Average User Rating",y="Price",kind="boxen",data=working_data2.sort_values("Size"));


# In[ ]:


# Taking values between 0 and 20 to draw more insights on the data avaliable for precise results
filtered_data = working_data2[working_data2['Price'].between(0, 20)]


# In[ ]:


#Now analysing this data set
sns.catplot(x="Average User Rating",y="Price",kind="boxen",data=filtered_data.sort_values("Size"));


# Now we could see that the games having price less than 5 have all kinds of reviews ranging from 1 to 5

# 

# In[ ]:


sns.catplot(x='Price',y='Average User Rating',kind = "box",data=filtered_data.sort_values("Size"));


# In[ ]:


#filtered_data2 by user rating count
filtered_data2 = filtered_data[filtered_data['User Rating Count'].between(0,50000)]


# In[ ]:


#Now analysing this data set
sns.catplot(x="User Rating Count",y="Price",kind="boxen",data=filtered_data2.sort_values("Size"));


# In[ ]:


ax=sns.catplot(x='Price',y='Average User Rating',kind = "box",data=filtered_data.sort_values("Size"));
ax.set(ylim=(2, 5));


# In[ ]:


ax=sns.catplot(x='Price',y='Average User Rating',kind = "box",data=filtered_data.sort_values("Size"));
ax.set(ylim=(3.5, 4.5));


# In[ ]:


sns.catplot(x="Average User Rating",y="Price",kind="bar",data=filtered_data);


# In[ ]:


sns.set(color_codes=True)
sns.distplot(filtered_data['Average User Rating']);


# In[ ]:


sns.set(color_codes=True)
sns.distplot(filtered_data['Average User Rating']);


# In[ ]:


sns.lmplot(x="Price",y="Average User Rating",data=filtered_data,y_jitter=0.3);


# In[ ]:


g=sns.FacetGrid(filtered_data2,col="Average User Rating",height=10,aspect=.5)
g.map(sns.barplot,"Price");


# In[ ]:


###########################################################################################################################


# # Final Conclusions - 
# 

# 1. On the analysis of genres we found out that, Price and Average User Rating are the two of the most determining factors for the processing and analysising data.

# 2. On further analysis of the genres we got to know that games below 20Rs or the free games have a remarkably higher User Rating Count and have been used by the majority of users. Addition to that games below 5Rs have a relative high user count and better review rating.

# 3. On dealing with the Size of a game it is noticed that free games with large size are generally preferred by players.

# 4. We could derive a final statement that games which have large size and are free will generally have greater than 4 as Average User Rating.
