#!/usr/bin/env python
# coding: utf-8

# # Take a first look at the data
# ________
# 
# The first thing we'll need to do is load in the libraries and datasets we'll be using. For today, I'll be using a dataset of events that occured in American Football games for demonstration, and you'll be using a dataset of building permits issued in San Francisco.
# 
# > **Important!** Make sure you run this cell yourself or the rest of your code won't work!

# In[1]:


# modules we'll use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read in all our data
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 


# The first thing I do when I get a new dataset is take a look at some of it. This lets me see that it all read in correctly and get an idea of what's going on with the data. In this case, I'm looking to see if I see any missing values, which will be reprsented with `NaN` or `None`.

# In[ ]:


#Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?
sf_permits.sample(5)
# your code goes here :)


# # See how many missing data points we have
# ___
# 
# Ok, now we know that we do have some missing values. Let's see how many we have in each column. 

# In[2]:


# your turn! Find out what percent of the sf_permits dataset is missing
missing_values_in_sf = sf_permits.isnull().sum()
(missing_values_in_sf.sum()/np.product(sf_permits.shape))*100


# # Figure out why the data is missing
# ____
#  
# This is the point at which we get into the part of data science that I like to call "data intution", by which I mean "really looking at your data and trying to figure out why it is the way it is and how that will affect your analysis". It can be a frustrating part of data science, especially if you're newer to the field and don't have a lot of experience. For dealing with missing values, you'll need to use your intution to figure out why the value is missing. One of the most important question you can ask yourself to help figure this out is this:
# 
# > **Is this value missing becuase it wasn't recorded or becuase it dosen't exist?**
# 
# If a value is missing becuase it doens't exist (like the height of the oldest child of someone who doesn't have any children) then it doesn't make sense to try and guess what it might be. These values you probalby do want to keep as NaN. On the other hand, if a value is missing becuase it wasn't recorded, then you can try to guess what it might have been based on the other values in that column and row. (This is called "imputation" and we'll learn how to do it next! :)
# 

# ## Your turn!
# 
# * Look at the columns `Street Number Suffix` and `Zipcode` from the `sf_permits` datasets. Both of these contain missing values. Which, if either, of these are missing because they don't exist? Which, if either, are missing because they weren't recorded?
# 
# > Here, it is really up to the content. In some places there may not be a Street Number Suffix. I believe a better way of deciding whether that a value is missing or does not exist shall be based on its overall presence percentage; if a value is NaN for over 70% of total row numbers, I will say that the value does not exist. Yet I will first obey the rules and try to see what happens when I remove NaNs and so on...

# # Drop missing values
# ___
# 
# If you're in a hurry or don't have a reason to figure out why your values are missing, one option you have is to just remove any rows or columns that contain missing values. (Note: I don't generally recommend this approch for important projects! It's usually worth it to take the time to go through your data and really look at all the columns with missing values one-by-one to really get to know your dataset.)  
# 
# If you're sure you want to drop rows with missing values, pandas does have a handy function, `dropna()` to help you do this. Let's try it out on our NFL dataset!

# In[ ]:


# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits.dropna()
# Oh no data left! how suprising!


# In[ ]:


# Now try removing all the columns with empty values. Now how much of your data is left?
drop_columns_sf=sf_permits.dropna(axis=1)

print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % drop_columns_sf.shape[1])


# Well, losing 3/4 of your columns while getting rid of NAs is not a very good way of analysing the data. I will try to fill those NaN values.

# # Filling in missing values

# Filling in missing values is also known as "imputation", and you can find more exercises on it [in this lesson, also linked under the "More practice!" section](https://www.kaggle.com/dansbecker/handling-missing-values). First, however, why don't you try replacing some of the missing values in the sf_permit dataset?
# 
# I will follow the way I defined above. before imputing, i will first try to see what values are actually missing. here my threshold will be 50%

# In[3]:


# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then replacing any remaining NaN's with 0

missing_percentages = (missing_values_in_sf / sf_permits.shape[0])*100
print(missing_percentages)


# In[4]:




sf_permits_toModify = sf_permits.copy()

for colname,perc in enumerate(missing_percentages):
    if perc > 70:
        print(missing_percentages.index[colname])
        #remove those columns one by one by the following
        sf_permits_toModify.drop(missing_percentages.index[colname],axis=1, inplace=True)
#this will tell me, or guide me for the columns whose NaN actually do not exist rather then missing. 
print("I do not see a reason to keep these columns in my dataframe after checking what they are")


# Before imputing, I need to see which columns are numbers and strings as I might use a "mean" or a "median" method to fill the NaNs. for this purpose I will check the type of the columns one by one

# In[5]:


missing_values_after_drop_sf = sf_permits_toModify.isnull().sum()
missing_percentages_after_drop = (missing_values_after_drop_sf / sf_permits_toModify.shape[0])*100
print(missing_percentages_after_drop)


# After the drop, what we see is it is only the Completed Date with 51% missing. We may need to take a look at this later on!

# In[6]:


for i in sf_permits_toModify.columns:
    if sf_permits_toModify[i].dtype != "object":
        print(i, sf_permits_toModify[i].dtype,"\n")


# Starting from the first numeric column, I want to understand whether there is a dominance pattern or equally shared. 

# In[7]:


from collections import Counter
Counter(sf_permits_toModify.iloc[:,1])


# Obviously, the Permit Type column is not made of continuous values. So this could actually be something that we could later on predict based on our other values. I will leave this columns like this for now. 
# 
# When I look at the Number of Stories existing and proposed, they match in a very high percentage.

# In[8]:


stories = sf_permits_toModify.iloc[:,16:18]
noNas_stories = stories.dropna()
C = np.where(noNas_stories.iloc[:,0] == noNas_stories.iloc[:,1],"yes","no")
yes_no=Counter(C)
print("the percentage of perfect match is: ",(yes_no["yes"]/(yes_no["yes"]+yes_no["no"]))*100)


# So, I believe having a correlation plot to see which columns are highly correlated so that we can merge them

# In[ ]:


merged = stories.iloc[:,0].fillna(stories.iloc[:,1])


# In[9]:


plt.matshow(sf_permits_toModify.corr())
plt.colorbar()
plt.xticks(range(len(sf_permits_toModify.corr().columns)), sf_permits_toModify.corr().columns, rotation='vertical')
plt.yticks(range(len(sf_permits_toModify.corr().columns)), sf_permits_toModify.corr().columns);
sf_permits_toModify.corr()


# In[16]:


df_c = sf_permits_toModify.corr()[sf_permits_toModify.corr()>0.9]
df_c =df_c[df_c<1]
#print(df_c)
for cols in sf_permits_toModify.corr():
    df_c1 = df_c[cols].notnull()

    for ii in range(len(df_c)):
        if df_c1[ii]==True:
            print('---')
            print('\n'+cols)
            print((list(sf_permits_toModify.corr())[ii]))
            
            






# In[ ]:




