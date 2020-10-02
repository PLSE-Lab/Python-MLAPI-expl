#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This kernel will do some basic data cleaning on the Ecommerce Daily Orders Data. Not sure if I caught everything, but would be glad to have feedback!

# Importing the relevant libraries:

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
import seaborn as sns
sns.set()


# Loading the data:

# In[ ]:


df = pd.read_csv("../input/ecommerce-bookings-data/ecommerce_data.csv")
df.head(10)


# In[ ]:


df.dtypes


# Checking the dataframe for missing data: 

# In[ ]:


df.isnull().sum()


# # Dealing with Outliers

# In[ ]:


df.hist(figsize = (10,6))
plt.subplots_adjust(hspace = 1, wspace = 1, top = 0.9)
plt.tight_layout()
plt.suptitle('Histograms of Features')


# We can see that there are appear to be some significant outliers in the `orders` column of our dataframe. Let's look a bit closer at the distribution of that column to get an idea where the outliers are located:

# In[ ]:


fig, ax = plt.subplots(2,2, figsize = (10,6))
fig.tight_layout()
fig.suptitle('Comparison of Distributions below Specific Quantiles of `order`')
fig.subplots_adjust(top = 0.85, hspace = 0.3, wspace = 0.3)
j = 0
i = 0
quantile_list = [0.8, 0.9, 0.95, 0.99]
for q in quantile_list:
    ax[i,j].hist(df['orders'][df['orders']<df['orders'].quantile(q)])
    ax[i,j].set_title("Below {}$^{{th}}$ Percentile".format(int(q*100)))
    if j < 1:
        j+=1
    else:
        i+=1
        j-=1


# It is clear that this column has a strong positive skew, which can often be expected when we look at something like purchases: most people will only buy a small number of items. Although the skew is inherent to this kind of data, we still have some substantial outliers. The largest order value was nearly 127,000! But as we saw above, the large majority of our data is was below even just 200 purchases.

# In[ ]:


df['orders'].describe().round(1)


# Based on the above findings, I am going to do two things that make our data easier to work with ***and*** preserve our ability to investigate high-volume purchases in the future. First, I will create a dummy variable that indicates if this particular city and date had a high-volume of purchases (i.e. greater than the 95th percentile).  Second, I will 'cap' the orders column at the 95th percentile. That is, I will replace all values above the 95th percentile with the value occuring at that point in the distribution. And we can see at the end that just under 5% of the rows have an indicator for high-volume purchases.

# In[ ]:


cutoff = df['orders'].quantile(0.95)
# Create a dummy indicating if the number of orders was greater than 95th percentile
df['high_volume'] = (df['orders']>cutoff)*1
df.loc[(df['orders']>cutoff), 'orders'] = cutoff
df['high_volume'].value_counts(normalize = True)


# # Completing Date Information
# 

# First, I will need to set the `date` column to pandas datetime format.

# In[ ]:


df['date'] = pd.to_datetime(df['date'])


# With a simple crosstab we can already see that not all cities have purchases on the same dates in the dataframe.

# In[ ]:


pd.crosstab(df['date'], df['city_id']).head(20)


# For example, city 25 has purchases on between July 10 - 13, but the other cities in this view do not. Likewise, further down the table we can see that cities 28 and 29 had no purchases on July 25-26, while all of the others did. Thus, if we want to create a panel time series from this data, we will need to add observations into the dataframe to fill those gaps, even if those rows are filled with zeroes. 

# To figure out which dates are missing from each city, I will create a function that identifies the minimum and maximum sale date observed in each city. Then, the function will compare the set of dates in the actual dataframe to another set of dates that includes all of the possible dates between the minimum and maximum. It will then store the dates without a match in a list.

# In[ ]:


# Defines a function to identify which dates are missing for each city
def missing_date_info(city, data):
    city_df = data.loc[data['city_id']==city, :] #create a temporary dataframe for a given city

    base = city_df['date'].min() #identifies the earliest date of observation
    maxdate = city_df['date'].max() #identifies the latest date of observation
    full_date_list = [base + timedelta(days=x) for x in range((maxdate-base).days)] # creates a list of dates spanning from earliest to latest that includes every day in between

    date_vals = sorted(set(city_df['date'])) #create a list of the date values currently in the dataframe for a given city
    date_missing = [i for i in full_date_list if i not in date_vals] #identifies the dates that are not currently represented in the dataframe
    return date_missing


# Now, I will create a dictionary that will be eventually filled with the information on the missing dates for each city, as well as information for the other columns of the dataframe. I will loop over `city_id`  and, for each city, identify the missing dates using the function defined above. Then, I will iterate through those dates and add new values to the dictionary. Ultimately, I will end up having a dictionary with five keys (representing the five columns of the dataframe) and each key will be paired with the corresponding values for the missing dates. Here, I assumed that if a date was missing from the dataframe, there were 0 orders. I then used this dictionary to create a new pandas dataframe.

# In[ ]:


missing_dict = {i : [] for i in df.columns}
for x in range(df['city_id'].max()+1):
    dates = missing_date_info(x, df) #gets the missing dates for each city   
    #Appends information to the new dictionary
    for d in dates:
        missing_dict['date'].append(d)
        missing_dict['product_id'].append(np.nan)
        missing_dict['city_id'].append(x)
        missing_dict['orders'].append(0)
        missing_dict['high_volume'].append(0)

# Creates a new dataframe with only the missing date rows + other columns    
missing_rows = pd.DataFrame(missing_dict)


# We can see below that the new dataframe consists of the missing dates and the rest of the columns from the original dataframe.

# In[ ]:


missing_rows


# Now, I will append the missing dates to the original dataframe and sort the values of the dataframe by `city_id` and `date`.

# In[ ]:


# Appends missing rows to the original dataframe and sorts the dataframe by city and date
df2 = df.append(missing_rows)
df2 = df2.sort_values(['city_id', 'date']).reset_index(drop = True)


# In[ ]:


print('df had {} rows.\nAnd df2 now has {} rows.'.format(len(df), len(df2)))


# As a final check to ensure that there are no longer any missing dates in the dataframe, I will make use of the function I created once again. This time, I will create a list called `test_list`, which will indicate, for each city, if there were any dates in the dataframe that were missing from the theoretical list of dates ranging from the minimum to the maximum (as I defined in the function earlier). If there were no missing dates, the function should have returned an empty list for each city. I will then test if this is the case by using `all(test_list)`, which will indicate if all of the items in the test_list were `True` (i.e. empty lists).

# In[ ]:


test_list = []
for x in range(df2['city_id'].max()+1):
    test_list.append(missing_date_info(x, df2) == [])
print('Are all of the missing date lists empty?\nResponse: {}'.format(all(test_list)))


# It seems like it was a success. Now, I will just export the cleaned dataframe as a csv for future work.

# In[ ]:


df2.to_csv("ecommerce_clean.csv", index = False)


# # Wrapping Up
# So that is how far I got with the data cleaning task. I am not sure if there were other problems that I may have missed, but I would love to get feedback on this. Thanks a lot for reading!
