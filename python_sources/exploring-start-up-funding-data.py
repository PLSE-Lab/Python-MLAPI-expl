#!/usr/bin/env python
# coding: utf-8

# # Exploring Startup Funding Data
# 
# Welcome to an exploratory data analysis of the current climate of startups, according to data from the popular business information website Crunchbase. The goal of this notebook is to hopefully teach you something new about the startup funding environment, while also providing some insight into pandas functions through tutorial-like steps.
# 
# 
# In particular, the questions I am curious to explore is:
#  
#  - What countries are most startups coming from?
#  - Is debt a factor in the success of Canadian startups?
# 
# 
# 
# Enjoy!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.set_option('mode.chained_assignment', None)

# Any results you write to the current directory are saved as output.


# **1. First, lets load the data using the `read_csv` function, and create a Data Series called *'startups'*.**

# In[ ]:



startups = pd.read_csv('/kaggle/input/startup-investments-crunchbase/investments_VC.csv', encoding= 'unicode_escape')


# **2. Next, to get a sense of what our data series looks like, lets use the `.head()` function and `.shape` function. These give us a preview of the first 5 rows, and tell us how many rows/coloumns there are in total, respectively.**
# 
# 
# 

# In[ ]:


startups.head()


# In[ ]:


startups.shape


# **You will see that our first output contains a table with information on each business in the dataset, such as its name, market, country of origin, and various figures related to funding, among others.**
# 
# **Our second ouput has two numbers: the first corresponds to rows, and the second to columns. In this case, rows represent a company and columns represent a category of data.**

# **3. Now, lets address my first exploratory question of what countries are producing the most startups.** 
# 
# **We'll use the `groupby` function. This allows us to split our dataset into certain groups, such as countries. From there, we can evaluate how many companies are *in* each group, with the `.size` function, which is known as a frequency count. Then, we'll order those values with the `sort_values` function. Within `sort_values`, we indicate that `ascending = False`, because we want our order to be high to low.**
# 
# **I want to highlight the importance of order in this line of code. This strategy is known as split-apply combine, and you must follow that order. `Groupby` is the 'split', `.size()` is the 'apply', and `sort_values` applies to the 'combine'.**
# 
# **We'll also use the `head()` function again, to give us an output of just the first 10 rows**

# In[ ]:


startups.groupby("country_code").size().sort_values(ascending=False).head(10)


# **4. If we want to make this a bit easier to see, we can utilize the `.plot` function, which gives us a wide array of graphs to choose from to visualize our data. One example is a `.pie` chart, which can be seen below.**
# 
# **Again, pay attention to the order. We only apply `.plot` once we have already 'split'(`groupby`) and 'applied'(`.size`) .**

# In[ ]:


startups.groupby("country_code").size().sort_values(ascending=False).head(10).plot.pie()


# **With the top ten startup countries now known, lets dive a little deeper into a certain country. Since I am Canadian, we'll do Canada.**
# 
# **5. Lets create a new DataSeries out of our previous *'startups'* series, that only includes startups with a country code of *'CAN'* (Canadian). This is called a *conditional selection*.**
# 

# In[ ]:


canadian_startups = startups[startups.country_code == 'CAN']
canadian_startups.head()


# **6. We'll use this new DataSeries to explore my second question: how does debt financing affect the success of a startup in Canada?**
# 
# **Before we answer that, lets look into a certain company. How do we find 'debt_financing' for the company "AQUA PURE"? **
# 
# **We'll have to use a function that allows us to select data. `loc` allows us to specify a cell we want, by taking two inputs. The first corresponds to a row of the index, and the second to a column. But first, using `set_index` we set the index to 'name', so we can navigate using the "AQUA PURE" business name.**

# In[ ]:




canadian_startups.set_index('name').loc["AQUA PURE",'debt_financing']


# It appears AQUA PURE had$3 million in debt-financing. Is this usual for the Clean Technology industry?
# 
# **7. To find out more about the average level of debt financing for Clean Technology and  all other markets, let's once again use the groupby function, and select the debt financing coloumn. We'll summarize our data by applying the `.mean()` operator, to obtain the average debt financing for each market. Once again, we'll use `sort_values()` and `head()` to make our output suitable to read.**
# 
# **Double check: is our order correct? 1. split (`groupby`) 2. apply (`.mean()`) 3. combine (`sort_values`). We're good to go.**
# 

# In[ ]:


canadian_startups.groupby(' market ')['debt_financing'].mean().sort_values(ascending=False).head(10)


# **Our data tells us that the Music market has the most debt for financing. The Clean Technology market has an average of $3,154,762 in debt financing, so the company AQUA PURE is a little under the industry average.**

# **Now, we want to take a deeper look at financing overall. Specifically, I am curious about the percentage of debt to overall fundraising, and how it affects the success of a company.**
# 
# **To analyze this, we'll focus on the related data attributes of *'total_funding_usd'*, *debt_financing*, and *'status'*, as well as general information like *'name'* and *'market'*.**
# 
# ***'status'* will be used because it tells us if a company was aquired or closed. Often, an acquired company is considered a successful startup.As well, closed companies may be considered a failed business. We'll make these assumptions for the sake of this analysis, in the absence of other indicators such as profit.**
# 
# **8. Lets select the data we want. If we want multiple columns of a DataSeries, we must type them out in a list `[]`.**

# In[ ]:


canadian_startups = canadian_startups[ ['name',' market ','status',' funding_total_usd ','debt_financing']]
canadian_startups.head()


# **9. To find out how debt_financing affects the chances of a startup eventually being acquired, a useful metric would be debt_financing as a percentage of total funding. To get this metric, let's create a new column that divides the *debt_financing* column by the *int_funding_total_usd column*.**
# 
# (Note: additional code was written (first two lines) to convert the *'funding_total_usd'* column from strings to integers and renamed *'int_funding_total_usd'*, so that we could make this calculation.)

# In[ ]:


# data conversion was needed. column was converted from string to integer.
canadian_startups[' funding_total_usd '] = canadian_startups[' funding_total_usd '].str.replace(',', '')
canadian_startups['int_funding_total_usd'] =  pd.to_numeric(canadian_startups[' funding_total_usd '],errors='coerce')

canadian_startups['debt_funding_percentage'] = (canadian_startups['debt_financing'] / canadian_startups['int_funding_total_usd'])
canadian_startups.head()


# **10. This time, we will use the split-apply-combine strategy to compare debt funding percentage levels for acquired companies versus closed companies. We will `groupby` *'status'*, select our newly created column *'debt_funding_percentage'* and the *int_funding_total_usd*, and summarize our data using `.mean` and see if we can make any meaningful observations.**

# In[ ]:


canadian_startups.groupby(['status'])['debt_funding_percentage','int_funding_total_usd'].mean()


# # Wrapping up
# **Overall, it looks like acquired companies recieve 2.9% of their funding through debt, while closed companies recieve 4.4%. This doesn't seem to be too much of a difference but it does still tell us that generally, startups that ended up closing relied more on debt.**
# 
# **Thank you for going through this analysis/tutorial! I hope you learned something new about startups and pandas!**
