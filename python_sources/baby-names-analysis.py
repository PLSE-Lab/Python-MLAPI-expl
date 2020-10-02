#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# TODO need to change to downloadable URL format
# 
dirname = '/kaggle/input'
filename = 'data.csv'
filepath = os.path.join(dirname, filename)


# EDA - Explorartoy Data analysis

# In[ ]:


df = pd.read_csv(filepath)


# Since our dataset is very simple we won't require to set any special flags or extra parameters, the default ones are good enough.
# 
# Let's start with the classic `df.head()` and `df.tail()` methods which are very handy for taking a sneak peek at the dataset.

# In[ ]:


df.head()


# In[ ]:


df.tail()


# These samples tells us several things:
# 
# * There are 4 columns (year, name, gender and count).
# * There are 1,957,046 rows.
# * Rows are sorted by year.
# * Female records are shown before male ones.
# * At least 5 parents in 2018 named their son 'Zzyzx'.
# 
# *Note: The dataset only includes names which have at least 5 records, this is for privacy reasons.*

# ### Unique Names
# 
# There are 98,400 unique names in the dataset. From those, 41,475 are male names, 67,698 are female ones and 10,773 are gender neutral.

# In[ ]:


# Unique names either gender.
df["name"].nunique()

# Unique names for male.
df[df["gender"] == "M"]["name"].nunique()

# Unique names for female.
df[df["gender"] == "F"]["name"].nunique()

# Unique names for gender neutral.
both_df = df.pivot_table(index="name", columns="gender", values="count", aggfunc=np.sum).dropna()
both_df.index.nunique()


# ### Top 10 Male and Female Names
# 
# To get the top 10 most used male and female names we are going to first filter the `dataframe` by gender.
# 
# Once we have a gender specific `dataframe` we wiil select only 2 fields, name and count. From there we will use the `groupby()` method on the name field and aggregate the results using a `sum()`.
# 
# Finally, we will sort the values on the count field in descending order and use the `head(10)` method to get the top 10 results.

# In[ ]:


# Step by step approach, the one-liners can be found below their respective tables.
only_gender_male = df[df["gender"] == "M"]
only_name_and_count_colmns = only_gender_male[["name", "count"]]
df_group_by_name = only_name_and_count_colmns.groupby("name")
df_group_by_name_sum = df_group_by_name.sum()
df_group_by_name_sum_sort_by_count = df_group_by_name_sum.sort_values("count", ascending=False)
df_group_by_name_sum_sort_by_count.head(10)


# In[ ]:


# In one liner format 
df[df["gender"] == "M"][["name", "count"]].groupby("name").sum().sort_values("count", ascending=False).head(10)


# In[ ]:


# One liner format for Female children
 df[df["gender"] == "F"][["name", "count"]].groupby("name").sum().sort_values("count", ascending=False).head(10)


# ### Top 20 Gender Neutral Names
# 
# This one was a bit challenging, first we need to pivot the `dataframe` so the names are the index, the genders will be the columns and the sum of all counts (per name, per gender) will be our values.
# 
# We are going to do this in small steps. First we pivot the table and drop the rows where the value is 0. This means rows where names are not present in either male or female categories.

# In[ ]:


df_pvt = df.pivot_table(index="name", columns="gender", values="count", aggfunc=np.sum).dropna()


# With the data in this shape we now know how many records each name has per gender.
# 
# Now we will only take into account those names that atleast have 50,000 records for each gender.

# In[ ]:


df_pvt_count_gt_50k = df_pvt[(df_pvt["M"] >= 50000) & (df_pvt["F"] >= 50000)]
df_pvt_count_gt_50k.head(20)


#  ### Highest and Lowest Years
# 
# Now we will know which years had the highest and lowest amount of records by gender and combined.

# In[ ]:


both_df = df.groupby("year").sum()
male_df = df[df["gender"] == "M"].groupby("year").sum()
female_df = df[df["gender"] == "F"].groupby("year").sum()

# Initializing list
data = []

# Combined Min (count and year)
both_df_min = both_df.min()["count"]
both_df_count = both_df.idxmin()["count"]

# Appending result to list
data.append(['Both Min',both_df_min, both_df_count ])

# Male Min (count and year)
male_df_min = male_df.min()["count"]
male_df_count = male_df.idxmin()["count"]

# Appending result to list
data.append(['Male Min',male_df_min, male_df_count ])

# Female Min (count and year)
female_df_min = female_df.min()["count"]
female_df_count = female_df.idxmin()["count"]

# Appending to list
data.append(['Female Min',female_df_min, female_df_count ])

# Combined Max (count and year)
both_df_max = both_df.max()["count"]
both_df_max_count = both_df.idxmax()["count"]

# Appending result to list
data.append(['Both Max',both_df_max, both_df_max_count ])

# Male Max (count and year)
male_df_max = male_df.max()["count"]
male_df_max_count = male_df.idxmax()["count"]

# Appending result to list
data.append(['Male Max',male_df_max, male_df_max_count ])

# Female Max (count and year)
female_df_max = female_df.max()["count"]
female_df_max_count = female_df.idxmax()["count"]

# Appending to list final value
data.append(['Female Max',female_df_max, female_df_max_count ])


# In[ ]:


pd.DataFrame(data, columns=["Gender and Attribute", "Total Count", "Year"])


# The year 1881 got the lowest records on the dataset, while the year 1957 got the highest records.
# 
# So far we got several interesting insights, it's time to create some pretty plots.

# ## Plotting the Data
# 
# For creating the plots we will use `seaborn` and `matplotlib`, the reason for this is that seaborn applies some subtle yet nice looking effects to the plots.
# 
# In this project we are only going to use line plots, which are very helpful for displaying how a value changes over time.
# 
# The first thing to do is to set some custom colors that will apply globally to each plot.
# 

# In[ ]:


# Those parameters generate plots with a mauve color.
sns.set(style="ticks",
        rc={
            "figure.figsize": [12, 7],
            "text.color": "white",
            "axes.labelcolor": "white",
            "axes.edgecolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "axes.facecolor": "#443941",
            "figure.facecolor": "#443941"}
        )


# 
# With our style declared we are ready to plot our data.
# 
# *Note: The next code blocks are more advanced than the previous ones. I also make heavy use of one-liners for efficiency reasons, but don't worry, I will explain what each line does.*
# 

# ### Counts by Year
# 
# Our first plot will consist on how the number of records has moved from 1880 to 2018.
# 
# First, we create new dataframes for male, female and combined.

# In[ ]:


both_df = df.groupby("year").sum()
male_df = df[df["gender"] == "M"].groupby("year").sum()
female_df = df[df["gender"] == "F"].groupby("year").sum()


# We plot our dataframes directly. The x-axis will be the index and the y-axis will be the total counts.

# In[ ]:


plt.plot(both_df, label="Both", color="yellow")
plt.plot(male_df, label="Male", color="lightblue")
plt.plot(female_df, label="Female", color="pink")


# ### Most Popular Names Growth
# 
# For our next plot we will observe how the all-time most popular names have grown over the years.
# 
# First, we merge values from male and female and pivot the table so the names are our index and the years are our columns. We also fill missing values with zeroes.

# In[ ]:


pivoted_df = df.pivot_table(index="name", columns="year", values="count", aggfunc=np.sum).fillna(0)


# Then we calculate the percentage of each name by year.

# In[ ]:


percentage_df = pivoted_df / pivoted_df.sum() * 100


# We add a new column to store the cumulative percentages sum.

# In[ ]:


percentage_df["total"] = percentage_df.sum(axis=1)


# We sort the dataframe to check which are the top values and slice it. After that we drop the `total` column since it won't be used anymore.
# 

# In[ ]:


sorted_df = percentage_df.sort_values(by="total", ascending=False).drop("total", axis=1)[0:10]


# We flip the axes so we can plot the data more easily.

# In[ ]:


transposed_df = sorted_df.transpose()


# In[ ]:


transposed_df.columns.tolist()


# We plot each name individually by using the column name as the label and Y-axis.

# In[ ]:


for name in transposed_df.columns.tolist():
    plt.plot(transposed_df.index, pivoted_df[name], label=name)


# We set our yticks in steps of 0.5%.

# In[ ]:


yticks_labels = ["{}%".format(i) for i in np.arange(0, 5.5, 0.5)]
plt.yticks(np.arange(0, 5.5, 0.5), yticks_labels)


# We add the final customizations.

# In[ ]:


plt.legend()
plt.grid(False)
plt.xlabel("Year")
plt.ylabel("Percentage by Year")
plt.title("Top 10 Names Growth")
plt.show()

