#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Looking for patterns in box office trends.

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


movies = pd.read_csv("/kaggle/input/bollywood-box-office-20172020/bollywoodboxoffice_raw.csv")
movies.info()


# In[ ]:


movies.head()


# Looks like the data needs to be cleaned somewhat.
# 
# I. The url columns will be dropped because they don't affect the analysis.
# 
# II. Some of the columns need to be split into multiple columns.
# 
# III. Columns should be renamed for simplicity
# 
# IV. Some of the text needs to be processed.

# In[ ]:


movies.drop(["movie_url", "movie_director_url"], axis = 1, inplace = True)


# In[ ]:


movies[["movie_release_date", "movie_runtime"]] = movies["movie_movierelease"].str.split("|", expand = True)
movies.drop("movie_movierelease", axis = 1, inplace = True)
movies.columns = movies.columns.str.replace("^movie_", '')
movies.head()


# In[ ]:


revenues = ["opening", "weekend", "firstweek", "total", "total_worldwide"]
types = {col: 'float' for col in revenues}

def replacer(x):
    x = x.str.replace("^: ", "")
    x = x.str.replace("cr$", "")
    x = x.str.replace('^---', "0")
    return x.str.replace("\* $", "")

movies[revenues] = movies[revenues].apply(replacer)
movies[revenues] = movies[revenues].astype(types)

movies[revenues].head()


# In[ ]:


movies[["date", "year"]] = movies["release_date"].str.split(",", expand = True)
movies[["day", "month"]] = movies["date"].str.split(" ", expand = True)

movies.drop(["release_date", "date"], axis = 1, inplace = True)

movies.head()


# In[ ]:


time = movies["runtime"].str.split()

time = time.apply(lambda x: [int(x1) for x1 in x if x1 != 'hrs' and x1 != 'mins'])

def calc_runtime(x):
    if len(x) == 1:
        return 60 * x[0]
    else:
        return (60 * x[0]) + x[1]

movies["runtime"] = time.apply(calc_runtime)

movies.head()


# In[ ]:


movies.rename({"total": "domestic", "total_worldwide": "worldwide"}, axis = 1, inplace = True)


# In[ ]:


def cat_length(x):
    if x <= 120:
        return "Less than 2hrs"
    elif (x > 120) and (x < 180):
        return "Between 2 and 3 hours"
    elif x > 180:
        return "Longer than 3 hours"

movies["cat_runtime"] = movies["runtime"].apply(cat_length)


# 
# I won't be dealing with the text data so those can stay as they are. The date data, gross amounts, and runtime data have all been cleaned up reasonably well. The gross amounts stay in crores of INR which are 10s of millions of INR.

# # Exploring the Data

# In[ ]:


months = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")
plt.figure(figsize = (16, 8))
sns.boxplot(x = "month", y = "domestic", data = movies, order = months)
plt.ylabel("Crores INR")
plt.xlabel("Month")
plt.title("Domestic Gross of Indian Movies by Month")


# In[ ]:


sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")
plt.figure(figsize = (16, 8))
sns.boxplot(x = "month", y = "worldwide", data = movies, order = months)
plt.ylabel("Crores INR")
plt.xlabel("Month")
plt.title("Worldwide Gross of Indian Movies by Month")


# 
# June seems to be the biggest month for movies in both Domestic and Worldwide markets. December also appears to be a decent month for movies at the theaters. Every other month had median values under 100cr intakes at the box office. June and December movies get good bumps from the worldwide box office. September is possibly the worst performing month though April and November aren't great either.

# ## 100CR Movies

# In[ ]:


hundred = movies[movies["worldwide"] >= 100]
hundred.shape


# In[ ]:


plt.figure(figsize = (7, 5))
sns.set_style("darkgrid")
plt.style.use("tableau-colorblind10")
sns.countplot(x = "year", data = hundred)
plt.ylim(0, 25)


# There was a considerable jump in 100cr movies form 2017 to 2018. There was an overall increasing trend in 100cr movies from 2017 to 2019. 2020 data is to hard to read from; it is low but it's not halfway through the year yet and Covid-19 is certainly going to cripple numbers.

# In[ ]:


plt.figure(figsize = (7, 5))
sns.set_style("darkgrid")
plt.style.use("tableau-colorblind10")
sns.countplot(x = "cat_runtime", data = movies)
plt.ylim(0, 120)
plt.xlabel("Runtime of Movie")


# There aren't a lot of movies that are shorter than 2hrs but there aren't any that are longer than 3hrs which is quite surprising.
# Now that I've taken a look at some of the group statistics, I'm going to take a look at some of the individual statistics. Biggest opening day and biggest opening weekend seem like a good place to start.
# 

# ## Biggest Opening Day and Opening Weekend

# In[ ]:


opening_weekend = movies.sort_values(by = ["weekend"], ascending = False).head(10)
first_day = movies.sort_values(by = ["opening"], ascending = False).head(10)


# In[ ]:


sns.set_style("darkgrid")
plt.style.use("fivethirtyeight")
f, ax = plt.subplots(2, 1, figsize = (16, 14), squeeze = False)

sns.barplot(x = 'opening', y = 'name', data = first_day, ax = ax[0][0], palette = "inferno")
sns.barplot(x = 'weekend', y = 'name', data = opening_weekend, ax = ax[1][0], palette = 'inferno')

ax[0][0].set_ylabel('')
ax[0][0].set_xlabel('Crores INR')
ax[0][0].set_title('Biggest Opening Day')

ax[1][0].set_ylabel('')
ax[1][0].set_xlabel('Crores INR')
ax[1][0].set_title('Biggest Opening Weekend')


# 
# Aside from Thugs of Hindustan and Bharat switching spots, the top 5 stay the same from opening day through the weekend. Padmaavat came from a 25cr or less opening day and turned into a 110+ cr opening weekend.
# 
