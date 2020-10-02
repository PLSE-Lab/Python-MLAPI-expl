#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[ ]:


from collections import Counter
import re
import os

from pylab import *
import pandas as pd
import seaborn as sns


# ## Data Import

# In[ ]:


df = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


df.count()


# It should be noted that there are 'nan' values, which need to be fixed.

# In[ ]:


df.info()


# ## Tasks:
# 1) most popular category <br>
# 2) app with the largest size<br>
# 3) app which hasn't been updated<br>
# 4) app with the leargest num of installs<br>

# <b> 1) most popular category

# In[ ]:


df2 = df.copy()


# In[ ]:


df2[["App", "Installs"]]


# In[ ]:


df2[["App", "Installs"]].info()


# In[ ]:


Counter(df2['Installs'])


# In[ ]:


# change "Installs" column to numeric

def conv_Installs(n_installs):
    if n_installs == 'Free':
        val = 0
    else:
        if '+' in n_installs:
            val = int(n_installs[:-1].replace(',', ''))
        else:
            val = int(n_installs)
    return val

df2["Installs"] = df2["Installs"].apply(lambda n_installs: conv_Installs(n_installs))


# In[ ]:


# sort by 'Installs', 'Rating'

df2 = df2.sort_values(by=["Installs", "Rating"], ascending=False)


# In[ ]:


# select the first n most popular "Apps"

rank = 1
prev_apps = []
for app, installs, rating in df2.iloc[:40][["App", "Installs", "Rating"]].values:
    if app not in prev_apps:
        print(f"Ranking {rank} | {app} | install: {installs}+ | rating: {rating}")
        prev_apps.append(app)
        rank += 1


# <b>2) app with the largest size

# In[ ]:


Counter(df2["Size"])


# In[ ]:


# remove rows with 'Varies with device'

indices_varies_with_device = (df2["Size"] != "Varies with device").values
df2 = df2[indices_varies_with_device]


# In[ ]:


# change 'Size' column to numerical values

def size_coverter(size):
    size = size.lower()
    num, eng = float(size[:-1].replace(',', '')), size[-1]
    
    eng2multiplier = {'k': 1000, 'm': 1000000, '+': 1}
    
    numerical_size = num * eng2multiplier[eng]
    return numerical_size

df2["Size"] = df2["Size"].apply(lambda s: size_coverter(s))


# In[ ]:


n = 0
prev_apps = []
for app, size in df2[["App", "Size"]].sort_values("Size", ascending=False).values:
    if app not in prev_apps:
        print(f"Ranking {n+1} | {app} | size: {int(size)}")
        n += 1
        prev_apps.append(app)
        
        if n == 10:
            break


# <b>3) app which hasn't been updated

# In[ ]:


import datetime


# In[ ]:


Counter(df2["Last Updated"])


# In[ ]:


def covt_date(unique_date):
    try:
        month, day, year = re.findall(r"(\w+) (\d+), (\d+)", unique_date)[0]
        pydate = datetime.datetime.strptime(f'{year}-{month}-{day}', '%Y-%B-%d')
    except IndexError:
        pydate = np.nan
    return pydate


# In[ ]:


df2["Last Updated"] = df2["Last Updated"].apply(lambda date: covt_date(date))


# In[ ]:


df2.shape


# In[ ]:


# remove rows with 'nan' 

df2 = df2.dropna(subset=["Last Updated"])
df2.shape


# In[ ]:


rank = 1
for app, last_updated in df2.sort_values(by="Last Updated").iloc[:10][["App", "Last Updated"]].values:
    print(f"Ranking {rank} | Last Updated: {last_updated} | App: {app}")
    rank += 1


# ## Data Cleaning

# <b> colum: Category

# In[ ]:


Counter(df["Category"])


# need to remove 1.9 <- must be typo

# In[ ]:


df["Category"].values.tolist().index('1.9')


# In[ ]:


df = df.drop([10472])  # removes typoed-row


# <b> colum: Rating

# In[ ]:


Counter(df["Rating"])


# In[ ]:


df = df.dropna(subset=["Rating"])


# In[ ]:


# confirms that Nan is removed.

Counter(df["Rating"])  


# <b> colum: Reviews

# In[ ]:


Counter(df["Reviews"])


# should change the type pf 'Reviews' to int

# In[ ]:


df["Reviews"] = df["Reviews"].astype(np.int)


# <b> colum: Size

# In[ ]:


Counter(df["Size"])


# - should remove 'Varies with device' (difficult to standardize..)
# - "Size" should be changed to numerical values

# In[ ]:


# remove rows with 'Varies with device'

indices_varies_with_device = (df["Size"] != "Varies with device").values
df = df[indices_varies_with_device]


# In[ ]:


# change 'Size' column to numerical values

def size_coverter(size):
    size = size.lower()
    num, eng = float(size[:-1].replace(',', '')), size[-1]
    
    eng2multiplier = {'k': 1000, 'm': 1000000, '+': 1}
    
    numerical_size = num * eng2multiplier[eng]
    return numerical_size

df["Size"] = df["Size"].apply(lambda s: size_coverter(s))


# <b> colum: Installs

# In[ ]:


# change "Installs" column to numeric

def conv_Installs(n_installs):
    if n_installs == 'Free':
        val = 0
    else:
        if '+' in n_installs:
            val = int(n_installs[:-1].replace(',', ''))
        else:
            val = int(n_installs)
    return val

df["Installs"] = df["Installs"].apply(lambda n_installs: conv_Installs(n_installs))


# <b> colum: Type

# In[ ]:


df["Type"].unique()


# <b> colum: Price

# In[ ]:


df["Price"].unique()


# In[ ]:


def convt_price(price: str):
    price = price.replace("$", '')
    price = np.float(price)
    return price


# In[ ]:


df["Price"] = df["Price"].apply(lambda price: convt_price(price))


# <b> colum: Content Rating

# In[ ]:


df["Content Rating"].unique()


# <b> colum: Genres

# In[ ]:


df["Genres"].unique()


# <b> colum: Last Updated   

# In[ ]:


def covt_date(unique_date):
    try:
        month, day, year = re.findall(r"(\w+) (\d+), (\d+)", unique_date)[0]
        pydate = datetime.datetime.strptime(f'{year}-{month}-{day}', '%Y-%B-%d')
    except IndexError:
        pydate = np.nan
    return pydate


# In[ ]:


df["Last Updated"] = df["Last Updated"].apply(lambda date: covt_date(date))


# <b> colum: Current Ver

# In[ ]:


# drop this column as 'Last Update' may represent this.


# In[ ]:


df = df.drop(columns=["Current Ver"])


# <b> colum: Android Ver

# In[ ]:


df["Android Ver"].unique()


# In[ ]:


# drop rows with 'Varies with device'

df = df[df["Android Ver"] != "Varies with device"]


# <b> final df.info()

# In[ ]:


df.info()  # after cleaned up


# ## Data Anlysis

# ### statistical property of each variable

# In[ ]:


category = pd.DataFrame()
category['category'] = Counter(df["Category"]).keys()
category['val'] = Counter(df["Category"]).values()
category = category.set_index("category")

category.plot.pie(y="val", figsize=(20, 5), );
plt.legend(loc="upper right", bbox_to_anchor=(2, 2));


# In the datset, there are lots of GAME, FAMILY categories

# In[ ]:


sns.distplot(df.Rating);


# many ratings are around 4.3

# ### linear relationships between numerical variables

# In[ ]:


# correlation

corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[ ]:


sns.pairplot(df);


# From the above scatter, correlation matrices, 
# - 'Installs' has a positive-linear relationship with 'Reviews', which makes sense. More download more reviews.

# ### relationship between 'Category' and 'Rating'

# In[ ]:


sns.catplot("Category", "Rating", data=df, kind="boxen", aspect=3);
plt.xticks(rotation=70);


# DATING-related apps:
# - has the lowest rating on average
# <br><br>
# 
# The following app users have very varying ratings (high variance): 
# - BUSINESS
# - DATING
# - FINANCE
# - HEATH_AND_FITNESS
# - LIFESTYLE
# - MEDICAL

# ### relationship between 'Type (Free/Paid)' & 'Rating' (& 'Content Rating')

# In[ ]:


sns.catplot("Type", "Rating", data=df, kind="boxen", aspect=0.7);


# Paid apps have a bit higher ratings. <br>
# Maybe because the paid apps were initially designed better, I suppose.
# 
# Nonetheless, there's not much difference by the free/paid

# In[ ]:


sns.catplot("Type", "Rating", data=df, kind="boxen", 
            hue="Content Rating",
            aspect=2);


# apps that allow 'Mature 17+' group for rating has particularly higher positive rating when it comes to 'Paid' apps

# ### relationship between 'Genres' & 'Rating' (& 'Content Rating')

# In[ ]:


sns.catplot("Genres", "Rating", data=df, kind="boxen", aspect=5)
plt.xticks(rotation=70);


# Note that box plots with almost no deviation may be caused due to a small dataset. -> less confident

# In[ ]:


sns.catplot("Genres", "Rating", data=df, kind="boxen", 
            col="Content Rating", col_wrap=1,
            aspect=5)
plt.xticks(rotation=70);


# Funny, 
# - 'DATING' apps for 'Teen' have a lower rating.
# - 'DATING' apps for 'Mature 17+' have a higher rating.
# 
# I guess after 17, people get to have a better chance of getting into some kind of relationship(?), which possibly increases the rating(?)

# ### relationship between 'Last Updated' and 'Rating'

# In[ ]:


sns.relplot("Last Updated", "Rating", data=df, alpha=0.3);


# ### relationship between 'Android Ver' and 'Rating'

# In[ ]:


sns.catplot("Android Ver", "Rating", data=df, kind="boxen", aspect=3)
plt.xticks(rotation=70);

