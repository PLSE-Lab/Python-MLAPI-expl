#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Reading in Data
credits_df = pd.read_csv("../input/credits.csv")
movies_df = pd.read_csv("../input/movies_metadata.csv")


# In[ ]:


# Group Movies by Genre and See Which Genres are the Most Profitable
movies_with_revenue_df = movies_df[movies_df["revenue"] > 0]
movies_with_revenue_df['profit'] = movies_with_revenue_df['revenue'] - movies_with_revenue_df['budget'].astype(float)
movies_grouped_genres = movies_with_revenue_df.groupby(["genres"])["profit"]
movies_grouped_genres.sum().sort_values(ascending = False)


# In[ ]:


# Group Movies by Production Companies and See Which Production Companies are the Most Profitable
movies_grouped_genres = movies_with_revenue_df.groupby(["production_companies"])["profit"]
movies_grouped_genres.mean().sort_values(ascending = False)


# In[ ]:


chernin_entertainment_df = movies_df[movies_df["production_companies"].apply(lambda x: True if type(x) is str and x.find("Chernin Entertainment") != -1 else False)]


# In[ ]:


chernin_entertainment_df["profit"] = chernin_entertainment_df["revenue"] - chernin_entertainment_df["budget"].astype(float)    


# In[ ]:


chernin_entertainment_df[['title', 'genres' , 'profit']].sort_values("profit", ascending = False)


# In[ ]:


chernin_entertainment_df['profit'].mean()


# In[ ]:


mystery_movies_df = movies_df[movies_df["genres"].apply(lambda x: True if type(x) is str and x.find("Mystery") != -1 else False)]


# In[ ]:


mystery_movies_df = mystery_movies_df[mystery_movies_df["revenue"] != 0]


# In[ ]:


mystery_movies_df_2013 = mystery_movies_df[mystery_movies_df["release_date"] > "2013-01-01"]


# In[ ]:


mystery_movies_df_2013['profit'] = mystery_movies_df_2013['revenue'] - mystery_movies_df_2013['budget'].astype(float)


# In[ ]:


mystery_movies_df_2013["profit"].mean()


# In[ ]:


plt.scatter([i for i in range(len(mystery_movies_df_2013['profit']))], mystery_movies_df_2013["profit"])


# In[ ]:


mystery_movies_df_2013[["title", "profit"]].sort_values("profit", ascending = False)


# In[ ]:


credits_df_jennifer_lawrence = chernin_entertainment_df = credits_df[credits_df["cast"].apply(lambda x: True if type(x) is str and x.find("Jennifer Lawrence") != -1 else False)]


# In[ ]:


lst = []
for i in credits_df_jennifer_lawrence["id"]:
    lst.append(i)


# In[ ]:


def filter_int(x):
    try:
        num = int(x)
        return num in lst
    except ValueError:
        return False


# In[ ]:


movies_df_jennifer_lawrence = movies_df[movies_df["id"].apply(filter_int)]


# In[ ]:


movies_df_jennifer_lawrence['profit'] = movies_df_jennifer_lawrence['revenue'] - movies_df_jennifer_lawrence['budget'].astype(float)


# In[ ]:


plt.scatter([i for i in range(len(movies_df_jennifer_lawrence['profit']))], movies_df_jennifer_lawrence['profit'])


# In[ ]:


movies_df_jennifer_lawrence[['title', 'profit']].sort_values("profit", ascending = False)


# In[ ]:


movies_df_jennifer_lawrence['doubled_budget'] = movies_df_jennifer_lawrence['revenue'] >= 2*movies_df_jennifer_lawrence['budget'].astype(float) 


# In[ ]:


movies_df_jennifer_lawrence = movies_df_jennifer_lawrence[movies_df_jennifer_lawrence['profit'] != 0]


# In[ ]:


movies_df_jennifer_lawrence[['title', 'profit', 'doubled_budget']].sort_values("profit", ascending = False)


# In[ ]:


chernin_entertainment_df = movies_df[movies_df["production_companies"].apply(lambda x: True if type(x) is str and x.find("Chernin Entertainment") != -1 else False)]
chernin_entertainment_df['profit'] = chernin_entertainment_df['revenue'] - chernin_entertainment_df['budget'].astype(float)


# In[ ]:


chernin_entertainment_df['doubled_budget'] = chernin_entertainment_df['revenue'] >= 2*chernin_entertainment_df['budget'].astype(float) 
chernin_entertainment_df[['title', 'profit', 'doubled_budget']].sort_values("profit", ascending = False)


# In[ ]:


paramount_pictures_df = movies_df[movies_df["production_companies"].apply(lambda x: True if type(x) is str and x.find("Paramount Pictures") != -1 else False)]


# In[ ]:


paramount_pictures_df["profit"] = paramount_pictures_df["revenue"] - paramount_pictures_df["budget"].astype(float)
paramount_pictures_df["doubled_budget"] = paramount_pictures_df['revenue'] >= 2*paramount_pictures_df['budget'].astype(float)


# In[ ]:


paramount_pictures_df[['title', 'profit', 'doubled_budget']].sort_values("profit", ascending = False)


# In[ ]:


paramount_pictures_science_fiction_df = paramount_pictures_df[paramount_pictures_df["genres"].apply(lambda x: True if type(x) is str and x.find("Science Fiction") != -1 else False)]


# In[ ]:


paramount_pictures_science_fiction_df = paramount_pictures_science_fiction_df[paramount_pictures_science_fiction_df["profit"] != 0]


# In[ ]:


paramount_pictures_science_fiction_df[['title', 'profit', 'doubled_budget']].sort_values("profit", ascending = False)


# In[ ]:


paramount_pictures_science_fiction_df[paramount_pictures_science_fiction_df['doubled_budget'] == True]["profit"].mean()


# In[ ]:


paramount_medium_budget_scifi_df = paramount_pictures_science_fiction_df[paramount_pictures_science_fiction_df["budget"].astype(float) >= 50000000]


# In[ ]:


paramount_medium_budget_scifi_df = paramount_medium_budget_scifi_df[paramount_medium_budget_scifi_df['budget'].astype(float) < 100000000]


# In[ ]:


paramount_medium_budget_scifi_df[['title', 'budget', 'profit', 'doubled_budget']].sort_values("profit", ascending = False)


# In[ ]:


paramount_medium_budget_scifi_df_2000 = paramount_medium_budget_scifi_df[paramount_medium_budget_scifi_df['release_date'] > "2000-01-01"]


# In[ ]:


paramount_medium_budget_scifi_df_2000[['title', 'release_date', 'budget', 'profit', 'doubled_budget']].sort_values("profit", ascending = False)


# In[ ]:


credits_df_natalie_portman = chernin_entertainment_df = credits_df[credits_df["cast"].apply(lambda x: True if type(x) is str and x.find("Natalie Portman") != -1 else False)]


# In[ ]:


credits_df_natalie_portman


# In[ ]:


lst = []
for i in credits_df_natalie_portman["id"]:
    lst.append(i)


# In[ ]:


movies_df_natalie_portman = movies_df[movies_df["id"].apply(filter_int)]


# In[ ]:


movies_df_natalie_portman['profit'] = movies_df_natalie_portman['revenue'] - movies_df_natalie_portman['budget'].astype(float)
movies_df_natalie_portman['doubled_budget'] = movies_df_natalie_portman['revenue'] >= 2*movies_df_natalie_portman['budget'].astype(float)


# In[ ]:


movies_df_natalie_portman[['title', 'profit', 'release_date', 'doubled_budget']].sort_values('profit', ascending = False)


# In[ ]:


movies_df_natalie_portman_scifi = movies_df_natalie_portman[movies_df_natalie_portman['genres'].apply(lambda x: True if type(x) is str and x.find("Science Fiction") != -1 else False)]


# In[ ]:


movies_df_natalie_portman_scifi[['title', 'profit', 'release_date', 'doubled_budget']]


# In[ ]:


chernin_entertainment_df


# In[ ]:


chernin_entertainment_df = movies_df[movies_df["production_companies"].apply(lambda x: True if type(x) is str and x.find("Chernin Entertainment") != -1 else False)]
chernin_entertainment_df['profit'] = chernin_entertainment_df['revenue'] - chernin_entertainment_df['budget'].astype(float)
chernin_entertainment_df['doubled_budget'] = chernin_entertainment_df['revenue'] >= 2*chernin_entertainment_df['budget'].astype(float) 


# In[ ]:


keyword_df = pd.read_csv("../input/keywords.csv")


# In[ ]:





# In[ ]:




