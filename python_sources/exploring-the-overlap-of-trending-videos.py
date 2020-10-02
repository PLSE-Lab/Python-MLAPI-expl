#!/usr/bin/env python
# coding: utf-8

# # Exploring the Overlap of Trending Videos between Countries

# In this notebook, we will look at the overlaps between the trending videos in the five countries in the dataset and explore which countries share similar interests on Youtube and whether such shared interests are mutual.

# In[1]:


# %load /home/mithrillion/default_imports.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
get_ipython().run_line_magic('matplotlib', 'inline')
import re
from collections import Counter
from functools import partial


# In[2]:


from itertools import product, combinations
from datetime import datetime


# We first load the dataset. The dataset is composed of two set of files, the trending videos data and video categories data. We first load the data for each country, then examine if the video categories data are different for each country.

# In[4]:


# load data
country_list = ["us", "ca", "de", "fr", "gb"]

def read_country_dat(code):
    dat = pd.read_csv("../input/{0}videos.csv".format(code.upper()), parse_dates=['publish_time'])
    dat = dat[dat['trending_date'].apply(lambda x: re.match(r'\d{2}\.\d{2}\.\d{2}', x) is not None)]
    dat['trending_date'] = dat['trending_date'].apply(lambda x: datetime.strptime(x, "%y.%d.%m"))
    return dat

dat = {code: read_country_dat(code) for code in country_list}


# In[4]:


def read_country_cat(code):
    cat = pd.read_json("../input/{0}_category_id.json".format(code.upper()), orient='columns')
    cat_exp = pd.DataFrame(list(cat['items'].values))
    cat_exp2 = pd.DataFrame(list(cat_exp['snippet'].values))
    cat_flat = pd.concat([cat_exp, cat_exp2], axis=1)
    return cat_flat

cat = {code: read_country_cat(code) for code in country_list}


# We first check if the category IDs from all countries match up, and whether each country use a different set of categories. This is important for comparing the category composition of overlapping videos between countries.

# In[5]:


def cat_merge(x, y, **kwargs):
    return pd.merge(x, y, on='id', how='outer', **kwargs)


# In[6]:


u = cat_merge(cat['us'][['id', 'title']], cat['ca'][['id', 'title']], suffixes=('_us', '_ca'))
u = cat_merge(u, cat['de'][['id', 'title']], suffixes=('', '_de'))
u = cat_merge(u, cat['fr'][['id', 'title']], suffixes=('', '_fr'))
u = cat_merge(u, cat['gb'][['id', 'title']], suffixes=('', '_gb'))
u.rename(columns={'title': 'title_de'}, inplace=True)
u


# As we see above, the categories in different countries maych up, except that "Nonprofits & Activism" category only appears in the US list.
# 
# Now that we know that countries share category IDs, we can easily calculate the overlapping videos between countries and their categories among trending videos. We first count the number of unique overlapping trending videos between any two countries.

# In[7]:


uid_lists = {code: set(dat[code]['video_id']) for code in country_list}


# In[8]:


style.use('default')
comp = np.array([(len(c0[1].intersection(c1[1])) if c0 != c1 else 0)
                 for c0, c1 in product(uid_lists.items(), repeat=2)]).reshape((5, 5))
plt.imshow(comp);
plt.colorbar();
plt.xticks(range(5), country_list);
plt.yticks(range(5), country_list);
plt.title('pairwise overlap of trending videos by country');
plt.show()


# However, the above graph is not terribly informative, because the number of unique trending videos for each country is not the same. (Interestingly, there are far more unique trending videos in the Canada, Germany and France section of the dataset than the US and Britain section)

# In[9]:


print("unique trending videos per country:")
print({n: len(l) for n, l in uid_lists.items()})


# Therefore, it makes more sense to compare the percentage of shared trending videos as a percentage of total shared videos in the country:

# In[10]:


plt.imshow(comp / np.array([len(c) for c in uid_lists.values()]).reshape((-1, 1)));
plt.colorbar();
plt.xticks(range(5), country_list);
plt.yticks(range(5), country_list);
plt.title('pairwise overlap % of trending videos by country (% of left column / y axis)');
plt.show()


# In this plot, the rows represent the percentage of trending videos the country in the row header shares with the countries in the column header. For instance, as we can see, the US shares the most trending videos with Canada, Briain shares the most trending videos with the US, followed by Canada, etc. Generally, it appears that shared language plays a big role in the number of shared videos here, as Germany and France do not share many trending videos with other countries. We also notice that the degree of shared interests is not always symmetrical, e.g. the shared videos between US and Britain makes up a smaller percentage of total trending videos in the US than in Britain.

# We are not only interested in the degree of shared interests overall, but also how much shared interests there are in each specific video category. We need some measures that are able to reflect differences in shared interests. We define:
# 
# **Unilateral Categorical Shared Interest (USCI)**: $\dfrac{\#\space shared\space videos\space in\space category}{\#\space total\space videos\space in\space category\space in\space other\space country}$
# 
# **Mutual Categorical Interest (MCI)**: $\dfrac{\#\space shared\space videos\space in\space category}{\#\space total\space videos\space in\space category\space in\space both\space countries}$
# 
# UCSI is an asymmetrical measure whereas MCI is symmetrical. In the following analyses we will generally prefer asymmetrical measures such as UCSI, as these measures can better take into account the gaps between number of unique trending videos in each country. We first calculate the mutual interest scores by video category:

# In[11]:


cat_uid_lists = {(code, cat): set(dat[code][dat[code]['category_id'] == cat]['video_id']) 
                 for code in country_list 
                 for cat in list(u.id.unique().astype(np.int))}


# In[12]:


icat2cat = dict(enumerate(list(u.id.unique().astype(np.int))))
cat2icat = {v: k for k, v in icat2cat.items()}
cat_ucsi = np.zeros((5, 5, 32))
cat_mci = np.zeros((5, 5, 32))
for ic0 in range(5):
    for ic1 in range(5):
        if ic0 != ic1:
            for icat in range(32):
                c0 = country_list[ic0]
                c1 = country_list[ic1]
                cat = icat2cat[icat]
                set0 = cat_uid_lists[(c0, cat)]
                set1 = cat_uid_lists[(c1, cat)]
                if len(set1) != 0:
                    cat_ucsi[ic0, ic1, icat] = len(set0.intersection(set1)) / len(set1)
                    cat_mci[ic0, ic1, icat] = len(set0.intersection(set1)) / len(set0.union(set1))


# We then visualise the UCSI matrix per category (same colour scale as the last graph):

# In[13]:


fig, ax = plt.subplots(8, 4, figsize=(16, 32))
plt.setp(ax, xticks=range(5), xticklabels=country_list, yticks=range(5), yticklabels=country_list)
icat = 0
for r in range(8):
    for c in range(4):
        cur_ax = ax[r, c]
        cat_name = u[u['id'] == str(icat2cat[icat])]['title_us'].iloc[0]
        cur_ax.set_title(cat_name)
        cur_ax.imshow(cat_ucsi[:, :, icat])
        icat += 1
plt.show()


# Each cell in the graph represents the UCSI score from the left side country to the bottom side country, i.e. number of shared videos didived by the total number of videos in that category in the **bottom** country.
# 
# We first observed that there are many categories that are rarely trending, or rarely trending in multiple countries, which is indicated by the lack of USCI data on the graph. We also see that the degree of shared interests do vary between different categories. Here are some of the observations:
# 
# 1. Accroding to the UCSI scores, the US and Britain are generally less interested in foreign-made videos than other countries in the dataset (as indicated by the dimmer colour in their rows).
# 2. Often, a large proportion of trending videos in the US is also trending in Canada. However, the converse is not true. However, without further examining where these videos are made, we are not able to tell whether this is because more US videos are internationally popular or because Canada makes more locally popular videos.
# 3. A large proportion of 'Nonprofits & Activism" videos from Britain is popular in all other countries (remeber that this category is only defined in the US category list, but now we see that this might be merely a missing value issue). This is rather interesting and we would like to know what these videos are later.
# 4. In the "Shows" category, the UCSI scores between Canada, Germany and France are quite high when it is zero everywhere else. We would like to know if this is one (or a few) show(s) skewing the results.
# 5. Canada and Germany shares a higher percentages of trending videos in the music category than other countries. We would like to know if this is backed up by facts, as from the language perspective, it makes more sense for Canada to share more trending music-related videos with the US, Britain or France.

# **Here we will try to address observation 3 and 5**. Starting with observation 3. We first select all videos in this category:

# In[19]:


acti = {code: dat[code][dat[code]['category_id'] == 29] for code in country_list}


# * Red: shared with GB

# In[33]:


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

bri = acti['gb']['title'].unique()
for code in country_list:
    print(color.BLUE + "Unique Nonprofits & Activism videos in [{0}]:\n".format(code) + color.END)
    for title in acti[code]['title'].unique():
        if title in bri:
            print(color.BOLD + color.RED + title + color.END)
        else:
            print(title)
    print("-" * 20)


# As we see here, it is not really surprising, given that there are only **four** videos in this category for Britain, one of which is trending in every country. Unfortunately, that video is neither British original nor a video to be proud of, so we cannot quite say that British activism is going strong on YouTube... This exposes two of the weaknesses of our measures:
# 
# 1. Unreliable when the number of videos is low;
# 2. It does not take into account the video's origin.

# **Now we try to address observation 5.** It appears that Canada and Germany shares more common interest in music than anyone else. Is that really the case?

# In[34]:


music = {code: dat[code][dat[code]['category_id'] == 10] for code in country_list}


# Count of trending videos in music category:

# In[37]:


for code in country_list:
    print("{0}: {1}".format(code, len(music[code])))


# Count of shared videos:

# In[45]:


for c0, c1 in combinations(country_list, r=2):
    titles = set(music[c0]['title']).intersection(set(music[c1]['title']))
    n = len(titles)
    print("Overlapping videos in music between [{0}] and [{1}]: {2}".format(c0, c1, n))


# As we see here, there are indeed a large number of overlapping trending videos between Canada and Germany in the first place. This plus the fact that there are fewer videos for these two countries in the music category makes the overlapping percentage especially high. Canada actually shares the most shared trending videos in music with France. Also worth mentioning is that despite the US and Britain being majority English-speaking like Canada, both of them have rather low number of shared videos with Germany.
# 
# Now let us look at the videos shared between Canada and Germany and see if there are anything notable in the data:

# In[54]:


for title in list(set(music['ca']['title']).intersection(set(music['de']['title'])))[:30]:
    print(title)
print("...")


# As we see, most of these videos are English songs or popular International songs from other countries. There are almost no original German songs on this list. This makes the large number of shared trending videos between these two countries even more interesting, as one would expect higher number of shared videos between Germany and other English-speaking countries. Whether this is down to "taste" or other reasons is worth further investigation.
# 
# One way to possibly gain some more insights into the phenomenom is to see which of the songs in the Canada-Germany list also made it onto the Germany-US or Germany-Britain list.
# 
# * Green: shared with US and GB
# * Blue: shared with US
# * Red: shared with GB

# In[53]:


de_us = set(music['de']['title']).intersection(set(music['us']['title']))
de_gb = set(music['de']['title']).intersection(set(music['gb']['title']))
for title in set(music['ca']['title']).intersection(set(music['de']['title'])):
    if title in de_us.intersection(de_gb):
        print(color.GREEN + title + color.END)
    elif title in de_us:
        print(color.BLUE + title + color.END)
    elif title in de_gb:
        print(color.RED + title + color.END)
    else:
        print(title)


# It is no surprise that most of the official music videos of popular songs are actually on all four countries' trending lists. What is surprising is that many live performance videos, covers and mixes are only able to make the list in Canada and Germany, but not the US and Britain.

# In[ ]:




