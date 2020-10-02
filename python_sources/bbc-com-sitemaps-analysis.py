#!/usr/bin/env python
# coding: utf-8

# # Analyzing BBC.com's XML Sitemaps' Four Million URLs
# XML sitemaps are boring. Really boring...  
# But if they come with the `lastmod` tag, and the site provides rich URLs, then they might be really interesting in understanding trends in publishing and content across time and across different languages, or cateogires (depending on what information is available in the URLs).  
# This is an exploration of almost four million URLs included in all of BBC.com's sitemaps. 

# In[ ]:


import pandas as pd
pd.options.display.max_columns = None
import dataset_utilities as du
import plotly.graph_objects as go
from ipywidgets import interact


# In[ ]:


bbc = pd.read_csv('/kaggle/input/news-sitemaps/bbc_sitemaps.csv',
                  parse_dates=['lastmod'], index_col='lastmod', usecols=['lastmod', 'loc'])
bbc.sample(5)


# The URLs have been imported using the [`sitemap_to_df`](https://advertools.readthedocs.io/en/master/advertools.sitemaps.html) function from [advertools](https://github.com/eliasdabbas/advertools).  
# The function can take a sitemap URL or a sitemap index, and goes through all of them, retreiving all URLs and any other tags available in the sitemap(s). 
# 
# `value_counts_plus` is a simple function that I wrote to give richer information to the pandas `value_counts` function, so I'll be using it when counting values, starting with the first thing after "https://www.bbc.com/", which seems to be the language of the page. 

# In[ ]:


du.value_counts_plus(bbc['loc'].rename('language').str.split('/').str[3], show_top=50)


# All of them are languages indeed, with the exception of "news", "sport", "newsround", and "mundo". Before going further and starting to extract data, let's first check if all the URLs are for the same domain and if there are any other sub-domains for example. 

# In[ ]:


bbc['loc'].str.contains('https://www.bbc.com/').all()


# Now that we know that they are all "bbc.com", we can safely take the part that follows that, and create a `slug` colum. I'm also prepending "english" to the other words mentioned above. 

# In[ ]:


bbc['slug'] = bbc['loc'].str.replace('https://www.bbc.com/', '')
bbc['slug'] = bbc['slug'].str.replace('^news|^sport|^newsround', 'english/\g<0>')
bbc.sample(5)


# Now that we have "english" in the slugs, we can split by "/" and get the first element and put it in the `lang` column. I also replaced "mundo" with "spanish".

# In[ ]:


bbc['lang'] = bbc['slug'].str.split('/').str[0].replace('mundo', 'spanish')
bbc.sample(5)


# In[ ]:


du.value_counts_plus(bbc['lang'], show_top=50)


# The slugs are separated by forward-slashes, and this is the main way in which we are going to extract information. They also come in different lengths (number of times a URL contains "/" + 1). So to add some structure to the process I'm adding a column that shows the length of the slug after splitting by "/" `slug_split_length`.

# In[ ]:


bbc['slug_split_length'] = bbc['slug'].str.split('/').str.len()
bbc.sample(7)


# We can now get an idea on the number of URLs for each length.

# In[ ]:


du.value_counts_plus(bbc['slug_split_length']).hide_index()


# After exploring all the lengths, I summarized the information in this table.  
# As you can see, all slugs start with "language", and all end with "title", which is the title of the article (separated by dashes and/or underscores). The other elements that are found are year, month, category, sport name, and a few others.  
# The following code cells go through every length, and explore the elements resulting from splitting the slugs, to come up with this table. You can skip to the section where we start to [add URLs](#add_urls) if you want

#    Slug length when split | Number of articles | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 
# --------------------------------:|--------------------------:|---|---|---|---|---|---|---|
#                    1 |     38          | language | NA | NA | NA | NA | NA | NA 
#                    2 | 725,691     | language | **title** |  NA | NA | NA | NA | NA 
#                    3 |  1,177,736 | language | **category** | **title** | NA | NA | NA | NA 
#                    4 | 419,816     | language | **"sport"** or **"news"** | **sport_name** | **title** | NA | NA | NA 
#                    5 |  1,562,777 | language | **category** | **year** or **"av"** or **"live"**| **month** or **sport_name** | **title** | NA | NA
#                    6 |  96,574      | language | **general category** | **category** | **year** | **month** | **title** | NA
#                    7 |  13,180      | language |**"simp"** or **"trad"** | **general_category** | **category** | **year** | **month** | **title**
# 

# Getting the number of unique elements after splitting a slug, is a quick way to make sure they are indeed article titles and not something else. In this case 725k is almost the same as the one you see in the table above for length two. 

# In[ ]:


format(bbc[bbc['slug_split_length']==2]['slug'].str.split('/').str[1].nunique(), ',')


# #### Slugs of length three:

# In[ ]:


bbc[bbc['slug_split_length']==3]['slug'].sample(15)


# In[ ]:


bbc[bbc['slug_split_length']==3]['slug'].str.split('/').str[1].value_counts()[:20]


# In[ ]:


format(bbc[bbc['slug_split_length']==3]['slug'].str.split('/').str[2].nunique(), ',')


# #### Slugs of length four:

# In[ ]:


bbc[bbc['slug_split_length']==4]['slug'].sample(10)


# These slugs with length four seem start with "english/sport" and "english/news", so let's quickly check for what percentage this is true.

# In[ ]:


bbc[bbc['slug_split_length']==4]['slug'].str.contains('english/sport/|english/news').mean()


# In[ ]:


bbc[bbc['slug_split_length']==4]['slug'].str.split('/').str[0].value_counts(dropna=False)


# Pretty much all of them. We can also quickly check the distribution of "sport" and "english" because the overwhelming majority fall under these categories. 

# In[ ]:


bbc[bbc['slug_split_length']==4]['slug'].str.split('/').str[1].value_counts(dropna=False)


# Now we can go through those and count the values that are available. You will see that the majority will contain sports names, and that there are many other occurrences of other categories of news and articles.  
# I manually went through them, and created a set `non_sports`, based on which I was able to come up with a list of the sports in the URLs. This will help in createing a regular expression to extract sports names; football, basketball, etc. 

# In[ ]:


len_4_index_1_2 = (bbc[bbc['slug_split_length']==4]
                   ['slug'].str.split('/').str[1:3]
                   .str.join('/')
                  )


# In[ ]:


len_4_index_1_2.value_counts()[:7]


# In[ ]:


non_sports = {
    'live', 'scotland', 'northern-ireland', 'get-inspired', 'wales', 'live', 'av',
    'sports-personality', 'supermovers', 'africa', 'england', 'audiovideo', 'england',
    'video_and_audio', 'features', 'live', 'live', 'special-reports', 'in-depth',
    'in-depth', 'business', 'world', 'ultimate-performers', 'scotland',
    'move-like-never-before', 'made-more-of-their-world', 'wales', 'syndication',
    'west-bank-hitchhiking', 'headlines', 'stadium', 'trump-kim-meeting',
    'deadcities', 'wedding-mixed-race', 'northern_ireland', 'wedding-dress', 'system',
    'the-vetting-files', 'brodsky', 'syndication', 'wedding-designers', 'education',
    'world-cup-russia-hopefuls', 'wedding-guests', 'uk-scotland', 'tianshu',
    'yorkshire-and-humberside', 'west',  'west-midlands', 'uk-scotland','students_diary',
    'students_experience', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
    '12', '2015', '2016', 'articles', 'asia', 'chart_uk', 'east',  'east-mids-and-lincs',
    'election', 'europe', 'globalnews', 'in_depth', 'london-and-south-east',
    'north-east-and-cumbria', 'north-west', 'on_britain', 'south', 'south-west',
    'special_reports','politics', 'qa', 'tenglong', 'technology'
}


# We can now explore the slugs of length five: 

# In[ ]:


bbc[bbc['slug_split_length']==5]['slug'].str.split('/').str[1].value_counts()[:10]


# The year the article was published ..../YYYY/... 

# In[ ]:


bbc[bbc['slug_split_length']==5]['slug'].str.split('/').str[2].value_counts()[:20]


# Months, following the year .../YYYY/MM/... 

# In[ ]:


bbc[bbc['slug_split_length']==5]['slug'].str.split('/').str[3].value_counts()[:20]


# In[ ]:


format(bbc[bbc['slug_split_length']==5]['slug'].str.split('/').str[4].nunique(), ',')


# Length six index one:

# In[ ]:


bbc[bbc['slug_split_length']==6]['slug'].str.split('/').str[1].value_counts()[:10]


# Length six, index two:

# In[ ]:


bbc[bbc['slug_split_length']==6]['slug'].str.split('/').str[2].value_counts()[:15]


# In[ ]:


bbc[bbc['slug_split_length']==6]['slug'].str.split('/').str[1:3].str.join('/').value_counts()[:20]


# Length six, index three, again, years, followed by months: 

# In[ ]:


bbc[bbc['slug_split_length']==6]['slug'].str.split('/').str[3].value_counts()[:15]


# In[ ]:


bbc[bbc['slug_split_length']==6]['slug'].str.split('/').str[4].value_counts()[:15]


# In[ ]:


format(bbc[bbc['slug_split_length']==6]['slug'].str.split('/').str[5].nunique(), ',')


# The same process again for slugs of length seven: 

# In[ ]:


bbc[bbc['slug_split_length']==7]['slug'].str.split('/').str[0].value_counts()


# In[ ]:


bbc[bbc['slug_split_length']==7]['slug'].str.split('/').str[1].value_counts()[:15]


# In[ ]:


bbc[bbc['slug_split_length']==7]['slug'].str.split('/').str[2].value_counts()[:15]


# In[ ]:


bbc[bbc['slug_split_length']==7]['slug'].str.split('/').str[3].value_counts()[:15]


# In[ ]:


bbc[bbc['slug_split_length']==7]['slug'].str.split('/').str[4].value_counts()[:15]


# In[ ]:


bbc[bbc['slug_split_length']==7]['slug'].str.split('/').str[5].value_counts()[:15]


# # [Add new columns](#add_urls)
# 
# As a summary here is the table showing the different elements of URLs and where they fall for different slugh lengths (it's the same as the one above)
# 
#    Slug length when split | Number of articles | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 
# --------------------------------:|--------------------------:|---|---|---|---|---|---|---|
#                    1 |     38          | language | NA | NA | NA | NA | NA | NA 
#                    2 | 725,691     | language | **title** |  NA | NA | NA | NA | NA 
#                    3 |  1,177,736 | language | **category** | **title** | NA | NA | NA | NA 
#                    4 | 419,816     | language | **"sport"** or **"news"** | **sport_name** | **title** | NA | NA | NA 
#                    5 |  1,562,777 | language | **category** | **year** or **"av"** or **"live"**| **month** or **sport_name** | **title** | NA | NA
#                    6 |  96,574      | language | **general category** | **category** | **year** | **month** | **title** | NA
#                    7 |  13,180      | language |**"simp"** or **"trad"** | **general_category** | **category** | **year** | **month** | **title**    

# We know that there are many slugs containing the pattern `/YYYY/MM/` so we can easily extract them:

# In[ ]:


bbc['year_month'] = bbc['loc'].str.extract('/(\d{4}/\d{2})/')[0].values


# In[ ]:


bbc.dropna(subset=['year_month']).sample(7)


# We know that sports names occur in slugs of length four and five, on index two and three respectively. So we can easily extract those elements, get their unique values, and remove the elements from `non_sports`. With this we can create our `sport_regex`.

# In[ ]:


sport_names_4 = set(bbc[bbc['slug_split_length']==4]['slug'].str.split('/').str[2].unique())
sport_names_5 = set(bbc[bbc['slug_split_length']==5]['slug'].str.split('/').str[3].unique())


# In[ ]:


sport_names_all = sport_names_4.union(sport_names_5).difference(non_sports)
sport_regex = '/(' + '|'.join(sport_names_all) + ')/'
sport_regex


# In[ ]:


bbc['sport'] = bbc['loc'].str.extract(sport_regex)[0].values


# In[ ]:


bbc.dropna(subset=['sport', 'year_month']).sample(7)


# In[ ]:


bbc['sport'].value_counts()[:10]


# There is a good portion of URLs that seem to have six digits right after the `/YYYY/MM/` pattern. Looking closely at them, and comparing them to the `lastmod` tag (which is the index of the DataFrame), you'll see that they are almost identical. It seems to me that these are the actual publishing dates of the articles.  
# 
# If we can extract these and compare them to the `lastmod` tag, we can get a view on how often the BBC update their content. 

# In[ ]:


import numpy as np
extracted_pub_date = (bbc['slug']
                      .str.extract('/([012][0-9][01][0-9][0123][0-9])_')[0]
                      .replace('00000[01]', np.nan, regex=True))
extracted_pub_date


# In[ ]:


bbc['pub_date'] = pd.to_datetime(extracted_pub_date, format='%y%m%d', errors='coerce', utc=True)
bbc.dropna(subset=['sport', 'pub_date']).sample(7)


# In[ ]:


bbc['pub_date'].notna().mean()


# It seems 38.7% of the articles contain a date, so now we can compare them to the index, by a simple subtraction, and counting the occurrences of the diffrent time differences.
# 
# Here we are subtracting the index from the `pub_date` column and counting the days.

# In[ ]:


du.value_counts_plus(bbc['pub_date'].sub(bbc.index).dt.days, dropna=True)


# So the overwhelming majority of the updated articles get updated within a day or two. Probably immediate corrections, typos, or simple mistakes. Which means that it seems that `lastmod` is a good proxy for publishing date.
# 
# 
# Now in order to extract the categories from the URLs, we need to identify where they are located, which we have already done.  The following list of tuples are for (length, index) of each URL. For slugs of length three, index one is where the category occurs, for length five, it is index one, and so on:

# In[ ]:


category_indexes = [(3, 1), (5, 1), (6, 2), (7, 3)]
category_indexes


# In[ ]:


categories = set()
for length, index in category_indexes:
    temp_categories = set(bbc[bbc['slug_split_length']==length]['slug'].str.split('/').str[index].unique())
    categories = categories.union(temp_categories)


# In[ ]:


categories_regex = '/(' + '|'.join(categories) + ')/'
categories_regex


# In[ ]:


bbc['category'] = bbc['loc'].str.extract(categories_regex)[0].values


# In[ ]:


bbc.dropna(subset=['category', 'sport', 'pub_date']).sample(7)


# Finally, the last part of the URLs is where the most important content is. The title of the article. We will split the slugs by "/", take the last element, replace leading six digits with the empty character, and do the same for articles ending with a dash and a bunch of numbers, and titles that are only numbers. Then we replace dashes and underscores with spaces, and we get a better easier to read article title. 

# In[ ]:


bbc['title'] = (bbc['slug']
                .str.split('/')
                .str[-1]
                .str.replace('^\d{6}_|-\d+$|^\d+$', '')
                .str.replace('_|-', ' '))


# In[ ]:


bbc.dropna(subset=['title']).sample(7)


# We can now start to look at annual, monthly, or weekly publishing trends for the languages that we are interested in. The following function takes up to three languages and plots the trend by the specified time frame. 

# In[ ]:


timeframe_key = dict(A='Year', M='Month', W='Week')
def compare_langs(lang1=None, lang2=None, lang3=None, timeframe='A', y_scale='linear'):
    title_lang = []
    fig = go.Figure()
    for lang in [lang1, lang2, lang3]:
        if lang is not None:
            df = bbc[bbc['lang']==lang].resample(timeframe)['loc'].count()
            fig.add_scatter(x=df.index, y=df.values, name=lang.title(),
                            mode='markers+lines')
            title_lang.append(lang.title())
    fig.layout.title = 'Articles per ' + timeframe_key[timeframe] + ': ' + ', '.join(title_lang)
    fig.layout.yaxis.type = y_scale
    fig.layout.paper_bgcolor = '#E5ECF6'
    return fig
        


# In[ ]:


compare_langs('english', 'russian', 'portuguese')


# In[ ]:


compare_langs('english', 'russian', 'portuguese', y_scale='log')


# In[ ]:


languages = [None] +  sorted(bbc['lang'].unique())


# This is an interactive version of the function, and you will be able to run it if you fork the notebook, because it needs a runningn Python process to work: 

# In[ ]:


interact(compare_langs, lang1=languages, lang2=languages, lang3=languages,
         timeframe=dict(Year='A', Month='M', Week='W'), y_scale=['linear', 'log']);


# After this the options are endless for what you can do. This was a basic preparation and categorization of the data, so the following steps are hopefully easier to do. 
