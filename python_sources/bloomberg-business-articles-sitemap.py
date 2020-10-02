#!/usr/bin/env python
# coding: utf-8

# # XML Sitemaps Analysis with Python
# 
# This is a quick tutorial on how to get and parse XML sitemaps into a `pandas` `DataFrame`, and how they can be analyzed.  
# 
# The `lastmod` tag provides the ability to check the publishing frequency of the website, and how it changed across years, months, days, or any time unit.  
# The `loc` tag which is basically the URL, can provide some limited information on the content, depending on what the website includes in the URL. The richer the URLs the more the information. URLs like http://example.com/article/12345 provide nothing interesting.
# 
# We first define a function to parse the sitemaps (it is actually taken from [scrapy's code](https://github.com/scrapy/scrapy/blob/master/scrapy/utils/sitemap.py) with many modifications). 
# This function takes a sitemap URL, goes through all the elements and puts them in a `DataFrame`. It would typically have the columns `"loc", "lastmod", "changefreq", "priority", "alternate"`, but not necessarily all of them.  
# If the sitemap happens to be a sitemap index, then it goes through all of the inclcuded sitemaps one by one, and returns everything in one `DataFrame`.
# 

# In[ ]:


import logging
from xml.etree import ElementTree
from urllib.request import urlopen

import pandas as pd


def sitemap_to_df(sitemap_url):
    xml_text = urlopen(sitemap_url)
    tree = ElementTree.parse(xml_text)
    root = tree.getroot()
        
    xml_df = pd.DataFrame()

    if root.tag.split('}')[-1] == 'sitemapindex':
        for elem in root:
            for el in elem:
                if el.text.endswith('xml'):
                    try:
                        xml_df = xml_df.append(sitemap_to_df(el.text), ignore_index=True)
                        logging.info(msg='Getting ' + el.text)
                    except Exception as e:
                        logging.warning(msg=str(e) + el.text)
                        xml_df = xml_df.append(pd.DataFrame(dict(sitemap=el.text),
                                                            index=range(1)), ignore_index=True)

    else:
        logging.info(msg='Getting ' + sitemap_url)
        for elem in root:
            d = {}
            for el in elem:
                tag = el.tag
                name = tag.split('}', 1)[1] if '}' in tag else tag

                if name == 'link':
                    if 'href' in el.attrib:
                        d.setdefault('alternate', []).append(el.get('href'))
                else:
                    d[name] = el.text.strip() if el.text else ''
            d['sitemap'] = sitemap_url
            xml_df = xml_df.append(pd.DataFrame(d, index=range(1)), ignore_index=True)
    if 'lastmod' in xml_df:
        xml_df['lastmod'] = pd.to_datetime(xml_df['lastmod'], utc=True)
    if 'priority' in xml_df:
        xml_df['priority'] = xml_df['priority'].astype(float)
    return xml_df

    


# # Bloomberg Business Articles Sitemaps
# 
# As an example I'll go through the sitemaps of Bloomberg business articles from 1991 to March 2020.   
# A quick look at their [sitemap index](https://www.bloomberg.com/feeds/businessweek/sitemap_index.xml) reveals that they have a template for their sitemaps as follows: 
# `https://www.bloomberg.com/feeds/businessweek/sitemap_{year}_{month}.xml`  
# It seems that they have a sitemap for each month. There are also video sitemaps.
# 
# I specified the column `lastmod` as the index, and parsed it as a date object, which makes the DataFrame a timeseries. This makes it easy to resample it to any required time unit.

# In[ ]:


bloomberg = pd.read_csv('../input/bloomberg-business-articles-urls/bloomberg_biz_sitemap.csv',
                        parse_dates=['lastmod'], index_col='lastmod',
                        low_memory=False)
bloomberg['priority'] = bloomberg['priority'].astype(float)
print(bloomberg.shape)
bloomberg


# 300k URLs seems like a lot, which makes it more interesting.  
# Some of the values of the index are `NaT` which is the missing date/time representation. After manually checking, I realized many of them were for video URLs. Let's quantify this:

# In[ ]:


bloomberg[bloomberg.index.isna()]['sitemap'].str.contains('video').mean()


# 99.8% of the sitemap URLs with a missing date contain the word "video", which means it is right. We can also check from the other direction (how many sitemap URLs with missing date and time correspond to URLs with "video"):

# In[ ]:


bloomberg[bloomberg.index.notna()]['sitemap'].str.contains('video').mean()


# ## Publishing Trends
# The first question I usually ask, is how often do they publish?  
# As this is a large time frame, we can first take a look at articles published per year, and then zoom in further to monthly trends. 

# In[ ]:


by_year_count = bloomberg.resample('A')['loc'].count()
by_year_count


# In[ ]:


by_month_count = bloomberg.resample('M')['loc'].count()
by_month_count


# In[ ]:


import plotly.graph_objects as go
fig = go.Figure()
fig.add_bar(x=by_year_count.index, y=by_year_count.values, showlegend=False)
fig.layout.template = 'none'
fig.layout.title = 'Bloomberg Business Articles Published per Year 1991 - 2020'
fig.layout.xaxis.tickvals = by_year_count.index.date
fig.layout.xaxis.ticktext = by_year_count.index.year
fig.layout.yaxis.title = 'Number of articles'
fig


# There are clear and very interesting trends in their publishing. It slowed down in 1996 after gowing gradually up. It shot up almost four times in 2001, then maintained a certain publishing level. The sharp drop after 2009 could be attributed to the financial crisis, as many financial institutions were having a difficult time. But Bloomberg? I mean Mike just spent half a billion on his presidential campaign...  
# The sudden drop in 2015, followed by a massive spike in 2019 are strange. It's not clear to me why. Maybe they just decided to focus much more on their website and publish more. There might be issues with the data. I'll try to check using dates in the URLs below, which indicates that the data might be valid, but it's still worth investigating, as the articles in 2019 were almost eighteen times more than 2018, and 2020 is on track to top that.  
# I'll assume that the data are correct, as this is just an exploration on what you can extract from sitemaps. 
# We can visualize the same trend, on a monthly basis and see: 

# In[ ]:


fig = go.Figure()
fig.add_bar(x=by_month_count.index, y=by_month_count.values, showlegend=False)
fig.layout.template = 'none'
fig.layout.title = 'Bloomberg Business Articles Published per Month 1991 - 2020'
fig.layout.yaxis.title = 'Number of articles'
fig


# Now the year 2019 looks really strange. If you zoom you'll see a jump from 159 articles in March, and then 2,360 articles the following month. It's highly unlikely that the editorial team grew that fast and immediately. Especially that they maintained that level of publishing throughout the year. They might have hired a huge number of people all of a sudden, made a large acquisition, or something like that. 
# March 2020, the month of Corona, shows the highest publishing level, which makes sense.  
# We can also see their weekday trends: 

# In[ ]:


(bloomberg
 .groupby(bloomberg.index.weekday)['loc']
 .count().to_frame()
 .rename(columns=dict(loc='count'))
 .assign(day=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
 .style.bar().format(dict(count='{:,}'))
)


# I would expect Monday to be the busiest day, but the difference seems massive.
# 
# ## URL Analysis
# Now let's take a look at the URLs and try to get some information from them. First, a quick look at a URL to see their structure:

# In[ ]:


bloomberg[:1]['loc'].values[0], bloomberg[:1]['loc'].index


# It seems this is the pattern that they use:  
# `bloomberg.com/news/{type}/{YYY-MM-DD}/slug-containing-important-keywords`  
# So let's see how many `type`s of articles they have: 

# In[ ]:


bloomberg['loc'].str.split('/').str[-3].value_counts()


# In[ ]:


bloomberg['loc'].str.split('/').str[-4].value_counts()


# Pretty much all of the URLs are `news`, so we can ignore that.  
# The last part of the URL is the one that contains the most important information. We can now replace dashes with spaces, and count the words in those slugs.

# In[ ]:


bloomberg['loc'].str.split('/').str[-1]


# In[ ]:


get_ipython().system('pip install advertools')


# ## Word Frequency / Text Analysis
# 
# For more information on the [`word_frequency`](https://advertools.readthedocs.io/en/master/advertools.word_frequency.html) function, you can check out my article on [text analysis for online marketing](https://www.semrush.com/blog/text-analysis-for-online-marketers/).

# In[ ]:


import advertools as adv
adv.word_frequency(bloomberg['loc'].dropna().str.split('/').str[-1].str.replace('-', ' ')).iloc[:20, :2]


# Most of the words are too generic. "China", "Fed", and "Trump" might be useful.  
# We can also run the same function by specifying `phrase_len` to be two or more. This counts the phrases that contain two words, where the previous function implicitly specified one.

# In[ ]:


adv.word_frequency(bloomberg['loc'].dropna().str.split('/').str[-1].str.replace('-', ' '),phrase_len=2).iloc[:20, :2]


# What the dot?!
# 

# In[ ]:


bloomberg['loc'].dropna()[bloomberg['loc'].dropna().str.contains('-dot')].str.split('/').str[-1][:10].values


# In some cases it makes sense, like the title that contains "dot-com era" for example. But in the others, they have decided to replaces dots "." with the word "dot". For example "Amazon vs. Jet.com" was transformed to "amazon-vs-dot-jet-dot-com". Strange. 
# 
# Now let's extract all URLs that contain "china", resample annually, and plot.

# In[ ]:


by_year_count_china = bloomberg[bloomberg['loc'].fillna('').str.contains('china', case=False)].resample('A')['loc'].count()


# In[ ]:


fig = go.Figure()
fig.add_scatter(x=by_year_count.index,
                y=by_year_count.values,
                name='All Articles',
                yaxis='y', 
                mode='lines+markers',
                marker=dict(size=10))
fig.add_scatter(x=by_year_count_china.index,
                y=by_year_count_china.values,
                name='China Articles',
                yaxis='y2',
                mode='lines+markers',
                marker=dict(size=10))
fig.layout.template = 'none'
fig.layout.title = 'BusinessWeek Articles Published per Year 2001 - 2020 (China vs All Topics)'
fig.layout.xaxis.tickvals = by_year_count.index.date
fig.layout.xaxis.ticktext = by_year_count.index.year
fig.layout.yaxis.title = 'All articles'
fig.layout.yaxis2 = dict(title='China', overlaying='y', side="right", position=1, anchor='free')


fig


# In most cases, the trend of publishing stories about China seems to be in line with all articles, as you would expect for such an important topic. Note that the scales are different for each of Y axes.  
# It's a bit misleading to see the big spike in articles on China, but the problem is mainly due to the big spike from 2018 to 2019 in all articles. In these cases it might be better to plot them on a log scale, so we can see percentage changes more clearly, which is what we actually want.  
# The above script is generalized below to a function that takes an arbitrary number of words, if you want to compare three or four or more. It optionally allows you to include the trend for all the articles, for perspective on how the topic(s) compare to the total, and specifying whether or not you want to have the Y-axis on a log scale: 

# In[ ]:


def plot_topic_vs_all(*topics, include_all=True, log_y=True):
    
    fig = go.Figure()
    if include_all:
        fig.add_scatter(x=by_year_count.index,
                        y=by_year_count.values,
                        name='All Articles',
                        yaxis='y', 
                        mode='lines+markers',
                        marker=dict(size=10))
    for topic in topics:
        topic_df = bloomberg[bloomberg['loc'].fillna('').str.contains(topic, case=False)].resample('A')['loc'].count()
        fig.add_scatter(x=topic_df.index,
                        y=topic_df.values,
                        name=topic + ' Articles',
                        mode='lines+markers',
                        yaxis='y2',
                        marker=dict(size=10))
    fig.layout.template = 'none'
    all_topics = ' vs. All Topics)' if include_all else ')'
    fig.layout.title = f'BusinessWeek Articles Published per Year 2001 - 2020 ({", ".join(topics)}' + all_topics
    fig.layout.xaxis.tickvals = topic_df.index.date
    fig.layout.xaxis.ticktext = topic_df.index.year
    fig.layout.yaxis.title = 'All articles'
    fig.layout.yaxis2 = dict(title=f"'{topics}' articles", overlaying='y',
                             side="right", position=1, anchor='free')
    if log_y:
        fig.layout.yaxis.type = 'log'
        fig.layout.yaxis2.type = 'log'


    return fig


# In[ ]:


plot_topic_vs_all('oil', 'china', include_all=True, log_y=True)


# In[ ]:


plot_topic_vs_all('trump', 'china', include_all=False, log_y=True)


# In[ ]:


plot_topic_vs_all('trump', 'huawei', 'china', include_all=False)


# In[ ]:


plot_topic_vs_all('trump', 'fed', include_all=False)


# ## Compare `lastmod` to Publishing Date
# 
# Article URLs contain the date of publishing. These can be compared to the `lastmod` tag value to see whether or not they match. This can serve three purposes: 
# 1. Check to see how often stories are updated. In a news website like this I don't expect stories to be updated a lot because they have a short shelf life. In other sites that provide less time-sensitive content (like recipes or exercise for example) you might expect them to do more updates. 
# 2. Check if there are big discrepancies in the data, as we will do for this case. 
# 3. When there is no `lastmod` like in the video URLs, this date can serve that purpose, and you can do the same exercise for video articles.
# Looking at some of the dates included in the URLs I realized some were dated in the year one thousand. 

# In[ ]:


bloomberg[bloomberg['loc'].str.contains('articles/1000')].iloc[:10, :1]['loc'].str[30:]


# The dates in `lastmod` seem to be correct, but not in the URLs. Let's change those to 2000

# In[ ]:


bloomberg['loc'] = bloomberg['loc'].str.replace('articles/1000-', 'articles/2000-')


# Now we can extract those numbers, put them in a new column `pub_date` as datetime objects.

# In[ ]:


bloomberg['pub_date'] = pd.to_datetime(bloomberg['loc'].str.extract('(\d{4}-\d{2}-\d{2})')[0])
bloomberg['pub_date']


# `dates_equal` compares the inferred `pub_date` with the `lastmod` tag.

# In[ ]:


bloomberg['dates_equal'] = bloomberg['pub_date'].dt.date == bloomberg.index.date


# In[ ]:


bloomberg['dates_equal'].mean()


# It seems only 24.7% of the values are equal. Let's create a new column that counts the difference between those two, for a better view and see if there are large differences.

# In[ ]:


bloomberg['days_to_update'] = bloomberg['pub_date'].dt.date - bloomberg.index.date


# In[ ]:


bloomberg.iloc[:5, [0, 1, 2, 3, 4, -3, -2, -1]]


# In[ ]:


bloomberg['days_to_update'].value_counts(normalize=True)[:15]


# The majority of the updates seem to happen one or zero days after publishing, which means not much changes happen after publishing.  
# 
# There are many other text mining and topic modeling techniques that can be run on this data. The fact that it has dates makes it even more interesting, because you can also compare the changes in the topics and keywords across time, and see if you can get better insights. 
