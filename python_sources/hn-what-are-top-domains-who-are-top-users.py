#!/usr/bin/env python
# coding: utf-8

# Thank You Kaggle +  BigQuery team for providing easy-access to public data-sets via Kaggle Kernels.  I'm sure this will lead to lot of interesting collaboration between Kaggle participants.
# 
# In this notebook, I have used [HackerNews open-dataset](https://www.kaggle.com/hacker-news/hacker-news/data) to study the following:
# 
# 1. Most popular domains and their time-series 
# 2. Overall time-series of posting pattern in HN
# 3. Domains which contribute to high-scoring posts
# 4. Most enthusiastic and most-valued and most-diverse users in HN
# 
# Overall, I ensured the notebook remains exploratory, aided by insightful visualizations.  
# Note:  Hope Kaggle team will soon add R support to Dataset Kernels so `R::ggplot()` can be used for visual analysis.

# In[ ]:


# Disable warnings in Anaconda
import warnings
warnings.simplefilter('ignore')

from google.cloud import bigquery
from ggplot import *
from fbprophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wordcloud
import seaborn as sns

from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

pd.options.display.float_format = '{:,.2f}'.format
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


client = bigquery.Client()


# ## 1. Most popular domains and their Time-series during 2017-18[](http://)

# In this nested-query below, I have considered domains (e.g. medium.com) from which content were posted in HN in 2017-18. I filtered only top domains (say atleast 100 posts from a domain made it to HN in 2017-18) and then I aggregate i.e. count the number of posts in HN, per week per domain, for each of the respective-domains.
# 
# Note: Regex-pattern is to extract the domain-name (i.e. the part after http://) from the full URL path. You can refer [Regex 101 site](https://regex101.com/r/QnD6Cd/1/) to understand how it works.

# In[ ]:


query = """
#standard-sql
SELECT
  domain, count_dom, week_year, COUNT(*) posts
FROM (
  SELECT
    week_year, domain, COUNT(*) OVER(PARTITION BY domain) count_dom
  FROM (
    SELECT
      TIMESTAMP_TRUNC(timestamp, WEEK) week_year,
      REGEXP_EXTRACT(url, '//([^/]*)/?') domain
    FROM
      `bigquery-public-data.hacker_news.full`
    WHERE
      url!='' AND EXTRACT(YEAR FROM timestamp) IN (2017, 2018)) )
WHERE
  count_dom> 100
GROUP BY
  1, 2, 3
ORDER BY
  2 DESC,3
"""


# In[ ]:


query_job = client.query(query)
iterator = query_job.result(timeout=30)
rows = list(iterator)

top_domains_df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
top_domains_df.head(5)


# In[ ]:


print("Shape of DataFrame: {0}\nNumber of Unique domains: {1}".format(top_domains_df.shape, 
                                                          len(top_domains_df.domain.unique())))


# Limiting only to top-domains below, in terms of number of posts since we can't visualize all unique domains in one plot.
# Also we won't be considering the latest week (BigQuery follows Sun-Sat week), since it isn't complete yet.
# 
# In this plot, I haved used [python version of ggplot](http://ggplot.yhathq.com/) - it is functional but doesn't fully support finer-plot-adjustments yet.

# In[ ]:


top_top_domains_df = top_domains_df[(top_domains_df.count_dom>3000) & 
                                        (top_domains_df.week_year < pd.datetime(2018,2,11))  ]

g = ggplot(top_top_domains_df, aes(x = 'week_year', y = 'posts', color = 'domain')) +geom_line(size  = 2) + facet_wrap('domain'); 

t = theme_gray()
t._rcParams['xtick.labelsize'] = 5
g + t


# Note: Domains are in alphabetical-order and not in the order of number of posts. 
# 
# **Medium** is trending high overall. **GitHub** posts, surprisingly, show a falling trend. **Techcrunch**-posts have spiked suddenly around mid of last-year (Jul-2017)
# 
# Among the main-stream media, **NYTimes** stays ahead solidly, in terms of following by tech-community. **Bloomberg** closely follows, with** The Guardian** and **Ars Technica** trailing further behind.

# ## 2. Overall time-series of HN posts

# In[ ]:


query = """
#standardSQL
SELECT DATE(timestamp) day, COUNT(*) posts FROM 
`bigquery-public-data.hacker_news.full`
WHERE EXTRACT(YEAR FROM timestamp) IN (2016, 2017, 2018)
GROUP BY 1
ORDER BY 1
"""


# In[ ]:


query_job = client.query(query)
iterator = query_job.result(timeout=30)
rows = list(iterator)

daily_posts_df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
daily_posts_df.head(5)


# Here I use [Facebook Prophet](https://facebook.github.io/prophet/docs/quick_start.html) to analyse the time-series. Prophet follows sklearn API model and use [stan](http://mc-stan.org/) behind the scenes for its statistical modeling

# In[ ]:


daily_posts_df.day = pd.to_datetime(daily_posts_df.day, format='%Y-%m-%d')
daily_posts_df_short = daily_posts_df[daily_posts_df.day.dt.year > 2016]
daily_posts_df_short['posts'] = np.log(daily_posts_df_short['posts'])
daily_posts_df_short.columns = ['ds', 'y']

# https://facebook.github.io/prophet/docs/quick_start.html#python-api
m = Prophet()
m.fit(daily_posts_df_short);


# In[ ]:


# Predict posts pattern for next 6 months time
future = m.make_future_dataframe(periods=180)

forecast = m.predict(future)
m.plot(forecast); # logarithm of number of posts


# In the log-plot of number of posts above, we can see a rougly falling trend, which continues onto 2018. 
# 
# We can further decompose the time-series to look for overall-trend, seasonality and cyclic patterns.
# In the first plot below, we can see the point-estimate along with confidence-interval for the forecast.
# In the second one, we can see that most number of posts happen in Tuesday/Wednesday, while Saturdays and Sundays show less very posts shared in HN.

# In[ ]:


m.plot_components(forecast);


# ## 3. Domains leading to high-scoring posts

# In[ ]:


# Here we consider posts with score > 20

query = """
SELECT REGEXP_EXTRACT(url, '//([^/]*)/?') domain, COUNT(*) n_posts, COUNTIF(score>20) n_posts_20
FROM `bigquery-public-data.hacker_news.full`
WHERE url!='' AND EXTRACT(YEAR FROM timestamp)=2017
GROUP BY 1 ORDER BY 3 DESC LIMIT 100
"""


# In[ ]:


query_job = client.query(query)
iterator = query_job.result(timeout=30)
rows = list(iterator)

fifty_score_df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
fifty_score_df.head(5)


# In[ ]:


temp_df = fifty_score_df[fifty_score_df['n_posts'] > 2000]
domain_list = temp_df.sort_values(['n_posts'], ascending=[0]).domain.values

fifty_score_df_melt = pd.melt(temp_df, 
                              id_vars=['domain'], 
                              value_vars=['n_posts', 'n_posts_20'])
#fifty_score_df_melt.head(5)


# In[ ]:


rcParams['figure.figsize'] = 12, 8
ax = sns.barplot(x="domain", y="value", hue="variable", data=fifty_score_df_melt,
                order = domain_list)
for item in ax.get_xticklabels():
    item.set_rotation(60)    
ax.set(xlabel = "Domain", ylabel = "(1) # of Posts (2)# of Posts with > 20 votes");    


# Here we can see that **Medium** while contributing to the most number of posts, trails behind **GitHub** and **NYTimes** when it comes to high-scoring posts. Of course, here I have chosen an arbitrary-cutoff of >20 points to evaluate, but I believe this will apply even if a higher cutoff gets chosen.

# Or we can go one level deeper, where we can visualize **histograms of scores** for each of the individual posts shared from top-domains.

# In[ ]:


query = query = """
SELECT domain, score FROM (
    SELECT REGEXP_EXTRACT(url, '//([^/]*)/?') as domain, score
    FROM `bigquery-public-data.hacker_news.full` 
    WHERE EXTRACT(YEAR FROM timestamp)=2017)
WHERE domain in ('medium.com', 'www.nytimes.com', 'github.com', 'www.bloomberg.com') 
"""


# In[ ]:


query_job = client.query(query)
iterator = query_job.result(timeout=30)
rows = list(iterator)

top_domains_hist_df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
#top_domains_hist_df.head(5)


# In[ ]:


g = top_domains_hist_df[top_domains_hist_df.domain == 'github.com']['score']
n = top_domains_hist_df[top_domains_hist_df.domain == 'www.nytimes.com']['score']
m = top_domains_hist_df[top_domains_hist_df.domain == 'medium.com']['score']
b = top_domains_hist_df[top_domains_hist_df.domain == 'www.bloomberg.com']['score']

rcParams['figure.figsize'] = 12, 8
sns.kdeplot(m.rename('Medium'), shade = True)
sns.kdeplot(g.rename('GitHub'), shade = True)
sns.kdeplot(n.rename('NYTimes'), shade = True)
sns.kdeplot(b.rename('Bloomberg'), shade = True)

plt.xlabel('Score received by the post');
plt.ylabel('Kernel density plot of posts\' scores in 2017');


# In[ ]:


top_domains_hist_df.groupby(['domain'])['score'].agg([np.mean, np.median])


# On an average, **Bloomberg** and **NYTimes** stand out, relatively, in terms of quality of posts - this can be seen in the KDE plot as well as average-score. On the other hand, **Medium**'s score is pulled down by low-scoring posts.

# ## 4. Most enthusiastic and most-valued and most-diverse users in HN community

# **HackerNews** along with **Reddit**, remain to be the sites most-frequnted by techies, to discuss about anything under the sun and freely exchange ideas. While anonymity is the norm in these sites, one can still get a rough-sense of someone's persona by skimming through the posts. 
# 
# Or one can use the power of BigQuery to surface the most interesting of these personas. In this query, let's limit to posts pattern during 365 days in year 2017.

# In[ ]:


query = """
SELECT `by` author, COUNT(DISTINCT domain) n_domains, SUM(score) total_score, AVG(score) avg_score, COUNT(*) AS n_posts FROM (
    SELECT `by`,REGEXP_EXTRACT(url, '//([^/]*)/?') as domain, score
    FROM `bigquery-public-data.hacker_news.full` 
    WHERE EXTRACT(YEAR FROM timestamp)=2017 AND `by` !='' AND url !='') 
GROUP BY 1 ORDER BY 5 DESC
LIMIT 5000
"""


# In[ ]:


query_job = client.query(query)
iterator = query_job.result(timeout=30)
rows = list(iterator)

top_users_df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
#top_users_df.head(5)


# ### Most enthusiastic posters from 2017

# In[ ]:


temp_df = top_users_df.sort_values(['n_posts'], ascending =[0]).head(10)
temp_df


# In[ ]:


rcParams['figure.figsize'] = 12, 8
ax = sns.barplot(x="n_posts", y="author", data=temp_df)
ax.set(xlabel = "Number of posts made by author in 2017", ylabel = "Author of post");  


# ### Most valued posters from 2017 ( i.e. high avg  score/ post)

# In[ ]:


temp_df = top_users_df.sort_values(['avg_score'], ascending =[0]).head(10)
temp_df


# In[ ]:


ax = sns.barplot(x="avg_score", y="author", data=temp_df)
ax.set(xlabel = "Average score per post", ylabel = "Author of post"); 


# ### Most diverse posters from 2017 ( i.e. content from diverse domains )

# In[ ]:


# Most diverse posters
temp_df = top_users_df.sort_values(['n_domains'], ascending =[0]).head(10)
temp_df


# In[ ]:


ax = sns.barplot(x="n_domains", y="author", data=temp_df)
ax.set(xlabel = "Number of unique-domains posted by author, 2017", ylabel = "Author of post"); 


# ### Most diverse posters, still with high average-scores

# In[ ]:


# Taking an arbitrary average of 100 score per post
temp_df = top_users_df[top_users_df.avg_score > 100].sort_values(['n_domains'], ascending =[0]).head(10)
temp_df


# In[ ]:


ax = sns.barplot(x="n_domains", y="author", data=temp_df)
ax.set(xlabel = "Number of unique-domains posted by top-scoring authors, 2017", ylabel = "Author of post"); 


# Hope you found the Kernel insightful.  Now, what more can be done with this dataset:
# 
# 1. Identifying network-patterns by studying interaction pattern between users
# 2. Topical-Modeling on articles-shared by mining language-semantics in title, comments section
# 3. PageRank analysis by generating an acyclic-graph with directed links from commenters to posters
# 4. Predicting home-page-chance from title of post
# 5. Understanding popularity (or) adoption of innovations with time-series of words used in posts' titles
# 
# and so on.
