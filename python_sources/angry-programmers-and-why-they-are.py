#!/usr/bin/env python
# coding: utf-8

# # So why they are so angry?

# Few changes in query compared to the original:
# - Comparison against the f-word is not case-sensitive
# - The f-word in the middle of the string is fine too
# - Length of the message is limited to $<50$
# - Results are grouped by message (case-insensitive)
# - Amount of hits for each message is counted
# - Ordered so the most frequent is at the top

# In[ ]:


query = """
SELECT message, COUNT(1) as hits
FROM (
    SELECT LOWER(message) as message
    FROM   `bigquery-public-data.github_repos.commits`
    WHERE  LENGTH(message) > 5
     AND   LENGTH(message) < 50
     AND   LOWER(message) LIKE '%fuck%'
)
GROUP BY message
ORDER BY hits DESC
"""


# In[ ]:


from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()


# In[ ]:


# See https://www.kaggle.com/sohier/efficient-resource-use-in-bigquery
def estimate(bq_client, query):
    # see https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun
    my_job_config = bigquery.job.QueryJobConfig()
    my_job_config.dry_run = True
    my_job = bq_client.query(query, job_config=my_job_config)
    BYTES_PER_GB = 2**30
    return my_job.total_bytes_processed / BYTES_PER_GB


# In[ ]:


estimate(client, query)


# Not that small, but manageable.

# In[ ]:


query_job = client.query(query)
rows = query_job.result()
rows = list(rows) # If you are sure you won't need the data twice, remove this one


# In[ ]:


why = pd.DataFrame((x.strip(),y) for x,y in rows)
why.columns = ('message', 'hits')


# In[ ]:


why.shape


# Okay, we got it.
# Let's see...

# In[ ]:


why[why.hits > 5]


# We can conclude that most of people just hate writing commit messages.  
# Many hate git, maybe for that same reason (why else, lol).  
# Some of them hate themselves and some don't understand what's happening. 

# Let's exclude the word and it's forms and see what we get.  *Sorry for the messy code.*

# In[ ]:


no_f = lambda s: 'fuck' not in s
temp = ((' '.join(filter(no_f, x[0].split())), x[1]) for i,x in why.iterrows())
why_censored = pd.DataFrame(word for word in temp if word[0])
why_censored.columns = ('message', 'hits') # I messed up and lost colnames in process
why_censored[why_censored.hits > 10]


# I also wonder what long messages occur many times.  
# *25 is but a magical number*

# In[ ]:


sorted(why.message[why.hits > 25], key=len, reverse=True)


# Okay, time to make some fancy word clouds.

# In[ ]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[ ]:


def df_to_dict(df):
    return  {x['message']: x['hits'] for i,x in df.iterrows()}


# In[ ]:


def split_keys(d):
    from collections import Counter
    d2 = Counter()
    for k,v in d.items():
        for word in k.split():
            d2[word] += d[k]
    return d2


# In[ ]:


def draw_cloud(freq_dict, figsize=(19,10.8),
               **cloud_args):
    cloud = WordCloud(**cloud_args)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(cloud.generate_from_frequencies(freq_dict))


# In[ ]:


draw_cloud(df_to_dict(why),
           background_color='white',
           width=1900, height=1080,
           max_font_size=300,
#          max_words=1000,
           relative_scaling=0.5)


# In[ ]:


draw_cloud(df_to_dict(why_censored),
           background_color='white',
           width=1900, height=1080,
           max_font_size=300,
#          max_words=1000,
           relative_scaling=0.5)


# In[ ]:


draw_cloud(split_keys(df_to_dict(why)), 
           background_color='white',
           width=1900, height=1080,
           max_font_size=300,
#          max_words=1000,
           relative_scaling=0.5)


# In[ ]:


draw_cloud(split_keys(df_to_dict(why_censored)), 
           background_color='white',
           width=1900, height=1080,
           max_font_size=300,
#          max_words=1000,
           relative_scaling=0.5)


# Fucking wallpapers are ready.
