#!/usr/bin/env python
# coding: utf-8

# # How to Check if a URL is Indexed by Google With Python
# How can you check if a certain URL is indexed or not by Google?  
# Although you can check the health and quality of indexing for your pages, no report tells you which pages are not indexed. Those pages might not be indexed because either Google thinks they are duplicates/spam, or Google simply can't see them for some reason. In any case, it is crucial to know whether your URLs are indexed. 
# 
# My suggested approach to find this out uses the official Google Custom Search Engine API to achieve this. 
# 
# For a more in-depth examples on how to use that for SEO research: 
# 
# - [Tutorial on how to use Google CSE API for SERP research](https://www.semrush.com/blog/analyzing-search-engine-results-pages/)
# - [Tutorial on using Google CSE and YouTube API for SERPs research on both platforms](https://www.kaggle.com/eliasdabbas/recipes-keywords-ranking-on-google-and-youtube)
# 
# The basic idea is simple. We send a query to the API, and analyze the response.  
# The important thing is to run the query using the `site:` operator together with the URLs that we want to check.  
# So, if we had `http://mysite.com/1`, `http://mysite.com/2`, and `http://mysite.com/3`, we would have to send requests for `site:http://mysite.com/1`, `site:http://mysite.com/2`, and `site:http://mysite.com/3` as the queries.
# 
# Trying the URLs alone or with another operator like `info:` would not work, because Google would return similar pages and URLs. With the `site:` operator, we are specifically asking for information about the domain or URL, and if it doesn't exist, we would have zero results. 
# 
# Here are the steps to set up an account to import data:
# 
# 1. [Create a custom search engine](https://cse.google.com/cse/). At first, you might be asked to enter a site to search. Enter any domain, then go to the control panel and remove it. Make sure you enable "Search the entire web" and image search. You will also need to get your search engine ID, which you can find on the control panel page.
# 2. [Enable the custom search API](https://console.cloud.google.com/apis/library/customsearch.googleapis.com). The service will allow you to retrieve and display search results from your custom search engine programmatically. You will need to create a project for this first.
# 3. [Create credentials for this project](https://console.developers.google.com/apis/api/customsearch.googleapis.com/credentials) so you can get your key.
# 4. [Enable billing for your project](https://console.cloud.google.com/billing/projects) if you want to run more than 100 queries per day. The first 100 queries are free; then for each additional 1,000 queries, you pay USD $5.
# 
# 

# In[ ]:


import advertools as adv
import pandas as pd

cx = 'YOUR_CUSTOM_SEARCH_ENGINE'
key = 'YOUR_KEY'
adv.__version__


# In[ ]:


wikipedia_urls = ['https://www.wikipedia.org/',  # search for this domain as a keyword
                  'https://www.wikipedia.org/wrong_page',  # search for this domain as a keyword (does not exist)
                  'site:https://www.wikipedia.org/', # search for this site/page (exists)
                  'site:https://www.wikipedia.org/wrong_again'] # search for this site/page (does not exist)


# The following code gets the data from Google. You simply pass in the list of queries as the parameter `q` and you get all results in one DataFrame. `cx` and `key` are used to authenticate, as mentioned above.

# In[ ]:


# wikipedia = adv.serp_goog(cx=cx, key=key, q=wikipedia_urls)


# In[ ]:


wikipedia = pd.read_csv('../input/wikipedia_serps.csv')
wikipedia[['searchTerms', 'rank', 'title', 'displayLink', 'formattedTotalResults']]


# The first two queries, namely `https://www.wikipedia.org/` and `https://www.wikipedia.org/wrong_page`, search for two URLs. Because there is no operator, they are treated as keywords. So, although the page `https://www.wikipedia.org/wrong_page` does not exist, Google tries to find something that might be relevant, and it actually found 22,300 pages for this query. 
# 
# The last two queries `site:https://www.wikipedia.org/` and `site:https://www.wikipedia.org/wrong_again` are not regular keywords. They ask for information about those specific URLs (make sure to include the full path with `https://www` so you don't get information about the domain). 
# 
# As you can see, we have one result for the URL that exists and NaN for the URL that does not.  
# And we are Done! 
# 
# For real life scenarios, you would have a large set of URLs to check and make sure they are indexed. All you have to do is create a list of those URLs and pass them as one argument to the `serp_goog` function (including the `site:` operator). The function handles looping and concatenating the responses into the final DataFrame.
# 
# The cost factor might become significant if you have hundreds of thousands of URLs, if you check them regularly. You can judget best, based on your budget and priorities.  
# My suggestion is to pick the top key pages that absolutely must be indexed and focus on those. Ideally, they would link to deeper pages, and ensure they are findable by search engine spiders.
