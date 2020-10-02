#!/usr/bin/env python
# coding: utf-8

# ## Reading in the data
# 
# There are basically three different data sets:
# - the meta data for each disinformation claim
# - the html page for each disinformation article
# - the raw text of a disinformation article as obtained by using the package `newspaper`
# 
# The meta data itself is already rich enough for analysis and it also contains a summary (in English) of each claim, few sentences why the claim is not true and a short abstract. 
# Of course, the html pages contain the raw texts as well but they also contain much more information about for example links contained in the text and meta information of the web page. Note that many articles were not downloaded from the original web page but from the web archive. Only newsarticles were downloaded but since the csv data also contains disinformation claims of type media objects which were not downloaded. Unfortunately, not all web pages could be downloaded, in that case most likely both the original and the web archive link were not working anymore.

# In[ ]:


import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ast


# ### The csv meta data

# In[ ]:


df = pd.read_csv("/kaggle/input/data/data.csv", index_col=0, 
                 converters={"keyword_name": ast.literal_eval, # these are list columns, need to use ast to actually get lists and not strings
                            "keyword_id": ast.literal_eval,
                            "country_id": ast.literal_eval,
                            "country_name": ast.literal_eval,
                            "has_parts": ast.literal_eval})

# make merging later easier
df["creative_work_id"] = df["creative_work_id"].str.replace("/", "", n=1).str.replace("/", "_")
df.head(3)


# In[ ]:


df.shape


# In total, there are 7369 news articles or media objects but only for the news articles the html/txt was downloaded:

# In[ ]:


df[df.type == "http://schema.org/NewsArticle"].shape


# In[ ]:


articles_text = [text_file for text_file in os.listdir("/kaggle/input/data/texts") if text_file.endswith(".txt")]

texts = []
for path in articles_text:
    with open(os.path.join("/kaggle/input/data/texts", path), "r") as file:
        texts.append(file.read())

data = pd.DataFrame().from_dict({"path": articles_text, "text": texts})

data["creative_work_id"] = data["path"].str.replace(".txt", "")

# merge the texts with the rest of the data
data.merge(df, on="creative_work_id", how="left").head(3)


# We can also load the html data instead:

# In[ ]:


articles_html = [text_file for text_file in os.listdir("/kaggle/input/data/html") if text_file.endswith(".html")]

html = []
for path in articles_html:
    with open(os.path.join("/kaggle/input/data/html", path), "r") as file:
        html.append(file.read())

html[0][0:500]


# One can for example use the package `newspaper` to extract information from the html:

# In[ ]:


# !pip install newspaper3k 


# In[ ]:


from newspaper import Article
from newspaper.outputformatters import OutputFormatter
from newspaper.configuration import Configuration

conf = Configuration()
conf.keep_article_html = True


# In[ ]:


art = Article('', keep_article_html=True)
art.set_html(html[0])
art.parse()

art.nlp()


# In[ ]:


art.keywords


# In[ ]:


art.summary


# 
# We can for example look for links on the web page:

# In[ ]:


import re
re.findall(r"(?:<a .*?href=\")(.*?)(?:\")", art.html)[0:12]

