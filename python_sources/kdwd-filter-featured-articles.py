#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import pandas as pd 
from tqdm import tqdm


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


file_path = "/kaggle/input/english-wikipedia-featured-articles-20200601/featured_articles_20200601.csv"
df_fa = pd.read_csv(file_path)
fa_page_ids = set(df_fa["page_id"].values)


# In[ ]:


file_path = "/kaggle/input/kensho-derived-wikimedia-data/page.csv"
df_page = pd.read_csv(file_path, keep_default_na=False)
df_page = df_page.set_index("page_id")


# In[ ]:


NUM_KLAT_LINES = 5_343_564
fa_pages = []
file_path = "/kaggle/input/kensho-derived-wikimedia-data/link_annotated_text.jsonl"
with open(file_path, "r") as fp:
    for line in tqdm(fp, total=NUM_KLAT_LINES):
        page = json.loads(line)
        if page["page_id"] in fa_page_ids:
            fa_pages.append(page)


# In[ ]:


fa_dataset = {
    "page_id": [],
    "page_title": [],
    "page_views": [],
    "intro_text": [],
}


# In[ ]:


for page in fa_pages:
    fa_dataset["page_id"].append(page["page_id"]) 
    fa_dataset["page_title"].append(df_page.loc[page["page_id"]]["title"])
    fa_dataset["page_views"].append(df_page.loc[page["page_id"]]["views"])
    fa_dataset["intro_text"].append(page["sections"][0]["text"])


# In[ ]:


df_fa_dataset = pd.DataFrame(fa_dataset)


# In[ ]:


df_fa_dataset


# In[ ]:


df_fa_dataset.to_csv("kdwd_featured_articles.csv", index=False)


# In[ ]:




