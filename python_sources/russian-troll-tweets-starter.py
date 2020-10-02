#!/usr/bin/env python
# coding: utf-8

# ## Initial EDA on the Russian tweets
# * Data from https://fivethirtyeight.com/features/why-were-sharing-3-million-russian-troll-tweets/
#     * 538 and the center did a great job at getting, sharing and analyzing the initial data, so this will be just the basic for getting the data into a more amenable form. 
#     
#     * In future - I'll add an additional sample of tweets from the same period or in English, to see if we can seperate  the distributions (not just between the groups/hashtags)

# In[ ]:


import pandas as pd 
import glob
import os
print(os.listdir("../input"))

PATH = "../input"


# ## Concatenate the dataframes
# * https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
# df = pd.concat((pd.read_csv(f) for f in all_files))
# 
# > os.path.join(path, "*.csv")
# 

# In[ ]:


filenames = glob.glob(os.path.join(PATH, "*.csv"))
print(filenames)
df = pd.concat((pd.read_csv(f) for f in filenames))
print(df.shape)
df.head()


# In[ ]:


## I'll keep the external id for now. We drop the harvested ID. 
df.drop("harvested_date",axis=1,inplace=True)


# *  Replace the existing external author id number with a shorter one, using pandas categoricals (not  really needed, but it's a tiny bit more mem.efficient, and can save data corruption if opened in excel. Also, I like this minor code snippet).
#     * https://stackoverflow.com/questions/38088652/pandas-convert-categories-to-numbers : Get categoricals mapping (str_id to int) 
# 
# * **BUG** - mismatch in # entities, disabling for now. 

# In[ ]:


# print(df.external_author_id.nunique()) # 2489
# df.external_author_id = pd.Categorical(df.external_author_id).codes
# print(df.external_author_id.nunique())  # # 2490

# # There's a mismatch - unknown where my bug is. Commenting out for now!


# In[ ]:


df.shape


# In[ ]:





# ### minor EDA
# 

# In[ ]:


df.dtypes


# In[ ]:


df.describe(include="all")


# In[ ]:





# ### 20%  duplicated posts!
# 
# * A lot of duplicated posts (may be due to retweets or lazyness)
# * Let's drop these for now - it makes author identification/resharing easier, but it's less interesting to us for some other purposes + makes it to oeasy

# In[ ]:


df.shape[0] - df.drop_duplicates(subset="content").shape[0]


# In[ ]:


df.drop_duplicates(subset="content",inplace=True)
print("df without duplicated content:",df.shape[0])


# ## Peek at the authors:
# * 2,848 authors
# * Relatively imbalanced (range from tens of thousands to a handful of tweets).
# * Unknown yet if there are  reallymultiple "writers" per authors or whether the same writers create content for multiple author.  (beyond the "external id" which doesn't necessarily capture the truth).
#     * Would be an interesting problem to practice *author identification* on! 

# In[ ]:


# how many unique authors?
# df.author.value_counts().shape[0]
df.author.nunique()


# In[ ]:


df.author.value_counts().head() # Top authors have tens of thousands of tweets
# df.author.value_counts(normalize=True).head() # top author are 1-2 % of all tweets 


# In[ ]:


(df.author.value_counts()<5).sum() # a few hundred with only a few posts


# ## Let's look at (English) language
# * We may want to keep only English language tweets. (As detected by Twitter's algo presumably)  = A bit of noise. (Note the many tweets that are in very rare languages. More likely they were'nt identified correctly)
# *  Makes it much more relevant for any future "Identify foreign propaganda/russian spy" type models
# 
# * Interestingly, Russian is the second most common language! 
#     * Would be interesting to leave the Russian tweets in , in future
# * We leave geography alone.

# In[ ]:


df.language.value_counts()


# In[ ]:


df_en = df.loc[df.language=="English"]
print(df_en.shape[0])


# In[ ]:


df_en.describe(include="all")


# ## 538 defined clusters/Categories:
# 
# * We'll drop the 8th, *"Unknown"* cluster (it's also the smallest).
# * we'll run a model (Externally?) to classify between the clusters.
#     * Presumably, 538 made the clusters based on simple word clusters/topics/LDA or similar, so this won't be very informative, but it's an easy thing to start with. 

# In[ ]:


df = df.loc[df.account_category != "Unknown" ]
df.account_category.value_counts(normalize=True)


# In[ ]:


df_en.account_category.value_counts()
print("original Unknown counts (for english only tweets)")
df_en = df_en.loc[df.account_category != "Unknown" ]
df_en.account_category.value_counts(normalize=True)


# 

# In[ ]:


## Model building & text features can go here:


# In[ ]:





# # Export the data

# In[ ]:


df.to_csv("russianTweet538Election.csv.gz",index=False,compression="gzip")


# In[ ]:


df_en.sample(frac=0.25).to_csv("russianTweet538Election_eng_sample.csv.gz",index=False,compression="gzip")

