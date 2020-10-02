#!/usr/bin/env python
# coding: utf-8

# ## FB ADS: text mining and exploratory data analysis
# 
# In this simple notebook, I will go though the "Political Advertisements from Facebook" dataset and will try to get some relevant insights out of it.
# 
# Disclaimer: 
# - In this notebook I make use of a python package for text analytics I'm working on called [texthero](https://github.com/jbesomi/texthero/). Texthero is still in alpha version and it's a work-in-progress.
# - Work-in-progress; I will keep working on it in the next few days.

# In[ ]:


import pandas as pd
get_ipython().system('pip install texthero -q')
import texthero as hero


# We start by loading and displaying the dataframe. As there are more than 3GB of data, this might take a while.

# In[ ]:


ads_df = pd.read_csv("/kaggle/input/political-advertisements-from-facebook/fbpac-ads-en-US.csv")
ads_df.head()


# We are left with 162'324 advertisements.

# In[ ]:


ads_df.shape


# For faster computation, we sub-sample 10k rows.

# In[ ]:


SAMPLE_SIZE = 10000
ads_df_ = ads_df.sample(SAMPLE_SIZE)


# Let's look at the different columns:

# In[ ]:


print(ads_df_.columns.values)


# In[ ]:


ads_df_.describe()


# In this notebook, we are interested in **columns with text data**. Let's find them.

# In[ ]:


def get_text_columns(df):
    text_columns = []
    for col in df.select_dtypes('object'):
        if (df[col].str.split().str.len() > 5).any():
            text_columns.append(df[col].name)
    return text_columns

get_text_columns(ads_df_)


# As supposed, among others we found column `message` and `title`. We could have found the same results by ourself, just by looking at the table description, doing programmatically is more fun.
# 
# We notice also the `entities` columns, this are entites from the text that have been extracted with a spefcific software. We might want to use our own methods for extract entities and compare it with their version.

# ##### What are the most common words in Ads title?

# In[ ]:


TOP_WORDS = 10

hero.top_words(ads_df_.title)[:TOP_WORDS]


# hmm, let's try again with lowercased text:

# In[ ]:


hero.top_words(ads_df_.title.str.lower())[:TOP_WORDS]


# As expected, we can spot some political words such as 'committe', 'international' and 'action'. Just by looking at the top words, we can have a feeling of the dataset.
# 
# But, there are many stopwords such as 'for', 'of', 'the' that does not help much. Let's get rid of them and try again.
# 
# For that, we make use of the powerful and handy pandas `pipe` function.

# In[ ]:


(
    ads_df_.title.str.lower()
    .dropna()
    .pipe(hero.remove_stopwords)
    .pipe(hero.top_words)[:10]
)


# ##### What are the most common words in the 'message'?
# 
# Let's repeat the same action again, this time on the message column. We will skip the temporary steps and look just at the final result without _stopwords_.

# In[ ]:


(
    ads_df_.message.str.lower()
    .dropna()
    .pipe(hero.remove_stopwords)
    .pipe(hero.top_words)[:10]
)


# Hm, that's look strange. `top_words` returns to us all the words present in every columns of the Pandas series. If, for instance, the series is composed of a single row with content `hello world!`, `top_words` first split into words, i.e `hello` and `word` and not `word!`, and then it count.
# 
# As the `message` columns contains html tags, the top_words refer to the words inside a tag (i.e `<p>` becomes `p`). We need therefore to get rid of html tags.

# In[ ]:


def remove_html_tags(s: pd.Series) -> pd.Series():
    """Remove all html entities from a pandas series"""
    
    # TODO. Consider this more sophisticated solution: ('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    
    return s.str.replace('<.*?>', '')

s = pd.Series("<p>Hello world!</p>")
remove_html_tags(s)


# In[ ]:


ads_df_['message'] = remove_html_tags(ads_df_['message'])

(
    ads_df_.message.str.lower()
    .dropna()
    .pipe(hero.remove_stopwords)
    .pipe(hero.top_words)[:10]
)


# Here we go. It's indeed interesting to notice how the top words for the title and the message are different. The top words of the message are `us`, `help`, `people`, `need` that sounds like words used in call-for-action sentence: "**We need to get your vote today!**"

# #### Show me some pics!
# 
# You are right; this notebook is boring. Let's add some images and colors. Let's start with old-style wordcloud:

# In[ ]:


hero.wordcloud(ads_df_.title)


# In[ ]:


hero.wordcloud(ads_df_.message)


# ##### Who are the principal advertiser?

# In[ ]:


ads_df['advertiser'].value_counts()[:10]


# A short google search reveals us that Beto O'Rourke is an American politicians (what a surprise!): "Robert Francis "Beto" O'Rourke is a Decocratic American politician who represented Texas's 16th congressional district in the United States House of Representatives from 2013 to 2019".

# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Beto_O%27Rourke%2C_Official_portrait%2C_113th_Congress.jpg/440px-Beto_O%27Rourke%2C_Official_portrait%2C_113th_Congress.jpg)

# ... and ACLU is the "[American Civil Liberties Union](https://www.aclu.org/)".
# 
# Among other, there is also the actual President of U.S.A J. Donald Trump with 1443 advertisements.

# ##### Show me some interesting data. What does the Donald Trump's ads says?
# 
# All right.

# In[ ]:


trump_df = ads_df[ads_df['advertiser'] == 'Donald J. Trump'].copy()
trump_df.title.unique()


# All the Trump's advertisements just contains `Donald J. Trump`. hmm, what about the content?

# In[ ]:


trump_df['message'] = remove_html_tags(trump_df['message'])

(
    trump_df.message.str.lower()
    .pipe(hero.remove_stopwords)
    .pipe(hero.top_words)[:10]
)


# What about "let's make America great again?"
# 
# Rather than looking at single terms, let's look at the top n-grams:

# In[ ]:


trump_df['noun_chunks'] = hero.nlp.noun_chunks(trump_df.message)
trump_df['noun_chunks'].head(2)


# In[ ]:


help(hero.nlp.noun_chunks)


# The `noun_chunks` functions return extra information regarding each noun_chunk found in the sentence, including the part-of-speech tagging and the start and end index of the noun chunk in the sentence.
# 
# Here, we are simply interested in getting all noun_chunks and find the most relevant ones:

# In[ ]:


trump_df['noun_chunks'].apply(lambda row: [r[0] for r in row]).explode().value_counts()[:20]


# Continuation coming soon.
