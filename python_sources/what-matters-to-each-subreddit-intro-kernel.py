#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool
from nltk.corpus import wordnet
import pylab as pl
import seaborn as sns

tqdm.pandas()


# # Introductory Note
# 
# ### Objectives
# * Mostly having fun with the data at hand
# * Seeing if we can uncover different trends from the various subreddits.
# * Use as a basic NLP tutorial
# * Giving people visualization ideas
# 
# ### On performance
# The data is small enough that I am charging everything in RAM and often using the code that I find the clearer even when it isn't the most RAM effective (eg not processing data in batch etc...). If you are interested for more efficient ways to process the data let me know.

# In[ ]:


df = pd.read_csv('/kaggle/input/1-million-reddit-comments-from-40-subreddits/kaggle_RC_2019-05.csv')


# # NLP: Basic text cleaning
# 
# 1. Tokenization
# 2. Pos-Tagging
# 3. Lemmas
# 
# Even tho I usually use `spacy` these days, I choose to use `NLTK` here since, when doing the tokenization and pos-tag together in multiprocess, it runs quite a bit faster.

# In[ ]:


lemmatizer = WordNetLemmatizer()

def worker_tagging(text):
    return nltk.pos_tag(word_tokenize(text))

with Pool(9) as p:
    # NOTE: if you run this code yourself, tqdm is strugging a bit to update with the multiprocessing
    # Do not worry if the progress bar gets stuck, it updates by BIG increments
    tagged = p.map(worker_tagging, tqdm(list(df.body)))

df['tagged'] = tagged


# In[ ]:


def get_wordnet_pos(treebank_tag):
    # nltk taggers uses the treebank tag format, but WordNetLemmatizer needs a different \ simpler format
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def lemmatize(lemmatizer, token_tag):
    lemmas = []
    for i, j in token_tag:
        j = get_wordnet_pos(j)
        try:
            lemmas.append(lemmatizer.lemmatize(i, j))
        except KeyError:  # some tags are not known by WordNetLemmatizer
            lemmas.append(i)
    return lemmas

df['tokens'] = df.tagged.progress_apply(lambda r: lemmatize(lemmatizer, r))


# ### Example code with spacy: lemmas
# 
# This is not the most efficient way to do it, I am not really making use of the fact that `nlp.pipe` returns a generator, but it is mostly for the curious and to show spacy multiprocessing capabilities.  
# As a matter of fact I think the following approach barely bits in RAM (16 Gb), if it does at all.
# 
# ```
# import spacy
# nlp = spacy.load('en')
# docs = nlp.pipe(df.body, n_process=10)
# df['tokens'] = [[i.lemma_ for i in r] for r in tqdm(docs)]
# ```

# # Tokens representativeness
# Given a word/token how representative is it of a given class/category (here subreddit).  
# Here I use a custom which is:  
# In average, how many times more is this token present in the given subreddit compared to the overall corpus.  
# The formula is pretty simple:  
# 
# >     R = (word_count(class) * nb_words(GLOBAL)) / (word_count(GLOBAL) * nb_words(class))
# 
# This constrains the value to
# `[0, nb_words(GLOBAL) / nb_words(subreddit)]`  
# `0` -> token never found in the corpus subset corresponding to my class  
# `nb_words(GLOBAL) / nb_words(subreddit)` -> token ONLY found in the corpus subset corresponding to my class
# 
# NOTE: there is plenty of other ways to find such words (e.g using TFIDF), but I wanted to stay on a very simple and intuitive formula. 

# ### Creating token counts
# We build our corpus token count and the token counts per category

# In[ ]:


from collections import Counter, defaultdict

CATEGORY_HEADER = 'subreddit'
tokens_counter = Counter()
category_tokens_counter = defaultdict(Counter)
_ = df.progress_apply(lambda r: (tokens_counter.update(r['tokens']), category_tokens_counter[r[CATEGORY_HEADER]].update(r['tokens'])), axis=1)
nb_tokens = sum(tokens_counter.values())
nb_token_per_category = {category: sum(c.values()) for category, c in category_tokens_counter.items()}


# ### Representativeness
# 
# We compute the representativeness for each couple `[token, subreddit]`  
# > Only the `TOPN_TOKENS` most frequents tokens are kept (you can play with the value) since tokens with a low count are too susceptible to out of the ordinary events and the representativeness metric doesn't make any sense (plenty of tokens would the reach the minimum and maximum values or R if they are present only a few times through the corpus)

# In[ ]:


TOPN_TOKENS = 3000
print('keeping %s tokens out of %s' % (TOPN_TOKENS, nb_tokens))

def representativeness(token, tokens_counter, category_tokens_counter, nb_tokens, nb_token_per_categ):
    representativeness_scores = {
        categ: category_tokens_counter.get(categ).get(token, 0) / tokens_counter.get(token) * nb_tokens / nb_token_per_categ[categ] for categ in category_tokens_counter.keys()
    }
    representativeness_scores['token'] = token
    representativeness_scores['token_count'] = tokens_counter.get(token)
    return representativeness_scores

representativeness_df = pd.DataFrame([representativeness(x[0], tokens_counter, category_tokens_counter, nb_tokens, nb_token_per_category) for x in tokens_counter.most_common(TOPN_TOKENS)])
representativeness_df.sort_values(by='token_count', inplace=True, ascending=False)


# ### Tokens ban
# We ban certain tokens specific to reddit formatting that were not processed correctly during Tokenization (certain URLS, markdown etc...) and that happen to be quite frequent. They were ot removed earlier because even tho they mostly add noise for the coming graph they can still be interesting to study (these tokens are also pretty category-specific)

# In[ ]:


BAN_SET = {'/','*','^'}

def ban_token(token, ban_set):
    return bool(set(token).intersection(ban_set))

representativeness_df['ban'] = representativeness_df.token.apply(lambda r: ban_token(r, BAN_SET))
representativeness_df = representativeness_df[representativeness_df.ban == False]
representativeness_df = representativeness_df.set_index('token')


# ### Top N most representatives tokens / subreddit

# In[ ]:


TOPN_PER_SUB = 12
MAX_VISUAL_TOKEN_LEN = 12

fig, axes = pl.subplots(7, 6, figsize=(16, 18), dpi=80, facecolor='w', edgecolor='k')
axes = axes.flatten()
[pl.setp(ax.get_xticklabels(), rotation=90) for ax in axes]
pl.subplots_adjust(wspace=0.8)
pl.subplots_adjust(hspace=0.4)
for i, subreddit in enumerate(category_tokens_counter.keys()):
    sorted_scores = representativeness_df[subreddit].sort_values()
    topn = sorted_scores.tail(TOPN_PER_SUB)
    xlabels = [i if len(i) < MAX_VISUAL_TOKEN_LEN else i[:MAX_VISUAL_TOKEN_LEN-2] + '..' for i in topn.index ]
    sns.barplot(topn.values, xlabels, ax=axes[i])
    axes[i].set_title(subreddit)
pl.title("Most over used words per subreddit (top %s words)" % TOPN_TOKENS)


# ### Representativeness distribution / subreddit

# In[ ]:


fig, axes = pl.subplots(7, 6, figsize=(16, 18), dpi=80, facecolor='w', edgecolor='k')
axes = axes.flatten()
[pl.setp(ax.get_xticklabels(), rotation=90) for ax in axes]
fig.tight_layout()
for i, subreddit in enumerate(category_tokens_counter.keys()):
    sorted_scores = representativeness_df[subreddit].sort_values()
    representativeness_df[subreddit].hist(ax=axes[i], bins=100)
    axes[i].set_title(subreddit)
    axes[i].set_yscale('log')
    axes[i].set_xlim(0, 60)
pl.title("Representativeness distribution / subreddit")

