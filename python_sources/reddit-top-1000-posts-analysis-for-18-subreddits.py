#!/usr/bin/env python
# coding: utf-8

# # Reddit Top 1000 Posts Analysis (for 18 Subreddits)
# 
# <p>&nbsp;</p>
# <img src="https://s3.eu-west-2.amazonaws.com/ammar-blog-post-images/2018/Nov/ytanalysisp.png" width=600>
# <p>&nbsp;</p>
# 
# [Reddit](https://reddit.com) is an American social news aggregation, web content rating, and discussion website. Registered members submit content to the site such as links, text posts, and images, which are then voted up or down by other members. Posts are organized by subject into user-created boards called "subreddits", which cover a variety of topics including news, science, movies, video games, music, books, fitness, food, and image-sharing. Submissions with more up-votes appear towards the top of their subreddit. As of February 2018, Reddit had 542 million monthly visitors (234 million unique users), ranking as the #3 most visited website in U.S. and #6 in the world, according to Alexa Internet. [\[Wikipedia\]](https://en.wikipedia.org/wiki/Reddit)
# 
# In this notebook, we will analyze the **top 1000 posts of 18** of the most popular subreddits on reddit. These subreddits are:
# * AskReddit
# * aww
# * books
# * explainlikeimfive
# * food
# * funny
# * GetMotivated
# * gifs
# * IAmA
# * Jokes
# * LifeProTips
# * movies
# * pics
# * Showerthoughts
# * todayilearned
# * videos
# * woahdude
# * worldnews
# 
# This analysis uses a [dataset](https://www.kaggle.com/ammar111/reddit-top-1000/home) which is a part of a [wider dataset](https://github.com/umbrae/reddit-top-2.5-million). Ths data of these datasets was pulled between August 15-20 of August **2013**.
# 
# ## Table of Contents
# * [Importing some packages](#p1)
# * [Reading the data](#p2)
# * [Getting a feel of the datasets](#p3)
# * [Preprocessing and data cleaning](#p4)
# * [Adding some features](#p5)
# * [Most common words in top-posts titles](#p6)
#     * [Most common words for all subreddits together](#p6-1)
#     * [Word cloud of the most common words](#p6-2)
# * [Most common words in body texts of some subreddits](#p7)
# * [Most common 2-grams in top-posts titles](#p8)
# * [Most common 2-grams for all subreddits together](#p8-1)
# * [2-grams word cloud](#p8-2)
# * [Most common 3-grams in top-posts titles](#p9)
# * [Most common 4-grams in top-posts titles](#p10)
# * [Data analysis and visulaization](#p11)
#     * [Distribution of title length](#p11-1)
#     * [Distribution of the number of comments](#p11-2)
#     * [Distribution of the number of upvotes and downvotes](#p11-3)
#     * [Distribution of score](#p11-4)
#     * [Correlation between variables](#p11-5)
# 
# 
# ## <a name="p1"></a>Importing some packages
# 

# In[ ]:


import numpy as np
import pandas as pd 
from textblob import TextBlob
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
import wordcloud
from collections import Counter
from pprint import pprint
import random
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# import re


# We make some configurations to enhance visualizations and appearance in general

# In[ ]:


pd.options.display.float_format = '{:.2f}'.format
sns.set(style="ticks")
plt.rc('figure', figsize=(8, 5), dpi=100)
plt.rc('axes', facecolor="#ffffff", linewidth=0.4, grid=True, labelpad=8, labelcolor='#616161')
plt.rc('patch', linewidth=0)
plt.rc('xtick.major', width=0.2)
plt.rc('ytick.major', width=0.2)
plt.rc('grid', color='#9E9E9E', linewidth=0.4)
plt.rc('text', color='#282828')
plt.rc('savefig', pad_inches=0.3, dpi=300)

# Hiding warnings for cleaner display
warnings.filterwarnings('ignore')

# Configuring some notebook options
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
# If you want interactive plots, uncomment the next line
# %matplotlib notebook


# ## <a name="p2"></a>Reading the data
# We read thr `csv` files containing the data of the subreddits we want to analyze. Each subreddit has its own dataset.

# In[ ]:


askReddit_df = pd.read_csv('../input/AskReddit.csv')
aww_df = pd.read_csv('../input/aww.csv')
books_df = pd.read_csv('../input/books.csv')
explainlikeimfive_df = pd.read_csv('../input/explainlikeimfive.csv')
food_df = pd.read_csv('../input/food.csv')
funny_df = pd.read_csv('../input/funny.csv')
getMotivated_df = pd.read_csv('../input/GetMotivated.csv')
gifs_df = pd.read_csv('../input/gifs.csv')
iAmA_df = pd.read_csv('../input/IAmA.csv')
jokes_df = pd.read_csv('../input/Jokes.csv')
lifeProTips_df = pd.read_csv('../input/LifeProTips.csv')
movies_df = pd.read_csv('../input/movies.csv')
pics_df = pd.read_csv('../input/pics.csv')
showerthoughts_df = pd.read_csv('../input/Showerthoughts.csv')
todayilearned_df = pd.read_csv('../input/todayilearned.csv')
videos_df = pd.read_csv('../input/videos.csv')
woahdude_df = pd.read_csv('../input/woahdude.csv')
worldnews_df = pd.read_csv('../input/worldnews.csv')

# We create these two lists for easier interaction with the datasets later
subreddits = [askReddit_df, aww_df, books_df, explainlikeimfive_df, food_df, funny_df,
              getMotivated_df, gifs_df, iAmA_df, jokes_df, lifeProTips_df, movies_df,
              pics_df, showerthoughts_df, todayilearned_df, videos_df, 
              woahdude_df, worldnews_df]

subreddit_names = ['AskReddit', 'aww', 'books', 'explainlikeimfive', 'food', 'funny',
                   'GetMotivated', 'gifs', 'IAmA', 'Jokes', 'LifeProTips', 'movies',
                   'pics', 'Showerthoughts', 'todayilearned', 'videos', 'woahdude', 'worldnews']


# ## <a name="p3"></a>Getting a feel of the datasets
# Let's see how the first rows of AskReddit dataset look like. All datasets have the same structure, the same columns.

# In[ ]:


askReddit_df.head()


# Then let's see some information about AskReddit dataset also

# In[ ]:


askReddit_df.info()


# We can see that there are 1,000 entries in the dataset. We can see also that there are missing values in some columns. For example, `link_flair_text` column has only 5 non-null values, which means that it has `1000 - 5 = ` 995 missing values. 
# 
# ## <a name="p4"></a>Preprocessing and data cleaning
# Before we deal with the missing-values issue, let's remove some columns that are not useful in our analysis

# In[ ]:


# We loop through all our subreddit datasets and remove some columns
# from each of them
for df in subreddits:
    df.drop(['link_flair_text', 'thumbnail', 'subreddit_id', 'link_flair_css_class', 
                       'author_flair_css_class', 'name', 'url', 'distinguished'],
                      axis=1, inplace=True)


# Now, let's see which columns in all of our datasets have missing values

# In[ ]:


for df, name in zip(subreddits, subreddit_names):
    # get the number of null values in each column of the dataset
    null_sum = df.isna().sum()
    # keep only the columns that have missing values
    null_sum = null_sum[null_sum > 0]
    print(name, 'dataset')
    for k,v in zip(null_sum.index, null_sum.values):
        print(k, ': ', v)
    print('-------------')


# We can see now that only `selftext` column has missing values in all of the datasets. This is probably because not all posts on reddit have body text (i.e. they just have titles and links). Now, let's replace each null value with an empty string to get rid of the missing-values problem

# In[ ]:


for df in subreddits:
    df['selftext'].fillna(value="", inplace=True)


# ## <a name="p5"></a>Adding some features
# Let's add more features that might be useful in analyzing our datasets. First let's add a column that represents the title length for each post

# In[ ]:


for df in subreddits:
    df['title_length'] = df['title'].apply(lambda x: len(x))


# Then, let's add a column that represents the number of fully capitalized words in the title of each post

# In[ ]:


def num_capitalized_word(s):
    c = 0
    for w in s.split():
        if w.isupper():
            c += 1
    return c

for df in subreddits:
    df['num_capitalized'] = df['title'].apply(num_capitalized_word)


# ## <a name="p6"></a>Most common words in top-posts titles
# Let's find out what are the most frequent words in top-posts titles. Are there some words that are common between the top posts of our subreddits?
# 
# There are two we better take care of in finding the most common words: first, we need to take care of contractions such as 'haven't' and 'you're'. To achieve consistency, we will convert them to their expanded form. This means that we will convert 'haven't' to 'have not',  'you're' to 'you are', etc. We will use a dictionary that maps the contraction to its expanded form. This dictionary is based on [this Wikipedia page](https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions) and I copied it from [this post on Stack Overflow](https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python).
# 
# You can see from the dictionary of contractions below that some contractions have more than one possible expansion (e.g. 'it'd' can be expanded as 'it would' or 'it had'). In this case, we will choose a random one of them.
# 
# Another thing we want to handle is [stop words](https://en.wikipedia.org/wiki/Stop_words) which are words very common words like 'the', 'have', 'we', 'and 'which'. We will remove these words before we extract the most common words from the datasets. We will use a list of stop words provided with `nltk` Python library for that.

# In[ ]:


contractions = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

# if a contraction has more than one possible expanded forms, we replace it 
# with a list of these possible forms
tmp = {}
for k,v in contractions.items():
    if "/" in v:
        tmp[k] = [x.strip() for x in v.split(sep="/")]
    else:
        tmp[k] = v
contractions = tmp


# Let's see the most common words for each subreddit. For every common word, the number of times it occured in top-posts titles is shown also.

# In[ ]:


tokenizer = RegexpTokenizer(r"[\w']+")
subreddit_words = []
for df, name in zip(subreddits, subreddit_names):
    all_titles = ' '.join([x.lower() for x in df['title']])
    for k,v in contractions.items():
        if isinstance(v, list):
            v = random.choice(v)
        all_titles = all_titles.replace(k.lower(), v.lower())
    words = list(tokenizer.tokenize(all_titles))
    words = [x for x in words if x not in stopwords.words('english')]
    subreddit_words.append(words)
    print('Most common words in ' + name, '*****************', sep='\n')
    pprint(Counter(words).most_common(35), compact=True)
    print()


# So, as we can see,
# * 'ever', 'know', 'best', etc. are common in AskReddit
# * 'dog', 'cat', 'little', 'baby', etc. are common in aww
# * 'made', 'cake', 'cheese', etc. are common in food
# * 'life', 'today', etc. are common in GetMotivated
# * 'us', 'korea', 'north', etc are common in worldnews
# * etc.
# 
# ### <a name="p6-1"></a>Most common words for all subreddits together
# Now let's see what are the most common words for the top posts of all the 16 subreddits we have

# In[ ]:


# flattening the list
subreddit_words_f = [x for y in subreddit_words for x in y]
print('Most common words in all subreddits', '*****************', sep='\n')
pprint(Counter(subreddit_words_f).most_common(35), compact=True)


# We can see that 'one', 'like', 'people', 'new', 'time', 'made', 'years', 'year', etc. are common in the titles of the top posts of our 16 subreddits together.
# 
# ### <a name="p6-2"></a>Word cloud of the most common words
# Now let's create a word cloud to visualize the most common words of our subreddits. The word size represents its frequency: the bigger the word, the more common it is. 

# In[ ]:


# a function to get custom colors for the word cloud
def col_func(word, font_size, position, orientation, font_path, random_state):
    colors = ['#b58900', '#cb4b16', '#dc322f', '#d33682', '#6c71c4', 
              '#268bd2', '#2aa198', '#859900']
    return random.choice(colors)

fd = {
    'fontsize': '32',
    'fontweight' : 'normal',
    'verticalalignment': 'baseline',
    'horizontalalignment': 'center',
}

for df, name, words in zip(subreddits, subreddit_names, subreddit_words):
    wc = wordcloud.WordCloud(width=1000, height=500, collocations=False, 
                             background_color="#fdf6e3", color_func=col_func, 
                             max_words=200,random_state=np.random.randint(1,8)
                            ).generate_from_frequencies(dict(Counter(words)))
    fig, ax = plt.subplots(figsize=(20,10))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(name, pad=24, fontdict=fd)


# ## <a name="p7"></a>Most common words in body texts of some subreddits
# Now let's choose some subreddits that have relatively few missing values in their `selftext` column, and see the most common words in these texts followed by a word cloud as above

# In[ ]:


selftext_subreddits = [askReddit_df, explainlikeimfive_df, iAmA_df, jokes_df]
selftext_subreddit_names = ['AskReddit', 'explainlikeimfive', 'IAmA', 'Jokes']
selftext_subreddit_words = []

for df, name in zip(selftext_subreddits, selftext_subreddit_names):
    selftexts = ' '.join([x.lower() for x in df['selftext']])
    for k,v in contractions.items():
        if isinstance(v, list):
            v = random.choice(v)
        selftexts = selftexts.replace(k.lower(), v.lower())
    words = list(tokenizer.tokenize(selftexts))
    words = [x for x in words if x not in stopwords.words('english')]
    selftext_subreddit_words.append(words)
    print(name, '*****************', sep='\n')
    pprint(Counter(words).most_common(25), compact=True)
    print()


# In[ ]:


for df, name, words in zip(selftext_subreddits, selftext_subreddit_names, selftext_subreddit_words):
    wc = wordcloud.WordCloud(width=1000, height=500, collocations=False, 
                             background_color="#002b36", color_func=col_func, 
                             max_words=200, random_state=np.random.randint(1,8)
                            ).generate_from_frequencies(dict(Counter(words)))
    plt.figure(figsize=(20,10))
    plt.imshow(wc, interpolation='bilinear')
    _ = plt.axis("off")
    _ = plt.title(name, fontdict=fd, pad=24)


# ## <a name="p8"></a>Most common 2-grams in top-posts titles
# What is 2-gram? 2-gram is a contiguous sequence of 2 words from a given text. For example, if our text is 'lorem ipsum dolor sit amet', then the 2-grams are: 'lorem ipsum', 'ipsum dolor', 'dolor sit', and 'sit amet'. So let's find the most common 2-grams in the titles of our subreddit top posts to see if they have common combinations of words. Note that here we keep contractions and stop words untouched.

# In[ ]:


from nltk.util import ngrams
subreddit_2ngrams = []
for df, name in zip(subreddits, subreddit_names):
    ng = [ngrams(tokenizer.tokenize(tw.lower()), 
                 n=2) for tw in df['title']]
    # flattening the list
    ng = [x for y in ng for x in y]
    subreddit_2ngrams.append(ng)
    print(name, '*****************', sep='\n')
    pprint(Counter(ng).most_common(25), compact=False)
    print()


# ### <a name="p8-1"></a>Most common 2-grams for all subreddits together
# Now let's get the most common 2-grams for all subreddits together, but in this case, we will remove 2-grams whose one of their words is a stop word

# In[ ]:


# flattening the list
subreddit_2ngrams_f = [x for y in subreddit_2ngrams for x in y]
# removing 2-grams that contain stop words
tmp = []
for n in subreddit_2ngrams_f:
    f = 0
    for w in n:
        if w in stopwords.words('english'):
            f = 1
    if f == 0:
        tmp.append(n)
subreddit_2ngrams_f = tmp
pprint(Counter(subreddit_2ngrams_f).most_common(50), compact=False)


# ### <a name="p8-2"></a>2-grams word cloud
# Now let's see the word cloud of the most common n-grams for each subreddit

# In[ ]:


for df, name, ngrams_2 in zip(subreddits, subreddit_names, subreddit_2ngrams):
    wc = wordcloud.WordCloud(width=2000, height=1000, 
                             collocations=False, background_color="black", 
                             colormap="Set3", max_words=66,
                             normalize_plurals=False,
                             regexp=r".+", 
                             random_state=7).generate_from_frequencies(dict(Counter([x + ' ' + y for x,y in ngrams_2])))
    plt.figure(figsize=(20,15))
    plt.imshow(wc, interpolation='bilinear')
    _ = plt.axis("off")
    _ = plt.title(name, fontdict=fd, pad=24)


# ## <a name="p9"></a>Most common 3-grams in top-posts titles
# Similar to the previous part, we will find now the most common 3-grams in each of our 16 subreddits

# In[ ]:


subreddit_3ngrams = []
for df, name in zip(subreddits, subreddit_names):
    ng = [ngrams(tokenizer.tokenize(tw.lower()), 
                 n=3) for tw in df['title']]
    # flattening the list
    ng = [x for y in ng for x in y]
    subreddit_3ngrams.append(ng)
    print(name, '*****************', sep='\n')
    pprint(Counter(ng).most_common(25), compact=False)
    print()


# ## <a name="p10"></a>Most common 4-grams in top-posts titles
# Now will find now the most common 4-grams in each of our 16 subreddits

# In[ ]:


subreddit_4ngrams = []
for df, name in zip(subreddits, subreddit_names):
    ng = [ngrams(tokenizer.tokenize(tw.lower()), 
                 n=4) for tw in df['title']]
    # flattening the list
    ng = [x for y in ng for x in y]
    subreddit_4ngrams.append(ng)
    print(name, '*****************', sep='\n')
    pprint(Counter(ng).most_common(25), compact=False)
    print()


# ## <a name="p11"></a>Data analysis and visulaization
# Let's explore our data by finding the distribution of some variables and looking at the relationships between them.
# 
# ### <a name="p11-1"></a>Distribution of title length
# First, let's examine the distibution of title length for each subreddit by using histogram plots. This allows us to see for example how many posts have title length between 30 and 40 characters, how many posts have title length between 40 and 50 characters, etc.

# In[ ]:


fig, axes = plt.subplots(6, 3, figsize=(20,30), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.5, wspace=0.4)
# fig.text(0.5, 0.04, 'Title Length', ha='center', fontdict={'size':'18'})
for ax, df, name in zip(axes.flat, subreddits, subreddit_names):
    sns.distplot(df["title_length"], kde=False, hist_kws={'alpha': 1}, color="#0747A6", ax=ax)
    ax.set_title(name, fontdict={'size': 16}, pad=14)
    ax.set(xlabel="Title length", ylabel="Number of posts")


# Note how the histogram of 'todayilearned' subreddit shows us that the top-posts titles of this subreddit are generally longer than other subreddits. We can see also that 'GetMotivated', 'gifs',  'Jokes', and 'woahdude' subreddits have generally shorter titles than other subreddits.
# 
# ### <a name="p11-2"></a>Distribution of the number of comments
# Similar to what we did for the distribution of title length, we now explore the distribution of the number of comments for each subreddit

# In[ ]:


fig, axes = plt.subplots(6, 3, figsize=(20,30))
fig.subplots_adjust(hspace=0.5, wspace=0.4)
# fig.text(0.5, 0.04, 'Number of Comments', ha='center', fontdict={'size':'18'})
# fig.text(0.04, 0.5, 'Number of Posts', va='center', rotation='vertical', fontdict={'size':'18'})
for ax, df, name in zip(axes.flat, subreddits, subreddit_names):
    sns.distplot(df["num_comments"], kde=False, hist_kws={'alpha': 1}, color="#0747A6", ax=ax)
    ax.set_title(name, fontdict={'size': 16}, pad=14)
    ax.set(xlabel="Number of comments", ylabel="Number of posts")


# Note that this time each plot has its own axes (i.e. the axes are not shared between all plots).
# 
# We can see that posts of 'AskReddit' subreddit have more comments than other subreddits. We can see also that posts of 'funny', 'IAmA', 'movies', 'pics', 'todayilearned', 'videos' and 'worldnews' have a large  number of comments relatively. Moreover, we notice that 'food', 'GetMotivated', 'Jokes', 'Showerthoughts' and 'woahdude' have the less comments than other subreddits.
# 
# For a clearer comparison, let's compare the medians of the number of comments for our subreddits

# In[ ]:


medians = []
for df in subreddits:
    medians.append(df['num_comments'].median())

plt.rc('axes', labelpad=16)
fig, ax = plt.subplots(figsize=(14,8))
d = pd.DataFrame({'subreddit': subreddit_names, 'num_comments_median': medians})
sns.barplot(x="subreddit", y="num_comments_median", data=d, palette=sns.cubehelix_palette(n_colors=24, reverse=True), ax=ax);
ax.set(xlabel="Subreddit", ylabel="Median");
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
plt.rc('axes', labelpad=8)


# ### <a name="p11-3"></a>Distribution of the number of upvotes and downvotes
# Now, we will explore the distribution of the number of upvotes and the distribution of the number of downvotes for each subreddit. We will plot both distributions together where the number of upvotes has the blue color and the number of downvotes has the orange color

# In[ ]:


fig, axes = plt.subplots(6, 3, figsize=(20,30))
fig.subplots_adjust(hspace=0.5, wspace=0.4)
for ax, df, name in zip(axes.flat, subreddits, subreddit_names):
    sns.distplot(df["ups"], kde=False, hist_kws={'alpha': 0.5}, color="#0747A6", ax=ax)
    sns.distplot(df["downs"], kde=False, hist_kws={'alpha': 0.5}, color="#FF5630", ax=ax)
    ax.set_title(name, fontdict={'size': 16}, pad=14)
    ax.set(xlabel="Number of upvotes/downvotes", ylabel="Number of posts")


# We can see that for some subreddits like 'books', 'food', 'explainlikeimfive', 'GetMotivated', 'Jokes', and 'Showerthoughts', posts have noticeably more upvote than downvotes, as you can see the blue shape is more to the right than the orange shape in this case. We can notice also that for subreddits like 'funny' and 'pics', posts have roughly similar number of upvotes and downvotes.
# 
# ### <a name="p11-4"></a>Distribution of score
# Now, let's examine the score variable which represents `number of upvotes - number of downvotes`. We will use violin plots for doing so. For a violin plot, the width of a violin represents the frequency. This means that if a violin is the widest between 300 and 400, then the area between 300 and 400 contains more data than other areas. Moreover, inside the violin, you can see statistical measures such as the median, as illustrated by the image below
# 
# ![](https://s3.eu-west-2.amazonaws.com/ammar-blog-post-images/2018/Nov/violin_plot.svg)

# In[ ]:


comments = pd.DataFrame(columns=['subreddit', 'score'])
n = []
l = []
for df, name in zip(subreddits, subreddit_names):
    cl = list(df['score'])
    l.extend(cl)
    n.extend([name] * len(cl))
comments['subreddit'] = pd.Series(n)
comments['score'] = pd.Series(l)


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 20))
sns.violinplot(x='score', y='subreddit', data=comments, scale='width', inner='box', ax=ax);


# ### <a name="p11-5"></a>Correlation between variables
# Now, we will see how some of our dataset variables are correlated with each other: for example, we would like to see how the number of comments and the score are correlated, meaning do they increase and decrease together (positive correlation)? Does one of them increase when the other decreases and vice versa (negative correlation)? Or are they not correlated?
# 
# Correlation is represented as a value between -1 and +1 where +1 denotes the highest positive correlation, -1 denotes the highest negative correlation, and 0 denotes that there is no correlation. We will use heatmaps to visualize correlation using colors

# In[ ]:


fig, axes = plt.subplots(8, 2, figsize=(20,60))
fig.subplots_adjust(hspace=0.7, wspace=0.3)
for ax, df, name in zip(axes.flat, subreddits, subreddit_names):
    sns.heatmap(df[['score', 'ups', 'downs', 'num_comments', 'title_length', 'num_capitalized']].corr(), annot=True, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)
    ax.set_title(name, fontdict={'size':'18'}, pad=14)
    ax.set(xlabel="", ylabel="")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


# For example, we can see that upvotes, downvotes, and score are highly correlated for 'GetMotivated' subreddit. Also, we can see that the number of comments is highly correlated with both upvotes and downvotes for 'IAmA' subreddit. Other things also can be concluded from these heatmaps.
# 
# ## End
# If you like this analysis, please consider to upvote it at the top of this page.
# Follow me on [Twitter](https://twitter.com/ammar_cel) to know when I publish something new, or visit [my website and blog](http://ammar-alyousfi.com?ref=redditAnalysis).
