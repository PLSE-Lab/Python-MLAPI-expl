#!/usr/bin/env python
# coding: utf-8

# # Overview of the reviews with color coded ngram wordcloud
# 
# Wordclouds are a nice way to visualize the most commonly used words and phrases in a document. However, when we get to product reviews, we also have a sentiment associated with this reviews. The goal of this kernel is to produce a word cloud representation for a product review, that is color coded based on the sentiment of the product, red for phrases appearing in negative reviews and green for positive. Let's get started: 

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')
import pandas as pd 
import sqlite3


# 1) We'll start with extracting the data, we need the reviews and their scores to make the wordclouds!

# In[2]:


con = sqlite3.connect('../input/database.sqlite')

# Extract the user, product, score and time of the review
rev_score = pd.read_sql_query("""
SELECT ProductId, Score, Summary, Text  
FROM Reviews
""", con)
rev_df = rev_score


# Just checking some basic information:  
# 2) The number of product that have at least one review  

# In[3]:


item_uniqs = rev_df['ProductId'].unique()
print('Total of %s products have been reviews'%str(len(item_uniqs)))


# 3) Look at the distribution of the number of reviews each product has  

# In[4]:



rev_cnts = dict(rev_df['ProductId'].value_counts())
hist(list(rev_cnts.values()), color='g', edgecolor = 'None')
xlabel('# reviews')
ylabel('# products')
show()


# Most products have a few reviews, making it difficult to gain knowledge from reviews text.

# 4) Get items with 10 or more reviews, to get a large enough sample to make the word_cloud

# In[5]:


high_reved = {k:v for k,v in rev_cnts.items() if v>10}
hist(list(high_reved.values()), bins=10, color='r', edgecolor = 'None')
xlabel('# reviews')
ylabel('# products')
show()


# 5) For each prodct, given the productId we can now extract the list of all their reviews. Let's get a list of the top 10 highly reviewed products and look at some of those

# In[6]:


top10_reved = list(high_reved.keys())[:10]
prod_1 = top10_reved[1]
#print(top10_reved)


# 6) For the given product get all their reviews and scores, here I picked the one with highest number of reviews:  

# In[7]:


prod_df = rev_df[rev_df['ProductId']==prod_1]


# 7) Explore the distribution of scores it recieves:

# In[8]:


from operator import itemgetter


# In[9]:


scores_1 = dict(prod_df['Score'].value_counts())
scores_set = sorted(scores_1.items(), key = itemgetter(0))

X, y = zip(*scores_set)
bar(X, y, align='center', width=0.5, color='r')
xticks(X, X)
ymax = max(scores_1.values()) 
ylim(0, ymax*1.1)

xlabel('Score')
ylabel('Counts')
show()


# Histogram clearly shows more 5 star reviews than other stars.

# 8) We need to convert the set of reviews for each product to a sequence of words,
#    removing stop words, numbers and lemmatizing.

# In[2]:


import nltk
import re
#from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Save stopwords into set for faster lookup
stops = set(stopwords.words('english'))

def review_to_wordlist( review, remove_stopwords=True ):
    # Function converts text to a sequence of words,
    # Returns a list of words.
 
    lemmatizer = WordNetLemmatizer()
    # 1. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    
    # 2. Convert words to lower case and split them
    words = review_text.lower().split()
    # 3. Remove stop words
    words = [w for w in words if not w in stops]
    # 4. Remove short words
    words = [t for t in words if len(t) > 2]
    # 5. lemmatizing
    words = [nltk.stem.WordNetLemmatizer().lemmatize(t) for t in words]

    return(words)


# ### Just to see an example of how review_to_wordlist works, looking at first review

# In[11]:


rev1 = prod_df['Text'].tolist()[0]
print(rev1)


# In[12]:


print(review_to_wordlist(rev1))


# As you can see the extra html tag characters have been removed, along with the numbers and the exclamation matsk. 

# 9) Convert each review to a list of words as a new column in the dataframe

# In[13]:


prod_df.is_copy = False
prod_df['tokenz'] = prod_df['Text'].apply(review_to_wordlist)


# 10) We split the reviews into positive and negative reviews, based on their score.  
# Using the common practice:
#     
#     1) Extremely Dissatisfied  
#     2) Dissatissfied  
#     3) Nutral  
#     4) Satissfied  
#     5) Extremly satissfied  
#     
#     
# So the score 3, doesn't really have much to offer. 

# In[14]:


pos_text = array([l for ls in prod_df[prod_df['Score']>3]['tokenz'].tolist() for l in ls])
neg_text = array([l for ls in prod_df[prod_df['Score']<3]['tokenz'].tolist() for l in ls])


# 11) More often than not, single words (unigrams) are not discriptive enough, and in contrast the longer group of words (ngrams) can better express the meaning of the document than unigrams.  
# 
# For that reason we look at collection of words with lenghts 1(unigrams), 2(bigrams) and 3(trigrams) all provided through nltk different modules.  
# 

# In[15]:


def get_ngram_freq(n_grams):
    """
    Function returns the frequncy of occurance of each ngram, given the list of ngrams
    """
    fdist = nltk.FreqDist(n_grams)
    for k,v in fdist.items():
        fdist[k] = v
    return fdist

def get_ngram_dist(text, n_ngram=1):
    '''
    Function gets the text and the desired lenght of ngrams 
    Returns, frequency of each ngram
    '''
    if n_ngram not in [1,2,3]:
        raise ValueError("Invalid ngram value. Expected one of: %s" % n_ngram)
    if n_ngram ==1:
        gs = nltk.word_tokenize(text)
        
    elif n_ngram == 2:
        gs = nltk.bigrams(text)
        
    elif n_ngram ==3:
        gs = nltk.trigrams(text)
    
    fdist = get_ngram_freq(gs)
        
    return fdist


# In[16]:


neg_fdist = get_ngram_dist(neg_text, 2)
pos_fdist = get_ngram_dist(pos_text, 2)


# Just choosing a subset of most frequent ngrams

# In[17]:


from operator import itemgetter
from collections import OrderedDict

def get_sub_most_frequnt(fdist, top_n):
    '''
    Returns the top_n frequented terms
    '''
    sorted_dist =sorted(fdist.items(), key=itemgetter(1), reverse=True)
    sub_sort = dict(sorted_dist[:top_n])

    sub_sort2 = {' '.join(k):v for k,v in sub_sort.items()}
    return sub_sort2


# In[18]:


neg_sub_sort = get_sub_most_frequnt(neg_fdist, 40)
pos_sub_sort = get_sub_most_frequnt(pos_fdist, 40)


# 12) Now as you can imagine, there are ngrams that might appear both in negative and positive reviews. Now are they correspond to a positvie experience about the product or the other way around? How should we assign a sentiment to these common set of ngrams?  
# 
# For this I took a simple approach, and for each common ngram, found the frequency of its occurance in positive and negative reviews and assinged the sentiment with higher frequency. 

# In[19]:


def get_sentiment_for_common_ngrams(neg_sorted, pos_sorted, com_ngrams):
    '''
    For ngrams appearing both in negative and positive reviews, 
    assign the sentiment with more frequency.
    '''
    
    com_2_n = []
    com_2_p = []

    
    for w in com_ngrams:
        cnt_n = neg_sorted[w]
        cnt_p = pos_sorted[w]
        if cnt_p > cnt_n:
            com_2_p += [w] 
        else:
            com_2_n += [w]
    
    return com_2_p, com_2_n


# 13) Finally get the positive and negative set of reviews, removing the common ngrams from the least frequent and adding it to the one with most repetition.

# In[20]:


def get_pos_neg_ngrams(neg_sorted, pos_sorted, com_ngrams):
    # Function gets, most common neg and pos ngrams 
    # 1) Get sets of pos, neg ngrams with common ngram appearing in only one of them
    
    com_2_p, com_2_n = get_sentiment_for_common_ngrams(neg_sorted, pos_sorted, com_ngrams)
    
    # 2) Returning unique ngrams
    neg_uniqes = list(set(neg_sorted.keys())- com_ngrams)+ com_2_n
    pos_uniqes = list(set(pos_sorted.keys())- com_ngrams)+ com_2_p
    
    # 3) Remove common ngrams from the one with least freq and add it to the one with most freq
    neg_uniq_dict = {k:neg_sorted[k] for k in neg_uniqes}
    pos_uniq_dict = {k:pos_sorted[k] for k in pos_uniqes}

    return pos_uniq_dict, neg_uniq_dict

def get_all_ngrams(uniq_pos_ngrams, uniq_neg_ngrams):
    # Combine dictionary of pos and neg ngrams to get the freq of all ngrams
    all_ngrams = uniq_pos_ngrams.copy()
    all_ngrams.update(uniq_neg_ngrams)
    return all_ngrams

def get_uniq_pos_neg_all_ngrams(neg_sub_sort, pos_sub_sort):
    # 1. Get the common ngrams appearing in both pos and neg reviews
   
    com_ngrams = set(neg_sub_sort.keys()) & set(pos_sub_sort.keys())
    # 2. Remove common ngrams from the one with least freq and add it to the one with most freq
    uniq_pos_ngrams, uniq_neg_ngrams = get_pos_neg_ngrams(neg_sub_sort, pos_sub_sort, com_ngrams)
    all_ngrams = get_all_ngrams(uniq_pos_ngrams, uniq_neg_ngrams)
    
    return uniq_pos_ngrams, uniq_neg_ngrams, all_ngrams


# In[21]:


uniq_pos_ngrs, uniq_neg_ngrs, all_ngrs = get_uniq_pos_neg_all_ngrams(neg_sub_sort, pos_sub_sort)


# 14) To have word sizes proportional to the number of their occurance, we should normalize the number their appearances by the max number of time one ngram has been repeated. 

# In[22]:


def get_normalized_frequecies(init_freq):
    # Normalize the occurance of ngram with the most frequent one
    max_cnt = max(init_freq.values())
    norm_freqs = {k:float(init_freq[k])/max_cnt for k in init_freq.keys() }
    return norm_freqs


# In[23]:


norm_freqs_all = get_normalized_frequecies(all_ngrs)
norm_freqs_neg = get_normalized_frequecies(uniq_neg_ngrs)
norm_freqs_pos = get_normalized_frequecies(uniq_pos_ngrs)


# 15) We have now the dict of the most frequently used positive and negative ngrams and all we need is plug and see! 
# 
# We only need to import a few library and coloring modules...  
# Install the package using: pip install wordcloud  

# In[24]:


#from PIL import Image
from wordcloud import (WordCloud, get_single_color_func, STOPWORDS)
class GroupedColorFunc(object):
    """
    Uses different colors for different groups of words. 
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)
        return self.get_color_func(word)(word, **kwargs)
    
    # Define functions to select a hue of colors arounf: grey, red and green
def red_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 100%%, %d%%)" % random.randint(30, 50)

def green_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(100, 100%%, %d%%)" % random.randint(20, 40)


# In[25]:


def plot_pos_neg_wordclouds(neg_ngrams_sort, pos_ngrams_sort):
    fig = plt.figure(figsize=(16,12))
    plt.subplot(121)

    wc1 = WordCloud(width=800, height=400, background_color="white", max_words=20, min_font_size=8)                    .generate_from_frequencies(neg_ngrams_sort)

    plt.imshow(wc1.recolor(color_func=red_color_func, random_state=3),
               interpolation="bilinear")
    axis("off")

    wc2 = WordCloud(width=800, height=400, background_color="white", max_words=20, min_font_size=8)                .generate_from_frequencies(pos_ngrams_sort)

    plt.subplot(122)

    plt.imshow(wc2.recolor(color_func=green_color_func, random_state=3),
               interpolation="bilinear")
    axis("off")
    show()


# In[26]:


plot_pos_neg_wordclouds(norm_freqs_neg, norm_freqs_pos)


# So this product is clearly about dog food and at least of the negative reviews reports a dog dying! that's quiet horrifying!  But let's see how the negative and positive phrase look like combined.

# In[27]:


def plot_allwords_wordclouds(norm_freqs_all):


    wc = WordCloud(width=1200, height=800, background_color="white", max_words=200, min_font_size=10)                    .generate_from_frequencies(norm_freqs_all)

    color_to_words = {
        # words below will be colored with a green single color function
        '#00ff00': uniq_pos_ngrs.keys(),
        # will be colored with a red single color function
        'red': uniq_neg_ngrs.keys()
    }

    # Words that are not in any of the color_to_words values
    # will be colored with a grey single color function
    default_color = 'grey'

   
    # Create a color function with multiple tones
    grouped_color_func = GroupedColorFunc(color_to_words, default_color)

    # Apply our color function
    wc.recolor(color_func=grouped_color_func)

    plt.figure(figsize=(16,12))

    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# In[28]:


plot_allwords_wordclouds(norm_freqs_all)


# ## How to interpret this figure? 
# It seems like overall this product has a more positive review ( which also was obvious from the histogram of the reviews) but additionally, we learn reviewers who did not like the product, have used phrases such as, digestive issues, get stucked and stopped giving! while the ones who loved the product, use phrases like,  love greenies, absolutely love and clean teeth ( and a few other phrases with the teeth suggesting it has something to do with the teeth?), to describe this product. 

# ## Conclusion:  
# 
# In this kernell I tried to offer a simple wordcloud representation of amazon reviews for a given prodcut.  I aimed to produce a wordcloud with ngrams, where words are colord based on their associated sentiment. A simple look at the wordcloud can tell you whether the product has mainly a positive or negative review, based on the majority of colors as well as, the most frequently used phrases in positive and negative reviews.
# 
# I hope you've enjoyed this product. Let me know if you had any question or suggestion! 

# In[29]:




