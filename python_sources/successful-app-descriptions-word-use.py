#!/usr/bin/env python
# coding: utf-8

# # Finding the words used by successful apps  
# 
# Consumers base their decide to download and based on a variety of information, from word of mouth to user reviewers to price. Developers want to maximize consumer engangment while at the same time minimizing the cost of producing an app. Here, I'm going to look at what words in app descriptions are associated with over and under-performing apps. I'm going to use number of people that rate an app is a proxy of app success and the size of the app in mb as a proxy for productions costs.

# In[ ]:


get_ipython().system('pip install inflect')
get_ipython().system('pip install regressors')

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Scientific libraries
from numpy import arange,array,ones
from scipy import stats
import numpy as np
import re
import operator
import statsmodels.api as sm
from sklearn import linear_model
import inflect
infl = inflect.engine()
from regressors import stats as reg_stats


# In[ ]:


appdata = pd.read_csv('/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv')


# First check the relationship between size and the number of user ratings. A log log transformation yields a nice relationship.

# In[ ]:


appdata_nonas = appdata[['Price', 'Size', 'User Rating Count', 'Description']].dropna()
appdata_nonas['log_rating_count'] = np.log(appdata_nonas['User Rating Count'])
appdata_nonas['log_size'] = np.log(appdata_nonas['Size'])

x = appdata_nonas['log_size']
y = appdata_nonas['log_rating_count']
plt.scatter(x, y, alpha = 0.5)
plt.title('The number of user rating depends on app size')
plt.xlabel('log Size(mb)')
plt.ylabel('log Number of User Ratings')
plt.show()


# It looks like the size of the app sets a maximum number of user ratings via a linear function, but that many apps fall below this line. Next, I'll use linear regression to create a metric for whether an app in over-performing or under-performing based on its size.

# In[ ]:


slope, intercept, r_value, p_value, std_err = stats.linregress(appdata_nonas['log_size'], appdata_nonas['log_rating_count'])
appdata_nonas['dev'] = ((appdata_nonas['log_rating_count'] - (intercept+slope*appdata_nonas['log_size']))/(intercept+slope*appdata_nonas['log_size']))
x = appdata_nonas['log_size']
y = appdata_nonas['dev']
plt.scatter(x, y, alpha=0.5)
plt.title('The number of user rating depends on app size')
plt.xlabel('log Size(mb)')
plt.ylabel('App performance relative expected')
plt.show()
slope, intercept, r_value, p_value, std_err = stats.linregress(appdata_nonas['log_size'], appdata_nonas['dev'])
print("Pvalue of app size on app performance: ", p_value)


# Here, points above one are apps that overperform for their size, and points below one are underperforming apps. It looks like my app performance measure doesn't depend on size, so I can move forward without size as a confounding factor. Next I'll check if Price might affect app performance.

# In[ ]:


appdata_cheap = appdata_nonas[appdata_nonas['Price'] < 3]
appdata_cheap.plot.scatter('Price', 'dev')
#slope, intercept, r_value, p_value, std_err = stats.linregress(appdata_nonas['Price'], appdata_nonas['dev'])
regr = linear_model.LinearRegression()
X = appdata_nonas[['Price']]
Y = appdata_nonas['dev']
regr.fit(X, Y)
coefficients = regr.coef_
pvals = reg_stats.coef_pval(regr, X, Y)
xlabels = ['Price']
reg_stats.summary(regr, X, Y, xlabels)


# Fortunately app performance and price aren't closely related, which means I can leave it out of my word analysis. Next I'm going to find which words are associated with overperforming apps, and which are associated with underperforming apps.  I'll do this in X steps:
# 
# 1) Process the app descriptions to make them uniform, this involves removing non-letter characters, homogenize case, and making all nouns singular.
# 
# 2) Count word occurance in app descriptions and find the 500 most frequently used words (as measure by the number of apps with the word)
# 
# 3) For each of the 500 words, code each app as either having, or not having, the word in its description.
# 
# 4) Regress word occurance with app performance, record the words that have a significantly positive, or negative, relationship with app performance.
#     
# 5) To ensure that these results are not unique to this set of apps, I perform the regression 1000 times with a random sample of half of the apps. For the final list of important descriptors I only keep words that were significant in over half of the samples.

# In[ ]:


def process_words_used_for_regression(df, column):
    word_list = []
    match_description_list = []
    for desc in df[column]:
        # remove common escaped characters
        desc = desc.replace('\\n', ' ')
        desc = desc.replace('\\u', ' ')
        desc = desc.replace('\\t', ' ')
        words = re.findall(r"[\w']+", desc)
        match_desc = ""
        for wd in words:
            if wd.isalpha() and len(wd)>2:
                word_upper = wd.upper()
                #make all nouns singular
                if infl.singular_noun(word_upper) != False:
                    word_upper = infl.singular_noun(word_upper)
                #inflection seems to handle some words incorrectly
                if word_upper == 'VARIOU':
                    word_upper = 'VARIOUS'
                if word_upper == 'THI':
                    word_upper = 'THIS'
                if word_upper == 'PROGRES':
                    word_upper = 'PROGRESS'
                if match_desc.find(" " + word_upper + " ") <0:
                    word_list.append(word_upper)
                match_desc += " " + word_upper
        match_desc += " "
        match_description_list.append(match_desc)
    df['match_string'] = match_description_list
    df['description_length'] = df['match_string'].str.len()
    word_list.sort(key=len, reverse = True)
    description_word_counts = dict()
    for word in word_list:
        if word in description_word_counts:
            description_word_counts[word] += 1
        else:
            description_word_counts[word] = 1
    sorted_words = sorted(description_word_counts.items(), key=operator.itemgetter(1), reverse = True)
    popular_words_list = [None]*500
    for i, word in enumerate(sorted_words):
        if i >= 500:
            break
        popular_words_list[i] = word[0]
        #print(word[0])
    for word in popular_words_list:
        match_word = " " + word + " "
        df[word] = np.where(df['match_string'].str.contains(match_word),1,0 )
    return(popular_words_list, description_word_counts, df)


# In[ ]:


def get_best_and_worst_words(words_list, df):
   X = df[words_list]
   Y = list(df['dev'])
   regr = linear_model.LinearRegression()
   regr.fit(X, Y)
   coefficients = regr.coef_
   pvals = reg_stats.coef_pval(regr, X, Y)

   word_coef_dict = dict()
   for count, word in enumerate(words_list): 
       word_coef_dict[word] = regr.coef_[count]
   word_pvalue_dict = dict()
   for count, word in enumerate(words_list): 
       word_pvalue_dict[word] = pvals[count+1]
       
   sorted_word_coef = sorted(word_coef_dict.items(), key=operator.itemgetter(1), reverse = True)
   best_words = set()
   worst_words = set()
   
   for entry in sorted_word_coef:
       word = entry[0]
       if word_coef_dict[word] > 0 and word_pvalue_dict[word] < 0.05:
           best_words.add(word)
           #print(word_coef_dict[word], entry[1], word_pvalue_dict[word])
       if word_coef_dict[word] < 0 and word_pvalue_dict[word] < 0.05: 
           #print(word_coef_dict[word], entry[1], word_pvalue_dict[word])
           worst_words.add(word)
   return(best_words, worst_words)
  


# In[ ]:


def bootstrap_good_bad_word_search(words_list, df, word_counts):
    good_terms = dict()
    bad_terms = dict()
    num_replicates = 1000
    for i in range(num_replicates):
        appdata_sub = df.sample(frac = 0.5)
        results = get_best_and_worst_words(words_list, appdata_sub)
        for term in results[0]:
            if term in good_terms:
                good_terms[term]+=1
            else:
                good_terms[term] = 1
        for term in results[1]:
            if term in bad_terms:
                bad_terms[term]+=1
            else:
                bad_terms[term] = 1

    def get_top_terms(term_list, apps_w_word, n):
        consistant_words_list = []
        for term in term_list:
            if term[1] > n / 2:
                consistant_words_list.append(term[0])
        return(consistant_words_list)
    sorted_terms = sorted(good_terms.items(), key=operator.itemgetter(1), reverse = True)
    overperforming_words = get_top_terms(sorted_terms, word_counts, num_replicates)
    sorted_terms = sorted(bad_terms.items(), key=operator.itemgetter(1), reverse = True)
    underperforming_words = get_top_terms(sorted_terms, word_counts, num_replicates)
    return overperforming_words, underperforming_words


# In[ ]:


def reset_dataframe(original_df):
    new_df = appdata[['Name','Price', 'Size', 'User Rating Count', 'Description']].dropna()
    new_df['log_rating_count'] = np.log(new_df['User Rating Count'])
    new_df['log_size'] = np.log(new_df['Size'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(new_df['log_size'], new_df['log_rating_count'])
    new_df['dev'] = ((new_df['log_rating_count'] - (intercept+slope*appdata_nonas['log_size']))/(intercept+slope*new_df['log_size']))
    return new_df


# In[ ]:


def print_word_context(word, df):
    df_sub_w_word = df[df[word]==1].sample(n = 5)
    desc_w_word = df_sub_w_word['match_string']
    for desc in desc_w_word:
        desc_length = len(desc)
        location = desc.find(" " + word + " ")
        print("\t", end = "")
        if location > 40:
            print(desc[location-50:location].lower(), end = "")
        else:
            print(desc[:location].lower(), end = "")
        if desc_length - location > 40:
            print(desc[location:location+50].lower())
        else:
            print(desc[location:].lower()) 


# In[ ]:


#Here we run the entire pipeline, outputting the words that most affect app performance
#this takes a little while
appdata_nonas = reset_dataframe(appdata)
words_list, word_counts, appdata_nonas = process_words_used_for_regression(appdata_nonas, 'Description')
best_words, worst_words = bootstrap_good_bad_word_search(words_list, appdata_nonas, word_counts)
print("Words used in overperforming apps:")
for word in best_words:
    print(word+", ", end = "")

print("\nWords used in underperforming apps:")
for word in worst_words:
    print(word+", ", end = "")


# Next we can take a look at the context these words are used in to see if any lessons can be drawn. Note that these descriptions have had non-alphabetical characters removed, as well as all one and two letter words, and all nouns are singular.

# In[ ]:


print("Words used in overperforming apps:")
for word in best_words:
    print(word+":")
    x = print_word_context(word, appdata_nonas)

print("\n")
print("Words used in underperforming apps:")
for word in worst_words:
    print(word+":")
    x = print_word_context(word, appdata_nonas)


# # What can we take away from these results?
# It's important to note that these results are correlative; they indicate which words successful apps use, but don't nessesarily indicate that using these words caused success. Nevertheless, there may be some lessons to be learned.
# 
# ## Words used by over-performing apps:
# MILLION: Apps typically use this in the context of extolling their "million's of users". So brag about the size of your user base.  
# HUNDRED: Used to describe the number of games modes or levels, hundreds connotes "very many" which consumers like.
# IPHONE is good but IPAD is bad. Make sure apps work on the iphone, and even if it works on the ipad don't mention it.  COM: Provide your website.  
# FREE: Everyone likes free!  
# VERSION: People like apps that are updated regularly. Conversely, developers update successful apps more often.  
# STAGE, PROGRESS: Emphasize progression systems, people want to know that you're offering depth of experience.  
# ALLIANCE, FORM, ONLINE: These words emphasize the social aspect of the app, and that you'll be interfacing with other people.  
# REVIEW: Users may be more likely to leave reviews if you ask them.  
# 
# ## Words used by under-performing apps:  
# 
# NOT: This used in a variety of contexts, but perhaps consumers don't like negative terms.  
# THREE: Also used to the number games modes, three connotes "not very many" which consumers don't like.
# BEAT: Perhaps users don't like the finality of this term, and prefer open ended challenges like "progress".  
# VARIOUS: Don't use vague terms.

# In[ ]:




