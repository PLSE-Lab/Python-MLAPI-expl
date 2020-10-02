#!/usr/bin/env python
# coding: utf-8

# # Sentiment intensity analysis of Alexa reviews using VADER
# 
# * VADER Summary : a lexicon-based sentiment intensity analyzer with crowd sourced dictionaries.
# 
# * Four Heuristics of VADER:
#     1. Emoticons (e.g. ! has value)
#     2. Capitalization (AMAZING vs Amazing)
#     3. Degree modifiers ('effing cute' vs 'sort of cute')
#     4. Shift in polarity due to but (e.g. I love you, but I don't want to be with you anymore)

# In[ ]:


import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()


# In[ ]:


dataset = '../input/amazon_alexa.tsv'
reviews_df = pd.read_csv(dataset, sep='\t', index_col=0, header=0)


# #### Method used:
# 
# * VADER compound polarity score is used
# * This score varies from -1 to +1
# * We count compound score 0 as positive marker

# In[ ]:


vader_score = []
vader_class = []

for review in reviews_df.verified_reviews:
    ss = sid.polarity_scores(review)
    compound_score = ss.get('compound')
    vader_score.append(compound_score)
    if (compound_score >= 0):
        vader_class.append(1)
    else:
        vader_class.append(0)
        
reviews_df['vader_score'] = vader_score
reviews_df['vader_class'] = vader_class


# #### Results (no pre-processing) : 90.76% (291 / 3150 samples labelled incorrectly)

# In[ ]:


reviews_df.loc[~(reviews_df['feedback'] == reviews_df['vader_class'])]


# #### Results (after pre-processing) : 90.86% (287 / 3150 samples labelled incorrectly)

# In[ ]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def remove_stopwords(text):
    #tokenization
    tokens = word_tokenize(text) 
    
    #stopwords removal
    temp = [word for word in tokens if word not in stopwords.words('english')]
    
    #detokenization
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in temp]).strip()


# In[ ]:


vader_score = []
vader_class = []

for review in reviews_df.verified_reviews:
    ss = sid.polarity_scores(remove_stopwords(review))
    compound_score = ss.get('compound')
    vader_score.append(compound_score)
    if (compound_score >= 0):
        vader_class.append(1)
    else:
        vader_class.append(0)
        
reviews_df['vader_score'] = vader_score
reviews_df['vader_class'] = vader_class


# In[ ]:


reviews_df.loc[~(reviews_df['feedback'] == reviews_df['vader_class'])]


# ## Discussion:
# 
# Stopwords removal does not yield better results because VADER uses words such as but in calculating the compuond score.

# In[ ]:


example = "It's got great sound and bass but it doesn't work all of the time. Its still hot or miss when it recognizes things"


# #### With stopwords
# "It's got great sound and bass but it doesn't work all of the time. Its still hot or miss when it recognizes things"

# In[ ]:


ss = sid.polarity_scores(example)
ss.get('compound')


# #### Stopwords removed
# "It's got great sound bass n't work time. Its still hot miss recognizes things"

# In[ ]:


ss = sid.polarity_scores(remove_stopwords(example))
ss.get('compound')


# ### References:
#     1. VADER Paper : http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf
#     2. Dataset : https://www.kaggle.com/sid321axn/amazon-alexa-reviews
#     3. Simpler Explanation of VADER : https://datameetsmedia.com/vader-sentiment-analysis-explained/

# In[ ]:




