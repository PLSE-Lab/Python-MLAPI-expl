#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Career Village Predictor
# 
# Matching aspiring students to professional help. This works by matching the tags in the question to professionals who have dealt with those tags before.
# 
# 

# First things first, we want to load in everything and import all the things.

# In[ ]:


get_ipython().system('pip install ranking')

import numpy as np
import pandas as pd
import seaborn as sns
from readability import Readability
from collections import defaultdict
from ranking import Ranking

import os
print(os.listdir("../input"))


# # Clustering with questions
# 
# The first thing we want to do is to try and get a feel for what kind of words are unique to each professional. To do this, we sum up the occurence of each word for a particular Professional/Question. We then know a word is unique to the professional if that word uses up significantly more of a person's lexicon than average.
# 
# Before we can look at that though, we need to load in everything. There are a ton of possibly useful features to look at, but for the time being I will ignore them for the sake of simplicity. 

# In[ ]:


questions = pd.read_csv("../input/questions.csv", delimiter=",")
questions = questions.drop(["questions_author_id", "questions_date_added"], axis=1)

scores = pd.read_csv("../input/question_scores.csv")

questions = pd.merge(
    questions,
    scores,
    left_on='questions_id',
    right_on='id',
    how="inner"  # Questions without answers are useless, so we leave them out
)

questions = questions.drop("id", axis=1)

questions.tail()


# Again, many of these features may be useful on a second pass-through.

# In[ ]:


professionals = pd.read_csv("../input/professionals.csv", delimiter=",")
professionals = professionals.drop(["professionals_date_joined", "professionals_headline"], axis=1)
professionals.tail()


# In[ ]:


answers = pd.read_csv("../input/answers.csv")
answers = answers.drop(["answers_date_added", "answers_id"], axis=1)

answers.head()


# Now that we have all of it loaded in, we want to merge all of this into one table for easier reading.

# In[ ]:


data = pd.merge(
    questions,
    answers,
    left_on='questions_id',
    right_on='answers_question_id',
    how="inner"
)
data = data.drop(["questions_id", "answers_question_id"], axis=1)

data = pd.merge(
    data, 
    professionals,
    left_on="answers_author_id",
    right_on="professionals_id",
    how="inner"
)

data = data.drop(["answers_author_id"], axis=1)
data.tail()


# Great! Now we have a convenient table containing questions, answers, and the professional's industry. Now lets take a look at what kind of industries we have access to.

# In[ ]:


(professionals.professionals_location.unique(), professionals.professionals_industry.unique())


# In[ ]:


(len(data.professionals_id.unique()), len(data.professionals_location.unique()),
 len(data.professionals_industry.unique()))


# It appears that there are a lot of professionals on this site (unsurprisingly). Lets see if we can improve this.

# In[ ]:


import string
# returns that last word in the string, in lowercase
def simplify_name(name):
    line = str(name).replace(",", " ")
    line = line.split()[-1]
    line = line.lower()
    return line 

# data['professionals_location']
data['professionals_location'] = data['professionals_location'].apply(simplify_name)
data['professionals_industry'] = data['professionals_industry'].apply(simplify_name)
data.tail()


# In[ ]:


(len(data.professionals_id.unique()), len(data.professionals_location.unique()),
 len(data.professionals_industry.unique()))


# Much better. There are now 10063 users that fit into 157 locations and 525 industries. Lets try and take a deeper look at the professionals themselves. What kind of words do they respond to? We are going to create a mapping of professional, to their personal word bucket they react to.

# In[ ]:


writers = list(data.professionals_id.unique())

# These dictionaries map the writer to the words that they engage with
writer_words = {writer:{} for writer in writers}
writer_tags = {writer:{} for writer in writers}

# for each row, add to the writer's word bucket
for _, question in data.iterrows():
    writer = question["professionals_id"]
    body = question["questions_body"].split()
    title = question["questions_title"].split()
    
    # Store the tags in one list and words in another
    # Store as a lower case so that things like Word and word are not held differently
    def simplify(s):
        s = s.lower()
        exclude = set(string.punctuation)
        s = ''.join(ch for ch in s if ch not in exclude)
        return s
    
    tags = [simplify(word) for word in (body+title) if len(word) > 0 and word[0] == "#"]
    body = [simplify(word) for word in (body+title) if len(word) > 0 and word[0] != "#"]
    
    # add the words to the writer's word cloud
    for word in body:
        if word in writer_words[writer]:
            writer_words[writer][word] += 1
        else:
            writer_words[writer][word] = 1
        
    # add the words to the writer's tag cloud
    for word in tags:
        if word in writer_tags[writer]:
            writer_tags[writer][word] += 1
        else:
            writer_tags[writer][word] = 1

writer_tags[data.professionals_id.unique()[1]]


# If we aren't careful here, we may end up recommending people who are more active. Rather, we should choose people whose fraction of vocabulary better fits with the target demographic. We will still filter out inactive and new users, since their sample of words is too small to make meaningful predictions.

# In[ ]:



total_tags = []
total_words = []

def normalize_dict(word_to_freq):
    """ Maps the counts from 0 to 1, divides all by the most frequent word/tag 
        Mutates and returns the original dict
    """
    max_freq = 0
    for word in word_to_freq:
        max_freq = max(max_freq, word_to_freq[word])
    
    if float(max_freq) <= 10 ** (-8):
        return dict()
    else:
        for word in word_to_freq:
            word_to_freq[word] /= float(max_freq)
    
    return word_to_freq
    

for writer in writers:
    # get the total number of tags they made
    totaltag = 0
    totalword = 0
    if writer in writer_tags:
        for tag in writer_tags[writer]:
            totaltag += writer_tags[writer][tag]
    if writer in writer_words:
        for word in writer_words[writer]:
            totalword += writer_words[writer][word]
            
    total_tags.append(totaltag)
    total_words.append(totalword)

# keep only the people with at least 50 tags associated to them (AKA only recommend users that have already been useful)
writer_totals = pd.DataFrame({"professional_id": writers, "total_tags": total_tags, "total_words": total_words})
writer_totals = writer_totals[writer_totals["total_tags"] > 50]

print(writer_totals.head())

writers = list(writer_totals["professional_id"])
total_tags = list(writer_totals["total_tags"])

norm_writer_tags = {writer:normalize_dict(writer_tags[writer]) for writer in writers}

norm_writer_tags['f65c1eac3d2846d1a05206be08477272']


# And now, we can begin to make some recommendations. We will parse the words in a new message, check it against all of the authors, once we do that, we return a list, in order of how well we beleive the question matches the writer's experience. We use the following error function to rank them. We sum up over all words that match up between the message and the existing word cloud for the author.
# 
# $$ E = \sum^{words}_n (p^w_i - p^m_i)^2 + \sum^{tags}_n (p^t_i - p^m_i)^2$$
# 
# Where $p^w$ is the word frequency, $p^t$ is that tag frequency, and $p^m$ is the frequency of the message. If $p^t$ or $p^w$ are missing, they are assumed to be $0$.

# In[ ]:


def _get_user_ranking_tag_similarity(words, other):
    """ (list of string, word bag) -> Error
    Gets an ordering of users to recommend. First user is most recommended.
    """
    # Read it into a normalized word frequency dictionary
    word_bag = defaultdict(int)
    for word in words:
        word_bag[word] += 1
    word_bag = normalize_dict(word_bag)
    
    error = 0 
    for word in word_bag:
        pw = other[word] if word in other else 0
        pm = word_bag[word]
        error += (pw - pm) ** 2
    
    return error


def get_user_ranking_tag_similarity(words, tags):
    """
    Given a list of words and a list of tags, ranks all of the professionals by what kind of questions they answered before
    """
    ranking = []
    for writer in norm_writer_tags:
        error = _get_user_ranking_tag_similarity(tags, norm_writer_tags[writer])
#         error += _get_user_ranking_tag_similarity(words, norm_writer_words[writer]) # Uncomment when I implement this
        ranking.append((writer, error))
    
    ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
    
    # Thanks stack overflow, gets the ranking
    users, scores = zip(*ranking)
    scores = [max(scores) - score for score in scores]
    
    ranking = list(Ranking(scores, start=1, reverse=True).ranks())
    
    ranking_df = pd.DataFrame({"prof_id": users, "error": scores, "rank":ranking})
    
    return ranking_df
    
    
results = get_user_ranking_tag_similarity(["college", "college", "professor"], ["college", "college", "professor"])
results.sample(10)
    

