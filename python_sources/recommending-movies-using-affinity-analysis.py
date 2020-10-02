#!/usr/bin/env python
# coding: utf-8

# **Affinity analysis** , or market basket analysis, is used to determine when objects occur frequently together. The aim is to discover when objects occur simultaneously. 
# In this example, we wish to work out when two movies are recommended by the same reviewers.
# The data for affinity analysis is often described in the form of a transaction. Intuitively, this comes from a transaction at a store - determining when objects are purchased together.
# 
# **Data** - https://grouplens.org/datasets/movielens/
# 
# **Algorithms** - Unsupervised because, unlike supervised learning, there is no correct answers and there is no teacher. Algorithms are left to their own devises to discover and present the interesting structure in the data.
# 
#  - Association rule learning - https://en.wikipedia.org/wiki/Association_rule_learning
#    
#  - Apriori - https://en.wikipedia.org/wiki/Apriori_algorithm
# 
# **Applications** Fraud detecetion, customer segmentation, software optimization, product recomandations,
# 

# In[31]:


# Code from 'Learning-Data-Mining - Chapter 4'

# Import
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files from MovieLens https://grouplens.org/datasets/movielens/
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[32]:


all_ratings = pd.read_csv('../input/rating.csv')
all_ratings["timestamp"] = pd.to_datetime(all_ratings['timestamp'])
all_ratings[:5]


# In[33]:


all_ratings.dtypes


# In[34]:


all_ratings.count()


# In[35]:


# Not all reviews are favourable! Our goal is "other recommended books", so we only want favourable reviews
all_ratings["favorable"] = all_ratings["rating"] > 3
all_ratings[10:15]


# In[36]:


# Sample the dataset. You can try increasing the size of the sample, but the run time will be considerably longer
ratings = all_ratings[all_ratings['userId'].isin(range(200))] 


# In[37]:


# We start by creating a dataset of each user's favourable reviews
favorable_ratings = ratings[ratings["favorable"]]
favorable_ratings[:5]


# In[38]:


# We are only interested in the reviewers who have more than one review
favorable_reviews_by_users = dict((k, frozenset(v.values)) for k, v in favorable_ratings.groupby("userId")["movieId"])
len(favorable_reviews_by_users)


# In[39]:


# Find out how many movies have favourable ratings
num_favorable_by_movie = ratings[["movieId", "favorable"]].groupby("movieId").sum()
num_favorable_by_movie.sort_values(by=["favorable"], ascending=False)[:5]


# In[40]:


# APRIORI Alogorithm
# STEP 1 . Create frequent itemsets
from collections import defaultdict

def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets, min_support):
    counts = defaultdict(int)
    for user, reviews in favorable_reviews_by_users.items():
        for itemset in k_1_itemsets:
            if itemset.issubset(reviews):
                for other_reviewed_movie in reviews - itemset:
                    current_superset = itemset | frozenset((other_reviewed_movie,))
                    counts[current_superset] += 1
    return dict([(itemset, frequency) for itemset, frequency in counts.items() if frequency >= min_support])


# In[41]:


import sys
frequent_itemsets = {}  # itemsets are sorted by length
min_support = 50

# k=1 candidates are the isbns with more than min_support favourable reviews
frequent_itemsets[1] = dict((frozenset((movie_id,)), row["favorable"])
                                for movie_id, row in num_favorable_by_movie.iterrows()
                                if row["favorable"] > min_support)

print("There are {} movies with more than {} favorable reviews".format(len(frequent_itemsets[1]), min_support))
sys.stdout.flush()
for k in range(2, 20):
    # Generate candidates of length k, using the frequent itemsets of length k-1
    # Only store the frequent itemsets
    cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users, frequent_itemsets[k-1],
                                                   min_support)
    if len(cur_frequent_itemsets) == 0:
        print("Did not find any frequent itemsets of length {}".format(k))
        sys.stdout.flush()
        break
    else:
        print("I found {} frequent itemsets of length {}".format(len(cur_frequent_itemsets), k))
        #print(cur_frequent_itemsets)
        sys.stdout.flush()
        frequent_itemsets[k] = cur_frequent_itemsets
# We aren't interested in the itemsets of length 1, so remove those
del frequent_itemsets[1]


# In[42]:


print("Found a total of {0} frequent itemsets".format(sum(len(itemsets) for itemsets in frequent_itemsets.values())))


# In[43]:


# Now we create the association rules. First, they are candidates until the confidence has been tested
candidate_rules = []
for itemset_length, itemset_counts in frequent_itemsets.items():
    for itemset in itemset_counts.keys():
        for conclusion in itemset:
            premise = itemset - set((conclusion,))
            candidate_rules.append((premise, conclusion))
print("There are {} candidate rules".format(len(candidate_rules)))


# In[44]:


print(candidate_rules[:5])


# In[45]:


# Now, we compute the confidence of each of these rules. This is very similar to what we did in chapter 1
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in favorable_reviews_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1
rule_confidence = {candidate_rule: correct_counts[candidate_rule] / float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule])
              for candidate_rule in candidate_rules}


# In[46]:


# Choose only rules above a minimum confidence level
min_confidence = 0.9


# In[47]:


# Filter out the rules with poor confidence
rule_confidence = {rule: confidence for rule, confidence in rule_confidence.items() if confidence > min_confidence}
print(len(rule_confidence))


# In[48]:


from operator import itemgetter
sorted_confidence = sorted(rule_confidence.items(), key=itemgetter(1), reverse=True)


# In[49]:


for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise, conclusion))
    print(" - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print("")


# In[50]:


movie_name_data = pd.read_csv('../input/movie.csv')
movie_name_data.head(5)


# In[51]:


def get_movie_name(movie_id):
    title_object = movie_name_data[movie_name_data["movieId"] == movie_id]["title"]
    title = title_object.values[0]
    return title


# In[52]:


get_movie_name(4)


# In[53]:


for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ", ".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
    print(" - Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print("")


# In[55]:


all_ratings.head(2)


# In[58]:


# Evaluation using test data
test_dataset = all_ratings[~all_ratings['userId'].isin(range(200))]
test_favorable = test_dataset[test_dataset["favorable"]]
#test_not_favourable = test_dataset[~test_dataset["favourable"]]
test_favorable_by_users = dict((k, frozenset(v.values)) for k, v in test_favorable.groupby("userId")["movieId"])
#test_not_favourable_by_users = dict((k, frozenset(v.values)) for k, v in test_not_favourable.groupby("UserID")["MovieID"])
#test_users = test_dataset["UserID"].unique()


# In[59]:


test_dataset[:5]


# In[60]:


correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)
for user, reviews in test_favorable_by_users.items():
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1
            else:
                incorrect_counts[candidate_rule] += 1


# In[61]:


test_confidence = {candidate_rule: correct_counts[candidate_rule] / float(correct_counts[candidate_rule] + incorrect_counts[candidate_rule])
                   for candidate_rule in rule_confidence}
print(len(test_confidence))


# In[62]:


sorted_test_confidence = sorted(test_confidence.items(), key=itemgetter(1), reverse=True)
print(sorted_test_confidence[:5])


# In[63]:


for index in range(10):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    premise_names = ", ".join(get_movie_name(idx) for idx in premise)
    conclusion_name = get_movie_name(conclusion)
    print("Rule: If a person recommends {0} they will also recommend {1}".format(premise_names, conclusion_name))
    print(" - Train Confidence: {0:.3f}".format(rule_confidence.get((premise, conclusion), -1)))
    print(" - Test Confidence: {0:.3f}".format(test_confidence.get((premise, conclusion), -1)))
    print("")

