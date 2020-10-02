#!/usr/bin/env python
# coding: utf-8

# The purpose of this Kernel is to explore the basics of Recommender Systems and to give the beginners some intuition with code examples. It covers some popular algorithms and strategies, but does not get deeply into advanced techniques or evaluation metrics (yet).  This notebook is inspired by Recommender Systems Course by University of Minessota.
# 
# I did not take huge datasets and tried to make everything easy-to-understand, but if you have any suggestions about making this Kernel better, please, share!

# ## Contents
# 
# 1. Introduction
# 2. Non-Personalized Recommender Systems
# 3. Personalized Recommender Systems
# 
#     3.1 Content-Based Filtering  
#     3.2 Collaborative Filtering
#     
#         3.2.1 User-user collaborative filtering
#         3.2.2 Item-item collaborative filtering
#         
#     3.3 Matrix factorization  
#     3.4 Hybrid Recommender Systems
# 4. Conclusion

# ## 1. Introduction

# (some background information and Wikipedia text)
# 
# A *recommender system* or a *recommendation system* (sometimes replacing "system" with a synonym such as platform or engine) is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item

# Recommender systems are utilized in a variety of areas including movies, music, news, books, research articles, search queries, social tags, and products in general. There are also recommender systems for experts, collaborators, jokes, restaurants, garments, financial services, life insurance, romantic partners (online dating), and Twitter pages.

# Recommender systems typically produce a list of recommendations in one of two ways:
# 
#  - Non-personalized approach
#  - Personalized approach

# Personalized-based recommender systems also have common subtypes:
# 
#  - Collaborative filtering recommender systems
#  - Content-based recommender systems

# All the techniques mentioned above have their own problems and pitfalls, which developers face creating and applying recommender systems to real-world problems. They must be taken into account while designing system architecture, and will be covered later in this work. Though the field of recommendation itself is relatively old, there are still no solutions that work perfectly for every case. Designing and evaluating a recommender system is hard, and requires a deep understanding of domain knowledge and data available, as well as constant experimenting and modification. First recommender systems appeared long time ago in 1990's, but the intense research started quite recently with the availability of better computational power and tremendous amounts of data coming from all different sources in internet.

# One of the events that energized research in recommender systems was the Netflix Prize. From 2006 to 2009, Netflix sponsored a competition, offering a grand prize of $1,000,000 to the team that could take an offered dataset of over 100 million movie ratings and return recommendations that were 10% more accurate than those offered by the company's existing recommender system
# 
# This competition energized the search for new and more accurate algorithms. The most accurate algorithm in 2007 used an ensemble method of 107 different algorithmic approaches, blended into a single prediction:
# 
# >Predictive accuracy is substantially improved when blending multiple predictors. Our experience is that most efforts should be concentrated in deriving substantially different approaches, rather than refining a single technique. Consequently, our solution is an ensemble of many methods.

# Beside classical approaches to recommendation with techniques described above, there are a lot of different cases that require modifications or special settings:
# 
#  - Group recommender systems
#  - Context-aware recommender systems
#  - Risk-aware recommender systems

# ## Common notation

# For convenience, we are going to use the same notation across this Kernel:

# ### Basic objects

# We need a way to talk about users, items, and the ratings matrix
# 
# $I$ - the set of items.<br>
# 
# $U$ - the set of users.<br>
# 
# $R$ - the ratings matrix or set of ratings.<br>
# 
# $u, v \in U$ - individual user.<br>
# 
# $i, j \in I$ - individual item.<br>
# 
# $r_{ui} \in R$ - a rating given by user $u$ for item $i$.<br>
# 
# $R_{u} \subset R$ - the set of ratings given by user $u$.<br>
# 
# $R_{i} \subset R$ - the set of ratings given for item $i$.<br>
# 
# $\vec{r_{u}}$ or $r_{u}$ - the ratings given by user $u$, as a vector with missing values for unrated items.
# We will often work with a normalized vector $\hat{r_{u}}$.<br>
# 
# $\vec{r_{i}}$ or $r_{i}$ - the ratings given for item $i$, as a vector with missing values for unrated items.<br>

# ### Scoring and ordering

# We are going to use it to describe scoring algorithms.
# 
# $s(i; u)$ - the score for item $i$ for user $u$.<br>
# 
# $s(i; u,q,x)$ - the score for item $i$ for user $u$ with query $q$ in context $x$.<br>
# 
# $O(I; u,q,x)$ - the ordering for items $I$ for user $u$ with query $q$ in context $x$.<br>

# ### Matrix factorization

# Notation for SVD and other decomposition techniques.
# 
# $R = P\Sigma Q^{T}$ - a factorization of the ratings matrix $R$ into a $|U| \times k$ user-feature preference matrix $P$ and a $|I| \times k$ item-feature relevance matrix $Q$.<br>
# 
# $\vec{p_{u}}$ or $p_{u}$ - the user feature vector over latent features.<br>
# 
# $\vec{q_{i}}$ or $q_{i}$ - the item feature vector.<br>

# ### 2. Non-personalized Recommendation

# Though non-personalized recommenders are rarely used in modern systems by themselves, they are still very powerful in combination with other algorithms, and, sometimes, the only available option. 
# 
# How can we make a recommendation for a user that we have little or no data about? 
# 
# That's where stereotype-based recommendations can be made, and most of the times we can take into account:
# 
#  - items popularity
#  - user demographic data 
#  - user actions during that particular session (for example, items in online-shop basket)

# #### Mean-based recommendation:

# One of the common approaches we can use is mean-based recommendation.
# 
# Basic mean is computed using the following formula:
# $$\mu = \frac{\Sigma_{u \in U_{i}}r_{ui}}{|U_{i}|}$$

# And can be used for recommending items with the highest rating. However, in order to make our recommendations more stable, we can use "damped" mean algorithm and add some "fake" global mean rating to our score. 

# $$s(i) = \frac{\Sigma_{u \in U_{i}}r_{ui} + \alpha\mu}{|U_{i}| + \alpha}$$

# Where $\alpha$ is a damping parameter, which represents the number of "fake" ratings we are adding. Because of that damping factor, recommender become less tend to make extreme recommendations.

# #### Associative rule recommendation:

# This approach is used to recommend items that are related to chosen one ("People who buy this also bought...") and, therefore, uses *reference item* to provide recommendations.

# The association rule formula is derived from Bayes theorem:
# 
# $$P(i|j) = \frac{P(i \vee j)}{P(j)}$$

# In this case, *j* is the *reference item*, and *i* is an item to be scored.

# We estimate probabilities by counting: $P (i)$ is the fraction of users in the system who
# purchased item i; $P(i\vee j)$ is the fraction that purchased both $i$ and $j$

# $$P(i|j) = \frac{P(i \vee j)}{P(j)} = \frac{|U_{i} \cap U{j}|/|U|}{|U_{j}| / |U|}$$

# The advanced version of this rule computes how much more likely someone is to rate an item $i$ when they rated $j$ than they would have if we do not know anything about whether they have rated $j$:
# 
# $$P(i|j) = \frac{P(i \vee j)}{P(i)P(j)}$$

# The following Python code produces some example of non-personalized data analysis based on Movie Lens movie ratings dataset:

# In[ ]:


# import libraries for data exploration and basic statistical functions

import pandas as pd
import numpy as np


# In[ ]:


data = pd.read_csv('../input/HW1-data.csv')
data


# Let's first calculate top movies by mean score:

# In[ ]:


# Top movies by mean score
means = data.iloc[:, 2:].mean().sort_values(ascending=False)
means


# And rating counts:

# In[ ]:


# Counts
counts = data.iloc[:, 2:].count()
counts.sort_values(ascending=False)


# Sometimes, we do not need to know precise rating to make recommendation. Therefore, we can define some ratings as positive (in this example, all the ratings >= 4):

# In[ ]:


# Top movies by percentage of positive marks
counts_positive = data.iloc[:, 2:][data.iloc[:, 2:] >= 4].count()
counts_positive.sort_values(ascending=False)
(counts_positive / counts).sort_values(ascending=False)


# Let's imagine we watched the movie "Toy Story" and we want to have a list of relevant movies to watch next. We can apply association rule here:

# In[ ]:


# Percentage of people who watched Toy Story also watched...
associative_product = '1: Toy Story (1995)'

watched_product = data.iloc[:, 2:][data[associative_product].notnull()].count()
(watched_product / data[associative_product].count()).sort_values(ascending=False)


# Beside association, we can also define similarity by basically measuring products corellation:

# In[ ]:


# Correlation between Toy Story and other movie ratings
data.iloc[:, 2:].corrwith(data[associative_product]).sort_values(ascending=False)


# Making recommendations above, we did not use data about user's gender. Statistically, men and women tend to like or dislike different kinds of movies, so, in order to make non-personalized recommendations more precise, we can take this information into account and see the difference:

# In[ ]:


# Means separate by gender
gender_column_name = 'Gender (1 =F, 0=M)'
male_means = data.iloc[:, 2:][data[gender_column_name] == 0].mean()
female_means = data.iloc[:, 2:][data[gender_column_name] == 1].mean()


# In[ ]:


# Male means
male_means.sort_values(ascending=False)


# In[ ]:


#Female means
female_means.sort_values(ascending=False)


# In[ ]:


# Overall mean ratings 
male_average_mean = data.iloc[:, 2:][data[gender_column_name] == 0].sum().sum() / data.iloc[:, 2:][data[gender_column_name] == 0].count().sum()
female_average_mean = data.iloc[:, 2:][data[gender_column_name] == 1].sum().sum() / data.iloc[:, 2:][data[gender_column_name] == 1].count().sum()
print("Male avg. mean: {} Female avg. mean: {}".format(male_average_mean, female_average_mean))


# In[ ]:


# Movies that female users rate highest above male raters

(female_means - male_means).sort_values(ascending=False)


# In[ ]:


# Movies that male users rate highest above female raters

(male_means - female_means).sort_values(ascending=False)


# In[ ]:


# Positive (> 4) ratings by male

counts_positive_male = data.iloc[:, 2:][(data >= 4)][data[gender_column_name] == 0].count()
counts_positive_male.sort_values(ascending=False)


# In[ ]:


# Percentage of positive ratings by male

counts_male = data.iloc[:, 2:][data[gender_column_name] == 0].count()
percentage_positive_male = (counts_positive_male / counts_male)
percentage_positive_male.sort_values(ascending=False)


# In[ ]:


# Positive (> 4) ratings by female

counts_positive_female = data.iloc[:, 2:][(data >= 4)][data[gender_column_name] == 1].count()
counts_positive_female.sort_values(ascending=False)


# In[ ]:


# Percentage of positive ratings by female

counts_female = data.iloc[:, 2:][data[gender_column_name] == 1].count()
percentage_positive_female = (counts_positive_female / counts_female)
percentage_positive_female.sort_values(ascending=False)


# In[ ]:


# Female-male difference in the liking percentage

(percentage_positive_female - percentage_positive_male).sort_values(ascending=False)


# In[ ]:


# Male-female difference in liking percentage
(percentage_positive_male - percentage_positive_female).sort_values(ascending=False)


# In[ ]:


# Difference between the average rating overall

female_average_mean - male_average_mean


# ## 3. Personalized Recommendation

# All the personalized recommendation require certain amount of data collected about users. Data could either be collected implicitly (products user click on, see) and explicitly (in forms of ratings, surveys, polls). Both methods are used widely and can be combined together depending on the system restrictions and type of recommendation system provide.

# ### 3.1 Content-based filtering

# Content-based filtering, also referred to as cognitive filtering, recommends items based on a comparison between the content of the items and a user profile. The content of each item is represented as a set of descriptors or terms, typically the words that occur in a document. The user profile is represented with the same terms and built up by analyzing the content of items which have been seen by the user.
# 
# Several issues have to be considered when implementing a content-based filtering system. First, terms can either be assigned automatically or manually. When terms are assigned automatically a method has to be chosen that can extract these terms from items. Second, the terms have to be represented such that both the user profile and the items can be compared in a meaningful way. Third, a learning algorithm has to be chosen that is able to learn the user profile based on seen items and can make recommendations based on this user profile.
# 
# The greatest advantage in content-based filtering systems is that the recommendations provided can easily be interpreted to user, because we always know what "features" about particular item made algorithm rate it higher.
# 
# When we have the representation of our item or user as a vector of features, we can measure use metrics such as cosine distance to measure similarity between user profile vector and item feature vector:
# 
# $$cos(p_{u}, q_{i}) = \frac{\Sigma_{t}p_{it}q_{ut}}{\sqrt{\Sigma_{t}q_{ut}^{2}}\sqrt{\Sigma_{t}p_{it}^{2}}}$$

# The example of content-based filtering applied to documents:

# In[ ]:


# importing raw data from excel file

raw_data = pd.read_excel("../input/cbf.xls")
raw_data


# In[ ]:


docs = raw_data.loc['doc1':'doc20', 'baseball':'family']
docs


# We have rating of two users. 
# 
# The value of 1.0 means the user liked the document, the value of 0 - disliked.
# 
# NaN means that the user never seen the document (and we have to predict rating)

# In[ ]:


user_ranks = raw_data.loc['doc1':'doc20', 'User 1':'User 2']
user_ranks.fillna(0, inplace=True)
user_ranks


# Let's use basic matrix multiplication to predict user interest in particular topic

# In[ ]:


user_profiles = np.array(docs).T @ np.array(user_ranks)
pd.DataFrame(user_profiles, docs.columns, user_ranks.columns)


# Next step is to calculate matrix of predicted user preferences for documents.

# In[ ]:


user_preferences = np.matmul(np.array(docs), user_profiles)
updf = pd.DataFrame(user_preferences, docs.index, user_ranks.columns)
updf


# We can see the predicted "ratings" of documents for User 1 & User 2

# In[ ]:


updf.loc[:, 'User 1'].sort_values(ascending=False)


# In[ ]:


updf.loc[:, 'User 2'].sort_values(ascending=False)


# You may have noticed that in our computation an article that had many attributes checked could have more influence on the overall profile than one that had only a few. doc 1 and doc 19 each have five attributes, while doc6, doc7, and doc18 only have 2 attributes each.
# 
# To fight this problem, we might want to normalize our ratings first.

# In[ ]:


normalized_docs = docs.div(docs.sum(axis=1).apply(np.sqrt), axis=0)
normalized_docs


# Normalized profiles now:

# In[ ]:


normalized_profiles = np.matmul(np.array(normalized_docs).T, np.array(user_ranks))
pd.DataFrame(normalized_profiles, docs.columns, user_ranks.columns)


# Using the same math, calculate new normalized user preferences:

# In[ ]:


normalized_preferences = np.matmul(np.array(normalized_docs), normalized_profiles)
npdf = pd.DataFrame(normalized_preferences, docs.index, user_ranks.columns)
npdf


# In[ ]:


npdf.loc[:, 'User 1'].sort_values(ascending=False)


# As we can see, preferences changed after normalization.
# 
# Another popular and very common approach is to apply TF-IDF technique to our documents.
# 
# TFIDF score is caclucated as a product of TF (term frequency) and IDF (inverse documents frequency), which makes more important things that appear frequently in this document, but rarely appear in other documents.

# In[ ]:


docs


# In[ ]:


DF = docs.sum(axis=0)
IDF = 1.0 / DF
np.array(IDF)


# In[ ]:


weighted_preferences = np.matmul(np.array(normalized_docs), np.multiply(np.array(normalized_profiles).T, np.array(IDF)).T)
pd.DataFrame(weighted_preferences, docs.index, user_ranks.columns)


# ### 3.2 Collaborative filtering

# Collaborative filtering, also referred to as social filtering, filters information by using the recommendations of other people. It is based on the idea that people who agreed in their evaluation of certain items in the past are likely to agree again in the future. A person who wants to see a movie for example, might ask for recommendations from friends. The recommendations of some friends who have similar interests are trusted more than recommendations from others. This information is used in the decision on which movie to see.

# Collaborative filtering often uses the concept of **neighbourhood** (the amount of people/items we base our prediction on). Making neighbourhoods too small results in not enough information for accurate prediction, and making them too big results in high computational complexity and letting noize in systems. Neighborhood size is a hyperparameter which needs to be tuned in every system. Distance between neighbors can be defined using such metrics as cosine similarity.

# One of the most common problems all collaborative filtering recommender systems face - a so called "cold start" problem, when we either:
# 
#  - do not have enough ratings for a new user to find neighbours
#  - do not have enough ratings for a new item to find neighbours
#  - have a completely new system without any data to make recommendations
#  
# In each of those cases, problems might be solved differently depending on the particular case and options available.

# #### 3.2.1 User-user collaborative filtering

# In user-user collaborative filtering, we provide a recommendation based on tastes of other users similar to us. The problem with that algorithm is that we need a lot of information about other people to provide correct recommendations, but the main benefits are effectivness and ability to provide new, unexpected, and, yet, good recommendatons.

# In order to account for user's tendecy to give higher/lower ratings, we will use normalization again. Algorithm for providing score based on user-user collaborative filtering is defined as:

# $$ p_{u,i} = \mu_{u} + \frac{\Sigma_{v \in N(u;i)}cos(u,v)(r_{v,i}-\mu_{v})}{\Sigma_{v \in N(u;i)}|cos(u,v)|}  $$

# An example of User-user collaborative filtering:

# In[ ]:


# library for visualization
import seaborn


# In[ ]:


data = pd.read_excel("../input/data.xls")
data


# In[ ]:


# user correlations
correlations = pd.DataFrame(data.transpose(), data.columns, data.index).corr()
seaborn.heatmap(correlations)


# For this example, we will make predictions for user 3867.
# 
# Our 'neighborhood' for a user - users with N highest correlations

# In[ ]:


# selecting 6 neighbors 

neighbours_3867 = correlations[3867].sort_values(ascending=False)[1:6]


# In[ ]:


recommendations = data.fillna(0)


# In[ ]:


recommendations.loc[neighbours_3867.index]


# In[ ]:


# calculations for top-5 movies

(recommendations.loc[neighbours_3867.index].multiply(
    neighbours_3867, axis=0).sum(axis=0) / (recommendations.iloc[:, :] != 0).multiply(
    neighbours_3867, axis=0).sum(axis=0)).sort_values(ascending=False)[:5]


# #### 3.2.2 Item-item collaborative filtering

# In item-item collaborative filtering, we provide a recommendation based on other items similar to us. The benefits of it, compared to user-user collaborative filtering, is that we usually need much less similarity computations (in most cases, there are much more users in systems than items). The most common pitfall - system provides can provide very obvious recommendations.

# Score provided by item-item filtering is computed using the following formula:
# 
# $$ s(i,u) = \mu_{i} + \frac{\Sigma_{j \in I_{u}}w_{ij}(r_{u,j}-\mu_{j})}{\Sigma_{j \in I_{u}}|w_{ij}|}  $$

# Example of item-item recommendation:

# In[ ]:


from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


ratings_raw_data = pd.read_excel('../input/iicf.xls')
data = pd.DataFrame(ratings_raw_data.iloc[0:20, 1:21])
data.index = ratings_raw_data.iloc[0:20,0]
data_means = data.mean(axis=0)
data


# In[ ]:


data.fillna(0, inplace=True)
similarities = pd.DataFrame(cosine_similarity(data.T, data.T), data.columns, data.columns)
seaborn.heatmap(similarities)


# In[ ]:


# Yeah, this probably can be done better with Pandas one-liners.

predictions = data.copy()
for user in data.index: #every user in index
    for movie in data.columns: #every movie for user
        mean = data_means[movie] #mean rate of this movie
        similar_movies = similarities[movie] # similar movies to this movie
        numerator = 0
        weights_sum = 0
        for sm in similar_movies.index: # for every similar movie
            weight = similar_movies[sm]
            rating = data.loc[user, sm]
            if weight > 0 and rating > 0: #which is non-negative (sim) and rated by user
                numerator += weight * (rating - mean)
                weights_sum += weight
        predictions.loc[user, movie] = mean + (numerator / weights_sum)


# Finally, we filled a table with our prediction. Let's see the results for user 755:

# In[ ]:


predictions.loc[755].sort_values(ascending=False)


# ### 3.3 Matrix factorization

# In matrix factorization techniques, we usually represent the rating matrix as a product of 3 other matrices.
# $$R = P\Sigma Q^{T}$$

# The benefits of those techniques are that they can dramatically improve system performance by reducing the necessary amount of space. Collaborative techniques can be later applied on decomposed matrices. 
# 
# This work does not cover factorization techniques in depth. (at least yet)

# ### 3.4 Hybrid recommender systems

# In hybrid recommender systems, recommendation is made usually based on scores provided by multiple recommender systems. The most common technique is to represent the final score as a linear combination of scores provided by other recommenders with according weights. 
# 
# Another option is so-called "switch" recommender system. Given some input, system decides, which of the available recommender engines is better to use for a recommendation in this particular situation. Such algorithm helps to overcome problems that exist in each recommender separately.
# 
# We also can use so-called "cascade" hybrid recommenders - the system where outputs of one recommendtion algorithm are used as inputs to other. 
# 
# There are dozens of ways to use hybrid recommender systems, and there are no common way for applying them to real world problem. Design and architecture of each of such systems depends on data available, domain field and requirements for a particular system.

# ## 4. Conclusion

# Building a good recommender system is not an easy task. Although some algorithms are considered "best practices", they all have their strenghs and weaknesses. Developing a recommender systems requires good understanding of domain users, data that can be collected, and purposes of our recommendation. Without knowing all the things mentioned above, it is impossible to design a good recommender system, no matter how complicated are algorithms you use. 

# However, there are common techniques that are commonly used by themselves and in combination. Modern recommender systems still use collaborative filtering and content filtering techniques, although nowadays this algorithms are used in combination and with application of such more advanced techniques as matrix factorization, neural networks and hybrid recommender systems.

# The field of recommender systems is constantly developing, providing us with new studies on context-based recommendation, risk-aware and group recommendations, as well as research in different evaluation methods and iterative factorization techniques. There are dozens of ways to design a recommender, and choosing "the best" approach is up to people who know why and how they want to make recommendations.
