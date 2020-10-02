#!/usr/bin/env python
# coding: utf-8

# # Music reviews and heavy metal subgenres
# 
# My dear fellow metalheads and music enthusiasts,
# 
# I've spent the weekend web mining a popular heavy metal webzine to gather some data on the world's best music genre. 
# The dataset consists of roughly 3000 reviews of album releases of international (popular and underground) heavy metal 
# bands. Each row in the dataset represents one album review. The columns contain information on the reviewer, the rating 
# given by the reviewer and the music subgenres the album falls into. 
# 
# ### Let's do some EDA (recommended listening: Iron Maiden - The Trooper)
# 
# ![](https://store.playstation.com/store/api/chihiro/00_09_000/container/CH/de/999/EP4067-NPEB01320_00-AVPOPULUSM003114/1550731490000/image?w=240&h=240&bg_color=000000&opacity=100&_version=00_09_000)
# 

# ## Structure of the data:

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import normaltest

data = pd.read_csv("../input/reviews.csv", index_col=0)


# Let's have a look at the data first:

# In[ ]:


data.head()


# Columns correspond to heavy metal subgenres, the reviewer and the grade given by the reviewer.
# A value of 1 in a column corresponding to a music genre denotes that the reviewer categorized the album under that subgenre. A single album can belong to more than one subgenre.

# In[ ]:


print('total number of reviews: ', len(data.index))


# In[ ]:


genres = list(data.columns[:-2])
print('Among the music genres are: ',end='')
for genre in genres[:7]:
    print(' '+genre+', ',end='')
print('...')

print('total number of genres: ',len(genres))


# In[ ]:


authors = list(data.AUTHOR.unique())
print('Review authors are denoted by letters: ',authors)


# Every review is written by one of 12 authors (denoted by letters A-L).

# In[ ]:


grades = [i for i in range(11)]
print('possible grades range from ',0,' (worst) to ', 10, ' (best) with "-" denoting a missing value')


#    ## What are the top 20 most reviewed music genres?

# In[ ]:


genre_abs = [data.loc[:,g].sum() for g in genres]
gc_data = pd.DataFrame({'genre' : genres, 'abs_freq': genre_abs}).sort_values('abs_freq',ascending=False)
gc_data = gc_data.iloc[:20,:]

# plot the top ten genres
plt.figure(figsize=(18,10))
plot = sns.barplot(x=gc_data.genre, y=gc_data.abs_freq)
plt.title("Top 20 most reviewed heavy metal subgenres")
plt.xlabel("Heavy metal subgenres")
plt.ylabel("Number of reviews")
t = plt.xticks(rotation=90)


# Aaaaand here's our first unanticipated result: **Black metal** has by far gotten the most reviews. Followed by genres which you would 
# more probably have expected to find among the first few places. What's the cause of this? Since I didn't mine all the reviews but just the reviews
# written by a subset of reviewers we might have oversampled reviews of black metal lovers. 

# ## Can we see correlation of genres in the data?
# 
# As a longtime metalhead you develop a working knowledge of which metal subgenres are closely related to each other (i.e. overlap) and which genres 
# are further apart. There is a very rough 2-part distinction between genres in which singers actually "sing" in a traditional
# way (like power metal, hard rock, etc.) and genres with growled or screamed vocals (e.g. death metal, black metal, etc.). 
# Let's look at a heatmap of the correlations between genres and see if we can rediscover some familiar connections. 

# In[ ]:


genre_corr = data.loc[:, gc_data.genre.tolist()].corr()
plt.figure(figsize=(16,12))
plot = sns.heatmap(data=genre_corr)


# What immediately catches our eye are the two bright spots close to the center (everything is symmetric so there is actually only one spot).
# Here we see a strong correlation between **melodic metal** and **power metal**. Makes sense! I have a hard time thinking of a non-melodic 
# power metal band. I don't think that the melodic metal genre is as well-defined as the power metal genre, but I can imagine that a reviewer
# of a power metal album would often choose melodic metal as a second category when looking for another genre label.
# 
# Other interesting (and expected) correlations are 
# * hard rock and classic rock
# * nwobhm and heavy metal
# * doom metal and sludge
# * power metal and heavy metal
# * modern metal and metalcore
# * metalcore and melodic death metal 
# 
# What I find remarkable is that **doom metal** looks to be correlated more strongly with **black metal** and **sludge** than with melodic 
# styles of metal like **heavy metal**. Usually doom metal itself falls roughly into the two categories of traditional and melodic doom 
# metal bands like Black Sabbath, Candlemass or Witchfinder General or modern and non-melodic bands closer to black metal. The overall high amount
# of black metal reviews leads me to conclude that doom cds were for the most part coming from the second family of doom bands. Maybe, looking at a 
# subset of more traditionally oriented reviewers could have yielded different correlations.

# ## Let's take a look at the distribution of ratings:
# 
# First we have to take care of missing rating values. 

# In[ ]:


print('There are ',len(data[data.RATING == '-']),' reviews with missing rating. We will exclude them from the dataset.')


# In[ ]:


data = data[data.RATING != '-']
data.RATING = data.RATING.astype(int)


# Let's plot the distribution of grades given by the reviewers.

# In[ ]:


grade_counts = [len(data[data.RATING == grade]) for grade in grades]
plt.figure(figsize=(14,10))
plot = sns.barplot(x=grades,y=grade_counts)
a = plt.title("Distribution of grades")
a = plt.xlabel("Grades")
a = plt.ylabel("Absolute frequency")


# In[ ]:


print('The median is ',data.RATING.median())
print('The mean is ', data.RATING.mean())
print('The skew is ',data.RATING.skew())


# The distribution is centered around the grade 7 with a longer left than right tail. It certainly doesn't look like a normal distribution to me. 
# Let's try to confirm our hypothesis with a D'Agostino and Pearson normality test (The null-hypothesis is that our collected data comes from a normally distributed
# variable).

# In[ ]:


stat, p_value = normaltest(data.RATING)

if p_value < 0.05:
    print("We reject the null-hypothesis with p_value ", p_value)
else:
    print("We confirm the null_hypothesis with p_value ", p_value)


# ## Is a reviewer more likely to give good grades to albums in his favourite subgenre?
# 
# This is an interesting question. Let's say you like metal in general but if you had to choose one single favourite genre, there would be no doubt about it - 
# *Majestic voices singing about glory and death on the battlefield, drums like the thundering charge of the cavalry, intertwined guitar and 
# keyboard solos like bolts of fire and ice from a wizard's wand* - If you had to choose **one** genre it would be **power metal** and
# you can name the 
# 3 first albums of every halfway popular power metal band of the last 20 years. 
# 
# Now, years later, you are a free editor of big heavy metal webzine and
# someone asks you to review a power metal cd. Are you more likely to give that cd a good grade because you generally like power metal bands better than
# other metal bands? Or are you more likely to give a new power metal cd a lower grade because most new releases just sound cheap to you compared to the 
# many stellar power metal releases of your old heroes? 
# 
# #### Let's find out!
# 
# First of all we have to think about how to determine every reviewers *favourite* subgenre. We will keep it simple and assume that every editor 
# can choose which cds to review. So, an editors favourite genre will most likely be the one where he has written the most reviews in. 

# In[ ]:


# returns the favourite genre of an author aut
def favourite(aut):
    rev = data.loc[data.AUTHOR == aut]
    genre_abs = pd.DataFrame({"genre": genres, "counts" : [rev.loc[:,g].sum() for g in genres]})
    return genre_abs.genre[genre_abs.counts.idxmax()]


# In[ ]:


favourite_genres = pd.DataFrame({"author" : authors, "favourite_genre" : [favourite(aut) for aut in authors]})
favourite_genres


# Let's think of the set of reviews as the disjoint union of "favourite reviews" and "non-favourite reviews". A "favourite review" is a review of an album that
# the reviewer categorized under his favourite subgenre. A "non-favourite review" is a review of an album that is categorized under different subgenres. For example 
# every review of author "A" of an album with label "modern_metal" (among other genres) is considered a favourite review.
# 
# Now, let's plot the distributions of favourite and non-favourite reviews:

# In[ ]:


favourite_ratings = pd.Series([])
non_favourite_ratings = pd.Series([])
for aut in authors:
    fav_gen = favourite(aut)
    favourite_ratings = favourite_ratings.append(data[(data.loc[:,fav_gen] == 1) & (data.loc[:,"AUTHOR"] == aut)].RATING)
    non_favourite_ratings = non_favourite_ratings.append(data[(data.loc[:,fav_gen] != 1) & (data.loc[:,"AUTHOR"] == aut)].RATING)
    
print('number of "favourite reviews": ',len(favourite_ratings))
print('number of "non-favourite reviews": ',len(non_favourite_ratings))

fav_cnts = pd.Series([sum(favourite_ratings == i) for i in range(11)])
non_fav_cnts = pd.Series([sum(non_favourite_ratings == i) for i in range(11)])

plt.figure(figsize=(14,10))
a = sns.distplot(favourite_ratings, bins=[0,1,2,3,4,5,6,7,8,9,10,11], hist=True, kde=False, norm_hist=True, label="favourite")
a = sns.distplot(non_favourite_ratings, bins=[0,1,2,3,4,5,6,7,8,9,10,11], hist=True, kde=False, norm_hist=True, label="non-favourite")
plt.title("relative frequency of ratings when reviewing favourite vs. non-favourite genres")
a = plt.legend()


# The distributions look somewhat similar but we can clearly see that ratings between 3 and 6 occur more frequently for non-favourite reviews. Conversely, ratings 
# between 7 and 9 occur more frequently for favourite reviews. Let's check if there really is a higher probability of giving an album in your favourite genre a good grade 
# than an album in your non-favourite genres. For this we use a Mann-Whitney U test. Even though the sample size is fairly large I don't think it is justified to use
# a traditional **t-test** for two reasons:
# 1. The probability distribution isn't very normal (as seen above)
# 2. We can't really be sure if our grading data is proper interval data
# 
# The good thing about **Mann-Whitney U** is that it doesn't assume normality of your data and works well with ordinal data. The null-hypothesis is that there is no difference
# in probability of giving good grades, whether an album belongs to your favourite or non-favourite genres. The alternative is that reviews in your non-favourite genres are
# less likely to get good grades.

# In[ ]:


stat, p_value = mannwhitneyu(non_favourite_ratings,favourite_ratings, alternative="less")

if p_value < 0.05:
    print("We reject the null-hypothesis with p_value ",p_value)
else:
    print("We confirm the null-hypothesis with p_value ", p_value)


# So as one might have suspected, a reviewer is more likely to give good ratings to albums in his favourite genre. 
# 
# This concludes my first glimpse into the data of heavy metal music reviews and as I'm writing this there are many more 
# ideas coming to my mind, what one could do with such data. If you made it this far I hope you had as much fun reading this
# analysis as I had in writing it. Thank you for reading and please let me know what you think in the comments.
# 
# ## Up the Irons! 

# 
