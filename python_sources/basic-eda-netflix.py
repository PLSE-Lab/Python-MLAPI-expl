#!/usr/bin/env python
# coding: utf-8

# Let's dig in! Give a thumbs up if you liked it! 
# 
# Let me know if you have some suggestions and other kind of analysis along similar lines. 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv("../input/Netflix Shows.csv", encoding='cp437')
data.head(10)


# ## NaN Check. 

# Lets check some data sanity. Number of NaN in the data and the columns contributing.

# In[ ]:


data.isnull().sum()


# ## NaN Handling.

# In[ ]:


data['user rating score'].dropna().plot(kind = "density")


# The data is not in normal distribution. Hence using mean if we were to replace NaN is not a good strategy. We need to replace by
# median! Let's check the difference. 

# In[ ]:


print("The median: {}".format(np.median(data['user rating score'].dropna())))
print("The mean : {}".format(np.mean(data['user rating score'].dropna())))


# For simplicity lets fill in the Nan of ratingLevel and user rating score with their respective median scores. 
# For ratingLevel since it is a categorical variable, we will take the sum of values and replace NaN with highest occurring value of those counts. 
# 
# 

# In[ ]:


user_rating_median = np.median(data['user rating score'].dropna())
data['user rating score'] = data['user rating score'].fillna(user_rating_median)


# In[ ]:


data['ratingLevel'].value_counts().head()


# In[ ]:


fillna_ratingLevel = "Parents strongly cautioned. May be unsuitable for children ages 14 and under."
data['ratingLevel'] = data['ratingLevel'].fillna(fillna_ratingLevel)


# In[ ]:


#Now after substituing lets check if everything has been done well. 
data.isnull().sum()


# ### Is user rating size necessary?

# Let's first see if the last column : user rating size has any variation. With a naked eye it doesn't seem like it!
# 

# In[ ]:


data['user rating size'].plot(kind = "density")


# Just a bi-modal data. If we were doing some prediction then this information could have been useful but in EDA,
# I dont think we can get any cool patterns from this. 
# 
# Let me know if you found any cool patterns. 
# I'm removing this feature for now. 
# 

# In[ ]:


data = data.drop('user rating size', axis=1)


# ## Year to no of movies released. 

# Let's first check for correlation between the two.

# In[ ]:


movies_in_year = data["release year"].value_counts().to_frame().reset_index()
movies_in_year.columns = ['release year', 'release number']

movies_in_year = movies_in_year.sort_values('release year', ascending = False)


# #### Correlation. 

# In[ ]:


movies_in_year.corr()


# In[ ]:


sns.barplot(x = "release year", y = 'release number', data = movies_in_year.head(15))


# Looks like there is a clear trend in the year and no of movies released in that year. Infact in the last two years, there seems to be an exponential increase!
# 
# We yet do not know upto which month this dataset is valid for in 2017. Hence considering 2017 for this wouldn't be fair. But, even then, it is almost close to number of movies released in 2015!

# ## Custom Scores!

# With an assumption that the ratings present were rating by adults (Obviously.... DUH!)
# Lets create a scoring system as below and see if people really like adult content.
# 
# 
# 

# In[ ]:


data['ratingLevel'].unique()


# ### Let's create a "adult-ness" score of a film!

# lets create a scoring system where we increment score if a word if in the list of following words. 
# Lets consider the following as words. 
# 

# In[ ]:



words = ['rude', 'sex', 'scary', 'violence', 'sex_related', 'adult', 'drug', 'sexual', 'nudity', 'parents', 'children']
#adding parents, children since it is indicative that this isn't for children.

def compute_score(data):
    score = 0
    for i in words:
        if(i in data):
            score = score + data.count(i)
    return(score)

data['adult_score'] = [compute_score(datum) for datum in data['ratingLevel']]


# In[ ]:


data['adult_score'].plot(kind = "density")


# In[ ]:


data['adult_score'].value_counts().plot(kind = "bar")


# Looks like people make sure there is some amount of adultness in films! 
# 
# And next obviously you need films for children and hence >300 instances where we see no adult-ness score!

# ### What kind of films are there?

# In[ ]:


data[data['adult_score'] > 3]


# I have seen that for score=0 there are many movies which were not rated. Hence lets filter those first and see which ones have 0.
# 

# In[ ]:



data_adultscore = data.copy()
data_adultscore = data_adultscore[data_adultscore['ratingLevel'] != "This movie has not been rated."]
data_adultscore[data_adultscore['adult_score'] ==0].head(10)


# In[ ]:


del(data_adultscore)


# ## And.... ofcourse the word cloud! 

# In[ ]:


from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='white',
                          width=1500,
                          height=1500
                         ).generate(" ".join(data['title']))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

