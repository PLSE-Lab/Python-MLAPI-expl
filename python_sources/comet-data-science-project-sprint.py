#!/usr/bin/env python
# coding: utf-8

# ## INTRODUCTION ##
# 
# 
# ----------
# 
# 
# Films or movies are one of the primary source of entertainment nowadays. It ranges from several genres having a number of directors and actors. The dataset used contains data regarding movies from the past 100 years from 1916 - 2016. The focus of this project is the budget used for the list of films. It aims to study the factors that are affected by the budget and the other variables related to it. Through graphical interpretation and explanations the goal is to answer the following questions:
# 
# 1. What genre has been most prevalent for the past years?
# 2. Do higher budgeted films result to higher ratings?
# 3. Do higher budgeted films generate more income?
# 
# The findings will be of relevance for the film makers as it may aid them to acquire more knowledge on the importance of  allotting just enough budget to make a good film while ensuring a decent income and the factors constituting to this.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns;
data_set = pd.read_csv("../input/movie_metadata.csv")
data_set = data_set.dropna()
data_set


# In[ ]:


#Question 1
data_set['genres'].value_counts()


# In[ ]:


#Question 2
bs_plot = sns.regplot(x="budget", y="imdb_score", data=data_set)


# In[ ]:


#Question 2
bs = {'Budget': data_set['budget'], 'IMDB_score': data_set['imdb_score']}
dataframe_budget_score = pd.DataFrame(bs)
dataframe_budget_score.corr('pearson')


# In[ ]:


#Question 3
bg_plot = sns.regplot(x="budget", y="gross", data=data_set)


# In[ ]:


#Question 3
bg = {'Budget': data_set['budget'], 'Gross': data_set['gross']}
dataframe_budget_gross = pd.DataFrame(bg)
dataframe_budget_gross.corr('pearson')


# ## DISCUSSION ##
# 
# 
# ----------
# 
# 
# Movies as the years progress has been gaining more and more popularity and is constantly growing in numbers. Directors, actors and filmmakers are increasing tenfold. The money and time allotted to these films should be given utmost importance and careful thought as this constitutes to the living and income of many. 
# 
# Removing the null values in the dataset, the findings were as follows: 
# The most prevalent genre of film made known to the public is Drama, as seen on the frequency count for the 'genres' column . Based on the findings illustrated above, the comparison between the two variables for question two and three showed a very weak positively increasing value for their pearson's r. This means that the correlation between the budget of the film with regards to the gross and ratings of the movie is not strong. Moreover, this signifies that these two factors are not entirely related and has no strong dependency on one another. Allotting more budget to a film does not ensure popularity, high ratings and more income for movies. Other factors should be looked at and studied in order to determine which ones could have a much stronger correlation with a movie's budget.
