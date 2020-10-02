#!/usr/bin/env python
# coding: utf-8

# <h1 style="font-size:30px; color:SteelBlue; text-align:center; margin-top:50px">Effect of Recession on Movie Genre</h1><hr>
# <p><span style="font-size:18px; color:black; margin-left:30px; margin-bottom:50px">The research question is to find whether there is a difference in the correlations between the genres of movies in recession years compared to non-recession years in U.S. (years during which U.S. experienced recession taken from wikipedia)</span></p>
# <p><span style="font-size:18px; color:black; margin-left:30px; margin-bottom:50px">Do the movies made in recession years show any changes in genre to reflect the prevailing sentiment or target particular group of audience who might more likely to watch movies given there are reduced jobs and more ideal time for people to engage in recreation tasks to keep the spirits high.</p><hr>
# 

# ### Importing libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plot
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ** Loading data using Pandas, ignoring the lines which give error**

# In[ ]:


get_ipython().run_cell_magic('capture', '', "data = pd.read_csv('../input/imdb.csv',error_bad_lines=False);")


# ## Getting ratings for only the Movies Category from IMDB movie dataset stored with variable name 

# In[ ]:


movie_data = data[data['type']=='video.movie']


# ## Plotting the rating against years in scatter plot

# In[ ]:


plot.figure(figsize=(11,8))
plot.scatter(movie_data['year'], movie_data['imdbRating']);


# <p style="font-size:20px; margin-bottom:20px;"><span style="color:royalblue">Changes in rating pattern:</span></p>
# <p><span style="font-size:18px; color:black; margin-left:20px; margin-bottom:50px">With more number of movies available for the viewers to watch, there is increase in count of movies which got less ratings as viewers expectation increased by late 90's.</span></p><hr>

# ## List of years during which U.S. experience recession (source wikipedia.org)

# In[ ]:


reces_years = [2007, 2008, 2009, 2003, 2002, 1991, 1990, 1981, 1980,
               1975, 1974, 1973, 1970, 1969, 1961, 1960, 1954, 1953, 
               1949, 1938, 1937, 1930, 1929, 1928]


# ## Creating DataFrame excluding recession years

# In[ ]:


remains_summed = pd.DataFrame()

# Count for movies with rating in the dataset in a given year
for i in range(1888,2018):
    if i in reces_years:
        continue
    try:
        remains_summed.at[i,'num_movies_with_rating'] =         round(movie_data[movie_data['year'] == i]['year'].count())
    except ValueError:
        continue


# In[ ]:


# creating column for average movie rating received for a given year
for i in range(1888,2018):
    if i in reces_years:
        continue
    try:
        remains_summed.at[i,'ave_rating_in_the_year'] =         round(movie_data[movie_data['year'] == i]['imdbRating'].mean())
    except ValueError:
        continue
        


# ## Most common movie ratings in non-recession years

# In[ ]:


g = sns.countplot(x='ave_rating_in_the_year',data=remains_summed)


# ## Creating dataframe including all the years

# In[ ]:


all_years = pd.DataFrame()

# Count for number of movies in a given year
for i in range(1888,2018):
    try:
        all_years.at[i,'num_movies_with_rating'] =         round(movie_data[movie_data['year'] == i]['year'].count())
    except ValueError:
        continue
 
#Average movie ratings for movies in a given year
for i in range(1888,2018):
    try:
        all_years.at[i,'ave_rating_in_the_year'] =         round(movie_data[movie_data['year'] == i]['imdbRating'].mean())
    except ValueError:
        continue


# ## Most common movie ratings for all the years between 1888 and 2017

# In[ ]:


sns.countplot(x='ave_rating_in_the_year',data=all_years)


# > ## Using selected which are more watched genre list for furthur analysis

# In[ ]:


genre_list = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography',
       'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
       'FilmNoir', 'GameShow', 'History', 'Horror', 'Music', 'Musical',
       'Mystery', 'News', 'RealityTV', 'Romance', 'SciFi', 'Short', 'Sport',
       'TalkShow', 'Thriller', 'War', 'Western']


# ## Creating genre counts of the movies released in a non-recession year

# In[ ]:


for k in genre_list:
    for i in range(1888,2018):
        if i in reces_years:
            continue
        try:
            remains_summed.at[i,k] =             (movie_data[movie_data['year'] == i][k]==1).sum()
        except ValueError:
            continue


# ## Creating correlation graph for genres

# In[ ]:


# calculating correlations between genres
correlations = remains_summed[genre_list].corr()

# creating mask to plot only lower triangle of the heatmap
mask = np.zeros_like(correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# plotting the heatmap
plot.figure(figsize=(20,12))
g = sns.heatmap(correlations*100, fmt='.0f', mask= mask, 
                cbar=False, cmap='coolwarm',annot=True,
                vmax=80);

# saving the heatmap
#g.figure.savefig('non_recess_year_heatmap.png')


# ** Observe the correlation of Adult genre with other genres, it shows weak positive correlation**
# <hr>

# In[ ]:


#adult genre correlation matrix
k = 9 #number of variables for heatmap
cols = correlations.nlargest(k, 'Action')['Action'].index
#cm = np.corrcoef(remains_summed[cols].values.T)

sns.set(font_scale=1.25)
sns.set_style('white')

mask = np.zeros_like(remains_summed[cols].corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

hm = sns.heatmap(remains_summed[cols].corr()*100, cbar=False, annot=True, square=True,
                fmt='.0f', annot_kws={'size': 15},
                 mask=mask)


# ## Clustermap of the genres

# In[ ]:


plot.figure(figsize=(25,18));
g = sns.clustermap(correlations, cmap='coolwarm');


# # Analyzing the recession years for change in correlation and clustering of the genres (recession years in U.S. as per wikipedia.org)

# ## Creating number of movies with rating and average rating in a given year

# In[ ]:


# creating an empty DataFrame
recession_years = pd.DataFrame()

# creating column with movies with rating for a given year
for i in reces_years:
    try:
        recession_years.at[i,'num_movies_with_rating'] =         round(movie_data[movie_data['year'] == i]['year'].count())
    except ValueError:
        continue

# creating column with average rating for the movies in a given year
for i in reces_years:
    try:
        recession_years.at[i,'ave_rating_in_the_year'] =         round(movie_data[movie_data['year'] == i]['imdbRating'].mean())
    except ValueError:
        continue


# ## Most common ratings in recession years

# In[ ]:


g = sns.countplot(x = 'ave_rating_in_the_year', data= recession_years);


# ## Generating count for genres of the movies in a given year

# In[ ]:


# Calculating the total count for genres in which all the
# movies in the year belongs too

for k in genre_list:
    for i in reces_years:
        try:
            recession_years.at[i,k] =             (movie_data[movie_data['year'] == i][k]==1).sum()
        except ValueError:
            continue


# ## Creating heatmap for movies genres released in the recession years, excluding the 'game show' genre

# In[ ]:


# creating list of genres to calculate correlation between them 
# excluding 'game show' (has no data for recession years)
genre_list = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography',
       'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
       'FilmNoir', 'History', 'Horror', 'Music', 'Musical',
       'Mystery', 'News', 'RealityTV', 'Romance', 'SciFi', 'Short', 'Sport',
       'TalkShow', 'Thriller', 'War', 'Western']

# creating correlations for the genres
correlations = recession_years[genre_list].corr()

# creating mask to display only lower left triangle for the heatmap
mask = np.zeros_like(correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# plotting the heatmap
plot.figure(figsize=(20,12))
g = sns.heatmap(correlations*100, fmt='.0f', mask=mask,
                cbar=False, cmap='coolwarm', annot=True)


# **The Adult genre's correlations changed from weak positive to strong positive rates in recession periods**
# <hr>

# <h1 style="font-size:30px; color:SteelBlue; text-align:center">Conclusion</h1><hr><br>
# 
# <p><span style="font-size:20px; color:black; margin-left:40px">The Genre which stands out the most is the Adult Genre. The correlation for adult genre during non-recession years has very weak positive correlation with other genres. But during the recession years the Adult genre has strong positive correlation with other genres. It could be thought as during recession years adult audience are target audience for movie makers, given more adults would be free due to reduced jobs in economy and more likely to engage in recreation tasks and watching movies would be a top choice for many people to keep the spirits high.</span><p>
# 
# <p><span style="font-size:20px; color:black; margin-left:40px">And other genres which show differences are War, Western, Filmnoir, News and Reality TV. And remaining genres show more or less similar correlation in both the periods.</span><p><hr>
