#!/usr/bin/env python
# coding: utf-8

# # DATS6103 - Intoduction to Data Mining
# ## Individual Project 2
# 
# ### By Fahim Ishrak
# 

# # Topic - Factors that Determines the Success of a Movie

# # Part 1- Loading the data and Pre-processing

# In[ ]:


#Importing Relevant Packages
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
rc = {'axes.facecolor':'white',
      'axes.grid' : True,
      'grid.color': '.8'}
plt.rcParams.update(rc)
from scipy import stats
from ast import literal_eval
import math
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ## The following csv file was downloaded from https://www.kaggle.com/rounakbanik/the-movies-dataset#movies_metadata.csv 
# ### This dataset is an ensemble of data collected from TMDB (The Movie DataBase) and GroupLens. The Movie Details, Credits and Keywords have been collected from the TMDB Open API. This dataset is made using the TMDb API and their API also provides access to data on many additional movies, actors and actresses, crew members, and TV shows. The API source can be found here https://www.themoviedb.org/documentation/api

# In[ ]:


# Reading the CSV files into the notebook

md = pd.read_csv('../input/movies_metadata.csv')
md.head()


# In[ ]:


md.info()


# In[ ]:


# Dropping the unnecessary columns 
md.drop(['adult','belongs_to_collection','homepage','imdb_id','original_title','overview','poster_path','tagline','video','status'],axis=1,inplace = True)


# In[ ]:


#Extracting the year and month to a seperate column from release_date

md['year'] = np.nan
md['month'] = np.nan
for j in range(0,len(md['release_date'])):
    if md['release_date'][j] != md['release_date'][j]:
        pass
    elif len(md['release_date'][j]) != 10:
        md.drop(j,axis=0,inplace=True)
    else:
        md['year'][j] = datetime.strptime(md['release_date'][j], '%Y-%m-%d').year
        md['month'][j] = datetime.strptime(md['release_date'][j], '%Y-%m-%d').month
        
md.drop(['release_date'],axis=1,inplace = True)


# In[ ]:


#Resetting index to continue pre-processing
md.reset_index(drop = True,inplace=True)


# In[ ]:


#Converting the dictionary style columns into lists

md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md['production_companies'] = md['production_companies'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md['production_countries'] = md['production_countries'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md['spoken_languages'] = md['spoken_languages'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[ ]:


md.head()


# In[ ]:


#Saving the first value of the list only from the genres column
for j in range(0,len(md['genres'])):
    if len(md['genres'][j]) ==0:
        md['genres'][j] = np.nan
    else:
        md['genres'][j] = md['genres'][j][0]


# In[ ]:


#Saving the first value of the list only from the production_companies column
for j in range(0,len(md['production_companies'])):
    if len(md['production_companies'][j]) ==0:
        md['production_companies'][j] = np.nan
    else:
        md['production_companies'][j] = md['production_companies'][j][0]


# In[ ]:


#Saving the first value of the list only from the production_countries column
for j in range(0,len(md['production_countries'])):
    if len(md['production_countries'][j]) ==0:
        md['production_countries'][j] = np.nan
    else:
        md['production_countries'][j] = md['production_countries'][j][0]


# In[ ]:


#Saving the first value of the list only from the spoken_languages column
for j in range(0,len(md['spoken_languages'])):
    if len(md['spoken_languages'][j]) ==0:
        md['spoken_languages'][j] = np.nan
    else:
        md['spoken_languages'][j] = md['spoken_languages'][j][0]


# In[ ]:


md.head()


# In[ ]:


#Dropping duplicate values
md.drop_duplicates(inplace=True)


# In[ ]:


#Dropping NaN values from these specific columns
col = ['budget','genres','id','original_language','popularity','year','revenue',
       'runtime','spoken_languages','title','vote_average','vote_count']
md.dropna(subset = col,how = 'any',inplace=True)


# In[ ]:


md.info()


# In[ ]:


#Setting the proper data type
md = md.astype({'budget':'float64'})
md = md.astype({'popularity':'float64'})
md = md.astype({'year':'int64'})
md = md.astype({'month':'int64'})


# In[ ]:


#Selecting movies after 1970
md = md.query('year > 1970 & year <2017')


# In[ ]:


# Replacing 0 with NaN 
md['budget'] = md['budget'].replace(0,np.nan)
md['revenue'] = md['revenue'].replace(0,np.nan)


# In[ ]:


md.head()


# In[ ]:


#Copying the dataset and dropping the remaining NaN values so that profit can be calculated accurately
md1 = md.copy()
md1.dropna(inplace=True)
md1['profit'] = md1['revenue'] - md1['budget']


# In[ ]:


md1.head()


# # Part 2- Visualization and Observations
# 

# ### Before Diving into the successful films, let's see some general figures at first

# In[ ]:


# Plotting Number of Movies Released Per Year
plt.figure(figsize=(25, 10))
plt.title("Number of Movies Released Per Year")
sns.countplot(x="year", data=md,palette = 'Blues')
plt.ylabel('Number of Movies')
plt.show()


# #### It is without question that the number of movies released per year is on the rise and will continue to follow this trend

# In[ ]:


# Plotting Number of Movies Released According to Genre
plt.figure(figsize=(23, 10))
plt.title("Number of Movies Released According to Genre")
sns.countplot(x="genres", data=md,order = md['genres'].value_counts().index)
plt.ylabel('Number of movies')
plt.show()


# #### Drama is the most produced film followed by Comedy, Action, Documentary, Horror and so on 

# In[ ]:


# Plotting Number of Movies Released According to Genre
plt.figure(figsize=(20, 10))
plt.title("Number of Movies Released by the Top Ten Countries")
sns.countplot(x="production_countries", data=md,order = md['production_countries'].value_counts()[0:10].index)
plt.xlabel('Countries')
plt.ylabel('Number of movies')
plt.show()


# #### The United States produced the most film while the rest produced less than 1/6th of the film compared to it

# In[ ]:


# Plotting Number of Movies Released According to Genre
plt.figure(figsize=(20, 10))
plt.title("Number of Movies by the Top Ten Language")
sns.countplot(x="original_language", data=md,order = md['original_language'].value_counts()[0:10].index)
plt.xlabel('Language')
plt.ylabel('Number of movies')
plt.show()


# #### Once again it's no surprise that most films are released in English, however the next ones in order are French, Japanese, Italian, Spanish and so on 

# In[ ]:


# Plotting Number of Movies Released According to Genre
plt.figure(figsize=(30, 10))
plt.title("Number of Movies Released by Top Ten Production Companies")
plt.xlabel('Production Companies')
plt.ylabel('Number of movies')
sns.countplot(x="production_companies", data=md,order = md['production_companies'].value_counts()[0:10].index)
plt.show()


# #### Universal Pictures and Paramount Pictures are close, followed by Twentieth Century fox, Columbia Picture, Warner Bros and so forth

# ### Now lets look at what effects the success of a movie

# ### The key variables to use in order to find out the success of a film would be the  profit, popularity and average vote. Lets see how they relate to each other at first.
# 

# In[ ]:


# Plotting Number of Movies Released According to Genre

plt.figure(figsize=(25, 15))
plt.subplot(311)
plt.title("Scatter plot of Average Vote and Popularity")
plt.ylim(0,80)
sns.regplot(x='vote_average', y ='popularity',data=md)


plt.subplot(312)
plt.title("Scatter plot of average vote and Profit")
#sns.regplot(x='revenue', y ='vote_average',data=md)
sns.regplot(x='profit', y ='vote_average',data=md1)


plt.subplot(313)
plt.title("Scatter plot of Profit and Popularity")
plt.ylim(0,80)
sns.regplot(x='profit', y ='popularity',data=md1)

plt.show()


# #### There is a clear correlation with the revenue generated and popularity but a comparatively small one of them with the vote average

# ### Let's see the general trend of the success of the movie industry as a whole

# In[ ]:


plt.figure(figsize=(25, 15))
plt.subplot(311)
plt.title("Popularity Trend")
sns.lineplot(x='year', y ='popularity' ,data=md, ci = None,color = 'r')


plt.subplot(312)
plt.title("Average Vote Trend")
sns.lineplot(x='year', y ='vote_average',data=md, ci = None)


plt.subplot(313)
plt.title("Profit Trend")
sns.lineplot(x='year', y ='profit',data=md1, ci = None,color = 'g')

plt.show()


# #### There are some peaks and troughs but each of the success factors are on the increase. So a movie released today is more likely to be successful than a movie released back in 1970

# ### Now let's see how each of the factor affects the success of a movie, starting with month

# In[ ]:


plt.figure(figsize=(25, 15))
plt.subplot(311)
plt.title("Average Vote with Genre")
sns.barplot(x='month', y ='vote_average',data=md,estimator=np.median,ci=None,palette = ('Paired'))


plt.subplot(312)
plt.title("Profit with Genre")
sns.barplot(x='month', y ='profit',data=md1,estimator=np.median,ci=None,palette = ('Paired'))


plt.subplot(313)
plt.title("Popularity with Genre")
sns.barplot(x='month', y ='popularity',data=md,estimator=np.median,ci=None,palette = ('Paired'))

plt.show()


# #### The vote average remains constant throughout the year.
# #### However the films released for the month of June(6), July(7) and December(12) tend to have a higher profit.
# #### Also the films released for the month of July(7) and December(12) tend to have a higher popularity.

# ### Now Let's see how genre effects the success of a film 

# In[ ]:


plt.figure(figsize=(25, 15))
plt.subplot(311)
plt.title("Average Vote with Genre")
sns.barplot(x='genres', y ='vote_average',data=md,estimator=np.median,ci=None)


plt.subplot(312)
plt.title("Profit with Genre")
sns.barplot(x='genres', y ='profit',data=md1,estimator=np.median,ci=None)


plt.subplot(313)
plt.title("Popularity with Genre")
sns.barplot(x='genres', y ='popularity',data=md,estimator=np.median,ci=None)

plt.show()


# #### Once again the average vote remains constant for each genre
# #### Animation and Adventure is the most profitable genre
# #### Adventure has the highest popularity, followed by Fantasy, Action, Animation

# ### Does the length of the movie have any noticeable effect?

# In[ ]:


plt.figure(figsize=(25, 15))
plt.subplot(311)
plt.title("Average vote with length of the movie")
plt.xlim(60,200)
sns.lineplot(x='runtime', y ='vote_average',data=md,ci=None)


plt.subplot(312)
plt.title("profit with length of the movie")
plt.xlim(60,200)
sns.lineplot(x='runtime', y ='profit',data=md1,ci=None)


plt.subplot(313)
plt.title("popularity with length of the movie")
plt.xlim(60,200)
sns.lineplot(x='runtime', y ='popularity',data=md,ci=None)

plt.show()


# #### There aren't any noticeable relation with the length and the success of a movie expect for the uptick of profit around the 190 min mark 

# ### Now let's take a look at the most successful and the least successful production companies
# ### Starting with the top ten successful ones 
# 

# In[ ]:


PC_by_profit = md1[['profit','production_companies']]
PC_by_profit.groupby(['production_companies']).median().sort_values(by = ['profit'],ascending = False).head(10).plot(kind = 'barh',figsize=(10,5))
plt.show()


# In[ ]:


PC_by_popularity = md[['popularity','production_companies']]
PC_by_popularity.groupby(['production_companies']).median().sort_values(by = ['popularity'],ascending = False).head(10).plot(kind = 'barh',figsize=(10,5))
plt.show()


# ### and the bottom ten successful ones 

# In[ ]:


PC_by_profit.groupby(['production_companies']).median().sort_values(by = ['profit'],ascending = False).tail(10).plot(kind = 'barh',figsize=(10,5))
plt.show()


# In[ ]:


PC_by_popularity.groupby(['production_companies']).median().sort_values(by = ['popularity'],ascending = False).tail(10).plot(kind = 'barh',figsize=(10,5))
plt.show()


# # Part 3- Conclusion
# ### The movie industry is truly robust and success depends on a lot of factors. The above insights gives us a sense of how some of the variables tend to be associated with success. If someone is deciding to make or invest in the movie industry then certain factors such as choosing the adventure or animation genre, releasing the movie in the month of June or July or December, using good production companies like Heydey Films or 1492 Pictures  and finally keeping the original language as English will likely make a good movie  

# In[ ]:




