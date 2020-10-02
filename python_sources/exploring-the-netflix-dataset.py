#!/usr/bin/env python
# coding: utf-8

# # Exploring the Netflix Dataset 

# Netflix has undoubtetdly grown to become an entertainment giant. Over the past few years, they've added content on a regular basis, with new titles released seemingly overnight. The dataset that we'll be exploring contains metadata on Netflix movies and TV shows. Exploring factors such as the number of movies released over time, the average duration of movies, and the top actors and actresses can give us an idea of how content has evolved on Netflix over time.  

# In[ ]:


#First, we must import libraries needed for analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#read the dataset
netflix = pd.read_csv('../input/netflix-shows-exploratory-analysis/netflix_titles.csv')
netflix.head()


# Each row in the dataframe represents a title on Netflix. The "show_id" column corresponds to that title's unique id. There are many factors we could explore, but for this analysis let's focus mainly on the "country", "cast", and "date_added" columns.

# ### Questions to explore

# #### In the United States, how many movies have been added, per year, since 2010?

# Since we want to break down the number of movies added per year, we'll use the "date_added" column.

# In[ ]:


netflix['date_added'].head()


# We can see that the data type (dtype) for the date_added column is an object. Pandas is handling the values as Python strings. We'll want to convert those values to datetime if we want them to be treated as dates.

# In[ ]:


netflix['date_added'] = pd.to_datetime(netflix['date_added'])
netflix['date_added'].head()


# With the data in the correct format, we can group the rows in the dataframe by year.

# In[ ]:


#we can filter by all movies released in the United States since 2010, and get the counts
movies_since_2010 = netflix[(netflix.date_added >= '2010-01-01') & (netflix.type == 'Movie')
                           & (netflix.country.str.contains('United States', case=False))]
movies_by_year = movies_since_2010.groupby(movies_since_2010.date_added.dt.year).show_id.count()

fig, ax = plt.subplots()

ax.bar(movies_by_year.index, movies_by_year)
ax.spines['right'].set_visible(False) #removing spines to minimize chartjunk
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_title('Movies added to Netflix (Released in U.S.)')
ax.tick_params(left=True, bottom=False)
ax.grid(False)
plt.show()


# Its no surprise that the increasing popularity of Netflix has coincided with the addition of more movie titles over the years. The number of titles added per year has increased since 2014. We can assume that the trend will continue in 2020.

# #### Which country has the most movie releases?

# We can gauge the popularity of Netflix in each country by examining the number of releases in each country. First, let's examine the column we want to break down: 'country'.

# In[ ]:


netflix[['title', 'country']].head()


# If a title is released in multiple countries, it is represented as one row in the dataframe. We actually want one row for each country and title. For example, in the first row in the above dataframe, we would like the "country" values to be separated into four rows: one for the United States, India, South Korea, and China. This will allow us to break down the number of releases by each country.

# I've found a function via stackoverflow that will be useful. For our purposes, each country in the "country" column will be stacked as a new row in the resulting dataframe. 

# In[ ]:


#from stackoverflow
def explode(df, lst_cols, fill_value='', preserve_index=False):
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values    
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:        
        res = res.reset_index(drop=True)
    return res


# In[ ]:


#since countries appear as comma separated values in the country column, we need to separate the values, and create new rows for each value
new = netflix.copy() #create a copy of the original dataframe
new = new.replace(np.nan, '', regex=True)
new['country'] = new.country.str.split(',')
new = explode(new, ['country']) #new dataframe with new rows added for each country
new['country'] = new.country.str.strip()
new[['title', 'country']].head()


# Now, each country has its own row in the dataframe, even if it is the same title. We are now able to group by the individual country.

# In[ ]:


results = new[new.type == 'Movie'].groupby('country').show_id.count().sort_values(ascending=True)[-11:] #group by the country column, get top ten countries by movie releases
results = results.drop(labels='') #dropping rows with blank values

fig, ax = plt.subplots()

ax.barh(results.index, results)
ax.set_title('Number of Netflix movies added')
ax.spines['right'].set_visible(False) #removing spines to minimize chartjunk
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(left=False, bottom=False) #remove ticks
ax.grid(False) #remove grid

plt.show()


# The United States has the most number of movies added on Netflix by a wide margin.

# #### What is the average duration of all movies?

# Does Netflix tend to add shorter or longer movies? Let's look at the "duration" column.

# In[ ]:


netflix[netflix.type == 'Movie'].duration.head()


# Since we want to treat the duration column as time values, we should remove the "min" from all of the rows. At first glance, we can apply the str.split() method to each row, and return the first element. 

# In[ ]:


def to_minutes(series): #function to return only the first element
    return series.split()[0]
    
        
netflix['duration'] = netflix.duration.apply(lambda x: to_minutes(x))


# In[ ]:


netflix[netflix.type == 'Movie'].duration.head()


# Now, the values in the duration column are formatted correctly.

# In[ ]:


netflix['duration'] = netflix.duration.astype('int') #to determine the average duration, must convert to int


# In[ ]:


netflix[netflix.type == 'Movie'].duration.mean() #average duration for all movies in the dataset


# The average duration for all movies released on Netflix is 99 minutes. Visualizing the distribution of movie durations can give us a more clear picture of the data.

# In[ ]:


sns.set(style='whitegrid')

sns.distplot(netflix[netflix.type == 'Movie'].duration).set_title('Distribution of Movie Durations')
plt.show()


# In[ ]:


sns.boxplot(netflix[netflix.type == 'Movie'].duration).set_title('Distribution of Movie Durations')
plt.show()


# The interquartile range appears to be fairly small and there are many outliers present.

# #### On average, has Netflix added shorter or longer movies over time?

# Let's examine the average duration of movies for each year.

# In[ ]:


duration_by_year = netflix[netflix.type == 'Movie'].groupby(netflix.date_added.dt.year).mean().duration
duration_by_year


# In[ ]:


fig, ax = plt.subplots()

ax.plot(duration_by_year)
ax.grid(False)
ax.set_title('Average duration of Netflix movies')
ax.spines['right'].set_visible(False) #removing spines to minimize chartjunk
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlabel('Year')
ax.set_ylabel('Average Duration')
plt.show()


# We can see that Netflix has added longer movies over time, on average.

# #### Which actors have appeared in the most titles?

# In[ ]:


netflix[['title','cast']].head()


# We have the same problem as when we examined the "country" column. Each cast member must be separated into individual rows. We can use the same explode function as before.

# In[ ]:


cast_df = netflix.copy()
cast_df = cast_df.dropna(axis=0, how='any', subset=['cast'])
cast_df['cast'] = cast_df.cast.str.split(',')
cast_df = explode(cast_df, ['cast']) #new dataframe with new rows added for each country
cast_df[['title', 'cast']].head()


# With a new "exploded" dataframe with a separate row for each cast member, we can group by the cast members.

# In[ ]:


cast_df.groupby('cast').count()


# With the data formatted as we need, we can now group by each individual cast member.

# In[ ]:


cast_df_filtered = cast_df[(cast_df.cast != '') & cast_df.country.str.contains('United States', case=False)]
grouped_cast = cast_df_filtered.groupby('cast').count().show_id.sort_values()[-11:] #get top ten cast members
grouped_cast


# In[ ]:


fig, ax = plt.subplots(figsize=(7,5))

ax.barh(grouped_cast.index, grouped_cast)
ax.grid(False)
ax.spines['right'].set_visible(False) #remove spines to minimize chartjunk
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_title('Top Actors on Netflix')
ax.set_xlabel('Number of Titles')
ax.set_xlim(left=10) #start x-axis at 10
plt.show()


# We've explored the Netflix dataset and now have a more firm understanding of the content distribution. Some topics for further exploration could be:
# 
# * Which actors/actresses tend to appear in the same movies/tv shows?
# * What is the average IMDB rating of Netflix content?
# * What are the most popular genres on Netflix?
