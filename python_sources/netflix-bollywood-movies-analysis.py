#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Important Python Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re


# In[ ]:


# Loading of Data
dframe = pd.read_csv("../input/netflix-shows/netflix_titles_nov_2019.csv")


# In[ ]:


dframe.head()


# ## Since we are doing our analysis on Bollywood Movies, so we will separate Bollywood Data(Indian Data) from rest of the data

# In[ ]:


# Filtering of Bollywood(Indian) Data from the whole dataframe

dframe_india = dframe[dframe['country']=='India']


# In[ ]:


dframe_india.shape


# In[ ]:


dframe_india['type'].unique()


# ## Since our focus is only on Bollywood Movies, so we will filter out TV Shows from our current dataset and do analysis only on Bollywood Movies

# In[ ]:


dframe_india = dframe_india[dframe_india['type']=='Movie']


# In[ ]:


dframe_india.shape


# In[ ]:


# removal of unwanted columns

dframe_india.drop(['country','type'],axis=1,inplace=True)


# In[ ]:


dframe_india.head()


# ### In this Dataframe, there is 1 column named **rating**. We will create a dictionary based on this rating column values and will create an another column in this Dataframe based on those dictionary values

# In[ ]:


# Creating a dictionary based on rating values

Movie_list = {'TV-Y7':'Child Movies',
              'TV-G':'Family Movies',
              'TV-PG':'Family Movies-Parental Guidance',
              'TV-14':'Family Movies-Parental Guidance',
              'TV-MA':'Adult Movies','TV-Y7-FV':'Child Movies',
              'PG-13':'Family Movies-Parental Guidance',
              'PG':'Family Movies-Parental Guidance',
              'R':'Adult Movies',
              'NR':'Unrated Movies',
              'UR':'Unrated Movies'}


# In[ ]:


# Adding a column named MOVIE TYPE in the existing Dataframe based on the mapping with dictionary values 

dframe_india['Movie Type'] = dframe_india['rating'].map(Movie_list)


# In[ ]:


dframe_india.head()


# In[ ]:


sns.countplot(y='Movie Type',data=dframe_india,palette='Set1')


# In[ ]:


# List of Child Movies in this Dataframe

dframe_india[dframe_india['Movie Type']=='Child Movies'].head(5)


# In[ ]:


# List of Adult Movies in this Dataframe

dframe_india[dframe_india['Movie Type']=='Adult Movies'].head(5)


# In[ ]:


# List of Family Movies with little Parental Guidance in this Dataframe

dframe_india[dframe_india['Movie Type']=='Family Movies-Parental Guidance'].head(5)


# In[ ]:


# List of Family Movies suitable for all ages in this Dataframe

dframe_india[dframe_india['Movie Type']=='Family Movies'].head(5)


# ## Now, our next step will be to categorize movies.
# 
# ### Here is the number of unique categories present in this Dataframe :
# 
# ('Comedies', 'Dramas', 'International Movies', 'Romantic Movies',
#        'Action & Adventure', 'Independent Movies', 'Thrillers',
#        'Documentaries', 'Sports Movies', 'Music & Musicals',
#        'Children & Family Movies', 'Horror Movies', 'Sci-Fi & Fantasy',
#        'Faith & Spirituality', 'LGBTQ Movies', 'Cult Movies',
#        'Classic Movies', 'Stand-Up Comedy')

# In[ ]:


# Creating a function to separate movies in different categories based on listed_in column present in this Dataframe

def category_separator(category,show_id):
    for i in (re.split(r',',category)):
        if i.strip() in dframe_india:
            dframe_india[i.strip()][dframe_india['show_id']==show_id]='YES'
        else:
            dframe_india[i.strip()]='NO'
            dframe_india[i.strip()][dframe_india['show_id']==show_id]='YES'


# In[ ]:


# Calling of category_separator function. This Loop will run for all the rows in the Dataframe and will separate all the movies in different categories.

for show_id, category in zip(dframe_india.show_id, dframe_india.listed_in): 
    category_separator(category,show_id)


# ## As you can see below that all the movies have now been categorized according to their Genre

# In[ ]:


pd.set_option('display.max_columns', None)

dframe_india.head(4)


# # Let's do some Data Visualisation...

# In[ ]:


# Total Number Comedy Movies present on NETFLIX

sns.countplot(x='Comedies',data=dframe_india,palette='Set2')
dframe_india['Comedies'].value_counts()


# In[ ]:


# Total Number Action & Adventure Movies present on NETFLIX

sns.countplot(x='Action & Adventure',data=dframe_india,palette='Set1')
dframe_india['Action & Adventure'].value_counts()


# In[ ]:


# Total Number Romantic Movies Movies present on NETFLIX

sns.countplot(x='Romantic Movies',data=dframe_india,palette='Set1')
dframe_india['Romantic Movies'].value_counts()


# In[ ]:


# Comedy Movies present in respective MOVIE TYPE categories

sns.countplot(y='Movie Type',hue='Comedies',data=dframe_india,palette='Set2')


# In[ ]:


# Romantic Movies present in respective MOVIE TYPE categories

sns.countplot(y='Movie Type',hue='Romantic Movies',data=dframe_india,palette='Set1')


# In[ ]:


# Horror Movies present in respective MOVIE TYPE categories

sns.countplot(y='Movie Type',hue='Horror Movies',data=dframe_india,palette='Set1')


# ## Now, Let's Play Around with this Dataframe..!!!!

# In[ ]:


# Comedy Bollywood Movies which we can see on NETFLIX

dframe_india['title'][dframe_india['Comedies']=='YES'].head(10)


# In[ ]:


# Action & Adventure Bollywood Movies which we can see on NETFLIX

dframe_india['title'][dframe_india['Action & Adventure']=='YES'].head(10)


# In[ ]:


# Romantic Movies Bollywood Movies which we can see on NETFLIX

dframe_india['title'][dframe_india['Romantic Movies']=='YES'].head(10)


# ## Okay let's do something more interesting !!
# ### Let's find out NON VEG COMEDY(Adult Comedy) bollywood movies present on NETFLIX :D :P

# In[ ]:


# Adult Comedy Bollywood Movies present on NETFLIX

dframe_india['title'][(dframe_india['Comedies']=='YES')&(dframe_india['Movie Type']=='Adult Movies')]


# ## Now, Let's find out Family Comedy Movies as well

# In[ ]:


# Family Comedy Bollywood Movies present on NETFLIX

dframe_india['title'][(dframe_india['Comedies']=='YES')&(dframe_india['Movie Type']=='Family Movies-Parental Guidance')].head(15)


# ## Interested in HORROR COMEDY BOLLYWOOD MOVIES present on NETFLIX ?? Here's is the list :

# In[ ]:


# Horror Comedy Bollywood Movies present on NETFLIX

dframe_india['title'][(dframe_india['Comedies']=='YES')&(dframe_india['Horror Movies']=='YES')].head(10)


# ## Want to see a SPORTS MOVIE ?? Here they are :

# In[ ]:


# Sports Bollywood Movies present on NETFLIX

dframe_india['title'][(dframe_india['Sports Movies']=='YES')].head(10)


# # Want to play more with these combinations ?
# 
# ## Why don't you try it yourself. It' very simple :)
# 
# ### Just try different combinations and play with it based on the genre of your choice and see the list of bollywood movies accordingly.
# 
# ### Just choose a movie category of your choice from below list(you can even choose multiple movie categories based on your interest like 'Romantic Movies' and 'Dramas'):
# 
# ('Comedies', 'Dramas', 'International Movies', 'Romantic Movies',
#        'Action & Adventure', 'Independent Movies', 'Thrillers',
#        'Documentaries', 'Sports Movies', 'Music & Musicals',
#        'Children & Family Movies', 'Horror Movies', 'Sci-Fi & Fantasy',
#        'Faith & Spirituality', 'LGBTQ Movies', 'Cult Movies',
#        'Classic Movies', 'Stand-Up Comedy')
#        
# ### Also, you can combine the above categories with Movie Type by selecting a type from below:
# ('Family Movies', 'Child Movies', 'Adult Movies','Family Movies-Parental Guidance')       
# 
# Example :
# 
# See results for ACTION & ADVENTURE + DRAMA + FAMILY MOVIE below :

# In[ ]:


# ACTION & ADVENTURE + DRAMA + FAMILY MOVIE present on NETFLIX

dframe_india['title'][(dframe_india['Action & Adventure']=='YES')&((dframe_india['Dramas']=='YES'))&(dframe_india['Movie Type']=='Family Movies-Parental Guidance')].head(15)


# ## What's Next....
# 
# ### Now, let's categorize movies based on the actors !!

# In[ ]:


dframe_india.cast.isnull().sum()


# In[ ]:


# Filling up the NULL values in cast column

dframe_india['cast'].fillna(value='Actors Not Known',inplace=True)


# In[ ]:


dframe_india.cast.isnull().sum()


# In[ ]:


dframe_india.head()


# In[ ]:


# Creating a function for categorizing movies based on the actors

def actor_separator(actors,show_id):
    for a in (re.split(r',',actors)):
        if a.strip() in dframe_india:
            dframe_india[a.strip()][dframe_india['show_id']==show_id] = 'YES' 
        else:
            dframe_india[a.strip()]='NO'
            dframe_india[a.strip()][dframe_india['show_id']==show_id]='YES'


# In[ ]:


# Calling of function actor_separator for all the rows of the dataframe to categorize movies based on the actors

for show_id,actors in zip(dframe_india['show_id'],dframe_india['cast']):
    actor_separator(actors,show_id)


# In[ ]:


dframe_india.head(3)


# ## Salman Khan Fans ?? Let's have a look at some data...

# In[ ]:


# Salman Khan's movies available on NETFLIX

sns.countplot(y='Movie Type',hue='Salman Khan',data=dframe_india,palette='Set1')


# In[ ]:


# Total Number of Salman Khan movies present on NETFLIX

sns.countplot(x='Salman Khan',data=dframe_india,palette='Set1')
dframe_india['Salman Khan'].value_counts()


# In[ ]:


# Salman Khan all movies at NETFLIX

dframe_india['title'][dframe_india['Salman Khan']=='YES']


# ## Let's talk about Shah Rukh Khan movies present at NETFLIX...

# In[ ]:


# Shah Rukh Khan's movies available on NETFLIX

sns.countplot(y='Movie Type',hue='Shah Rukh Khan',data=dframe_india,palette='Set1')


# In[ ]:


# Total Number of Shah Rukh Khan movies present on NETFLIX

sns.countplot(x='Shah Rukh Khan',data=dframe_india,palette='Set1')
dframe_india['Shah Rukh Khan'].value_counts()


# In[ ]:


# Shah Rukh Khan's ROMANTIC MOVIES

dframe_india['title'][(dframe_india['Shah Rukh Khan']=='YES')&(dframe_india['Romantic Movies']=='YES')]


# In[ ]:


# Shah Rukh Khan's COMEDY MOVIES

dframe_india['title'][(dframe_india['Shah Rukh Khan']=='YES')&(dframe_india['Comedies']=='YES')]


# In[ ]:


# Shah Rukh Khan's ACTION & ADVENTURE MOVIES

dframe_india['title'][(dframe_india['Shah Rukh Khan']=='YES')&(dframe_india['Action & Adventure']=='YES')]


# In[ ]:


# Shah Rukh Khan's ADULT Category movies

dframe_india['title'][(dframe_india['Shah Rukh Khan']=='YES')&(dframe_india['Movie Type']=='Adult Movies')]


# ## Akshay Kumar movies...

# In[ ]:


# Akshay Kumar's movies available on NETFLIX

sns.countplot(y='Movie Type',hue='Akshay Kumar',data=dframe_india,palette='Set1')


# In[ ]:


# Total Number of Akshay Kumar movies present on NETFLIX

sns.countplot(x='Akshay Kumar',data=dframe_india,palette='Set1')
dframe_india['Akshay Kumar'].value_counts()


# In[ ]:


# Akshay Kumar's COMEDY MOVIES

dframe_india['title'][(dframe_india['Akshay Kumar']=='YES')&(dframe_india['Comedies']=='YES')]


# In[ ]:


# Akshay Kumar's ROMANTIC MOVIES

dframe_india['title'][(dframe_india['Akshay Kumar']=='YES')&(dframe_india['Romantic Movies']=='YES')]


# In[ ]:


# Akshay Kumar's ACTION & ADVENTURE MOVIES

dframe_india['title'][(dframe_india['Akshay Kumar']=='YES')&(dframe_india['Action & Adventure']=='YES')]


# ## Sunny Leone Adult Category Movies

# In[ ]:


# Sunny Leone's ADULT CATEGORY MOVIES

dframe_india['title'][(dframe_india['Sunny Leone']=='YES')&(dframe_india['Movie Type']=='Adult Movies')]


# In[ ]:




