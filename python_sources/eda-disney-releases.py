#!/usr/bin/env python
# coding: utf-8

# ## EDA on Disney releases

# 1. Inspect and clean data
# 2. Total number of movies by genre
# 3. Density plot of genres by release year
# 4. Top 10 largest grossing films
# 5. Biggest earners by genre

# Import libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3


# In[ ]:


### File path and pd read csv
disney_movies_path = pd.read_csv('../input/disney-movies/disney_movies.csv')
dis = disney_movies_path


# Create sqlite3 connection and query object

# In[ ]:


### Create dqlite3 connection object and query object
e = pd.read_sql_query
conn = sqlite3.connect('disney_movies.db')
dis.to_sql('disney', conn, if_exists='replace', index=False)


# **Inspect and clean data.**

# In[ ]:


### Inspect and clean data if necessary
print (dis.dtypes)
print (dis.isna().sum())
print (dis.genre.head(n=20))
print (dis.mpaa_rating.head(n=20))
dis.replace = {'genre': 'Not Known', 'mpaa_rating':'Not Rated' }
dis_full_clean = dis.fillna(dis.replace)
dis = dis_full_clean
print (dis.isna().sum())


# In[ ]:


print ('Total number of movies included in the data is', dis['movie_title'].count())


# In[ ]:


print ('Earliest release in the data is', dis['release_date'].min())


# In[ ]:


print ('Most recent release in the data is', dis['release_date'].max())


# **2. Number of movies per genre**

# In[ ]:


### Counts of genre
total_genre = e('''
                
                    SELECT COUNT(movie_title) AS total, genre
                    FROM disney
                    GROUP BY genre
                    ORDER BY total
                    
                    ''', conn)


# In[ ]:


### Plot of genre totals
plt.figure(figsize=(16,8))
chart = sns.barplot(x='genre', y='total', data=total_genre )
plt.xticks(rotation=45)
plt.show()


# In[ ]:


print (len(dis.loc[dis['genre']=='Comedy']),'comedies made from 1937 - 2016')


# Disney makes and releases more comedies than any other genre, with adventure and action making up biggest part of the other fraction of total movies made.

# **3. Density plot of genres per release year**

# In[ ]:


### Genre per release year
dis['release_year'] = dis['release_date'].str[:-6]
dis.to_sql('disney', conn, if_exists='replace', index=False)
genre_release = e('''
                  
                      SELECT genre, COUNT(genre), release_year
                      FROM disney
                      WHERE genre NOT LIKE 'None'
                      GROUP BY genre, release_year
                      ''', conn)


# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1587285904951' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Di&#47;DisneyGenreRelease&#47;Sheet1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='DisneyGenreRelease&#47;Sheet1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Di&#47;DisneyGenreRelease&#47;Sheet1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1587285904951');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# The number of adventure movies release per year have been fairly consistent since the mid 1908's. Comedy saw a big spike of releases through the nineties and has since droppped off over the past 10 years. We can also see through the first few decades of releases, the genres were limited to adventures, musicals and dramas.

# **4. Top 10 largest grossing films**

# In[ ]:


### 10 biggest grossing films
largest_gross = e('''
                  
                         SELECT inflation_adjusted_gross, movie_title
                         FROM disney
                         ORDER BY inflation_adjusted_gross DESC
                         LIMIT 10
                         
                         ''', conn)


# In[ ]:


### Plot of top 10 largest grossing films
plt.figure(figsize=(15,7))
chart =sns.barplot(x='movie_title', y='inflation_adjusted_gross', data=largest_gross, palette='plasma')
plt.xlabel('Movie Title')
plt.ylabel('Total Gross [$]')
plt.xticks(rotation=45)
plt.show()


# Snow White is by far the biggest grossing film Disney has ever had, the gross has been adjusted to recognise inflation over the past century and I believe, but can't be certain, the gross includes the gross made by all several versions released. This would go some way to explaining the meteoric difference between Snow White and the gross made by other films, the fact that it has 80+ years of earnings and the sum of several releases worth of earnings concurrently.

# **5. Biggest earners by genre**

# In[ ]:


### Biggest earners by genre
genre_gross = e('''
                
                    SELECT genre, SUM(inflation_adjusted_gross) AS total
                    FROM disney
                    GROUP BY genre
                    ORDER BY total DESC
                    
                    
                    ''', conn)


# In[ ]:


### Plot of genre with total gross
plt.figure(figsize=(16,6))
chart =sns.barplot(x='genre', y = 'total', data=genre_gross, palette='Spectral'  )
plt.xticks(rotation=45)
plt.xlabel('Genre')
plt.ylabel('Gross [$]')
plt.show()


# The adventure genre has totaled the biggest gross for Disney so far, although adventure isn't the genre that has the largest number of releases, we showed this to be comedy in section 1. However, we did show in section 2 that adventure movies have been made consistently for a longer period of time than any other genre and have remained popular.
