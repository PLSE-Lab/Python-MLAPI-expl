#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
movies = pd.read_csv('../input/rating.csv')
movies['timestamp'] =  pd.to_datetime(movies['timestamp'])
movies['month']= movies['timestamp'].dt.month
movies['day']= movies['timestamp'].dt.day
movies['year']= movies['timestamp'].dt.year
movies['day_of_the_year'] = movies['timestamp'].dt.dayofyear
movies['day_of_the_week'] =  movies['timestamp'].dt.dayofweek + 1
movies['hour'] =  movies['timestamp'].dt.hour
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


movies_rat = movies[movies.columns][movies['rating'] > 4.0 ]


# In[ ]:


movies_rat2 = movies[movies.columns][(movies['rating'] < 4.01) & (movies['rating']>3.0) ]
movies_rat3 = movies[movies.columns][(movies['rating'] < 3.01) & (movies['rating']>2.0) ]
movies_rat4 = movies[movies.columns][(movies['rating'] < 2.01) & (movies['rating']>1.0) ]
movies_rat5 = movies[movies.columns][movies['rating'] < 1.01]


# In[ ]:


def histogram_by_property(prop,ticks, file_name, x_size, start_ticks = 0):
    plt.figure(figsize = (x_size,9))
    plt.xticks(rotation = 90)
    count,binss,patches = plt.hist(prop.values,                                   color="#3F5D7D",                                    bins=np.arange(start_ticks,ticks + 2)-0.5,                                   edgecolor = "black",                                   alpha = 0.5,                                   label = file_name)

    plt.xticks(np.arange(start_ticks,ticks + 1))
    plt.legend(loc='upper right',prop = {'size' : '20'})
    #plt.savefig(file_name+".png",\
    #            bbox_inches = "tight")
    plt.show()
    plt.clf()


# In[ ]:


import matplotlib.colors as colors
import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize = (12,3))
plt.subplot(1,5,1)
plt.xticks(rotation = 90)
count,binss,patches = plt.hist(movies_rat.year,                                   color="#3F5D7D",                                    bins=np.arange(1996,2018),                                   edgecolor = "black",                                   alpha = 0.5,                                   label = 'Number per month')

plt.xticks(np.arange(1996,2017))
#plt.legend(loc=8,prop = {'size' : '14'})
#plt.colorbar()
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off

plt.subplot(1,5,2)
plt.xticks(rotation = 90)
count,binss,patches = plt.hist(movies_rat2.year,                                   color="#3F5D7D",                                    bins=np.arange(1996,2018),                                   edgecolor = "black",                                   alpha = 0.5,                                   label = 'Number per day')

plt.xticks(np.arange(1996,2017))
#plt.colorbar()
#plt.legend(loc=8,prop = {'size' : '14'})
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
plt.subplot(1,5,3)
plt.xticks(rotation = 90)
count,binss,patches = plt.hist(movies_rat3.year,                                   color="#3F5D7D",                                    bins=np.arange(1996,2018),                                   edgecolor = "black",                                   alpha = 0.5,                                   label = 'Number per day of the week')

plt.xticks(np.arange(1996,2017))
#plt.legend(loc=8,prop = {'size' : '14'})
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
#plt.colorbar()
plt.subplot(1,5,4)
plt.xticks(rotation = 90)
count,binss,patches = plt.hist(movies_rat4.year,                                   color="#3F5D7D",                                    bins=np.arange(1996,2018),                                   edgecolor = "black",                                   alpha = 0.5,                                   label = 'Number per hour')

plt.xticks(np.arange(1996,2017))
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
#plt.legend(loc=8,prop = {'size' : '14'})

plt.subplot(1,5,5)
plt.xticks(rotation = 90)
count,binss,patches = plt.hist(movies_rat5.year,                                   color="#3F5D7D",                                    bins=np.arange(1996,2018),                                   edgecolor = "black",                                   alpha = 0.5,                                   label = 'Number per hour')

plt.xticks(np.arange(1996,2017))
#plt.legend(loc=8,prop = {'size' : '14'})
#plt.colorbar()
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
plt.show()


# In[ ]:


plt.figure(figsize = (12,9))
plt.subplot(4,1,1)
plt.xticks(rotation = 90)
count,binss,patches = plt.hist(movies.month,                                   color="#3F5D7D",                                    bins=np.arange(1,14),                                   edgecolor = "black",                                   alpha = 0.5,                                   label = 'Number per month')

plt.xticks(np.arange(1,12 + 1))
plt.legend(loc=8,prop = {'size' : '14'})
#plt.colorbar()

plt.subplot(4,1,2)
plt.xticks(rotation = 90)
count,binss,patches = plt.hist(movies.day,                                   color="#3F5D7D",                                    bins=np.arange(1,33),                                   edgecolor = "black",                                   alpha = 0.5,                                   label = 'Number per day')

plt.xticks(np.arange(1,31 + 1))
#plt.colorbar()
plt.legend(loc=8,prop = {'size' : '14'})
plt.subplot(4,1,3)
plt.xticks(rotation = 90)
count,binss,patches = plt.hist(movies.day_of_the_week,                                   color="#3F5D7D",                                    bins=np.arange(1,9),                                   edgecolor = "black",                                   alpha = 0.5,                                   label = 'Number per day of the week')

plt.xticks(np.arange(1,7 + 1))
plt.legend(loc=8,prop = {'size' : '14'})
#plt.colorbar()
plt.subplot(4,1,4)
plt.xticks(rotation = 90)
count,binss,patches = plt.hist(movies.hour,                                   color="#3F5D7D",                                    bins=np.arange(1,26),                                   edgecolor = "black",                                   alpha = 0.5,                                   label = 'Number per hour')

plt.xticks(np.arange(1,24 + 1))
plt.legend(loc=8,prop = {'size' : '14'})
#plt.colorbar()
plt.show()


# In[ ]:


histogram_by_property(movies.month, 12 , 'Ratings per month', 13, 1)


# In[ ]:


histogram_by_property(movies.day_of_the_week, 7 , 'Ratings per day of week', 8, 1)


# In[ ]:


movie_names = pd.read_csv('../input/movie.csv')
print(movie_names.head())
movie = movie_names['title'][movie_names['title'].str.contains('Christ',regex = False)==True]


# In[ ]:


movies_to_test = movies[movies.columns][movies['movieId']  == 592]
histogram_by_property(movies_to_test.month, 12 , 'Ratings per  month', 13, 1)
histogram_by_property(movies_to_test.hour, 24 , 'Ratings per hour of the day', 13, 1)


# In[ ]:


def bar_plot(xaxis_values, data, name,step = 1,              color = "g" ):
    
    plt.xticks(rotation = 90)
    plt.figure(figsize = (90 * np.log10(xaxis_values*2),90))
    plt.xticks(np.arange( 1, xaxis_values + 2, step),rotation='vertical')

    indices = np.arange(xaxis_values)

    plt.bar(indices,data,color=color,            label = 'Destinations',edgecolor="black",             linewidth = 0.5, alpha = 0.5)
    
    plt.legend(loc='upper right',prop = {'size' : '20'})
    plt.savefig(name+".png",bbox_inches = "tight")
    plt.clf()


# In[ ]:


#per_day_month = pd.DataFrame({'cnt' : data.groupby(['day','month','year'])['rating'].count()}).reset_index()


# In[ ]:


result = pd.merge(movies, movie_names, on =['movieId'])


# In[ ]:


ratings_df = pd.DataFrame({'cnt' : movies_to_test.groupby(['day','month'])['rating'].count()}).reset_index()


# In[ ]:


#ratings_df[ratings_df.columns][ratings_df['day'] == 25]
#ratings_df.sort_values('cnt', ascending = False)


# In[ ]:



plt.figure(figsize = (13,11))
plt.subplot(2,2,1)
cnts, xedges, yedges, img = plt.hist2d(movies.day.values,                                       movies.hour.values,                                       norm = colors.LogNorm(),                                       bins =[len(movies.day.unique()),len(movies.hour.unique())])

p = plt.colorbar()
p.set_label('Day x hour')
plt.subplot(2,2,2)
cnts, xedges, yedges, img = plt.hist2d(movies.day.values,                                       movies.day_of_the_week.values,                                       norm = colors.LogNorm(),                                       bins =[len(movies.day.unique()),len(movies.day_of_the_week.unique())] )
p = plt.colorbar()
p.set_label('Day x day of the week')
plt.subplot(2,2,3)
cnts, xedges, yedges, img = plt.hist2d(movies.hour.values,                                       movies.month.values,                                       norm = colors.LogNorm(),                                       bins =[len(movies.hour.unique()),len(movies.month.unique())] )
p = plt.colorbar()
p.set_label('Hour x month')
plt.subplot(2,2,4)
cnts, xedges, yedges, img = plt.hist2d(movies.day.values,                                       movies.month.values,                                       norm = colors.LogNorm(),                                       bins =[len(movies.day.unique()),len(movies.month.unique())] )
p = plt.colorbar()
p.set_label('Day x month')
plt.show()


# In[ ]:


import matplotlib.colors as colors
import matplotlib.pyplot as plt
plt.figure(figsize = (13,9))
cnts, xedges, yedges, img = plt.hist2d(movies_to_test.day.values,                                       movies_to_test.month.values,                                       norm = colors.LogNorm(),                                       bins =[len(movies_to_test.day.unique()),len(movies_to_test.month.unique())] )
plt.colorbar()
plt.show()

