#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Project Description
Whether or not you like football, the Super Bowl is a spectacle.
There's drama in the form of blowouts, comebacks, and controversy in the games themselves.
There are the ridiculously expensive ads, some hilarious, others gut-wrenching, thought-provoking, and weird. 
The half-time shows with the biggest musicians in the world
And in this project, you will find out how some of the elements of this show interact with each other.
You will answer questions like:

What are the most extreme game outcomes?
How does the game affect television viewership?
How have viewership, TV ratings, and ad cost evolved over time?
Who are the most prolific musicians in terms of halftime show performances?
'''


# In[ ]:


# Import pandas
import pandas as pd

# Load the CSV data into DataFrames
super_bowls = pd.read_csv('../input/super_bowls.csv')
tv = pd.read_csv('../input/tv.csv')
halftime_musicians = pd.read_csv('../input/halftime_musicians.csv')

# Display the first five rows of each DataFrame
display(super_bowls.head())
display(tv.head())
display(halftime_musicians.head())


# In[ ]:


From the visual inspection of TV and halftime musicians data, there is only one missing value displayed,
but I've got a hunch there are more. The Super Bowl goes all the way back to 1967,
and the more granular columns (e.g. the number of songs for halftime musicians) probably weren't tracked reliably over time.
Wikipedia is great but not perfect.

An inspection of the .info() output for tv and halftime_musicians shows us that there are multiple columns with null values.


# In[ ]:


# Summary of the TV data to inspect
tv.info()

print('\n')

# Summary of the halftime musician data to inspect
halftime_musicians.info()


# In[ ]:



For the TV data, the following columns have missing values and a lot of them:

total_us_viewers (amount of U.S. viewers who watched at least some part of the broadcast)
rating_18_49 (average % of U.S. adults 18-49 who live in a household with a TV that were watching for the entire broadcast)
share_18_49 (average % of U.S. adults 18-49 who live in a household with a TV in use that were watching for the entire broadcast)
For the halftime musician data, there are missing numbers of songs performed (num_songs) for about a third of the performances.

get_ipython().set_next_input('There are a lot of potential reasons for these missing values. Was the data ever tracked');get_ipython().run_line_magic('pinfo', 'tracked')
Is the research effort to make this data whole worth it? Maybe.
Watching every Super Bowl halftime show to get song counts would be pretty fun.
But we don't have the time to do that kind of stuff now!
Let's take note of where the dataset isn't perfect and start uncovering some insights.

Let's start by looking at combined points for each Super Bowl by visualizing the distribution. 
Let's also pinpoint the Super Bowls with the highest and lowest scores.


# In[ ]:


# Import matplotlib and set plotting style
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')


# Plot a histogram of combined points
plt.hist(super_bowls['combined_pts'])
# ... YOUR CODE FOR TASK 3 ...
plt.xlabel('Combined Points')
plt.ylabel('Number of Super Bowls')
plt.show()

# Display the Super Bowls with the highest and lowest combined scores
display(super_bowls[super_bowls['combined_pts'] > 70])

display((super_bowls[super_bowls['combined_pts'] < 30]).sort_values(['combined_pts'],ascending= True))


# In[ ]:


# Plot a histogram of point differences
plt.hist(super_bowls.difference_pts)
plt.xlabel('Point Difference')
plt.ylabel('Number of Super Bowls')
plt.show()

# Display the closest game(s) and biggest blowouts
display(super_bowls[super_bowls['difference_pts'] == 1])
display(super_bowls[super_bowls['difference_pts'] >= 35])


# In[ ]:


# Join game and TV data, filtering out SB I because it was split over two networks
games_tv = pd.merge(tv[tv['super_bowl'] > 1], super_bowls, on='super_bowl')

# Import seaborn
import seaborn as sns
# ... YOUR CODE FOR TASK 5 ...

# Create a scatter plot with a linear regression model fit
sns.regplot(x='difference_pts',y='share_household',data=games_tv)


# In[ ]:


# Create a figure with 3x1 subplot and activate the top subplot
plt.subplot(3, 1, 1)
plt.plot(tv.super_bowl , tv.avg_us_viewers, color='#648FFF')
plt.title('Average Number of US Viewers')

# Activate the middle subplot
plt.subplot(3, 1, 2)
plt.plot(tv.super_bowl, tv.rating_household, color='#DC267F')
plt.title('Household Rating')

# Activate the bottom subplot
plt.subplot(3, 1, 3)
plt.plot(tv.super_bowl, tv.ad_cost, color='#FFB000')
plt.title('Ad Cost')
plt.xlabel('SUPER BOWL')

# Improve the spacing between subplots
plt.tight_layout()


# In[ ]:



It turns out Michael Jackson's Super Bowl XXVII performance,
one of the most watched events in American TV history,
was when the NFL realized the value of Super Bowl airtime and decided they needed to sign big name acts from then on out.
The halftime shows before MJ indeed weren't that impressive,
which we can see by filtering our halftime_musician data.


# In[ ]:


# Display all halftime musicians for Super Bowls up to and including Super Bowl XXVII
halftime_musicians[halftime_musicians.super_bowl <= 27]


# In[ ]:


# Count halftime show appearances for each musician and sort them from most to least
halftime_appearances = halftime_musicians.groupby('musician').count()['super_bowl'].reset_index()
halftime_appearances = halftime_appearances.sort_values('super_bowl', ascending=False)

# Display musicians with more than one halftime show appearance
print(halftime_appearances[halftime_appearances['super_bowl']>1])


# In[ ]:


From our previous inspections, the num_songs column has lots of missing values:

A lot of the marching bands don't have num_songs entries.
For non-marching bands, missing data starts occurring at Super Bowl XX.
Let's filter out marching bands by filtering out musicians with the word "Marching" in them and the word "Spirit" 
(a common naming convention for marching bands is "Spirit of [something]").
Then we'll filter for Super Bowls after Super Bowl XX to address the missing data issue,
then let's see who has the most number of songs.


# In[ ]:


# Filter out most marching bands
no_bands = halftime_musicians[~halftime_musicians.musician.str.contains('Marching')]
no_bands = no_bands[~no_bands.musician.str.contains('Spirit')]

# Plot a histogram of number of songs per performance
most_songs = int(max(no_bands['num_songs'].values))
plt.hist(no_bands.num_songs.dropna(), bins = 10)
plt.xlabel('Number of Songs Per Halftime Show Performance')
plt.ylabel('Number of Musicians')
plt.show()

# Sort the non-band musicians by number of songs per appearance...
no_bands = no_bands.sort_values('num_songs', ascending=False)
# ...and display the top 15
display(no_bands.head(15))


# In[ ]:




