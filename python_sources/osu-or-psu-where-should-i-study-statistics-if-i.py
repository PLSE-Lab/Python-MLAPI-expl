#!/usr/bin/env python
# coding: utf-8

# # OSU or PSU: Where should I study statistics (if I go to grad school)?
# ## An analysis of educators.
# ## Or, scraping web stuff with Python for fun because I am a nerd.
# **Michael Sieviec**
# 
# **August 22nd, 2019**
# 
# Contents:
# 1. [Hi](#hi)
# 2. [Getting the details](#deets)
# 3. [Some visualizations](#viz)
# 4. [Some real comparisons](#comp)
#  * [Mann-Whitney U tests for overall ratings](#mann)
#  * [T-tests for rating population means](#ttest)
# 5. [Conclusion](#bye)
# 
# 
# # Hi <a name = 'hi'></a>
# 
# I have a simple question: if I indeed decide to go to grad school, should I go to Oregon State or Portland State? I could rely on my gut feelings towards some obvious questions such as: do I want to live in Corvallis? Am I really *that* tired of Portland? How stressful and frustrating was it *really* being an undergrad at PSU?
# 
# These are all valid questions, but perhaps I should ask some more quantifiable ones (or, at the very least, ones that can be answered with some certainty), like: which program has what I want to study? Is a PhD in statistics the same as a PhD in "Mathematical Sciences" (what)? Are the teachers at OSU or PSU better? The first is fairly easy to answer&mdash;all it takes is looking in each schools' catalog. The solution to the second is maybe arbitrary, maybe not (why have an MS in statistics and no PhD?).
# 
# The last seemed trickiest, but I found a way, oh yes I did.
# 
# # Getting the details <a name = 'deets'></a>
# My answer came initially in the forms of a popular professor rating site and [a script I found on GitHub](https://github.com/Rodantny/Rate-My-Professor-Scraper-and-Search). These laid the groundwork for my, uh, research, I guess. So, let's find out where I should study graduate-level statistics (if I do)!
# 
# Here is the part of the project where I scrape a list of all the professors at both schools and somewhat tediously narrow the results. I know not every teacher is listed on this particular site, but it's the only quantified measure of quality I have.
# 
# I can't scrape straight from kaggle&mdash;they won't let scripts connect to other sites (rightfully so, I imagine)&mdash;so I had to do it on my computer and upload the cleaned data here. Trust me when I say the whole process involved a lot of troubleshooting and  cleaning, but I'm better off for it because now I know how to use requests, Beautiful Soup, and Selenium to get data from webpages.
# 
# When compiling the data, I cared about:
# 
# * If a teacher was current faculty or adjunct
# * If they had been rated for a statistics class they taught
# * What the highest level of class they had been rated for was
# 
# Naturally, these criteria thinned my results, but I ended up with some good stuff.

# In[ ]:


import pandas as pd

all_profs = pd.read_csv('../input/all_profs.csv')


# And the last few touches of cleaning.

# In[ ]:


all_profs['overall_rating'] = all_profs['overall_rating'].apply(lambda x: float(x)) # transform
all_profs = all_profs[['school', 'name'] + 
                      [col for col in all_profs if col not in ['school', 'name']]] # move things around


# In[ ]:


all_profs[all_profs.school == 'Oregon State University']


# In[ ]:


all_profs[all_profs.school == 'Oregon State University'].shape


# In[ ]:


all_profs[all_profs.school == 'Portland State University']


# In[ ]:


all_profs[all_profs.school == 'Portland State University'].shape


# That's what it looks like&mdash;13 teachers at OSU and 15 at PSU. Not great, not terrible.
# 
# # Some visualizations <a name = 'viz'></a>
# Interactive graphs are tight, so here are some.

# In[ ]:


import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Box(y = all_profs[all_profs.school == 'Oregon State University'].overall_rating,
              name = 'Oregon State University'))
fig.add_trace(go.Box(y = all_profs[all_profs.school == 'Portland State University'].overall_rating,
              name = 'Portland State University'))
fig.update_layout(title_text = 'Figure 1: Box Plots of Overall Ratings for Statistics Professors/Instructors at OSU and PSU',
                  yaxis_title_text = 'Overall Rating',
                  showlegend = False)
fig.show()


# Oof, this plot is not looking as good for PSU what with that low median and q1. Let's keep poking around!

# In[ ]:


#histogram
fig = go.Figure()
fig.add_trace(go.Histogram(x = all_profs[all_profs.school == 'Oregon State University'].overall_rating,
              name = 'Oregon State University',
                           xbins = dict(size = 0.25)))
fig.add_trace(go.Histogram(x = all_profs[all_profs.school == 'Portland State University'].overall_rating,
              name = 'Portland State University',
                           xbins = dict(size = 0.25)))
fig.update_traces(opacity=0.75)
fig.update_layout(title_text = 'Figure 2: Histogram of Overall Ratings', # title of plot
                  xaxis_title_text = 'Overall Rating', # xaxis label
                  yaxis_title_text = 'Count', # yaxis label
                  bargap = 0.1, # gap between bars of adjacent location coordinates
                  bargroupgap = 0.05 # gap between bars of the same location coordinates
)
fig.show()


# Confirming the boxplot, PSU is just dominating that low-mid rating range, but they are putting up a fight in the top-most spots.
# 
# This isn't exactly a fair fight, though. Why?

# In[ ]:


# total number of OSU ratings
all_profs[all_profs.school == 'Oregon State University'].number_of_ratings.sum()


# In[ ]:


# total number of PSU ratings
all_profs[all_profs.school == 'Portland State University'].number_of_ratings.sum()


# Portland State teachers have more than twice as many ratings. So what does *that* look like?

# In[ ]:


#number of ratings plots
#box plot
fig = go.Figure()
fig.add_trace(go.Box(y = all_profs[all_profs.school == 'Oregon State University'].number_of_ratings,
              name = 'Oregon State University',
              marker = dict(size = 10)))
fig.add_trace(go.Box(y = all_profs[all_profs.school == 'Portland State University'].number_of_ratings,
              name = 'Portland State University',
              marker = dict(size = 10)))
fig.update_layout(title_text = 'Figure 3: Box Plots of Number of Ratings',
                  yaxis_title_text = 'Number of Ratings',
                  showlegend = False)
fig.show()


# Yeah that is measly for OSU: median number of teacher ratings is 6 and q1 ***two***. And their max of 95 is less than 2/3 of the PSU max of 146. Seems like maybe they haven't had enough ratings for their numbers to come down as much as with PSU.

# In[ ]:


#histogram
fig = go.Figure()
fig.add_trace(go.Histogram(x =  all_profs[all_profs.school == 'Oregon State University'].number_of_ratings,
              name = 'Oregon State University',
                           xbins = dict(size = 5)))
fig.add_trace(go.Histogram(x = all_profs[all_profs.school == 'Portland State University'].number_of_ratings,
              name = 'Portland State University',
                           xbins = dict(size = 5)))
fig.update_traces(opacity=0.75)
fig.update_layout(title_text = 'Figure 4: Histogram of Number of Ratings', # title of plot
                  xaxis_title_text = 'Number of Ratings', # xaxis label
                  yaxis_title_text = 'Count', # yaxis label
                  #bargap = 1, # gap between bars of adjacent location coordinates
                  bargroupgap = 0.1 # gap between bars of the same location coordinates
)
fig.show()


# Both are right-skewed, but OSU's numbers are a bit more bottom-heavy. So what do these look like by level?

# In[ ]:


# scatter overall
fig = go.Figure()
fig.add_trace(go.Scatter(x = all_profs[all_profs.school == 'Oregon State University'].highest_level_rated,
                         y = all_profs[all_profs.school == 'Oregon State University'].overall_rating,
                         name = 'Oregon State University',
                         mode = 'markers',
                         marker = dict(size = 20),
                         hovertext = all_profs[all_profs.school == 'Oregon State University'].name))
fig.add_trace(go.Scatter(x = all_profs[all_profs.school == 'Portland State University'].highest_level_rated,
                         y = all_profs[all_profs.school == 'Portland State University'].overall_rating,
                         name = 'Portland State University',
                         mode = 'markers',
                         marker = dict(size = 20),
                         hovertext = all_profs[all_profs.school == 'Portland State University'].name))
fig.update_traces(opacity=0.75)
fig.update_layout(title_text = 'Figure 5: Scatter Plot of Overall Ratings by Class Level', # title of plot
                  yaxis_title_text = 'Overall Rating', # yaxis label
                  bargroupgap = 0.1, # gap between bars of the same location coordinates
                  xaxis = go.layout.XAxis(tickvals = [200, 300, 400, 500]),
                  xaxis_title_text = 'Class Level')
fig.show()


# There isn't much overlap between the class levels that have rated teachers, just a tiny bit in the 400 and 500 levels, which will have to do. I'll just assume the 200 level PSU classes and 300 level OSU classes are roughly equivalent for the sake of comparison.

# In[ ]:


# scatter number
fig = go.Figure()
fig.add_trace(go.Scatter(x = all_profs[all_profs.school == 'Oregon State University'].highest_level_rated,
                         y = all_profs[all_profs.school == 'Oregon State University'].number_of_ratings,
                         name = 'Oregon State University',
                         mode = 'markers',
                         marker = dict(size = 20),
                         hovertext = all_profs[all_profs.school == 'Oregon State University'].name))
fig.add_trace(go.Scatter(x = all_profs[all_profs.school == 'Portland State University'].highest_level_rated,
                         y = all_profs[all_profs.school == 'Portland State University'].number_of_ratings,
                         name = 'Portland State University',
                         mode = 'markers',
                         marker = dict(size = 20),
                         hovertext = all_profs[all_profs.school == 'Portland State University'].name))
fig.update_traces(opacity=0.75)
fig.update_layout(title_text = 'Figure 6: Scatter Plot of Number of Ratings by Class Level', # title of plot
                  yaxis_title_text = 'Number of Ratings', # yaxis label
                  bargroupgap = 0.1, # gap between bars of the same location coordinates
                  xaxis = go.layout.XAxis(tickvals = [200, 300, 400, 500]),
                  xaxis_title_text = 'Class Level')
fig.show()


# Wow, PSU dominates the number of ratings in the higher level classes.

# # Some real comparisons <a name = 'comp'></a>
# First, let's have a peek at some of the descriptive statistics of the numbers we're looking at.

# In[ ]:


all_profs.groupby(['school', 'highest_level_rated']).overall_rating.describe()


# PSU average overall ratings by level seem to be a little more tightly grouped. Consistency (for better or worse), perhaps?
# 
# ## Mann-Whitney U tests for overall ratings <a name = 'mann'></a>
# 
# Our sample sizes are small&mdash;some less than 5&mdash;for the overall ratings in each school/level grouping, so we're going with the Mann-Whitney U test to compare them at each level. We could do a Friedman test on the mean ratings in the table above with schools as one factor and class levels as the other, but a straight ANOVA-style test seems like it would be missing some important information&mdash;the number of observations (teachers) varies quite a bit between school and level.
# 
# It looks like the OSU mean is less for the 400 and 500 levels but greater for the 200/300 level. We'll be generous and set our significance level at 0.1 for each test.

# In[ ]:


from scipy import stats
# 2/300 level
osu_300 = all_profs[(all_profs.school == 'Oregon State University') & (all_profs.highest_level_rated == 300)].overall_rating
psu_200 = all_profs[(all_profs.school == 'Portland State University') & (all_profs.highest_level_rated == 200)].overall_rating
stats.mannwhitneyu(osu_300, psu_200, alternative = 'greater')


# In[ ]:


# 400 level
osu_400 = all_profs[(all_profs.school == 'Oregon State University') & (all_profs.highest_level_rated == 400)].overall_rating
psu_400 = all_profs[(all_profs.school == 'Portland State University') & (all_profs.highest_level_rated == 400)].overall_rating
stats.mannwhitneyu(osu_400, psu_400, alternative = 'less')


# In[ ]:


# 500 level
osu_500 = all_profs[(all_profs.school == 'Oregon State University') & (all_profs.highest_level_rated == 500)].overall_rating
psu_500 = all_profs[(all_profs.school == 'Portland State University') & (all_profs.highest_level_rated == 500)].overall_rating
stats.mannwhitneyu(osu_500, psu_500, alternative = 'less')


# Well, the average overall ratings for each school do not appear to be distributed differently at any level, suggesting the teachers may be of the same quality after all.
# 
# # T-tests for rating population means
# 
# We're going to test a slightly different thing here: whether or not the individual ratings that are given are distributed differently for each school and class level. I ended up using Selenium and Beautiful Soup to get this data, which was arduous but rewarding.
# First, we want to verify we have the numbers to support conducting a t-test.
# 

# In[ ]:


all_profs.groupby(['school', 'highest_level_rated']).number_of_ratings.describe()


# Only count 2 (teachers) * mean 3 (number of ratings) = 6 ratings at the 400-level for OSU is quite small, but that's more than 5 so it looks like we're good to go. Let's prepare the data.

# In[ ]:


all_ratings = pd.read_csv('../input/all_ratings.csv') # load in individual ratings
all_ratings.head()


# All that's contained is the id and a rating. We want to check that we did indeed get *all* of the ratings for each teacher  by checking the number rows against total number of ratings.

# In[ ]:


all_ratings.shape[0] ==  sum(all_profs.number_of_ratings)


# Perfect, now we merge the sets.

# In[ ]:


all_merged = pd.merge(all_profs, all_ratings, on = 'id')
all_merged.head()


# And on to the tests.

# In[ ]:


# 2/300 level
stats.ttest_ind(all_merged[(all_merged.school == 'Oregon State University') & (all_merged.highest_level_rated == 300)].ratings,
                all_merged[(all_merged.school == 'Portland State University') & (all_merged.highest_level_rated == 200)].ratings,
                equal_var = False)


# It turns out there is sufficient evidence to suggest the means are different at 0.1 significance as p/2 < 0.1 with t > 0, so we will say ratings for 200/300-level OSU statistics teachers are higher on average than for PSU statistics teachers.

# In[ ]:


# 400 level
stats.ttest_ind(all_merged[(all_merged.school == 'Oregon State University') & (all_merged.highest_level_rated == 400)].ratings,
                all_merged[(all_merged.school == 'Portland State University') & (all_merged.highest_level_rated == 400)].ratings,
                equal_var = False)


# And the opposite is true for the 400-level. p/2 < 0.1 and t < 0 so we will say the average rating at PSU is higher than at OSU.

# In[ ]:


# 500 level
stats.ttest_ind(all_merged[(all_merged.school == 'Oregon State University') & (all_merged.highest_level_rated == 500)].ratings,
                all_merged[(all_merged.school == 'Portland State University') & (all_merged.highest_level_rated == 500)].ratings,
                equal_var = False)


# Finally, the same is true for the 500-level. PSU statistics teachers are rated more highly on average (p/2 < 0.1 and t < 0).
# 
# # Conclusion <a name = 'bye'></a>
# It's important to note the limitations of the data I used when considering the results, such as: 
# 
# * It doesn't include every teacher
# * The categories that suggested PSU were better were also strangely populated:
#   1. The 400-level had barely any OSU teachers and the 500 barely any PSU teachers
#   2. The number of ratings was highly uneven
# * 200/300-level classes may not be exactly comparable
# * Common sense also dictates that many people will not go out of their way to talk about an experience unless it was particularly remarkable, in either a good or bad way. As such, we're really only looking at a fraction of opinions on each teacher.
# 
# However, all other things being equal, I guess I should go to PSU.
