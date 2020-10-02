#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis of the Olympic Games Dataset

# In[ ]:


get_ipython().system('pip install upsetplot')


# In[ ]:


import numpy as np
import pandas as pd
pd.set_option('expand_frame_repr', False)
pd.set_option('max_colwidth', 140)
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import upsetplot
from plotnine import *
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="specify_your_app_name_here")
from geopy.extra.rate_limiter import RateLimiter
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2)
import scipy as sp

# ignore MatplotLibDeprecation warning: Passing one of 'on', 'true', 'off', 'false' as a boolean is deprecated.
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category = matplotlib.cbook.mplDeprecation)


# ## 1. Read in the data and get a basic idea of the structure of the dataset

# In[ ]:


olympic_data = pd.read_csv("../input/athlete_events.csv")
print(olympic_data.shape)
print(olympic_data.describe())
olympic_data.head()


# ## 2. Timeline of Summer and Winter Olympic Games
# We know that the modern Olympic Games are held every four years, with the Summer and Winter Games alternating by occurring every four years but two years apart from each other. Has this always been the case?

# In[ ]:


event_and_year = olympic_data[['Year', 'Season']].drop_duplicates().sort_values('Year')
(ggplot(event_and_year, aes(x = 'Year', y = 'Season', colour = 'Season')) +
    geom_line() +
    geom_point() +
    scale_x_continuous(breaks = range(event_and_year['Year'].min(), event_and_year['Year'].max()+1, 4)) +
    labs(title = "Timeline of Summer and Winter Olympic Games") +
    theme(plot_title = element_text(size = 10, face = "bold"),
          axis_text_x = element_text(angle = 45),
          axis_text = element_text(size = 7),
          legend_position = "none"))


# **Conclusions**  
# 1. Summer Olympic Games started in 1896.  
# 2. Winter Olympic Games started 28 years later, in 1924.  
# 3. Summer events happened every 4 years, with 3 exceptions: 
#     * between 1904 and 1908 summer events took place every 2 years (1906 was an Intercalated Game event)
#     * there were no events between 1912 and 1920 (8 year hiatus, caused by WW1)
#     * there were no events between 1936 and 1948 (12 year hiatus, caused by WW2)  
# 4. Winter events happened every 4 years, with 2 exceptions:
#     * between 1992 and 1994 winter events took place every 2 years
#     * there were no events between 1936 and 1948 (12 year hiatus, caused by WW2)   
# 5. 1992 was the last year when both Summer and Winter events happened in the same year.

# ## 3. Multi-sport athletes  
# ### 3a. Are there athletes that participate in both Summer and Winter Olympic Games?
# Intuitively, one might assume that athletes train in a single sport, and therefore participate in a single season of the games. Is this true?  
# 
# For every athlete, we find out if they participated in Summer events only, Winter events only, or both. UpSet plots make it easy to visualise set intersections.  

# In[ ]:


# Keep distinct pairs of athlete IDs and Season they participated in
athlete_and_season = olympic_data[['ID', 'Season']].drop_duplicates()
# Convert data frame to wide format
athlete_and_season['Participated'] = True
athlete_and_season_wide = athlete_and_season.pivot(index='ID', columns='Season', values='Participated').fillna(False)
# Construct the series for UpSet plotting
columns = list(athlete_and_season_wide.columns)
athlete_and_season_upset = athlete_and_season_wide.groupby(columns).size()

# UpSet plot
upsetplot.plot(athlete_and_season_upset, sort_by = "cardinality") 
plt.suptitle('Athlete participation in seasonal events')
plt.show() 

# Get the summary data behind the UpSet plot
athlete_and_season_summary = athlete_and_season_wide.groupby(columns).size().reset_index(name='Num athletes').sort_values('Num athletes',  ascending=False)
num_athletes = athlete_and_season.ID.nunique()
athlete_and_season_summary['All athletes'] = num_athletes
athlete_and_season_summary['% of all athletes'] = athlete_and_season_summary['Num athletes'] * 100 / athlete_and_season_summary['All athletes']
print(athlete_and_season_summary.to_string(index=False))


# Quick introduction to UpSet plots (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4720993/)  
#  * UpSet visualises set intersections in a matrix layout. The combination matrix identifies the intersections, while the bars above it encode the size of each intersection (cardinality).  
#  * In our case, the sets are Winter and Summer events. The vertical bars show number of athletes (y axis) who participated in Summer events only (first bar), Winter events only (second bar), or both events (third bar). The filled circles underneath each vertical bar show which sets contribute to that bar.
# 
# **Conclusions**
# 1. 86% of all athletes participated in Summer events only.
# 2. 13.9% of all athletes participated in Winter events only.
# 3. 0.1% of all athletes participated in both Summer and Winter events.  
# 
# ### 3b. What sports did athletes who participated in both Summer and Winter events compete in?
# We now know that indeed the majority of athletes participate in only one season of the Olympic Games. However, 163 athletes participated in both, which makes them interesting to us for further analysis. Which sports do these athletes compete in?

# In[ ]:


# Get the raw data on athletes who participated in both seasons only
athlete_and_season_wide = athlete_and_season_wide.reset_index()
double_season_athletes = athlete_and_season_wide[(athlete_and_season_wide['Summer']==True) & (athlete_and_season_wide['Winter']==True)][['ID']]
double_season_data = pd.merge(double_season_athletes, olympic_data, on = 'ID')
# Keep distinct pairs of athlete IDs and the sports they competed in
double_season_sport = double_season_data[['ID', 'Sport']].drop_duplicates()
# Convert data frame to wide format
double_season_sport['Participated'] = True
double_season_sport_wide = double_season_sport.pivot(index='ID', columns='Sport', values='Participated').fillna(False)
# Construct the series for UpSet plotting
columns = list(double_season_sport_wide.columns)
double_season_sport_upset = double_season_sport_wide.groupby(columns).size()

# UpSet plot
upsetplot.plot(double_season_sport_upset, sort_by = "cardinality", show_counts='%d') 
plt.suptitle('Sports of multi-season athletes')
plt.show() 


# **Conclusions**
# 1. The most popular combination of sports that athletes participate in is <span style="color:blue">Bobsleigh</span> in Winter and <span style="color:red">Athletics</span> in Summer.
# 2. The second most popular combination of sports is <span style="color:blue">Speed Skating</span> in Winter and <span style="color:red">Cycling</span> in Summer.
# 3. Some athletes participate in both Summer and Winter events but compete in only one sport - Ice Hockey or Figure Skating - meaning that these two sports are played in both Summer and Winter Games.  
# 
# ### 3c. Are Ice Hockey and Figure Skating the only two sports that are played in both Summer and Winter Games?

# In[ ]:


sports_both_seasons = olympic_data.groupby('Sport').filter(lambda x: x['Season'].nunique() == 2)
sports_both_seasons['Sport'].unique()


# **Further conclusion**  
# Alpinism is also present during both Summer and Winter Games, but no athlete competed in this sport during both seasons.

# ### 3d. Are there athletes that won medals in multiple sports?
# Now that we've seen that there are some athletes who compete in multiple sports, one final follow-up question would be whether any of those also win medals in multiple sports.

# In[ ]:


# Get a data set where a medal was won
medal_data = olympic_data[olympic_data.Medal.notnull()]
# Get a list of athlete IDs who won medals in multiple sports
multi_medal_data = medal_data.groupby('ID').filter(lambda x: x['Sport'].nunique() > 1)
multi_medal_athletes = multi_medal_data['ID'].unique()
multi_medal_athletes_df = pd.DataFrame({'ID':multi_medal_athletes})
# Get raw data only on athletes who won medals in multiple sports
multi_medal_sports_data = pd.merge(medal_data, multi_medal_athletes_df, on = 'ID')
print(str(multi_medal_sports_data.ID.nunique()) + ' athletes won medals in multiple sports.')
# Keep distinct pairs of athlete IDs and the sports they won medals in
multi_medal_sports = multi_medal_sports_data[['ID', 'Sport']].drop_duplicates()
# Convert data frame to wide format
multi_medal_sports['Won'] = True
multi_medal_sports_wide = multi_medal_sports.pivot(index='ID', columns='Sport', values='Won').fillna(False)
# Construct the series for UpSet plotting
columns = list(multi_medal_sports_wide.columns)
multi_medal_sports_upset = multi_medal_sports_wide.groupby(columns).size()

# UpSet plot
upsetplot.plot(multi_medal_sports_upset, sort_by = "cardinality", show_counts='%d')
plt.suptitle('Sports of multi-sport medallists')
plt.show() 


# In[ ]:


# Function that returns the data on athletes who won specific combinations of sports
def get_multi_medal_athlete_data(sports):
  sports_data = medal_data[medal_data.Sport.isin(sports)]
  sports_data_athletes = sports_data.groupby('ID').filter(lambda x: x['Sport'].nunique() == 2)
  sports_data_athlete = sports_data_athletes['ID'].unique()
  return(medal_data[medal_data.ID.isin(sports_data_athlete)])


# In[ ]:


get_multi_medal_athlete_data(['Shooting', 'Gymnastics'])


# In[ ]:


get_multi_medal_athlete_data(['Bobsleigh', 'Boxing'])


# In[ ]:


get_multi_medal_athlete_data(['Handball', 'Swimming'])


# **Conclusions**
# 1. Interestingly enough, 86 athletes won medals in multiple sports.  
# 2. Most of them won medals in similar sports (e.g. Water Polo and Swimming, Diving and Swimming), but some won medals in what appear to be very dissimilar sports.  
# 3. Using the get_multi_medal_athlete_data() function, we can see who these athletes were, by providing an array of sports in which they won medals.
#     * This way we find out that for example Edward Patrick Francis "Eddie" Eagan won gold medals in both Boxing and Bobsleigh, which is quite impressive.

# ## 4. The home-field advantage
# In sports, the home-field advantage is the advantage a team enjoys from being on its usual playing field. In the Olympic Games specifically, athletes from the host country have a slightly esier time qualifying for the games, as for example host countries are guaranteed a spot in each team sport. So overall, host countries have more athletes participating in the games compared to when they are not hosting.
# 
# Then the question arises as to whether the so-called home-field advantage is simply due to the larger numbers of athletes participating, or is it that the host country actually performs better, perhaps due to a moral advantage of being home and having a larger crowd cheering them on?
# 
# We will start this analysis by looking at the relationship between number of athletes and number of medals, and then proceed to determine whether host countries not just win more medals overall, but actually perform better.
# 
# ### 4a. What is the relationship between number of athletes and number of medals won?
# Here we explore the relationship between the number of athletes that a country brings to the Olympic Games and the number of medals that country wins.

# In[ ]:


# Fix known bugs in the noc_regions dataset
regions = pd.read_csv("../input/noc_regions.csv")
regions.loc[regions['NOC'] == 'BOL', 'region'] = 'Bolivia'
regions.loc[regions['region'] == 'Singapore', 'NOC'] = 'SGP'
# Add the region name into the athlete_events dataset
olympic_data_with_region = pd.merge(olympic_data, regions, on = 'NOC')
print(olympic_data_with_region.shape)
# Are there any NOCs that do not have a corresponding region?
print(olympic_data_with_region.isnull().sum())
# There are 21 events that do not have a region - let's see which NOCs those are.
print(olympic_data_with_region[olympic_data_with_region.region.isnull()]['NOC'].unique())
# 3 NOCs don't have a corresponding region.
# ROT = Refugee Olympic Team
# UNK = Unknown
# TUV = Tuvalu
# All these 3 "regions" are mentioned in the "notes" columns. So we can coalesce the "region" and "notes" columns.
olympic_data_with_region['region'] = olympic_data_with_region.region.combine_first(olympic_data_with_region.notes)


# From the output above we can see that for 21 data points, the "region" column is NaN, meaning that the NOC in the athlete_events table did not have a corresponding value in the noc_regions table. We can further see that these 3 NOC regions are ROT, UNK, and TUV, and upon looking them up in the noc_regions table we can see that their long names are recorded in the "notes" column instead. We coalesce the "notes" and "regions" columns to make sure we don't introduce missing values in the "regions" column, as it will be needed for further analysis.
# 
# One thing to be careful about is the presence of team sports in our dataset. When a coutry wins a medal in a team sport, there is an entry for every team member. However, for the purpose of this analysis, we want to know how many medals a country (not each athlete) won, therefore we only want to count team medals once.  
# 
# In order to accomplish this, we cannot simply group by country and count the rows. Instead, we could count the unique combinations of Games + Event + Medal, since those are unique per country's medal.  
# 
# For example, Romania won a gold medal in fencing in the 2016 Summer Olympics at Rio de Janeiro. The fencing team was made up of 4 team members, each with an entry in the athlete dataset. The way we know these 4 entries belong to the same medal for Romania, is because they all have the same Game listed (Summer 2016) + Event (Fencing Women's epee, Team) + Medal (Gold).

# In[ ]:


# For every Olympic Games edition, get the number of athletes brought by each country
num_athletes = olympic_data_with_region.groupby(['Games', 'region'])['ID'].nunique().reset_index(name='num_athletes')
print(num_athletes.shape)

# For every Olympic Games edition, get the number of medals won by each country
# Create a unique_medal field which is a concatenation of the year+season+event+medal
medal_data_with_region = olympic_data_with_region[olympic_data_with_region.Medal.notnull()]
medal_data_with_region['unique_medal'] = medal_data_with_region.Games.astype(str) + medal_data_with_region.Event.astype(str) + medal_data_with_region.Medal.astype(str)
num_medals = medal_data_with_region.groupby(['Games', 'region'])['unique_medal'].nunique().reset_index(name='num_medals')
print(num_medals.shape)

# Merge into a single data frame
# If a country didn't win any medals, list 0.
event_summary = pd.merge(num_athletes, num_medals, on = ['Games', 'region'], how = 'left').fillna(0)

# Let's visualise the relationship between number of participating athletes and number of medals won
(ggplot(event_summary, aes(x = 'num_athletes', y = 'num_medals')) +
    geom_point() +
    geom_smooth(method='lm')+
    labs(x = "Number of participating athletes", y = "Number of medals won",
         title = "Relationship between number of athletes a country brings to a game and number of medals won"))


# Visually, it looks like number of athletes and number of medals correlate positively, but we want to test this statistically.
# 
# Pearson's correlation test assumes (1) the presence of a linear relationship between the variables, as well as the (2) absence of outliers, and (3) homogeneity of variance. From the scatterplot above, we can tell that there is a linear relationship between our variables. As for the presence of outliers, that is a bit more difficult to tell. From the scatterplot above, there seem to be some data points that fall far from the regression line. However, since those data points are not physically impossible to occur and we do not have a clear basis on which to remove them, it might be preferable to keep all data points in and opt for a non-parameteric test such as Spearman's rank-order correlation which is not sensitive to outliers.
# 
# In order to test homogeneity of variances, we can use Levene's test.

# In[ ]:


sp.stats.levene(event_summary['num_athletes'], event_summary['num_medals'])


# Levene's test is not significant meaning that we failed to reject the null hypothesis that there is no difference in variances between the two variables. In other words, we do not violate the assumption of homoscedasticity.

# In[ ]:


# What do the distributions of the 2 variables look like?
plt.subplot(1, 2, 1)
plt.hist(event_summary['num_athletes'], bins=30)
plt.xlabel('num_athletes')
plt.subplot(1, 2, 2)
plt.hist(event_summary['num_medals'], bins=30)
plt.xlabel('num_medals')
plt.tight_layout()
plt.show()


# So although we meet the assumptions of linearity and homoscedasticity, the distribution of the two variables indicates a heavy skew and the presence of outliers. In this case, it might be preferable to go with the non-parametric Spearman's rank-order correlation. The test indicates that there is indeed a significant positive correlation between our variables, meaning that as the number of athletes increases, so does the number of medals won.

# In[ ]:


# Spearman's rank-order correlation
coef, p = sp.stats.spearmanr(event_summary['num_athletes'], event_summary['num_medals'])
print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
alpha = 0.05
if p > alpha:
    print('Number of athletes and medals are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
    print('Number of athletes and medals are correlated (reject H0) p=%.3f' % p)


# ### 4b. Are countries performing better when they are hosting?
# Now that we know that as the number of athletes increases, so does the number of medals, we can assume that since host countries can participate with more athletes, they will have more medals overall, just because of the number of athletes - testing this hypothesis could be another analysis. But the more interesting question for now is whether they actually perform better when they are hosting. Is there some kind of moral advantage that translates to better performance?
# 
# In order to measure a country's performance, we compute the number of medals they won per participating athlete (mpa = medals-per-athlete). The dataset contains the host city for each event but not the country, therefore the first step is to find out which country each city is in.

# In[ ]:


# Get a list of cities that hosted the Olympic Games and the countries they are in.
cities_df = pd.DataFrame({'City': olympic_data['City'].unique()})
cities_df['location'] = cities_df['City'].apply(geocode, language='en')
cities_df['point'] = cities_df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
cities_df['Country'] = cities_df['point'].apply(lambda point: geolocator.reverse(point, language = 'en').raw['address']['country'])
host_loc = cities_df[['City', 'Country']]
host_loc.head()


# In[ ]:


# Get the NOC of the host country (to make sure the country names are spelled the same as the "region")
host_loc_with_noc = pd.merge(host_loc, regions, left_on = 'Country', right_on = 'region', how = 'left')
print(host_loc_with_noc[host_loc_with_noc.region.isnull()])
# Turns out that United Kingdom, PRC, and B&H are spelled differently in the noc_regions dataset. We must correct these to continue with the analysis.
# United Kingdom is spelled UK
# PRC is spelled China
# B&H is spelled Bosnia and Herzegovina
host_loc.loc[host_loc['Country'] == 'United Kingdom', 'Country'] = 'UK'
host_loc.loc[host_loc['Country'] == 'PRC', 'Country'] = 'China'
host_loc.loc[host_loc['Country'] == 'B&H', 'Country'] = 'Bosnia and Herzegovina'
# Re-run merge after data was fixed
host_loc_with_noc = pd.merge(host_loc, regions, left_on = 'Country', right_on = 'region', how = 'left')
print(host_loc_with_noc[host_loc_with_noc.region.isnull()])


# For our analysis, we would like to have the same country spelling between the country of the host and the country of the athlete. The output above shows that the spellings were different for 3 countries, and so we went and fixed that by changing the names in the new host country dataset we created to match the names in the noc_regions dataset.  
# 
# Now that we made sure that the host country names match the noc_regions dataset, we can append the country name in our original athlete dataset.

# In[ ]:


olympic_data_with_host = pd.merge(olympic_data_with_region, host_loc, on = 'City')
print(olympic_data_with_host.shape)
olympic_data_with_host.head()


# Our second step is to build the dataset we need for analysis. For every country, we want to compute the medals-per-athlete for the years when they hosted, and for the years when they did not host.

# In[ ]:


olympic_data_with_host['unique_medal'] = olympic_data_with_host.Games.astype(str) + olympic_data_with_host.Event.astype(str) + olympic_data_with_host.Medal.astype(str)


# For every country, find out the year when they hosted
host_year = olympic_data_with_host[['Country', 'Year', 'Season']].drop_duplicates().sort_values(['Season', 'Year'])

# For every host country, find out how many medals they won per athlete when they hosted
host_medals = olympic_data_with_host[(olympic_data_with_host.region == olympic_data_with_host.Country) & olympic_data_with_host.Medal.notnull()].groupby(['region', 'Year', 'Season'])['unique_medal'].nunique().reset_index(name='num_medals')
host_athletes = olympic_data_with_host[(olympic_data_with_host.region == olympic_data_with_host.Country)].groupby(['region', 'Year', 'Season'])['ID'].nunique().reset_index(name='num_athletes')
host_summary = pd.merge(host_athletes, host_medals, on = ['region', 'Year', 'Season'])
host_summary['mpa'] = host_summary['num_medals'] / host_summary['num_athletes']
host_summary = host_summary[['region', 'Year', 'Season', 'mpa']]
host_summary.columns = ['Country', 'Year', 'Season', 'host_mpa']

# Merge the two datasets
host_data = pd.merge(host_year, host_summary, on = (['Country', 'Year', 'Season']))
print(host_data.head())

# For every host country, find out how many medals they won per athlete when they did not host
non_host_medals = olympic_data_with_host[(olympic_data_with_host.region != olympic_data_with_host.Country) & olympic_data_with_host.Medal.notnull()].groupby(['region', 'Year', 'Season'])['unique_medal'].nunique().reset_index(name='num_medals')
non_host_athletes = olympic_data_with_host[(olympic_data_with_host.region != olympic_data_with_host.Country)].groupby(['region', 'Year', 'Season'])['ID'].nunique().reset_index(name='num_athletes')
non_host_summary = pd.merge(non_host_athletes, non_host_medals, on = ['region', 'Year', 'Season'])
non_host_summary['mpa'] = non_host_summary['num_medals'] / non_host_summary['num_athletes']
non_host_summary = non_host_summary[['region', 'Year', 'Season', 'mpa']]
non_host_summary.columns = ['Country', 'Year', 'Season', 'nonhost_mpa']
# Only keep data for countries that were hosts at some point
host_countries = host_summary['Country'].drop_duplicates()
host_countries_df = pd.DataFrame({'Country':host_countries})
non_host_summary = non_host_summary[non_host_summary.Country.isin(host_countries_df['Country'])]
non_host_summary.head()


# Our goal is to find out whether host countries perform better when they host compared to when they don't host. There are many ways in which this problem could be approached (e.g. compare medals-per-athlete in host year with the average in all other years, or compare medals-per-athlete in host year with medals-per-athlete in the previous year they participated) but for this analysis we will choose to treat this problem more like an anomaly detection problem. So we want to know whether the year when they hosted, the performance was anomalous as compared to their typical behaviour over the other years. 
# 
# We will use the empirical cumulative distribution function (ecdf) to visualise how the performance of each country is distributed - and what proportion of olympic games participations resulted in up to a specified number of medals per athlete.

# In[ ]:


# ECDF for a single country - Finland
fin = non_host_summary[non_host_summary.Country == 'Finland']
x = np.sort(fin['nonhost_mpa'])
y = np.arange(1, len(x)+1) / len(x)
plt.plot(x, y, marker = '.', linestyle = 'none')
plt.xlabel('Number of medals per athlete')
plt.ylabel('% of olympic games')
plt.margins(0.02) # keep data off plot edges
plt.show()


# From Finland's ecdf we can tell that, for example:
# * in 50% of the olympic games they participated in, they won up to 0.13 medals per athlete
# * in 90% of the olympic games they participated in, they won up to 0.40 medals per athlete (in other words, 9 times out of 10, Finland won up to 0.4 medals per athlete).
# 
# We can try to visualise the ecdf plots for every host country. There are 22 host countries in total, so the plot will be busy, but it's just meant to give us an idea of how countries compare to each other.

# In[ ]:


# For every country, sort the number of medals ascending
non_host_medals_raw_sorted = non_host_summary[['Country', 'nonhost_mpa']].sort_values(['Country', 'nonhost_mpa'])
# For every country, get the number of entries (i.e. number of olympic games in which they participated and won)
lenx = non_host_medals_raw_sorted.groupby('Country').size().reset_index(name = "lenx")
# For every country, compute the ECDF
ecdf_for_all = pd.merge(non_host_medals_raw_sorted, lenx, on = 'Country')
ecdf_for_all['row_number'] = ecdf_for_all.groupby(['Country', 'lenx']).cumcount()+1
ecdf_for_all['ecdf'] = ecdf_for_all['row_number'] / ecdf_for_all['lenx']

(ggplot(ecdf_for_all, aes(x = 'nonhost_mpa', y = 'ecdf', colour = 'Country')) +
    geom_point(stat = 'identity') +
    labs(title = 'ECDF plot of number of medals per athlete won by countries in years when they did not host',
         x = 'Number of medals per athlete',
         y = '% of olympic games') +
    scale_colour_discrete(name = 'Country'))


# Now that we have the ecdf for the medals-per-athlete each country won when they did not host, we can choose a threshold that we would like to compare the medals-per-athlete each country won when they did host. For this analysis we will pick the 90th percentile. We will consider the performance of the host year anomalous if it falls above the 90th percentile of that country's general performance.
# 
# For example, if we return to Finland, we know from the ecdf that 9 times out of 10 Finland won up to 0.4 medals per athlete. So if Finland wins more than 0.4 medals per athlete when they host the olympic Games, we know that there is only a 1 in 10 chance of this happening, and that this is anomalous.
# In the year when they hosted, Finland won 0.08 medals per athlete, which is a much poorer performance than their 0.4 cutoff.
# 
# We can now go ahead and get the medals-per-athlete value for each country for ecdf = 0.9 and compare it with their medals-per-athlete in the year when they hosted.

# In[ ]:


# Function to get the 90th percentile 
# If the country's ecdf doesn't contain 0.9 we use interpolation to get the mpa value at 0.9 ecdf
def my_interp(x, a=0.9):
    X = np.sort(x)
    e_cdf = np.arange(1, len(X)+1) / len(X)
    if a in e_cdf:
        s = pd.Series(X, index=e_cdf)
        res = s[a]
    else:
        X = np.append(X, np.nan)
        e_cdf = np.append(e_cdf, a)
        s = pd.Series(X, index=e_cdf)
        inter = s.interpolate(method='index')
        res = inter[a]
    return(res)

df = pd.DataFrame(columns = ['Country', 'mpa_percentile90'])
for country in host_countries:
    c = ecdf_for_all[ecdf_for_all.Country == country]
    d = {'Country': [country], 'mpa_percentile90': [my_interp(c['nonhost_mpa'])]}
    res = pd.DataFrame(data=d)
    df = df.append(res)

# Final
final = pd.merge(host_summary, df, on = 'Country')
final['change'] = final['host_mpa'] - final['mpa_percentile90']
final.sort_values('change', ascending=False)


# A positive value in the "change"column above indicates that the performance in the host year was better than the cutoff performance for each country in their non-host years.
# 
# **Conclusions**
# 1. Out of 51 Olympic Games, only in 6 of them did the host country perform unusually well relative to their typical performance.
# 2. In the vast majority of cases (88%), host countries don't perform unusually well.
# 3. It might be the case that the home-field advantage comes down to the larger than usual number of athletes a country can bring to the games, but not to its better performance. This remains an open question that would require further investigation.

# ## 5. Country differences  
# 
# ### 5a. What sports are different countries best at?
# Since we have data on the country each athlete represented, we can find out whether different countries are better (i.e. win more medals) at different sports.  
# 
# We start by counting the number of medals each country won in each sport, and select the top sport for each country. From there, for every sport, we get the list of countries that have historically been best at that sport.

# In[ ]:


# For every country, get a list of the sports in which they won medals, and the number of medals won
medal_data_with_region = olympic_data_with_region[olympic_data_with_region.Medal.notnull()]
# Create a unique_medal field which is a concatenation of the year+season+event+medal
medal_data_with_region['unique_medal'] = medal_data_with_region.Games.astype(str) + medal_data_with_region.Event.astype(str) + medal_data_with_region.Medal.astype(str)
medal_data_with_region.head()
country_medals = medal_data_with_region.groupby(['region', 'Sport'])['unique_medal'].nunique().reset_index(name='Number of medals').sort_values(['region', 'Number of medals'], ascending=[True, False])
# For every country, get the sport at which they won the most medals
country_top_sport = country_medals.groupby('region').first().reset_index()
country_top_sport.head()


# In[ ]:


# Get the number and list of countries that won most medals in each sport
sport_top_countries = country_top_sport.groupby('Sport').size().reset_index(name = 'Num countries').sort_values('Num countries')
list_of_countries_per_sport = country_top_sport.groupby('Sport')['region'].apply(np.unique).to_frame()
sport_top_countries_with_list = pd.merge(sport_top_countries, list_of_countries_per_sport, on = 'Sport')
sport_top_countries_with_list.columns = ['Sport', 'Num countries best at it', 'Countries who won most medals in this sport']
sport_top_countries_with_list


# **Conclusions**  
# It is interesting to inspect the table above and see which countries win the most medals in which sport. This can potentially give us some insight into which sports are most popular in each country, or which sports countries invest most in.
# For example,
# 1. Romania and Japan both won most of their olympic medals in Gymnastics.
# 2. India and Pakistan both won most of their olympic medals in Hockey.
# 3. Egypt, Iraq, and North Korea all won most of their olympic medals in Weightlifing.
