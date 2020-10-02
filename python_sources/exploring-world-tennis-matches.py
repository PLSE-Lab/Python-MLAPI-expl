#!/usr/bin/env python
# coding: utf-8

# # Exploring World Tennis Matches
# 
# In this notebook we will explore the match aspects of this dataset. We will probe the basic dataset attributes and hopefully uncover some interesting effects from the data! This exploratory data analytics notebook is recommended for beginners and those interested in probing this dataset further. Feel free to fork this notebook and/or copy the code here and explore further on your own!
# 
# ![](https://i.imgur.com/xgKeain.jpg)

# # Data munging

# In[ ]:


import pandas as pd
matches = pd.read_csv("../input/world-tennis-odds-database/t_odds.csv")
pd.set_option('max_columns', None)
matches.head(3)


# In[ ]:


matches.date = pd.to_datetime(matches.date)


# # Tennis matches, when and where
# 
# To start with, let's examine when and where tennis matches occur.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

f, axarr = plt.subplots(2, 2, figsize=(14, 7))
plt.suptitle('Tennis Matches by...', fontsize=18)
f.subplots_adjust(hspace=0.5)

(matches.day
     .value_counts()
     .reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
).plot.bar(color='steelblue', ax=axarr[0][0])
axarr[0][0].set_title("Day of Week")

sns.kdeplot((pd.to_datetime(matches.date)
     .dropna()
     .rename(None)
     .map(lambda v: v.hour)
), ax=axarr[0][1])
axarr[0][1].set_title("Hour of Day")

sns.kdeplot((matches.date
     .dropna()
     .rename(None)
     .map(lambda v: v.hour)
), ax=axarr[0][1])
axarr[1][0].set_title("Hour of Day")

(matches.set_index('date').resample('W').count()).day.plot.line(linewidth=1, ax=axarr[1][0])
axarr[1][0].set_title("Week of Year")

(matches.set_index('date')
     .resample('D')
     .count()
     .day
     .reset_index()
     .where(lambda srs: (srs['date'] >= pd.Timestamp('2016-01-01')) & (srs['date'] < pd.Timestamp('2017-01-01')))
     .dropna()
     .day
     .reset_index(drop=True)
).plot.line(ax=axarr[1][1])
axarr[1][1].set_title("Day of Year (2016)")

sns.despine()


# Interestingly enough there is a strong daily cycle to tennis matches: they peak on Tuesday, and decline every day after that. I imagine this may be due to the way tournaments are scheduled: if a tournament takes a week to play, then it will ramp up on Tuesday, but then slow down as more people are eliminated towards the end of the week, until only the last few matches need to played.
# 
# Aside from a strong dip at the very end of the year, as well as some identifiable "rest weeks" throughout the year, pro tennis's famous grind never lets up. Games are played almost all the time throughout the year! It's also interesting to see a strong effect in terms of hour of day.

# In[ ]:


import geoplot as gplt
import geoplot.crs as gcrs
import geopandas as gpd

countries = gpd.read_file("../input/countries-of-the-world/countries.json")
countries = countries.set_index('name')

matches_by_country = gpd.GeoDataFrame(matches
    .assign(country=matches.country.str.title().map(lambda n: n.replace("-", " ").replace('Usa', 'United States of America')))
    .groupby('country')
    .count()
    .join(countries)
    .reset_index()
    .dropna())

gpd.GeoDataFrame(
    matches_by_country
)

import matplotlib.pyplot as plt

gplt.choropleth(
    matches_by_country,
    hue='url', k=5, cmap='Greens',
    projection=gcrs.Robinson(),
    linewidth=1, edgecolor='black',
    figsize=(14, 10),
    legend=True, legend_kwargs={'loc': 'lower left', 'fontsize': 14}
)
ax = plt.gca()
ax.set_global()
plt.title('Tennis Games Played by Country', fontsize=20)
pass


# North America, Europe, and some South American countries plus China lead the world in terms of the number of tennis matches they host. These are the most tennis-happy countries it seems. Meanwhile, basically no matches are played in the interior of Africa. Other expections, like North Korea, are less surprising.

# In[ ]:


import seaborn as sns
sns.set_style('whitegrid')

matches_by_country.set_index('country').url.sort_values(ascending=False).head(10).plot.bar(
    figsize=(14, 7), fontsize=16, color='mediumaquamarine'
)
plt.title('Top Ten Tennis Playing Countries', fontsize=20)


# This chart shows how commanding a percentage of all games played occur in the United States. The United States is a convenient place to be an aspiring tennis pro, period&mdash;many low-level players only have the money to compete in events close to them, and being the States raises your chances of being in touching distance of a tourney near you by quite a lot!

# # Tournaments and games

# In[ ]:


matches = matches.assign(
    tour=matches.tournament_name.map(lambda v: 'Men' if ('men' in v or 'atp' in v) else 'Women'),
    level=matches.tournament_name.map(lambda v: 'ITF' if 'itf' in v else 'Pro')
)


# In[ ]:


sns.set_style("white")
f, axarr = plt.subplots(1, 2, figsize=(14, 4))

matches.tour.value_counts().plot.bar(ax=axarr[0], fontsize=16, color='mediumaquamarine')
axarr[0].set_title("Games by Gender", fontsize=20)

matches.level.value_counts().plot.bar(ax=axarr[1], fontsize=16, color='mediumaquamarine')
axarr[1].set_title("Games by Level", fontsize=20)

sns.despine()


# The Women's Tour (WTA) surprisingly has far, far fewer tournaments total than the Men's Tour (ATP) in this dataset. I am not sure whether or not this is due to a decreased level of competition on the women's tour, but I doubt it; I think it might be that bookies are much more likely to offer betting odds on men's games than on women's ones! Since this is a betting dataset, that would about explain it.
# 
# The ITF level is the tour level for "aspiring professionals", so to speak, and the bottom-most rung of the tennis ladder. This right chart confirms my suspicion about the biased-ness of the dataset, as there are far, far more ITF games played than Pro games played. It's just that the latter are more rarely betted upon.

# In[ ]:


matches.tournament_name.value_counts().head(20).plot.bar(figsize=(14, 6), 
                                                         color='mediumaquamarine',
                                                         fontsize=16)
plt.gca().set_title("Tournaments by Games Played", fontsize=20)
sns.despine()


# The four grand slams (the Australian Open, Wimbledon, the US Open, and the French Open) lead the way in the number of matches. Generally speaking, the higher the "level" of the tournament, the more matches are played as a part of it. This chart basically just shows off what the most prestegious titles on the tours are!

# # Rare events
# 
# Finally, just for fun, let's take a quick look at rare events&mdash;walk-offs, etcetera&mdash;that can occur during a tennis match.

# In[ ]:


import missingno as msno
import numpy as np
sns.set_style('whitegrid')
msno.bar(
    matches.loc[:, ['no_set_info', 'missing_bookies', 'retired_player', 
                    'cancelled_game', 'walkover', 'awarded_player']].replace(0, np.nan),
    figsize=(12, 6)
)


# It seems that players retire pretty often, and games are cancelled reasonably often. Walkoovers are games in which the other player fails to show. No set info probably means that the game's scoreline wasn't precisely recorded (which is a no-no! but happens sometimes in very low-level tournaments). I don't really know what missing bookies are. Awarded player is times when a match is stopped and an umpire forcefully declares a winner, usually due to something egregious (like a fistfight). Luckily this doesn't happen very often!

# In[ ]:


matches.query('awarded_player == 1').iloc[[1]]


# ...[for example in the above case one of the players pelted a line judge with a ball](https://www.usatoday.com/story/sports/tennis/2013/08/01/olga-puchkova-line-judge-citi-open-paula-ormaechea/2610725/)!

# # Further ideas
# 
# That's all for here folks! Now that you hopefully have a decent understanding of the match setup and domain knowledge surrounding this dataset, what can you do with the hereto-untouched betting odds? Can you figure out a way to beat those pesky bookies?
