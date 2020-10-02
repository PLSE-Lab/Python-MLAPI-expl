#!/usr/bin/env python
# coding: utf-8

# ## **1. Introduction** <a id="1"></a>

# <a><img style="float: right;" src="https://pbs.twimg.com/media/D_SZjUSWkAENmvg.jpg" width="300" /></a>
# Traditionally, the game of basketball had clearly defined player positions with archetypal body profiles. In simplistic terms, the small players would dribble the ball and pass, slightly taller shooting guards would hit 3 pointers, the medium sized small forwards would attack the basket and the bigs would hustle for rebounds. 
# 
# However, the lines between player positions and roles have increasingly been blurring around the NBA. This change can be attributed to the successful implementation of '[small-ball](https://en.wikipedia.org/wiki/Small_ball_(basketball)' style of play by the Miami Heat and Golden State Warriors in the early and middle '10 respectively.
# 
# 
# These changes inevitably meant that players bodies had to adapt. This notebook looks at how the physical profiles of NBA players have changed over two decades between 1996 and 2019.

# ### **1.1 Table of Contents** <a id="1.1"></a>
# * [1. Introduction](#1)
#    * [1.1 Table of Contents](#1.1)
#    * [1.2 Notebook Preparation](#1.2)
# * [2. Data Preparation](#2)
#    * [2.1 Creation of Additional Variables](#2.1)
# * [3. Height and Weight Exploration](#3)
#    * [3.1 Main Variable Analysis](#3.1)
#    * [3.2 Height and Weight over Time](#3.2)
#    * [3.3 Size and Geography](#3.3)
#    * [3.4 Height and Weight Impact on Performance](#3.4)
#    * [3.5 Weight Transformations](#3.5)
# * [4. Conclusion](#4)

# ### 1.2 Notebook Preparation <a id="1.2"></a>

# In[ ]:


# Import libraries

import pandas as pd
import numpy as np 
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go


# In[ ]:


# Set notebook options

pd.set_option('precision',2)
pd.options.display.max_columns = 30

import warnings
warnings.filterwarnings("ignore")


# ## **2. Data Preparation** <a id="2"></a>

# The below section provides an initial exploration of the data.

# In[ ]:


# Import the data as a DataFrame and check first 5 rows

df = pd.read_csv('../input/nba-players-data/all_seasons.csv', index_col=0)
country_codes = pd.read_csv("../input/country-code/country_code.csv", index_col=0)

df.head()


# In[ ]:


# Check data types and if any records are missing

df.info()


# No records appear to be missing. However, the following edits will have to be made:
# 1. Create ```drafted``` column with boolean 0/1 logic 
#    * Complete
# 2. Convert ```draft_year``` column to integer type
#    * Complete
# 3. Convert ```season``` column to integer type
#    * Complete

# In[ ]:


df.describe()


# Upon first inspection, the following items stand out:
# 1. Maximum ```gp``` (games played) number is 85. The NBA season is 82 games, so the extra games will have to be examined
#    * After examining the dataset in detail, we find 14 players who have recorded more than 82 games played. [This Reddit thread](https://www.reddit.com/r/nba/comments/a8bsyn/in_the_200304_season_theo_ratliff_and_shareef/) suggests the number is higher than 82 because these players have been traded mid-season. For this reason the records will not be adjusted in any way.
# 2. ```net_rating``` values are between -200 and 300. This seems odd and the formula for calculating ```net_rating``` will have to be reviewed
#    * Official ```net_rating``` calculation from the NBA: **NetRtg = 100*((Points)/(POSS) - 100*((Opp Points)/(Opp POSS))**
#    * It is still unclear how Bruce Bowen managed to earn himself a net rating of 300, but overall it seems that the outliers are players that had very small statistical contributions and usaully played just 1 game. Therefore, these records will be removed.

# In[ ]:


categoricals = df.select_dtypes(exclude=[np.number])
categoricals.describe()


# Upon first inspection, the following items stand out:
# 1. There are only 2235 unique players over the course of 24 seasons. Wow... This illusatrates how competitive it is to make it to the NBA.
# 2. There are 36 unique team abbreviations. The league in 2019 consisted of 30, so the extra 6 team will have to be examined.
#    * The extra teams are now defunct: Seattle SuperSonics, Charlotte Bobcats, etc.
# 3. The NBA Draft of 2019 has only 2 rounds, so 8 unique rounds number seems interesting.
#    * It turns out that [historically the Draft had way more rounds](https://en.wikipedia.org/wiki/NBA_draft) (going all the way up to 21 actually). Only in 1989 did the NBA settle for the two round system.
#    * The 8th unique value in the ```draft_round``` column is the ```undrafted``` variable.
# 4. There are 60 individuals called up in the 2019 NBA Draft format (30 in each of the 2 rounds), however, the data shows that there are 75 unique draft numbers.
#    * This again is a relic of the past NBA Draft format.

# ### 2.1 Creation of Additional Variables <a id="2.1"></a>

# In[ ]:


country_codes = country_codes[['Country_name', 'code_3digit']]
country_codes = country_codes.rename({'Country_name': 'country'}, axis=1) 
country_codes['country'] = country_codes['country'].replace({'United States of America': 'USA', 'Russian Federation':'Russia',
                                                             'Venezuela (Bolivarian Republic)':'Venezuela', 'Korea (South)':'South Korea',
                                                             'Tanzania, United Republic of':'Tanzania','Macedonia, Republic of':'Macedonia',
                                                             'Congo, (Kinshasa)':'Democratic Republic of the Congo',
                                                             'Congo (Brazzaville)':'Congo','Iran, Islamic Republic of':'Iran',
                                                             'Virgin Islands, US':'US Virgin Islands',
                                                             })


# In[ ]:


# Create drafted column wiht boolean logic

df['drafted'] = np.where(df['draft_year'] != 'Undrafted', 1, 0)


# In[ ]:


# Convert draft_year column into a date type

# Replace Undrafted with NaN
df['draft_year'] = df['draft_year'].replace(r'Undrafted', np.nan, regex=True)

# Convert the column data type to date
df['draft_year'] = pd.to_datetime(df['draft_year'])


# In[ ]:


# Convert season column to an integer type

df['season'] = pd.to_datetime(df['season'].str[:4])


# In[ ]:


# Fix country names

df['country'] = df['country'].replace({'Great Britain':'United Kingdom','England':'United Kingdom','Scotland':'United Kingdom',
                                       'Bosnia & Herzegovina':'Bosnia and Herzegovina','Bosnia':'Bosnia and Herzegovina',
                                       'Cabo Verde':'Cape Verde','St. Vincent & Grenadines':'Saint Vincent and Grenadines'})


# In[ ]:


print('Complete')


# ## **3. Height and Weight Exploration** <a id="3"></a>

# ### 3.1 Main Variable Analysis <a id="3.1"></a>

# In[ ]:


# Calculate heigh and weight averages

main_variables = df.groupby('player_name', as_index=False).agg({'player_height': 'mean', 'player_weight':'mean'})


# In[ ]:


# Visualise distribution of height and weight data
# Source for average US male heigh: https://en.wikipedia.org/wiki/Average_human_height_by_country
# Source for average US male weight:https://en.wikipedia.org/wiki/Human_body_weight

fig, ax = plt.subplots(1,2,figsize=(16, 8),sharey=True)
plt.subplots_adjust(wspace=0.05)

sns.distplot(main_variables ['player_height'], ax=ax[0], label='_nolegend_', kde=False)
sns.distplot(main_variables ['player_weight'], ax=ax[1], label='_nolegend_', kde=False)
ax[0].axvline(main_variables ['player_height'].mean(), color='#c9082a', label='NBA Mean')
ax[1].axvline(main_variables ['player_weight'].mean(), color='#c9082a', label='NBA Mean')

# Add lines for average adults to compare
ax[0].axvline(175.3, color='#17408b', label='Average US Male Adult')
ax[1].axvline(88.8, color='#17408b', label='Average US Male Adult')

ax[0].yaxis.set_label_text('Count')
ax[0].xaxis.set_label_text('Height (cm)')
ax[1].xaxis.set_label_text('Weight (kg)')
plt.suptitle('Distribution of Height and Weight Data', fontsize=16)
plt.legend(loc='upper right', bbox_to_anchor=(0.98, 1.06), frameon=False)
sns.despine(ax=ax[1], left=True)
sns.despine(ax=ax[0])

plt.show()


# Both, height and weight, are distributed normally in the NBA. However, it is clear that NBA athletes stand out in terms of their height and weight when compared to regular adult males.

# In[ ]:


fig = go.Figure(data=go.Scatter(x=main_variables['player_weight'],
                                y=main_variables['player_height'],
                                mode='markers',
                                text=main_variables['player_name'],
                                marker=dict(color='#17408b')
                                ))

fig.update_layout(
    title='NBA Player Height and Weight (for interactive exploration)',
    xaxis_title='Weight (kg)',
    yaxis_title='Height (cm)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig.show()


# This table utalises Plotly visualisation library. One big advatage of a Plotly table is inherent responsiveness without having to write additional lines of code. This allows to hover over the data points and observe:
# * The tallest player ever to step on the NBA floor was Gheorghe Muresan with a hight of 231cm
# * The shortest player is Muggsy Bogues standing at 160cm
# * Sim Bhullar is by far the heaviest player weighting 163kg (he also made history by [being the first NBA player of Indian descent](https://www.sactownroyalty.com/2020/4/7/21211433/sacramento-kings-history-april-7-sim-bhullar-breaks-a-barrier-first-indian-nba-player))

# In[ ]:


plt.figure(figsize=(16, 8))

sns.regplot(x='player_weight', y='player_height', data=main_variables, color='#17408b')

plt.title('Relationship Between Player Height and Weight', fontsize=16)
plt.ylabel('Height (cm)')
plt.xlabel('Weight (kg)')
sns.despine()

plt.show()


# In[ ]:


corr = stats.pearsonr(main_variables['player_height'], main_variables['player_weight'])

print(f"Correlation between player height and weight is: {corr[0]}. The statistical significance of this relationship is {corr[1]}")


# Not surprisingly, height and weight are two closely related variables.

# In[ ]:


corr_over_seasons = df.groupby('season')[['player_weight','player_height']].corr().unstack().iloc[:,1]

fig = go.Figure()
fig.add_trace(go.Scatter(x=corr_over_seasons.index, 
                         y=corr_over_seasons.values,
                         mode='lines',
                         name='lines',
                         line=dict(color='#17408b')
                        ))

fig.update_layout(
    title='NBA Player Height and Weight Correlation Each Season',
    xaxis_title='Season',
    yaxis_title='Correlation',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig.show()


# However, if we break down the correlation by season, we can see that it is steadily decreasing. This is an early indicator of the changing body types of NBA players.

# ### 3.2 Height and Weight over Time <a id="3.2"></a>

# In[ ]:


fig, ax = plt.subplots(figsize=(16, 8))
ax2 = ax.twinx()

sns.lineplot(x=df['season'], y='player_height', data=df, marker="o", ax=ax, ci=None, label='Height', color='#17408b')
sns.lineplot(x=df['season'], y='player_weight', data=df, marker="o", ax=ax2, ci=None, label='Weight', color='#c9082a')

plt.title('Average Height and Weight Each Season', fontsize=16)
ax.xaxis.set_label_text('Season')
ax.yaxis.set_label_text('Height (cm)')
ax2.yaxis.set_label_text('Weight (kg)')

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.show()


# In[ ]:


df['bmi'] = df['player_weight'].values / (df['player_height'].values ** 2) * 10000

plt.figure(figsize=(16, 8))

sns.lineplot(x=df['season'], y='bmi', data=df, marker="o", ci=None, color='#17408b')

plt.title('Average BMI Each Season', fontsize=16)
plt.ylabel('BMI')
plt.xlabel('Season')
sns.despine()

plt.show()


# In[ ]:


# Height and weight based on draft year

first_season_drafted = df.query('season == draft_year')

h_w_by_draft_year = first_season_drafted.groupby('draft_year')['player_height', 'player_weight'].mean()

fig, ax = plt.subplots(figsize=(16, 8))
ax2 = ax.twinx()

sns.lineplot(x=h_w_by_draft_year.index, y='player_height', data=h_w_by_draft_year, marker="o", ax=ax, ci=None, label='Height', color='#17408b')
sns.lineplot(x=h_w_by_draft_year.index, y='player_weight', data=h_w_by_draft_year, marker="o", ax=ax2, ci=None, label='Weight', color='#c9082a')

plt.title('Average Height and Weight of Draft Class', fontsize=16)
ax.xaxis.set_label_text('Draft Class')
ax.yaxis.set_label_text('Height (cm)')
ax2.yaxis.set_label_text('Weight (kg)')

lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.show()


# In[ ]:


# BMI based on draft year

bmi_draft = h_w_by_draft_year['player_weight'] / (h_w_by_draft_year['player_height'] ** 2) * 10000
bmi_draft = pd.DataFrame(bmi_draft, columns=['bmi'])

plt.figure(figsize=(16, 8))

sns.lineplot(x=bmi_draft.index, y=bmi_draft['bmi'], data=bmi_draft, marker="o", color='#17408b')

plt.title('Average BMI of Draft Class', fontsize=16)
plt.ylabel('BMI')
plt.xlabel('Draft Class')
sns.despine()

plt.show()


# In[ ]:


# Weight and player age

plt.figure(figsize=(16, 8))

w_by_age = df.groupby(['age'])['player_weight'].agg(['mean', 'count'])
w_by_age['outliers'] = np.where(w_by_age['count']<=100, 1, 0)
w_by_age = w_by_age.loc[w_by_age['outliers']==0]

sns.lineplot(x=w_by_age.index, y='mean', data=w_by_age, marker='o', color='#17408b')

plt.title('Average Player Weight for by Age', fontsize=16)
plt.ylabel('Average Weight (kg)')
plt.xlabel('Age')
sns.despine()

plt.show()


# ### 3.3 Size and Geography <a id="3.3"></a>

# In[ ]:


# Select player height and weight when they entered the league
relevant_fields = df[['player_name', 'player_height', 'player_weight', 'country', 'season']]
player_first_season = relevant_fields.loc[relevant_fields.groupby(['player_name']).season.idxmin()]

# Group by country and count the number of players for each country
df_geography = player_first_season.groupby('country', as_index=False).agg(
                                                                          {'player_height': 'mean', 'player_weight':'mean', 'player_name':'count'}
                                                                          ).rename({'player_name': 'count'}, axis=1) 

# Select countries that have at least five unique players
df_geography = df_geography[df_geography['count'] >= 5]

# Add country codes for Plotly visualisation
df_geography = pd.merge(df_geography, country_codes, how='inner', on='country')


# In[ ]:


height_map = go.Choropleth(
    locations = df_geography['code_3digit'],
    z = df_geography['player_height'],
    text = df_geography['country'],
    colorscale = 'Blues',
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'Player Height (cm)'
)

weight_map = go.Choropleth(
    locations = df_geography['code_3digit'],
    z = df_geography['player_weight'],
    text = df_geography['country'],
    colorscale = 'Blues',
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_title = 'Player Weight (kg)',
    visible=False
)

data = [height_map, weight_map]

updatemenus = list([
    dict(type="buttons",
         buttons=list([   
            dict(label = 'Height Map',
                 method = 'update',
                 args = [{'visible': [True, False]},
                         {'title': 'Average NBA Player Height by Country'}]),

            dict(label = 'Weight Map',
                 method = 'update',
                 args = [{'visible': [False, True]},
                         {'title': 'Average NBA Player Weight by Country'}])
        ]),
    )
])

layout = dict(updatemenus=updatemenus,
             title_text='Average NBA Player Height by Country',
             geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type='equirectangular'),
             margin=dict(l=0, r=0, b=0)
             )

fig = dict(data=data, layout=layout)

plotly.offline.iplot(fig)


# ### 3.4 Height and Weight Impact on Performance <a id="3.4"></a>

# In[ ]:


df_corr = df[df['season'] != '2019-01-01']
df_corr = df[['gp','pts','reb','ast','net_rating','usg_pct','player_weight', 'player_height']]

# Compute the correlation matrix
corr = df_corr.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(250, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


# In[ ]:


g = sns.PairGrid(df_corr, y_vars=['reb','ast'], x_vars=["player_height", "player_weight"], height=3, aspect=2)
g.map(sns.regplot, color='#17408b')
#g.set(ylim=(-1, 11), yticks=[0, 5, 10])
plt.show()


# Neither height or weight seem to have significant impact on the total games played and average points scored. However, as expected, height and weight do impact average assist and rebound statistics.

# In[ ]:


corr_w_r = df.groupby('season')[['player_weight','reb']].corr().unstack().iloc[:,1]
corr_w_a = df.groupby('season')[['player_weight','ast']].corr().unstack().iloc[:,1]

corr_h_r = df.groupby('season')[['player_height','reb']].corr().unstack().iloc[:,1]
corr_h_a = df.groupby('season')[['player_height','ast']].corr().unstack().iloc[:,1]


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=corr_w_r.index, y=corr_w_r.values,
                    mode='lines',
                    name='Weight and Rebounds'))

fig.add_trace(go.Scatter(x=corr_w_a.index, y=corr_w_a.values,
                    mode='lines',
                    name='Weight and Assists'))

fig.add_trace(go.Scatter(x=corr_h_r.index, y=corr_h_r.values,
                    mode='lines',
                    name='Height and Rebounds'))

fig.add_trace(go.Scatter(x=corr_h_a.index, y=corr_h_a.values,
                    mode='lines',
                    name='Height and Assists'))

fig.update_layout(
    title='Correlation Coefficient Comparison Over Time',
    xaxis_title='Season',
    yaxis_title='Coefficient',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig.show()


# Rebound coefficients are stable. However, the negative correlation for assists (the higher or heavier the player is the less assists he makes) has been reducing since 2011. This illustrates how the game is changing and the bigs are becoming a more integral to the overall playmaking of the team.

# ### 3.5 Weight Transformations <a id="3.5"></a>

# In[ ]:


# Largest body weigth transformations

# Compute weight pertentage change values for each player by age
w_pct_change = df.groupby(['season', 'player_name'], as_index=False)['player_weight'].mean().sort_values(['player_name', 'season'])
w_pct_change_values = w_pct_change.groupby(['player_name'])['player_weight'].apply(lambda x: x.pct_change())
w_pct_change = pd.concat([w_pct_change, w_pct_change_values], axis=1).fillna(0)
w_pct_change.columns = ['season', 'player_name', 'weight', 'pct_change']

counts = w_pct_change['player_name'].value_counts()
w_pct_change = w_pct_change[w_pct_change['player_name'].isin(counts.index[counts > 5])]

w_pct_change['sig_cng'] = np.where((w_pct_change.groupby('player_name')['pct_change'].transform('max') > 0.15) | (w_pct_change.groupby('player_name')['pct_change'].transform('min') < -0.15), 1, 0)


# In[ ]:


a = w_pct_change[w_pct_change['sig_cng'] == 1]


# In[ ]:


plt.figure(figsize=(16, 8))

sns.lineplot(x="season", y="weight", data=a, units='player_name', estimator=None, hue='player_name')

plt.title('Percentage Change of Players Weight', fontsize=16)
plt.ylabel('Percentage Change')
plt.xlabel('Season')
sns.despine()

plt.show()


# ## **4. Conclusion** <a id="4"></a>

# <a><img style="float: right;" src="https://preview.redd.it/30szuy4xkjhz.jpg?auto=webp&s=012cd6b49ab610c3e37e7f3db9e34cd4f29a988b" width="300" /></a> This notebook provides data supported evidence on how the bodies of NBA players have changed over time.
