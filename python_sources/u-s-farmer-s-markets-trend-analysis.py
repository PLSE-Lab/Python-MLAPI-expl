#!/usr/bin/env python
# coding: utf-8

# # US Farmers Markets
# 
# ##### - The goal of this analysis is to determine a trend to the distribution of farmers markets across the US.
#     - We will then be comparing the pattern of this trend against the distribution of wealth across the US.
#     
# ##### - In order to make this an effective analysis we will:
#      - Clean the data to a suitable format.
#      - Then we will perform exploratory data analysis using visualization techniques.
#      - From this, we will try to build a picture of an overall trend between farmers markets in high and low-income areas.

# #### Importing the relevant libraries for data analysis and visualisation.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Read the data into pandas dataframes.

# In[ ]:


ci = pd.read_csv('../input/farmers-markets-in-the-united-states/wiki_county_info.csv')
fm = pd.read_csv('../input/farmers-markets-in-the-united-states/farmers_markets_from_usda.csv')


# # Data Cleaning - county income (ci dataframe)

# In[ ]:


ci.head(2)


# #### We can see the following from the .info() method:
#     - There are some null values that need to be addressed.
#     - Not all of the columns will be relevant to us.
#     - We need to change the Dtype of rows 3,4,5,6,7.
#     - We should drop the number column as it just acts as another index.

# In[ ]:


ci.info()


# #### Removing columns that we don't need:
#  - 'number' column because it just acts as an extra index 
#  - 'county' column because we will be conducting the analysis at a state level.

# In[ ]:


ci.drop(['number','county'],axis=1,inplace=True)


# #### Addressing the null values.

# In[ ]:


print(f'number of rows in ci: {len(ci)}\n')
print(f'number of null values by column in ci: \n\n{ci.isnull().sum()}')


# Given that the percentage of rows containing null values is much less than 1% of the total data,
# we can just drop any row containing a null value.

# In[ ]:


rows_to_drop = [i for i in range(len(ci)) if ci.iloc[i].isnull().sum()>0]
for i in rows_to_drop:
    ci.drop(i,axis=0,inplace=True)
ci.isnull().sum()


# #### Changing the Dtype of the relevant columns to more appropriate values.

# In[ ]:


ci.info()


# In[ ]:


ci.head(5)


#  Changing the Dtype of 'per capita income','median household income','median family income' to float:
#     - We must address the dollar sign and the commas before changing the Dtype.

# In[ ]:


def get_value(x):
    return int(''.join(x.split('$')[1].split(',')))
def get_number(x):
    return int(''.join(x.split(',')))


# In[ ]:


for i in ['per capita income','median household income','median family income']:
    ci[i] = ci[i].apply(lambda x:get_value(x))

for i in ['population','number of households']:
    ci[i] = ci[i].apply(lambda x:get_number(x))


# In[ ]:


ci.info()


# In[ ]:


ci.head()


# # Data Cleaning - farmers markets (fm dataframe)

# In[ ]:


fm.head(3)


# #### We can see the following from the .info() method.
#     - For this analysis we will want to keep columns 10,20,21,58 in order to compare with the ci dataframe
#         - Because we care about the location at the county and state level, we can remove the street, city and zip column.
#     - There are alot of null values that need to be addressed.
#     - We will need to change the Dtypes of row 58
#     - updateTime column should only be the year as this gives us enough information as to whether the source is outdated.

# In[ ]:


fm.info()


# #### Choosing relevant columns

# In[ ]:


fm = fm[['MarketName','State','x','y','updateTime']]


# In[ ]:


fm.head(5)


# #### Addressing the null values.

# In[ ]:


print(f'number of rows in ci: {len(fm)}\n')
print(f'number of null values by column in ci: \n\n{fm.isnull().sum()}')


# We must remove the rows that contain a null value for the county column since we need this information to use the two dataframes together. Given that county is a categorical variable we cannot estimate its value so we ideally will just remove the rows. Because the number of null value 'County' rows are less than 1% this won't have an effect on their accuracy of our data.
# 
# (the heatmap below shows where the are null values in white)

# In[ ]:


sns.heatmap(fm.isnull())


# In[ ]:


fm.dropna(subset=['x','y'],axis=0,inplace=True)
fm.isnull().sum()


# In[ ]:


fm.head()


# #### Changing the Dtypes of the updateTime columns.

# In[ ]:


fm.info()


# In[ ]:


def get_year(x):
    return pd.to_datetime(x).year

fm['year updated'] = fm['updateTime'].apply(lambda x:get_year(x))
fm['details'] = 'State: '+fm['State']+' --- '+'Name: '+fm['MarketName']
fm.drop('updateTime',axis=1,inplace=True)


# In[ ]:


fm.head()


# In[ ]:


ci.head()


# #### Creating a table to work from for the trend analysis.

# In[ ]:


no_markets_per_state = pd.DataFrame(fm['State'].value_counts())
no_markets_per_state.columns = ['no. farmers markets']
no_markets_per_state.sort_values('no. farmers markets').tail()


# In[ ]:


state_level = ci[['State','per capita income','median household income','median family income','population','number of households']].groupby('State').mean()


# In[ ]:


markets= pd.concat([state_level,no_markets_per_state], axis=1)

markets['state'] = markets.index

markets.drop('median family income',axis = 1, inplace = True) # Remove median family income as we will use per capita income
markets.drop(index = 'Virgin Islands', inplace = True) # remove the duplicate row for Virgin Islands


# #### Replace the NaN values with the respective column averages.

# In[ ]:


for i in markets.drop('state',axis=1).columns:
    markets[i].fillna(markets[i].mean(),inplace=True)


# In[ ]:


markets.isnull().sum()


# In[ ]:





# #### The table that we will be using to analyse trends in the data.

# In[ ]:


markets.head()


# #### Creating a list of state codes associated with the state names of our table.
#  - We can then pass this list as the 'locations' argument in the chloropleth plots.

# In[ ]:


state_income = ci[['State','per capita income']] 
av_state_per_capita_income = state_income.groupby('State').mean()
statenames = list(av_state_per_capita_income.index)
statenames[48] = 'Virgin Islands'

states = {'Alaska': 'AK',
 'Alabama': 'AL',
 'Arkansas': 'AR',
 'American Samoa': 'AS',
 'Arizona': 'AZ',
 'California': 'CA',
 'Colorado': 'CO',
 'Connecticut': 'CT',
 'District of Columbia': 'DC',
 'Delaware': 'DE',
 'Florida': 'FL',
 'Georgia': 'GA',
 'Guam': 'GU',
 'Hawaii': 'HI',
 'Iowa': 'IA',
 'Idaho': 'ID',
 'Illinois': 'IL',
 'Indiana': 'IN',
 'Kansas': 'KS',
 'Kentucky': 'KY',
 'Louisiana': 'LA',
 'Massachusetts': 'MA',
 'Maryland': 'MD',
 'Maine': 'ME',
 'Michigan': 'MI',
 'Minnesota': 'MN',
 'Missouri': 'MO',
 'Northern Mariana Islands': 'MP',
 'Mississippi': 'MS',
 'Montana': 'MT',
 'North Carolina': 'NC',
 'North Dakota': 'ND',
 'Nebraska': 'NE',
 'New Hampshire': 'NH',
 'New Jersey': 'NJ',
 'New Mexico': 'NM',
 'Nevada': 'NV',
 'New York': 'NY',
 'Ohio': 'OH',
 'Oklahoma': 'OK',
 'Oregon': 'OR',
 'Pennsylvania': 'PA',
 'Puerto Rico': 'PR',
 'Rhode Island': 'RI',
 'South Carolina': 'SC',
 'South Dakota': 'SD',
 'Tennessee': 'TN',
 'Texas': 'TX',
 'Utah': 'UT',
 'Virginia': 'VA',
 'Virgin Islands': 'VI',
 'Vermont': 'VT',
 'Washington': 'WA',
 'Wisconsin': 'WI',
 'West Virginia': 'WV',
 'Wyoming': 'WY'}

statecodes = []
for i in statenames:
    statecodes.append(states[i])


# # Exploring the distribution of farmers markets in the US.

# In[ ]:


import plotly.express as px

state_farms = pd.DataFrame(fm['State'].value_counts())
state_farms.head()
state_farms.rename(columns={'State':'Number of farmers markets'}, inplace = True)

px.bar(state_farms, x=state_farms.index, y='Number of farmers markets')


# In[ ]:


import plotly.graph_objects as go


fig = go.Figure(data=go.Scattergeo(
        lon = fm['x'],
        lat = fm['y'],
        mode = 'markers',
        text = fm['details'],
        marker = dict(size = 1,opacity = 1,reversescale = True,autocolorscale = False,symbol = 0, 
                      line = dict(width=1,color='rgba(102, 102, 102)'),colorscale = 'icefire',cmin = 0)
                )
        )
    

fig.update_layout(
        title = 'US farmers markets',
        geo_scope='usa'
    )
fig.show()


# #### Observations:
#  - We can see that the southern region has a relatively moderate density of farmers markets throughout, with the states further to the east having a higher density.
#  - The mid-west states have a relatively low density of farmers markets throughout, however the states further to the east increase in density.
#  - The farmers markets of the western states are concentrated along the coastal states, with quite a low density across the rest of the region.
#  - The north-eastern states have the highest density of farmers markets throughout all of the states.
#  

# # Exploring income-related trends in the data.

# #### Note:
#  - New York and California behave as outliers in this analysis.
#  - The R^2 values of the linear regression lines are low in absolute terms, but we can compare them relatively when considering the full context.
#      - The number of farmers markets is largely dependent on factors that are difficult to measure, such as:
#          - The difficulty of obtaining a farmers market license, which varies by state.
#          - Demand shocks.
#          - Human choice and circumstances.

# #### The relationship between the 'per capita income' and the 'no. farmers markets':

# In[ ]:


fig = px.choropleth(data_frame=av_state_per_capita_income,
                    locations=statecodes,
                    locationmode="USA-states",
                    color='per capita income',
                    color_continuous_scale = 'Blues',
                    scope="usa",
                    title = 'US States by average income per capita.'
                   )

fig.show()


# #### Observations:
#  - We can see that the southern states on average are earning less income than the other states. 
#  - Furthermore, it seems to be the northern states that are earning more.
#  - The coastal states in both the north-east and the west earning the most. 

# In[ ]:


fig = px.scatter(data_frame = markets,
           x = 'per capita income',
           y = 'no. farmers markets',
           size = 'population',
           color = 'population',
           color_continuous_scale = 'Blues',
           trendline = 'ols',
           trendline_color_override = 'red',
           hover_data = markets.columns,
           title = 'Relationship between per capita income and no. farmers markets.'
       )

fig.show()


# In[ ]:


fig = px.density_contour(markets,
                         x="per capita income",
                         y="no. farmers markets"
                        )

fig.update_traces(contours_coloring="fill",
                  contours_showlabels = True,
                 colorscale = 'Blues')

fig.show()


#  - Looking at the trend we can see that there is a slight positive correlation between a state's per capita income and the number of farmers markets in that state.
#  - In terms of the spread of the data, the points are not closely-packed.
#      - The R^2 values of 0.001 indicates that this linear regression trend line is not of good fit to the data.
#  - The linear regression line predicts that for a 1000 dollar increase in a state's per capita income results in 0.85 more farmer's markets in that state.

# #### Looking at the relationship between 'median household income' and the 'no. farmers markets':

#  - Looking at the trend we can see that there is a positive correlation between that median household income of the state and the number of farmers markets within that state.
#  - In terms of the spread of the data, again the data is quite spread out.
#      - The R^2 value of 0.085 implies that there is quite a loose linear association between the state median household income and the number of farmers markets within that state.
#  - The linear regression line predicts that for a 1000 dollar increase in the state median household income there will be 1.1 new farmers markets within that state.

# In[ ]:


fig = px.scatter(data_frame = markets,
           x = 'median household income',
           y = 'no. farmers markets',
           trendline = 'ols',
           size='population',
           color = 'population',
           color_continuous_scale = 'emrld',
           hover_data = markets.columns,
           title = 'Relationship between median household income and no. farmers markets per state.'
          )

fig.show()


# In[ ]:


fig = px.density_contour(markets,
                         x="median household income",
                         y="no. farmers markets"
                        )

fig.update_traces(contours_coloring="fill",
                  contours_showlabels = True,
                 colorscale = 'Blues')

fig.show()


# # Exploring the population-related trends in the data.

# #### The relationship between 'population' and 'no. farmers markets':

#  - Looking at the trend we can see that there is a positive correlation between the population of a state and the number of farmers markets within that state.
#  - In terms of the spread of the data, the data is more closely-packed, but still there is a weak linear association between the population of a state and the number of farmers markets within that state.
#      - The R^2 value of 0.16 shows that this linear model loosely fits the data.
#  - The linear regression line predicts that a state population increase of 1000 will result in 1.6 new farmers markets within that state.

# In[ ]:


fig = px.choropleth(data_frame=markets,
                    locations=statecodes,
                    locationmode="USA-states",
                    color='population',
                    color_continuous_scale = 'greens',
                    scope="usa",
                    title = 'US States by population.'
                   )

fig.show()


# In[ ]:


fig = px.scatter(data_frame = markets,
           x = 'population',
           y = 'no. farmers markets',
           trendline = 'ols',
           size='median household income',
           color = 'median household income',
          color_continuous_scale = 'Blues',
           hover_data = markets.columns,
           title = 'Relationship between population and no. farmers markets per state.'
          )

fig.show()


# In[ ]:


fig = px.density_contour(markets,
                         x="population",
                         y="no. farmers markets"
                        )

fig.update_traces(contours_coloring="fill",
                  contours_showlabels = True,
                 colorscale = 'Blues')

fig.show()


# # From this trend analysis we can draw the following conclusions:
# 
# Overall trends:
#  - The number of farmers markets within a state has the strongest correlation with the population of that state.
#     - This can be expected, since a higher population increases both demand and supply of farmers markets.
#  - The income of a state may have a positive effect on the number of farmers markets within the state but the correlation is still quite weak.
#     - This weak correlation could be explained by the fact that supply and demand of farmer's markets depends on investment and consumption decisions respectively. These dependencies are based on human behaviour and cultural influences, therefore they can both be quite unpredictable.
# 
# Links to observations:
#  - This analysis predicts that the farmers markets in the U.S will be concentrated in areas with a higher population and income level. 
#  - Looking at the analysis:
#      - It appears that the combination of high-income and high-population in the western and north-eastern coastal states are responsible for the high concentration of farmers markets in these areas.
#      - The relatively high per capita income of the northern states appears to be a factor in the higher concentration of farmers markets in the area.
#      - The relatively high population of the south-eastern states seems to play a factor in the higher concentration of farmers markets in the area.
