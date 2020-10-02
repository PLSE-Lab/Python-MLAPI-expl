#!/usr/bin/env python
# coding: utf-8

# # This analysis looks to answer the following questions in a statistical rigorous way
# ### 1. Is the temperature rising? 
# ### 2. How fast? 
# ### 3. How does this differ across the world

# In[2]:


#functions and import

import numpy as np
import pandas as pd

# display precision to 5 decimal points
pd.set_option('precision',5)

import statsmodels.api as sm

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# In[4]:


#reading data in

global_temp = pd.read_csv("../input/GlobalTemperatures.csv", parse_dates = [0], infer_datetime_format=True)

global_temp.dtypes, global_temp.shape


# In[38]:


# basic visualization of the data
global_avg = go.Scatter(
    x = global_temp.dt,
    y = global_temp.LandAverageTemperature,
    name = 'Average Temp'
)

global_avg_uncertainty = go.Scatter(
    x = global_temp.dt,
    y = global_temp.LandAverageTemperatureUncertainty,
    name = 'Uncertainty',
    yaxis='y2'
)

layout = go.Layout(
    title='Global Average Temperature and Measurement Uncertainty',
    yaxis=dict(
        title='Average Temperature'
    ),
    yaxis2=dict(
        title='Measurement Uncertainty',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right'
    )
)

data = [global_avg, global_avg_uncertainty]

fig = go.Figure(data=data, layout=layout)
plot_url = py.iplot(fig)


# ### Observations
# * Uncertainty is relatively high prior to 1850, and making statistical inference less reliable
#     * For subsequent analysis, will filter out data prior to 1850
# * Temperature trend is expectedly seasonal. The seasonal trend is of secondary importance when it comes to examining the overall trends in temperature. 
#     * For subsequent analysis, will aggregate data to yearly metrics to remove seasonality

# In[6]:


global_temp['year'] = global_temp.dt.apply(lambda x : x.year)
# , as_index=False
global_temp_by_year = global_temp[global_temp['year'] >= 1850].groupby('year').agg(
        {
            'LandAverageTemperature': ['mean', 'std', 'max', 'min']
        }
    )

global_temp_by_year.head()


# In[7]:


global_avg = go.Scatter(
    x = global_temp_by_year.index,
    y = global_temp_by_year.LandAverageTemperature['mean'],
    name = 'Mean Average Temp'
)

global_avg_min = go.Scatter(
    x = global_temp_by_year.index,
    y = global_temp_by_year.LandAverageTemperature['min'],
    name = 'Min Average Temp'
)

global_avg_max = go.Scatter(
    x = global_temp_by_year.index,
    y = global_temp_by_year.LandAverageTemperature['max'],
    name = 'Max Average Temp'
)

global_avg_std = go.Scatter(
    x = global_temp_by_year.index,
    y = global_temp_by_year.LandAverageTemperature['std'],
    name = 'Standard Deviation',
    yaxis='y2'
)

layout = go.Layout(
    title='Global Average Temperature by year',
    yaxis=dict(
        title='Average Temperature'
    ),
    yaxis2=dict(
        title='Standard Deviation',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        side='right'
    )
)

data = [global_avg, global_avg_min, global_avg_max, global_avg_std]

fig = go.Figure(data=data, layout=layout)
plot_url = py.iplot(fig)


# ### Temperatures exhibit increasing trend
# * All of max, min and average global temperature appear to be increasing over time
# * The variability over the year (standard deviation) appears to be decreasing over time, albeit not by much
#     * This is interesting, because it is counter to the claim that climate change has been linked to more extreme temperature patterns
#     * It is possible that the averaging process used to aggregate individual regional measurement to obtain the global average has dampened the actual variability (possible follow up analysis that we won't explore here)
#     
# ### To Assess the rate at which the temperature has been increasing, we will use a linear regression fit

# In[11]:


X = sm.add_constant(global_temp_by_year.index)
mod = sm.OLS(global_temp_by_year['LandAverageTemperature']['mean'], X)

res = mod.fit()

print(res.summary())
print(res.params)
print(res.pvalues)


# In[12]:


act = go.Scatter(
    x = global_temp_by_year.index,
    y = global_temp_by_year.LandAverageTemperature['mean'],
    name = 'Actual Average Temp'
)

model = go.Scatter(
    x = global_temp_by_year.index,
    y = res.predict(),
    name = 'Model Average Temp'
)


layout = go.Layout(
    title='Global Average Temperature by year',
    yaxis=dict(
        title='Average Temperature'
    )
)

data = [act, model]

fig = go.Figure(data=data, layout=layout)
plot_url = py.iplot(fig)


# #### Global average temperature is rising 0.02 degrees per year
# * regression coefficient p-value is very small at 3.7e-51, meaning that the probability that we observe this slope given that the null hypothesis is true (temperature is not rising over time) is very low
# * there appears to be two regimes of temperature rise. From 1980 onward, the temperature trends appears to have accelerated and have a higher slope
#     * To see that we will fit two separate models, one from 1850 - 1980, and one from 1980 onwards

# In[13]:


X = sm.add_constant(global_temp_by_year.index)
mod = sm.OLS(global_temp_by_year['LandAverageTemperature']['mean'][0:131], X[0:131])
mod1 = sm.OLS(global_temp_by_year['LandAverageTemperature']['mean'][131:], X[131:])
res = mod.fit()
res1= mod1.fit()

print(res.summary())
print(res1.summary())


# In[14]:


act = go.Scatter(
    x = global_temp_by_year.index,
    y = global_temp_by_year.LandAverageTemperature['mean'],
    name = 'Actual Average Temp'
)

model = go.Scatter(
    x = global_temp_by_year.index[0:131],
    y = res.predict(),
    name = 'Model Average Temp'
)

model1 = go.Scatter(
    x = global_temp_by_year.index[131:],
    y = res1.predict(),
    name = 'Model1 Average Temp'
)

layout = go.Layout(
    title='Global Average Temperature by year',
    yaxis=dict(
        title='Average Temperature'
    )
)

data = [act, model, model1]

fig = go.Figure(data=data, layout=layout)
plot_url = py.iplot(fig)


# #### Rate of temperature increase appears to be accelerating
# * From 1850 to 1980, the temperature was rising 0.006 degrees per year
# * From 1980 onwards, the rate of increase was 0.0275 degress per year
# 
# #### The above is based on the highly aggregated global average. To be more precise, let's take a look at the analysis at the city level, where the temperature measurements will be more self consistent

# In[15]:


# reading in city level data

temp_by_cities = pd.read_csv("../input/GlobalLandTemperaturesByCity.csv", parse_dates = [0], infer_datetime_format=True)

temp_by_cities.dtypes, temp_by_cities.shape


# In[16]:


# derive year
# group by year, country, city, calulate average to eliminate seasonality
# run regression for each
# get coefficient, and p value

temp_by_cities['year'] = temp_by_cities.dt.apply(lambda x : x.year)
# , as_index=False
temp_by_cities_year = temp_by_cities[temp_by_cities['year'] >= 1850].groupby(['Country', 'City', 'year']).agg(
        {
            'AverageTemperature': ['mean']
        }
    )


# In[18]:


# define a custom aggreagtion functions that runs a regression and extracts the slope and the p value
def fit_linear_model(g):
    g = g.dropna()
    X = sm.add_constant(g.index.get_level_values('year'))
    mod = sm.OLS(g['AverageTemperature'], X)
    res = mod.fit()
    
    col_names = ['coef', 'coef p-val']
    
    return(pd.Series((res.params['x1'], res.pvalues.x1), index = col_names)) #x1 is the default for unnamed coefficient

temp_by_cities_year.columns = temp_by_cities_year.columns.get_level_values(0)
    
temp_by_cities_year_regr = temp_by_cities_year.groupby(['Country', 'City'])    .apply(fit_linear_model)


temp_by_cities_year_regr.head()


# In[23]:


# plot the distribution of p-value to see if we can reject null hypothesis (slope is 0)
trace = go.Histogram(x=temp_by_cities_year_regr['coef p-val'],
                    name='P-val',
                    marker=dict(
                        color='rgb(49,130,189)')
                )
layout = go.Layout(
    title='Distribution of P-val of regression slope',
    yaxis=dict(
        title='count'
    ),
    xaxis=dict(
        title='P-val'
    )
)
            
data = [trace]
fig = go.Figure(data=data, layout=layout)
plot_url = py.iplot(fig)


# #### Most of the P Values are extremely small, with the largest value coming in at 0.0029. This suggests that we can be reasonably confident in rejecting the null hypothesis in the linear regression fits to the temperature trends at the city level.

# In[25]:


# plot the distribution of actual slopes (rate of the temperature increases over time)
trace = go.Histogram(x=temp_by_cities_year_regr['coef'],
                    name='slope',
                    marker=dict(
                        color='rgb(49,130,189)')
                )
layout = go.Layout(
    title='Distribution of rate of temperature increase across all cities',
    yaxis=dict(
        title='count'
    ),
    xaxis=dict(
        title='rate of temperature increase'
    )
)
            
data = [trace]
fig = go.Figure(data=data, layout=layout)
plot_url = py.iplot(fig)


# ### All of the temperature slopes are positive with low p-val. Temperatures are indeed rising across the world
# ### What are the cities and countries with fastest (and slowest) rate of temperature increase?

# In[34]:


# plot the cities and countries with highest and lowest rate of temperature increase
# define a utility function for plotting pareto charts

def plot_pareto(source, colx, coly, colz, colw, asc):
    """Utility function for plotting pareto bar charts
    
    depends on pandas, plotly
    import pandas as pd
    import plotly.offline as py
    py.init_notebook_mode(connected=True)
    import plotly.graph_objs as go
    import plotly.tools as tls

    Args:
        source (df): Pandas dataframe of the data to be plotted
        colx (str): name of the col to group source by
        coly (str): ['count', 'sum', 'mean', 'std', 'max', 'min', uniques] - aggregation operations to apply to the grouped df
        colz (str): name of the col to apply the aggregation function on
        colw (int): number of results to display in the chart
        asc (str): ['Ascending', 'Descending'] - the order of the pareto chart

    Returns:
        A plot.ly chart
    """
    sort_order = (asc == 'Ascending')
    temp = source
    grouped = temp.groupby(colx)

    if(coly in ['count', 'sum', 'mean', 'std', 'max', 'min']):
        grouped = grouped.agg(
            {
                colz : [coly]
            }
        )
    elif(coly == 'uniques'):
        grouped = grouped.apply(
            lambda g: pd.Series(g[colz].unique().size, index = pd.MultiIndex.from_product([[colz],[coly]]))
        )



    grouped = grouped.reset_index().sort_values([(colz, coly)], ascending=sort_order).head(colw)        .sort_values([(colz, coly)], ascending = (not sort_order))

#             print(grouped)

    trace = go.Bar(
        y=grouped[colx],
        x=grouped[colz][coly],
        name=colx,
        marker=dict(
            color='rgb(49,130,189)'
        ),
        orientation = 'h'
    )
    layout = go.Layout(
        title=coly + ' of ' + colz + ' by ' + colx,
        yaxis=dict(
            title=colx,
            type = "category",
#                     categoryorder = "category descending"
            tickformat =".3f"
        ),
        xaxis=dict(
            title=coly + ' of ' + colz
        ),
        margin=dict(
            l = 200
        )
    )
    
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    plot_url = py.iplot(fig)


# #### Cities with the fastest rate of temperature increase

# In[36]:


plot_pareto(temp_by_cities_year_regr, 'City', 'mean', 'coef', 20, 'Descending')


# #### Cities with the slowest rate of temperature increase

# In[31]:


plot_pareto(temp_by_cities_year_regr, 'City', 'mean', 'coef', 20, 'Ascending')


# #### Countries with the fastest rate (average of cities) of temperature increase

# In[32]:


plot_pareto(temp_by_cities_year_regr, 'Country', 'mean', 'coef', 20, 'Descending')


# #### Countries with the slowest rate (average of cities) of temperature increase

# In[33]:


plot_pareto(temp_by_cities_year_regr, 'Country', 'mean', 'coef', 20, 'Ascending')

