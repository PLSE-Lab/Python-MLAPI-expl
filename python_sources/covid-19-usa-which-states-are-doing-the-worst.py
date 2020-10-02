#!/usr/bin/env python
# coding: utf-8

# ## United States Covid-19: Where should we be the most concerned?

# ## Intro

# This notebook is meant to be a beginner-approachable look at the covid spread in the USA, grouped by state. It does some basic data cleaning, visualizations, and exponential fit analysis.
# 
# The notebook uses US covid spread data from [this kaggle dataset](http://https://www.kaggle.com/sudalairajkumar/covid19-in-usa), which originally pulls the data from the [Covid Tracking Project](http://https://covidtracking.com/). We're inundated with news and data about covid every day, and no one analysis will answer all of the questions we have about this virus. With that in mind, this notebook choses to focus on the following questions:
# 
# General Status Questions
#  - Which states have the most cases? Deaths? Tests?
#  - Which states have the highest death rates?
#  - Which states are growing at an exponential rate vs a linear rate?
#  - Which states have the highest base for their exponential fit (which states are growing at the fastest exponential rate)?
# 
# COVID Policy Questions [TODO]
#  - How are states with a shelter in place doing compared to states without?
#  - How does case rate change post shelter in place?
#  - How does the [social distancing scoreboard](http://https://www.unacast.com/covid19/social-distancing-scoreboard) correlate to growth rate?
#  - How would forecasts look with/without social distancing?
#  
# Economic Downstream Questions [TODO]
#  - How does covid relate to unemployment? Open table data?
#  
# I will be updating this notebook as I go, so expect some sections to be missing/incomplete. I welcome comments if you have thoughts!

# ## Data Imports and Cleaning
# 
# Grab the data from Kaggle and all the python packages we'll need for analysis.

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from scipy import stats


# In[ ]:


df_original = pd.read_csv('../input/covid19-in-usa/us_states_covid19_daily.csv')
df_original.head()


# In[ ]:


ax = sns.heatmap(df_original.isnull(),yticklabels=False,cbar=False)
ax.set(xlabel='columns', ylabel='rows (white if null)', title='Checking Dataset for Null Values')
plt.show()


# Looking at the data, we see a fair number of null values we'll need to fill in and some columns we might want to add. Specifically, we'll clean the data by:
#  - changing the string dates to datetimes for easier use
#  - filling in NaN values via forward filling, setting the first date to 0 positive cases/deaths if that value is NaN
#  - Adding a death rate column, where death rate = num deaths/num positive cases
#  - Renaming some columns and dropping columns we won't use.

# In[ ]:


df_cleaning = df_original.copy()

# Update column types.
df_cleaning['date'] = df_cleaning['date'].apply(lambda x: dt.datetime.strptime(str(x), '%Y%m%d'))

# Rename colums for easier use.
df_cleaning = df_cleaning.rename(columns={'total':'total_tests_inc_pending', 
                                          'totalTestResults':'total_tests', 
                                          'death':'deaths'})
# Drop columns that aren't needed.
df_cleaning = df_cleaning.drop(columns=['dateChecked',
                                        'hash',
                                        'fips', 
                                        'deathIncrease',
                                        'hospitalizedIncrease',
                                        'negativeIncrease',
                                        'positiveIncrease',
                                        'totalTestResultsIncrease'])

state_dict = {}
for state in df_cleaning.state.unique():
    # Process each state separately, mostly to do the forward filling for NaNs by state.
    state_df = df_cleaning[df_cleaning['state']==state].copy()
    state_df = state_df.sort_values(by='date', ascending=True)
    state_df = state_df.reset_index(drop=True)
    state_df.loc[0] = state_df.loc[0].fillna(0)
    state_df.index = state_df.index + 1
    state_df = state_df.fillna(method='ffill')
    state_dict[state] = state_df
    
# Rejoin all states to make one large dataframe.
df = pd.DataFrame()
for state_df in state_dict.values():
    df = pd.concat([df, state_df])
df= df.reset_index()

# Add additional feature columns.
df['death_rate'] = (df['deaths'] / df['positive'])
# NaN values occur here when there are 0 positive cases and thus no deaths, can fill with 0s.
df['death_rate'] = df['death_rate'].fillna(0) 


# Now that we've filled in all the NaNs, let's do a visual check to make sure all values in the dataframe are indeed defined (in this case, we expect the heatmap to be all black).

# In[ ]:


ax = sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
ax.set(xlabel='columns', ylabel='rows (white if null)')
plt.show()


# And finally, let's take a look at our dataframe and see that all the columns we want are now there and look roughly reasonable!

# In[ ]:


df.head()


# Excellent. We can now start to answer some concrete questions.

# ## Which States are Doing the Best? The Worst?
# 
# Below (hidden) are some functions to create consistent plots across the set of comparisons we're making. Then we can look at the data: which states have the most and least deaths, cases, and tests?

# In[ ]:



def format_plot(fig):
    """
    Format figures for standard appearance.
    """
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left', fontsize='large')

def best_and_worst(df, rank_on, y_label, states_per_plot=5, more_is_worse=True):
    """
    Plots states_per_plot number of best and worst faring states, ranked on the "rank_on" column.
    Labels both y axes with the y_label string provided.
    the more_is_worse boolean indicates if a larger number makes a state worse or better off.
    """
    most_recent_day = df.date.max()
    if more_is_worse:
        ranked = df[df['date']==most_recent_day].sort_values(by=rank_on, ascending=False)
    else:
        ranked = df[df['date']==most_recent_day].sort_values(by=rank_on, ascending=True)
    worst = ranked.head(states_per_plot).state
    best = ranked.tail(states_per_plot).state

    fig=plt.figure(figsize=(20,5))
    plt.subplot(1, 2, 1)
    for state in worst:
        state_df = df[df['state']== state].copy()
        plt.plot(state_df.date, state_df[rank_on], label=state)
        plt.title('Worst States')
        plt.ylabel(y_label)
        format_plot(fig)
    plt.subplot(1, 2, 2)
    for state in best:
        state_df = df[df['state']== state].copy()
        plt.plot(state_df.date, state_df[rank_on], label=state)
        plt.title('Best States')
        plt.ylabel(y_label)
        format_plot(fig)
        
    plt.show()
    


# ### Which states have the most/least positive cases?

# In[ ]:


best_and_worst(df=df, rank_on='positive', y_label='Number of Positive Cases')


# ### Which states have the most/least number of deaths?

# In[ ]:


best_and_worst(df=df, rank_on='deaths', y_label='Number of Deaths')


# ### Which states have performed the most/least tests?

# In[ ]:


best_and_worst(df=df, rank_on='total_tests', y_label='Number of Completed Tests', more_is_worse=False)


# ### Which states have the highest/lowest death rate?

# In[ ]:


best_and_worst(df=df, rank_on='death_rate', y_label='Death Rate')


# ## Linear vs Exponential Growth
# 
# One of the main concerns about covid growth is: are the number of cases growing linearly, or exponentially? Here we'll look at if the growth looks more linear or exponential, by state.
# 
# We'll start with the example of New York, since it currently has the largest number of cases by far.

# Sidenote: as a friendly math reminder, if $y = e^{mx} e^b$ is a good fit for the data, then
# \begin{align}
# y &= e^{mx} e^b \\
# e^{log(y)} &= e^{m x + b} \\
# log(y)&=m x + b \\
# \end{align}
# 
# meaning $log(y)=m x + b$ will a good linear fit.

# Hidden below is code to fit and plot linear and exponential functions to a state's number of positive cases over time.

# In[ ]:


def get_lin_exp_fits(state_df, col='positive'):
    # Calculate linear and exponential fits.
    linear_coeffs = stats.linregress(x=state_df.index, y=state_df[col])
    positive_values = state_df[col]>0
    exp_coeffs = stats.linregress(x=state_df.index[positive_values], y=np.log(state_df.loc[positive_values, col]))
    return linear_coeffs, exp_coeffs

def plot_lin_vs_exp(state_df, col='positive', y_label='Number of Positive Cases'):
    """ Calculate linear and exponential fits 
    """
    linear_coeffs, exp_coeffs = get_lin_exp_fits(state_df, col=col)
    
    # Plot the results.
    fig=plt.figure(figsize=(20,5))
    plt.subplot(1, 3, 1)
    plt.plot(state_df.date, state_df[col], label="data")
    plt.plot(state_df.date,  linear_coeffs[1] + state_df.index*linear_coeffs[0], label="linear prediction")
    plt.title('Linear Fit')
    plt.ylabel(y_label)
    format_plot(fig)
    plt.subplot(1, 3, 2)
    plt.plot(state_df.date, state_df[col], label="data")
    plt.plot(state_df.date,  np.exp(exp_coeffs[1])  * (np.exp(exp_coeffs[0])**state_df.index), label="exp prediction")
    plt.title('Exponential Fit')
    plt.ylabel(y_label)
    format_plot(fig)
    plt.subplot(1, 3, 3)
    plt.plot(state_df.date, np.log(state_df[col]), label="data")
    plt.plot(state_df.date,  exp_coeffs[1] + state_df.index*exp_coeffs[0], label="exp prediction")
    plt.title('Exponential Fit, Log Plot')
    plt.ylabel('Log ' + y_label)
    format_plot(fig)


# Taking a look at New York, we can see that indeed, a linear curve doesn't quite grow fast enough, and that an exponential fit is much more accurate.

# In[ ]:


plot_lin_vs_exp(state_df=state_dict["NY"])


# In[ ]:


plot_lin_vs_exp(state_df=state_dict["NY"], col='deaths', y_label='Number of Deaths')
print("New York State")


# Even Washington, where social distancing began much sooner, still looks somewhat exponential, although the fit quality of linear vs. exponential growth is less clear.

# In[ ]:


plot_lin_vs_exp(state_df=state_dict["WA"])
print("Washington State:")


# So how can we determine which states are growing at an exponential rate, vs a linear rate? We can look at the quality of a linear vs exponential fit by taking a look at the r squared value of each fit.

# In[ ]:


linear, exponential = get_lin_exp_fits(state_dict["NY"])
print ("linear r-squared:", linear.rvalue**2)
print ("exponential r-squared:", exponential.rvalue**2)


# Here we see that the r squared value for the exponential fit is much higher, and thus likely a better fit for New York case growth over time. This begs the question: which states currently trend towards exponential case growth? The hidden code below calculates linear and exponential fits for all states.

# In[ ]:


all_states = df.state.unique()

agg_df = pd.DataFrame(columns=['state','linear_slope','linear_intercept','linear_r2','exp_base','exp_mult','exp_r2'])

for state in all_states:
    state_df = state_dict[state]
    # Only consider states with at least one positive case
    if state_df.positive.abs().sum() != 0:
        linear, exp = get_lin_exp_fits(state_dict[state])
        row = pd.DataFrame(data={'state':[state],
                                 'linear_slope':[linear.slope],
                                 'linear_intercept':[linear.intercept],
                                 'linear_r2':[linear.rvalue**2],
                                 'exp_base':[np.exp(exp.slope)],
                                 'exp_mult':[np.exp(exp.intercept)],
                                 'exp_r2':[exp.rvalue**2]})
        agg_df = agg_df.append(row, ignore_index=True)

agg_df = agg_df.sort_values(by='exp_base', ascending=False)
agg_df.head()
    


# In[ ]:


plt.plot(agg_df.linear_r2, agg_df.exp_r2,'o')
plt.ylabel("Exponential Fit R Squared")
plt.xlabel("Linear Fit R Squared ")
plt.title("Linear and Exponential R Squared Values, Per State")
plt.show()


# Plotting the linear vs exponential r squared values per state, we see that in general exponential fits tend to do better, while linear fits have a wide variety of success rates.

# In[ ]:


fig=plt.figure(figsize=(20,5))
agg_df = agg_df.sort_values(by='linear_r2')
plt.bar(agg_df.state,agg_df.linear_r2)
plt.xlabel("State")
plt.ylabel("Linear Fit R Squred Value")
plt.title("States Ranked by Linear Fit R Squared")
plt.show()


# Let's take a closer look at a state that has worse linear growth (one on the left), like Texas:

# In[ ]:


plot_lin_vs_exp(state_df=state_dict["TX"])
print("Texas: Less linear, more exponential")


# And contrast that with a more linear state (one on the right) like Virginia:

# In[ ]:


plot_lin_vs_exp(state_df=state_dict["VI"])
print("Virginia: More linear, less exponential")


# Whether or not a state's number of cases is growing at an exponential rate is one thing, but what about the rate of that exponential growth? Below, we rank states based on the exponent base in their exponential fit.

# In[ ]:


fig=plt.figure(figsize=(20,5))
agg_df = agg_df.sort_values(by='exp_base')
plt.bar(agg_df.state,agg_df.exp_base)
plt.xlabel("State")
plt.ylabel("Exponential Fit Basee Value")
plt.title("States Ranked by Exponential Fit Base")
plt.show()


# There is some variation in the exponential base value, ranging from close to 1.0 to as high as 1.5. As you'd expect, the states with lower exponential base values are the ones with the highest linear fit r squared values (and thus look the most linear). There doesn't look like there's much correlation between the exponential fit r squared values and the exponential base, but all of those r squared values are so high that the difference between them is negligible.

# In[ ]:


fig=plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(agg_df.linear_r2, agg_df.exp_base,'o')
plt.ylabel("Exponential Fit Base")
plt.xlabel("Linear Fit R Squared ")
plt.title("Linear R Squared vs Exponential Base, Per State")
plt.subplot(1,2,2)
plt.plot(agg_df.exp_r2, agg_df.exp_base,'o')
plt.ylabel("Exponential Fit Base")
plt.xlabel("Exponential Fit R Squared ")
plt.title("Exponential R Squared vs Exponential Base, Per State")
plt.show()

