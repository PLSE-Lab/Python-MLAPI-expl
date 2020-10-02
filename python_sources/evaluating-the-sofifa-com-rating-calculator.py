#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The website sofifa.com has a player rating calculator [here](https://sofifa.com/player/calculator/20801) which takes player attributes as inputs and spits out ratings for that player in each position on the pitch.These ratings are only guesses though, and they can be wrong. For example, at the time of writing, Cristiano Ronaldo's rating at RW/LW is 91, while the calculator gives him a rating of 94 in these positions. This project looks at the calculated ratings for every player in every position and tries to come up with more accurate values. We find that even a simple linear regression of the predicted values (according to the sofifa.com calculator) against the actual values gives us a better model.
# 
# # Methods
# 
# Most of the work in this project was in implementing sofifa.com's rating calculator in Python, and vectorising it so that it could take a dataframe as an input. This part is too long and complicated to be displayed in a notebook, but you can look at the GitHub repository [here](https://github.com/kevinheavey/fifa18_player_rating_calculator). The gist of sofifa's method is that they use a different linear combination of a player's main attributes (e.g. acceleration, finishing etc.) for each position, and then apply a bonus based on the player's international reputation, and then check that the player's position rating isn't above their potential.
# 
# In the code below we use a dataframe of predicted ratings (`calculated_ratings`) that were calculated using our Python player rating calculator. If you clone the GitHub repository you can plug the `complete_player_data` dataframe below into `calculate_ratings_from_frame`and see that it gives the same values as in `calculated_ratings`. 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
get_ipython().run_line_magic('matplotlib', 'inline')

# In FIFA 18, a player's position rating is unaffected by being prefixed with "right", "left" or "centre".
# That is, a player's rating in rdm will always be the same as in cdm and ldm.
# So, we only need to use the root positions below
root_positions = ['gk', 'rwb', 'rb', 'rcb', 'rdm', 'rm', 'rcm', 'ram', 'rf', 'rw', 'rs']

complete_player_data = pd.read_csv('../input/fifa-18-more-complete-player-dataset/complete.csv', encoding='latin1')
actual_ratings = complete_player_data[['ID'] + root_positions]
calculated_ratings = pd.read_csv('../input/fifa-18-calculated-ratings/calculated_ratings.csv')


# In[ ]:


# We want long format data: one row for each position rating
calculated_ratings_long_format = calculated_ratings.melt(id_vars='ID', var_name='position', value_name='predicted')
actual_ratings_long_format = actual_ratings.melt(id_vars='ID', var_name='position', value_name='actual')

comparison = (actual_ratings_long_format
              .merge(calculated_ratings_long_format, on=['ID', 'position'])
              .assign(error=lambda df: df['actual'] - df['predicted']))


# # Comparison between predicted values and actual values

# Here's what the `comparison` dataframe looks like:

# In[ ]:


comparison.nlargest(10, 'predicted')


# ## Let's look at how the predicted ratings line up with the actual ratings for each position
# 
# Firstly and briefly, the mean square error for the entire model, without regard to particular positions:

# In[ ]:


mse = ((comparison['error']) ** 2).mean()
mse


# Next, look at the correlation between predicted ratings and actual ratings for each position:

# In[ ]:


idx = pd.IndexSlice
r_squared_scores = (comparison
                    .groupby('position')
                    [['actual', 'predicted']]
                    .corr()
                    [['predicted']]
                    .loc[idx[:, 'actual'], :]
                    .reset_index(1, drop=True)
                    .rename(columns={'predicted':'correlation'}))
r_squared_scores


# Impressively high! Let's look at the scatterplots of actual vs predicted ratings for each position:

# In[ ]:


sns.lmplot(data=comparison, x='predicted', y='actual', col='position', hue='position', col_wrap=3)


# This is pretty good performance, though if you look closely you'll see that the least squares line appears to always (or almost always) have a non-zero intercept, suggesting that the calculator can be improved by simply adding a constant for each position
# 
# Consider the slope and intercept parameters of an OLS model for every position, where the actual rating is the y variable and the predicted rating is the x variable:

# In[ ]:


param_dict = {}
for position in root_positions:
    df = comparison[comparison['position'] == position]
    params = smf.ols(formula='actual ~ predicted + 1', data=df).fit().params
    param_dict[position] = params
pd.concat(param_dict).to_frame('value')


# The important take-away from the above table is that the OLS model has an intercept parameter approximately equal to -1 **for every position except GK.** Since our calculator sticks to integer values we'll round off these intercepts, and subtract 1 from the predicted value wherever `position != 'gk'`.
# 
# New comparison dataframe:

# In[ ]:


comparison_new = (comparison
                  .assign(predicted_new=comparison['predicted']
                          .where(comparison['position'] == 'gk',
                                 comparison['predicted'] - 1))
                  .assign(error_new=lambda df: df['actual'] - df['predicted_new']))
comparison_new.nlargest(10, 'predicted_new')


# ## Now, our new mean square error:

# In[ ]:


mse_new = (comparison_new['error_new'] ** 2).mean()
mse_new


# We reduced the MSE from 1.05 to 0.11! Not bad for simply subtracting 1 from the non-goalkeeper ratings.

# # Conclusion
# 
# The player rating calculator on sofifa.com is pretty good, and you can make it better by taking one point off the ratings for outfield positions. There are still some differences between the predicted and actual values though, and it would be worth investigating if the data can explain any of these differences.
