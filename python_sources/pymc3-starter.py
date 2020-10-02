#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt

results = pd.read_csv('../input/WRegularSeasonCompactResults.csv')
teams = pd.read_csv('../input/WTeams.csv')


# # PyMC3 Starter
# 
# This notebook is intended to show how a simple but effective model might be fit with `PyMC3`.  We will model the "strength" of team as being normally distributed around 0, and assume that each game the probability of a team winning is the (logit of the) difference of the strengths.
# 
# Specifically, if team $i$ has strength $t_i$, then the chance of beating team $j$ is
# $$
# p(\text{outcome}_{i, j} = 1) \sim \operatorname{Bernoulli}\left(p=\frac{1}{1 + \exp{(t_i - t_j)}}\right).
# $$
# 
# The maximum likelihood estimator of this model would be 1-hot encoding all the teams and fitting logistic regression to it.  This would likely be overfit.  By using `PyMC3`, we will also get uncertainty bounds on the strength of each team, so we can optimize to some other fitness function if we want.

# ## Loading the data
# 
# We are going to just use 2017 data, and compare the results to the AP rankings at the end of the season.

# In[ ]:


results_2017 = results[results.Season == 2017].reset_index()
teams_2017 = teams[teams.TeamID.isin(results_2017.WTeamID) | teams.TeamID.isin(results_2017.LTeamID)].reset_index(drop=True)
teams_2017['idx'] = teams_2017.index
teams_2017 = teams_2017.set_index('TeamID')


# Now map the winning and losing teams to the index in the `teams_2017` numpy array.

# In[ ]:


wloc = np.array(teams_2017.loc[results_2017.WTeamID, 'idx']).squeeze()
lloc = np.array(teams_2017.loc[results_2017.LTeamID, 'idx']).squeeze()


# ## Defining the model
# 
# This is a very simple model to define in `PyMC3`!  If we wanted the trace to record values for `p` as well, we could wrap it in a call to `pm.Deterministic`.

# In[ ]:


with pm.Model() as model:
    team_rating = pm.Normal('rating', mu=0, sd=1, shape=teams_2017.shape[0])
    p = pm.math.sigmoid(team_rating[wloc] - team_rating[lloc])
    # data is organized so the first team always won
    outcome = pm.Bernoulli('outcome_obs', p=p, observed=tt.ones_like(p))


# We can inspect the model!

# In[ ]:


model


# Now we take some samples from the model:

# In[ ]:


with model:
    trace = pm.sample()


# ## Evaluating the model
# 
# Each team's rating is normally distributed, but we can compare those by their mean.  Note that just because the mean of one team is higher than another's, it does not mean that team will win more often (particularly if the standard deviations are funny).
# 
# Here is the plot of estimates for all the teams:

# In[ ]:


pm.traceplot(trace);


# That is fairly busy, but we can get the 10 teams with the highest estimated mean strength:

# In[ ]:


top_ten = trace['rating'].mean(axis=0).argsort()[-10:]
teams_2017.set_index('idx').loc[top_ten][-1::-1].reset_index(drop=True)


# The true top 10 [from the AP News](https://www.google.com/search?q=women%27s+ncaa+basketball+2017&oq=women%27s+ncaa+basketball+2017&aqs=chrome..69i57j0l5.7760j0j1&sourceid=chrome&ie=UTF-8#sie=lg;/g/11c5bjb1bb;3;/g/11dxbvlspv;rn;fp;1) at the end of 2017 was:
# 
# 1. Connecticut
# 2. Notre Dame
# 3. South Carolina
# 4. Maryland
# 5. Baylor
# 6. Stanford
# 7.  Mississippi St
# 8. Oregon St
# 9. Duke
# 10. Florida St
