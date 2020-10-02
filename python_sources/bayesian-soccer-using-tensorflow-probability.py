#!/usr/bin/env python
# coding: utf-8

# # Multilevel Bayesian Model for Soccer Match Predictions

# This kernel introduces multilevel bayesian modelling with [TensorFlow Probability](https://www.tensorflow.org/probability) applied to soccer match predictions.  
# 
# It may be seen as an extension of a [previous kernel](https://www.kaggle.com/fernandoramacciotti/bayesian-model-using-greta) that I built using the [R's package called greta](https://greta-stats.org) and it is essentially based on a [PyMC3](https://docs.pymc.io/) model, explained in detail in this [article by Gijs Koot](http://gijskoot.nl/bayesian/sports/soccer/predictions/pymc3/2018/02/07/knvb-model.html).
# 
# On top of it, I would like to acknowledge [Junpeng Lao](https://junpenglao.xyz/)'s help to understand better TensorFlow Probability's features as well as translating the model (as in this [GitHub's issue](https://github.com/tensorflow/probability/issues/601))

# ---
# The basically idea of this multilevel model is to model each team, home and away, probability density of score a goal. The final result of the match is, therefore, the difference between scored home and away goals.  
# 
# The framework starts by modelling attacking and defense rate for each team, but tries to consider a home advantage for the home team. Therefore, the model assumptions are the following:
# 
# The purpose of this work is to model the posterior distribution of home and away goals for each match as follows:
# 
# **likelihoods**
# \begin{align}
# homegoals &\sim Poisson(\exp{(homediff)}) \\
# awaygoals &\sim Poisson(\exp{(awaydiff)})
# \end{align}
# 
# 
# **priors**
# \begin{align}
# homediff &= AttRate_{home} - DefRate_{away} + HomeAttAdvantage_{home} \\
# awaydiff &= AttRate_{away} - DefRate_{home} - HomeDefAdvantage_{home} \\
# \end{align}  
# 
# 
# **atack rate**
# \begin{align}
# AttRate_i &= BaseAtt + AttRate_{i, non-centered} * \tau^{att}_i \\
# BaseAtt &\sim Normal(0, 1) \\
# AttRate_i &\sim Normal(0, 1) \\
# \tau^{att}_i &\sim Gamma(2, 2) \\
# \end{align} 
# 
# **defense rate**
# \begin{align}
# DefRate_i &= DefRate_{i, non-centered} * \tau^{def}_i \\
# DefRate_i &\sim Normal(0, 1) \\ 
# \tau^{def}_i &\sim Gamma(2, 2) \\
# \end{align} 
# 
# **home attacking advantage**
# \begin{align}
# HomeAttAdvantage_{home} &= BaseHomeAtt + HomeAttRate_{i, non-centered} * \tau^{HomeAtt}_i \\
# BaseHomeAtt &\sim Normal(0, 1) \\
# HomeAttRate_i &\sim Normal(0, 1) \\
# \tau^{HomeAtt}_i &\sim Gamma(2, 2) \\
# \end{align} 
# 
# **home defense advantage**
# \begin{align}
# HomeDefAdvantage_{home} &= BaseHomeDef + HomeDefRate_{i, non-centered} * \tau^{HomeDef}_i \\
# BaseHomeDef &\sim Normal(0, 1) \\
# HomeDefRate_i &\sim Normal(0, 1) \\
# \tau^{HomeDef}_i &\sim Gamma(2, 2) \\
# \end{align}
# 
# where $i = 1, \dots, n$, where $n$ is the number of teams playing the league (usually 20)
# 
# ---

# ## Coding

# In[ ]:


import numpy as np
import pandas as pd
import sqlite3
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm.auto import tqdm

tfd = tfp.distributions
tfb = tfp.bijectors

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('max_columns', 50)

import warnings
warnings.filterwarnings('ignore')


# ### Read SQL Data 

# In[ ]:


# data is stored in SQL DB, so we need to establish a connection with it
# let's use SQLite3 and Pandas for it

# create engine for connection
engine = sqlite3.connect('../input/soccer/database.sqlite')

# connect and get dfs
matches = pd.read_sql_query('SELECT * FROM Match', con=engine)
teams = pd.read_sql_query('SELECT * FROM Team', con=engine)
countries = pd.read_sql_query('SELECT * FROM Country', con=engine)


# In[ ]:


matches.head()


# In[ ]:


teams.head()


# In[ ]:


countries.head()


# In[ ]:


# let's merge all dataframes
df = pd.merge(matches, teams, left_on=['home_team_api_id'], right_on=['team_api_id'])
df = df.merge(teams, left_on=['away_team_api_id'], right_on=['team_api_id'],
              suffixes=('_home', '_away'))
df = df.merge(countries, left_on=['country_id'], right_on=['id'])

df.sample(5)


# ### Selecting a league to work on

# Let's select Italian's Serie A League, season 2015/2016, just to illustrate how the model works.
# 
# We will model the 1st leg of each match (i.e. 19 rounds/matches) and predict the second. Of course the model and final standings can be predicted match by match (and perhaps that's the more intuitive way to understand bayesian update) but we will stick with this approach just for illustration purpose.

# In[ ]:


def get_league_year_data(df, season, country):
    mask_league = df.season == season
    mask_country = df.name == country
    return df.loc[mask_league & mask_country].reset_index(drop=True)


# In[ ]:


# params
season = '2015/2016'
country = 'Italy'
df_italy = get_league_year_data(df, season, country)

df_italy.sample(5)


# Our training data will be first leg of each match, i.e. up to round 19 (or stage 19)

# In[ ]:


def get_training_data(df_season, stage):
    mask_stage = df_season.stage <= stage
    return df_season.loc[mask_stage]

# params
training_stage = 19
df_train = get_training_data(df_italy, training_stage)

# check max stage
df_train.stage.max() # 19


# ## Partial standings

# In[ ]:


# let's have a look at partial standings up to stage 19
def get_standing(df_season, partial=None):
    '''function to calculate (partial) standings.
    If `partial` params is passed, then the standing is calculated up to that stage.
    
    Win: 3pts
    Draw: 1pt
    Lose: 0pt
    '''
    aux = df_season.copy()
    if partial is not None:
        aux = aux.loc[aux.stage <= partial]
    # create flag for each results, home and away
    aux['home_win'] = aux['home_team_goal'] > aux['away_team_goal'] # home win
    aux['draw'] = aux['home_team_goal'] == aux['away_team_goal'] # draw
    aux['away_win'] = aux['home_team_goal'] < aux['away_team_goal'] # away win
    
    # columns for standings
    # we will groupby team and count home and away performance
    rename_home = {
        'team_long_name_home': 'Team',
        'home_team_goal': 'H-GF', # home goals for
        'away_team_goal': 'H-GA', # home goals against
        'home_win': 'H-W',        # home wins
        'draw': 'H-D',            # home draws
        'away_win': 'H-L',        # home losses
        'stage': 'H-Played',      # home played
    }
    # away is inverted
    rename_away = {
        'team_long_name_away': 'Team',
        'home_team_goal': 'A-GA', # away goals for
        'away_team_goal': 'A-GF', # away goals against
        'home_win': 'A-L',        # away wins
        'draw': 'A-D',            # away draws
        'away_win': 'A-W',        # away losses
        'stage': 'A-Played',      # away played
    }
    # agg to calculate (all sums except played that is count)
    home_agg = {col: 'sum' for col in rename_home.values() 
                if 'Played' not in col 
                if 'Team' not in col}
    home_agg['H-Played'] = 'count'
    away_agg = {col: 'sum' for col in rename_away.values() 
                if 'Played' not in col 
                if 'Team' not in col}
    away_agg['A-Played'] = 'count'
    # generating DFs
    home_tmp = aux.rename(columns=rename_home).groupby('Team').agg(home_agg).astype(int)
    away_tmp = aux.rename(columns=rename_away).groupby('Team').agg(away_agg).astype(int)
    # adding partial results
    standings = pd.concat([home_tmp, away_tmp], axis=1)
    # get overall stats
    standings['Played'] = standings['H-Played'] + standings['A-Played']
    standings['W'] = standings['H-W'] + standings['A-W']
    standings['D'] = standings['H-D'] + standings['A-D']
    standings['L'] = standings['H-L'] + standings['A-L']
    # goals
    standings['GF'] = standings['H-GF'] + standings['A-GF']
    standings['GA'] = standings['H-GA'] + standings['A-GA']
    # goals diff
    standings['H-GD'] = standings['H-GF'] + standings['H-GA']
    standings['A-GD'] = standings['A-GF'] + standings['A-GA']
    standings['GD'] = standings['H-GD'] + standings['A-GD']
    # points
    standings['H-Pts'] = standings['H-W'] * 3 + standings['H-D'] * 1
    standings['A-Pts'] = standings['A-W'] * 3 + standings['A-D'] * 1
    standings['Pts'] = standings['H-Pts'] + standings['A-Pts']
    # sort by pts, then Wins, then GF
    cols_order = ['Pts', 'Played', 'W', 'D', 'L', 'GF', 'GA', 'GD',
                  'H-Pts', 'H-Played', 'H-W', 'H-D', 'H-L', 'H-GF', 'H-GA', 'H-GD',
                  'A-Pts', 'A-Played', 'A-W', 'A-D', 'A-L', 'A-GF', 'A-GA', 'A-GD',]
    return standings.sort_values(by=['Pts', 'W', 'GF'], ascending=False)[cols_order]


# In[ ]:


get_standing(df_italy, 19)


# (results are ok, validated [here](https://www.worldfootball.net/schedule/ita-serie-a-2015-2016-spieltag/19/))
# 
# We expect, intuitively, that the teams with most Goal For (`GF`) get higher estimated attack rate than those with fewer goals scored. The same (inverted) logic applied for defense rate, i.e. teams with least goals against should have stronger defense rate.
# 
# For home advantage, we should see Fiorentina and Roma, for example, with stronger estimated parameters for both attack and defense at home.

# # Modelling

# ### Label Encode

# In[ ]:


# we should first label encode the teams
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(np.concatenate((df_train['team_long_name_home'].unique(), 
                       df_train['team_long_name_away'].unique())))

df_train['home_team_new_id'] = le.transform(df_train['team_long_name_home'])
df_train['away_team_new_id'] = le.transform(df_train['team_long_name_away'])


# ### Bookkeeping

# In[ ]:


home_team = df_train['home_team_new_id'].values
away_team = df_train['away_team_new_id'].values
home_score = tf.cast(df_train['home_team_goal'], tf.float32)
away_score = tf.cast(df_train['away_team_goal'], tf.float32)
scores = (home_score, away_score)
num_home_teams = df_train['team_long_name_home'].nunique()
num_away_teams = df_train['team_long_name_away'].nunique()


# ### Model with TFP

# In[ ]:


# thanks for Junpeng Lao's help (https://github.com/tensorflow/probability/issues/601)

Root = tfd.JointDistributionCoroutine.Root
def model():  # <== need to be a model with no input and no return
    # Home attack rate
    attack_hyper = yield Root(tfd.Sample(tfd.Normal(loc=0., scale=1.), 1))
    attack_hyper_sd = yield Root(tfd.Sample(tfd.Gamma(concentration=2., rate=2.), 1))
    attack_rate_nc = yield Root(tfd.Sample(tfd.Normal(loc=0., scale=1.), num_home_teams))
    attack_rate = attack_hyper + attack_rate_nc * attack_hyper_sd
    # Away defense rate
    defense_hyper_sd = yield Root(tfd.Sample(tfd.Gamma(concentration=2., rate=2.), 1))
    defense_rate_nc = yield Root(tfd.Sample(tfd.Normal(loc=0., scale=1.), num_away_teams))
    defense_rate = defense_rate_nc * defense_hyper_sd
    # Home attack advantage
    home_attack_hyper_sd = yield Root(tfd.Sample(tfd.Gamma(concentration=2., rate=2.), 1))
    home_attack_hyper = yield Root(tfd.Sample(tfd.Normal(loc=0., scale=1.), 1))
    home_attack_nc = yield Root(tfd.Sample(tfd.Normal(loc=0., scale=1.), num_home_teams))
    home_attack_advantage = home_attack_hyper + home_attack_nc * home_attack_hyper_sd
    # Home defense advantage
    home_defense_hyper_sd = yield Root(tfd.Sample(tfd.Gamma(concentration=2., rate=2.), 1))
    home_defense_hyper = yield Root(tfd.Sample(tfd.Normal(loc=0., scale=1.), 1))
    home_defense_nc = yield Root(tfd.Sample(tfd.Normal(loc=0., scale=1.), num_home_teams))
    home_defense_advantage = home_defense_hyper + home_defense_nc * home_defense_hyper_sd
    # Likelihood
    home_diff = tf.gather(attack_rate, home_team, axis=-1) -               tf.gather(defense_rate, away_team, axis=-1) +               tf.gather(home_attack_advantage, home_team, axis=-1)

    away_diff = tf.gather(attack_rate, away_team, axis=-1) -               tf.gather(defense_rate, home_team, axis=-1) -               tf.gather(home_defense_advantage, home_team, axis=-1)
    home_goals = yield tfd.Independent(tfd.Poisson(log_rate=home_diff), 1)
    away_goals = yield tfd.Independent(tfd.Poisson(log_rate=away_diff), 1)
  
model_jd = tfd.JointDistributionCoroutine(model)


# ## Inference Prep

# In[ ]:


# we need log prob for running MCMC chains
unnomarlized_log_prob = lambda *args: model_jd.log_prob(list(args) + [
    home_score[tf.newaxis, ...], away_score[tf.newaxis, ...]])

# number of parallel chains
num_chains = 5

# initial states (random samples from model)
initial_state = model_jd.sample(num_chains)[:-2] # except last two, which we are estimating


# In[ ]:


# space constraints
# identity (no constraint) for all rate params
# exp for standard deviations params (tau), since it must be positive
unconstraining_bijectors = [
  tfb.Identity(),
  tfb.Exp(),
  tfb.Identity(),
  tfb.Exp(),
  tfb.Identity(),
  tfb.Exp(),
  tfb.Identity(),
  tfb.Identity(),
  tfb.Exp(),
  tfb.Identity(),
  tfb.Identity(),
]


# In[ ]:


@tf.function(autograph=False)
def run_chain(init_state, step_size, number_of_steps=1000, burnin=50):

  def trace_fn(_, pkr):
    return (
        pkr.inner_results.inner_results.target_log_prob,
        pkr.inner_results.inner_results.leapfrogs_taken,
        pkr.inner_results.inner_results.has_divergence,
        pkr.inner_results.inner_results.energy,
        pkr.inner_results.inner_results.log_accept_ratio
    )
  unrolled_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
      tfp.mcmc.TransformedTransitionKernel(
          inner_kernel=tfp.mcmc.NoUTurnSampler(
              target_log_prob_fn=unnomarlized_log_prob,
              step_size=step_size),
          bijector=unconstraining_bijectors),
    num_adaptation_steps=burnin,
    step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(  # pylint: disable=g-long-lambda
        inner_results=pkr.inner_results._replace(step_size=new_step_size)
    ),
    step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
    log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
    )

  # Sampling from the chain.
  mcmc_trace, sampler_stats = tfp.mcmc.sample_chain(
      num_results=number_of_steps,
      num_burnin_steps=burnin,
      current_state=init_state,
      kernel=unrolled_kernel,
      trace_fn=trace_fn)
  return mcmc_trace, sampler_stats


# ### MCMC

# In[ ]:


get_ipython().run_cell_magic('time', '', '# params\nnchain_adapt = 100\ninitial_state = list(model_jd.sample(num_chains)[:-2])\n\nnumber_of_steps = 2000\nburnin = 500\ninit_step_size = [tf.ones_like(x) for x in initial_state]\n\n# mcmc\nmcmc_trace, sampler_stats = run_chain(initial_state, init_step_size, \n                                      number_of_steps, burnin)')


# ## Analysis

# In[ ]:


# forest plots
def forest_plot(num_chains, num_vars, var_name, var_labels, samples):
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 15), sharey=True, gridspec_kw={'width_ratios': [3, 1]})
    for var_idx in range(num_vars):
        values = samples[..., var_idx]
        rhat = tfp.mcmc.diagnostic.potential_scale_reduction(values).numpy()
        meds = np.median(values, axis=-2)
        los = np.percentile(values, 5, axis=-2)
        his = np.percentile(values, 95, axis=-2)

        for i in range(num_chains):
            height = 0.0 + var_idx + 0.05 * i
            axes[0].plot([los[i], his[i]], [height, height], 'C0-', lw=2, alpha=0.5)
            axes[0].plot([meds[i]], [height], 'C0o', ms=1.5)
        axes[1].plot([rhat], [height], 'C0o', ms=4)

    axes[0].set_yticks(np.arange(0, num_vars))
    axes[0].set_ylim(0, num_vars)
    axes[0].grid(which='both')
    axes[0].invert_yaxis()
    axes[0].set_yticklabels(var_labels)
    axes[0].xaxis.set_label_position('top')
    axes[0].set(xlabel='95% Credible Intervals for {}'.format(var_name))

    axes[1].set_xticks([1, 2])
    axes[1].set_xlim(0.95, 2.05)
    axes[1].grid(which='both')
    axes[1].set(xlabel='R-hat')
    axes[1].xaxis.set_label_position('top')

    plt.show()


# Let's plot the parameters that are non-centered and the ones that are actually not pooled, i.e. each team has its own estimated parameter and are indepentend (somehow).
# 
# From the plots below, we can conclude that a few teams have strong attacking rates: Napoli, Juventus, Fiorentina and Roma. This is great, since they are, up to round 19, the teams which scored the most goals.
# 
# The defense rate also is pretty fair estimated, since we have Juventus, Inter and Napoli as with highest estimated parameters - and from the standings we see that they are the teams with least goals against. It is interesting to notive that Frosinone has the lowest estimated defense rate and in fact they concealed the most goals.
# 
# Regarding home advantages, it is a bit more uniform, but we can see Roma, as expected, with a slightest home advantage (on both attacking and defense).

# In[ ]:


forest_plot(num_chains, len(le.classes_), 'Attack Rate NC', le.classes_, mcmc_trace[2])


# In[ ]:


forest_plot(num_chains, len(le.classes_), 'Defense Rate NC', le.classes_, mcmc_trace[4])


# In[ ]:


forest_plot(num_chains, len(le.classes_), 'Home Attack Advantage NC', le.classes_, mcmc_trace[7])


# In[ ]:


forest_plot(num_chains, len(le.classes_), 'Home Defense Advantage NC', le.classes_, mcmc_trace[10])


# ## Predictions

# Now that we are satisfied with our estimated parameters, let's run predictions from all remaining games.
# 
# It is worth noticing that is a prediction with estimated parameters up to round 19, but in practice, the parameters should be updated at every match or round and so the predictions.

# ### Map "future" matches

# In[ ]:


# we need a df with future matches and properly encoded teams
df_future = df_italy.loc[df_italy.stage > 19]

# encode
df_future['home_team_new_id'] = le.transform(df_future['team_long_name_home'])
df_future['away_team_new_id'] = le.transform(df_future['team_long_name_away'])

# bookkeeping
home_team_future = df_future['home_team_new_id'].values
away_team_future = df_future['away_team_new_id'].values


# ## Create "Future Model"  
# this model is just the workaround to run posterior predictive, i.e. generate future match results given the trace previously estimated

# In[ ]:


# future model
# we only need to change the last two parameters, which now receive future matches home and away team codes

def future_model():  # <== need to be a model with no input and no return
  # Home attack rate
  attack_hyper = yield Root(tfd.Sample(tfd.Normal(loc=0., scale=1.), 1))
  attack_hyper_sd = yield Root(tfd.Sample(tfd.Gamma(concentration=2., rate=2.), 1))
  attack_rate_nc = yield Root(tfd.Sample(tfd.Normal(loc=0., scale=1.), num_home_teams))
  attack_rate = attack_hyper + attack_rate_nc * attack_hyper_sd
  # Away defense rate
  defense_hyper_sd = yield Root(tfd.Sample(tfd.Gamma(concentration=2., rate=2.), 1))
  defense_rate_nc = yield Root(tfd.Sample(tfd.Normal(loc=0., scale=1.), num_away_teams))
  defense_rate = defense_rate_nc * defense_hyper_sd
  # Home attack advantage
  home_attack_hyper_sd = yield Root(tfd.Sample(tfd.Gamma(concentration=2., rate=2.), 1))
  home_attack_hyper = yield Root(tfd.Sample(tfd.Normal(loc=0., scale=1.), 1))
  home_attack_nc = yield Root(tfd.Sample(tfd.Normal(loc=0., scale=1.), num_home_teams))
  home_attack_advantage = home_attack_hyper + home_attack_nc * home_attack_hyper_sd
  # Home defense advantage
  home_defense_hyper_sd = yield Root(tfd.Sample(tfd.Gamma(concentration=2., rate=2.), 1))
  home_defense_hyper = yield Root(tfd.Sample(tfd.Normal(loc=0., scale=1.), 1))
  home_defense_nc = yield Root(tfd.Sample(tfd.Normal(loc=0., scale=1.), num_home_teams))
  home_defense_advantage = home_defense_hyper + home_defense_nc * home_defense_hyper_sd
  # Likelihood (this is what is changed, so that tf.gather works as expected)
  home_diff = tf.gather(attack_rate, home_team_future, axis=-1) -               tf.gather(defense_rate, away_team_future, axis=-1) +               tf.gather(home_attack_advantage, home_team_future, axis=-1)

  away_diff = tf.gather(attack_rate, away_team_future, axis=-1) -               tf.gather(defense_rate, home_team_future, axis=-1) -               tf.gather(home_defense_advantage, home_team_future, axis=-1)
  home_goals = yield tfd.Independent(tfd.Poisson(log_rate=home_diff), 1)
  away_goals = yield tfd.Independent(tfd.Poisson(log_rate=away_diff), 1)
  
future_model_jd = tfd.JointDistributionCoroutine(future_model)


# ## Posterior Predictive

# In[ ]:


get_ipython().run_cell_magic('time', '', '# posterior predictive given mcmc trace\ndists, _ = future_model_jd.sample_distributions(value=mcmc_trace)\n# take only the last two (home and away goals)\nhome_goals_future, away_goals_future = dists[-2], dists[-1]')


# In[ ]:


# sample from taken distributions
# the shape from a single sample() pass is shape=(2000, 5, 190)
# so, we essentially have 2k * 5 = 10k samples for each of the remaining 190 matches (10 matches per round * 19 rounds)
home_goals_pred = home_goals_future.sample()
away_goals_pred = away_goals_future.sample()


# In[ ]:


home_goals_pred.shape


# ## Results

# In[ ]:


# generate dataframe with results
from scipy import stats

# probability of each result (mean over all chains, which)
df_future['proba_home'] = (tf.sign(home_goals_pred - away_goals_pred) == 1).numpy().mean(axis=(0, 1))
df_future['proba_draw'] = (tf.sign(home_goals_pred - away_goals_pred) == 0).numpy().mean(axis=(0, 1))
df_future['proba_away'] = (tf.sign(home_goals_pred - away_goals_pred) == -1).numpy().mean(axis=(0, 1))


# In[ ]:


cols_to_display = ['stage', 'date', 
                   'team_long_name_home',
                   'team_long_name_away',
                   'proba_home', 'proba_draw', 'proba_away']
df_future[cols_to_display].head(10)


# We essentially have 10k competitions simulated. Let's what are the final standings distributions.

# In[ ]:


# get numpy preds
home_goals_pred_np = home_goals_pred.numpy().reshape(-1, 190)
away_goals_pred_np = away_goals_pred.numpy().reshape(-1, 190)


# **Example of simulated competition**

# In[ ]:


# example of simulated competition 
def get_simulated_competition(df_1, df_2, preds, simulation_ix=10):
    # re
    df_1_nodup = df_1.loc[:,~df_1.columns.duplicated()]
    df_2_nodup = df_2.loc[:,~df_2.columns.duplicated()]
    # stage preds
    df_2_nodup['home_team_goal'] = preds[0][simulation_ix] # home
    df_2_nodup['away_team_goal'] = preds[1][simulation_ix] # away
    
    full_df = pd.concat([df_1_nodup, df_2_nodup], axis=0, 
                        ignore_index=True, sort=False)
    final_standing = get_standing(full_df)
    
    return final_standing

get_simulated_competition(df_train, df_future, (home_goals_pred_np, away_goals_pred_np))


# In[ ]:


def get_simulated_standing_dist(df_1, df_2, preds, n_simulations=10):
    # placeholder
    sim_results = pd.DataFrame()
    for sim in tqdm(range(n_simulations)):
        # get standing
        final_standing_sim = get_simulated_competition(
            df_1, df_2, preds, simulation_ix=sim)
        final_standing_sim['position'] = np.arange(1, len(le.classes_)+1) # positions in order
        final_standing_sim['sim_number'] = sim                            # sim_number
        # concat with results df by index
        sim_results = sim_results.append(final_standing_sim[['position', 'sim_number']])
    
    return sim_results


# In[ ]:


simulations = get_simulated_standing_dist(
    df_train, df_future, (home_goals_pred_np, away_goals_pred_np),
    n_simulations=10000)


# In[ ]:


# source: http://gijskoot.nl/bayesian/sports/soccer/predictions/pymc3/2018/02/07/knvb-model.html
rankings_agg = simulations.reset_index().groupby('Team').position.value_counts(normalize=True).unstack(1).fillna(0)
rankings_agg = rankings_agg.assign(expected=rankings_agg @ np.arange(len(le.classes_) + 1, 1, -1))    .sort_values("expected", ascending=False).drop("expected", axis = "columns")

plt.figure(figsize=(12, 8))
sns.heatmap(rankings_agg, annot=True, fmt=".0%", cbar=False, cmap="hot", square=True)
plt.title('Predictions | As of round 19');


# The rankings probabilities are very spread out, as expected, since there is half tournament to go.

# **Actual final standing**

# In[ ]:


get_standing(df_italy)


# ---
# Please feel free to reach me out in case of any doubts and for any suggestion.  

# In[ ]:




