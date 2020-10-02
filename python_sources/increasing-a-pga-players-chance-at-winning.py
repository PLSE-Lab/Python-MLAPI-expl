#!/usr/bin/env python
# coding: utf-8

# # How does a PGA player increase their odds of winning?
# # Introduction
# In this kernel, I'll use PGA ratings for things like power, short game, accuracy, etc. to try and determine which are the most important in predicting a PGA match winners throughout the 2010-2016 seasons. Through using a non-linear regression, I can determine the marginal effects of each explanatory parameter in increasing the probability of at least one win.
# # Data processing
# The first step would be to transpose the data structure from a listwise panel set to a columnized dataset, but only for the statistics column. Special thanks to [Mike Sadowski](https://www.kaggle.com/mesadowski/moneyball-golf-scikit-learn-and-tf-estimator-api)'s kernel for the process code.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import Logit as logit

dataset = pd.read_csv('../input/pga-tour-20102018-data/PGA_Data_Historical.csv')


# I know from the data descriptions on Kaggle that there are 1450+ variables, but I want to reduce it to those that I think will best fit my model.

# In[ ]:


#Mike's code for transposing the variables out
df = dataset.set_index(['Player Name', 'Variable', 'Season'])['Value'].unstack('Variable').reset_index()

#Creating a list for the variables I want to keep
model_vars = ['Player Name', 
              'Season',
              'Top 10 Finishes - (1ST)',
              'Scoring Rating - (RATING)',
              'Accuracy Rating - (RATING)',
              'Short Game Rating - (RATING)',
              'Putting Rating - (RATING)',
              'Power Rating - (RATING)']


# In[ ]:


df=df[model_vars]
df.head(4)


# In[ ]:


df['Top 10 Finishes - (1ST)'].fillna(0, inplace = True)
df=df.dropna()


# In[ ]:


#Column renaming and converting data types
df.rename(columns={'Top 10 Finishes - (1ST)':'wins', 'Scoring Rating - (RATING)':'score','Accuracy Rating - (RATING)':'accu', 'Short Game Rating - (RATING)':'sg', 'Putting Rating - (RATING)':'putt','Power Rating - (RATING)':'power', 'Scoring Rating - (RATING)':'score'}, inplace = True)
for col in  df.columns[2:]:
   df[col] = df[col].astype(float)
df.info()

#Creating a binary win variable where 1 = at least 1 win and 0 = no wins
df['win'] = df['wins']
df['win'].replace([2,3,4,5], 1, inplace = True)
df['win'].value_counts()


# In addition, it will be interesting to visualize the homogeneity of player ratings. I would expect that most every professional golf player centers around a specific rating, though the power rating can vary greatly by age.

# In[ ]:


plt.subplot(2,3,1)
df['accu'].plot.hist()
plt.xlabel('Accuracy Rating')
plt.subplot(2,3,2)
df['score'].plot.hist()
plt.xlabel('Scoring Rating')
plt.subplot(2,3,3)
df['sg'].plot.hist()
plt.xlabel('Short Game Rating')
plt.subplot(2,3,4)
df['putt'].plot.hist()
plt.xlabel('Putting Rating')
plt.subplot(2,3,5)
df['power'].plot.hist()
plt.xlabel('Power Rating')
plt.tight_layout()
print('Figure 1: Homogeneity in Ratings')
plt.show()


# # Logit MLE
# The logit or probit model is used when the variable of interest is a discrete binary parameter. Given that a new variable was generate that indicates at **least 1** win, this will be the dependent variable. I am interested in how certain variables, specifically ratings, are related to the probability of winning PGA matches. Below I specify the explanatory variables as well as the methodology of the logit estimation procedure.
# ## Method
# In the PGA historical dataset, there are only 5 ratings listed for each player. The empirical model follows and includes these ratings,
# * Dependent: win, where 1 = won at least 1 match, 0 = won no matches
# * Explanatory: driving accuracy (%), power rating, short game rating, putt rating, scoring rating; where all ratings are 1-10 continuous
# 
# The logit model uses iterations to converge to the beta parameters that maximize the probability of a win in our case. The logit model stems from the [Random Utility Model (RUM)](https://link.springer.com/article/10.1007%2FBF00133443?LI=true), but this overview will be skipped and I'm assuming that the reader is familiar with a latent class varaible. For this model, the probability that a player win's at least 1 match is such that the latent class variable is greater than 0,
# $$Pr(WIN_i) = Pr(\beta_0 + \beta_1 accu_i + \beta_2 power_i + \beta_3 sg_i + \beta_4 putt_i + \beta_5 score_i + \epsilon_i > 0)$$
# 
# $$= Pr(-\beta_0 - \beta_1 accu_i - \beta_2 power_i - \beta_3 sg_i - \beta_4 putt_i - \beta_5 score_i < \epsilon_i)$$
# 
# $$ = Pr(-\beta 'X_i < \epsilon_i)$$
# 
# Where $-\beta 'X_i$ are vectors of our explanatory parameters. From here, the cumulative probability is standardized by $\sigma = 1$. More rearranging,
# 
# $$=Pr(\epsilon_i <\beta 'X_i)$$
# 
# Where $Pr(\epsilon_i <\beta 'X_i)=\Phi(\beta 'X_i)$, therefore:
# 
# $$Pr(WIN)=\Phi(\beta 'X_i)$$
# 
# $$Pr(LOSE)=1-\Phi(\beta 'X_i)$$
# 
# For the logistic regression, the CDF is,
# 
# $$\Phi(\beta 'X_i)=\frac{e^{\beta 'X_i}}{1+e^{\beta 'X_i}}$$
# 
# The likelihood function then becomes the product of outcomes, and the log likelihood is,
# 
# $$\max L=\prod(\Phi(\beta 'X_i))^y_i(1-\Phi(\beta 'X_i))^{1-y_i}$$
# 
# $$\max lnL=\sum y_i(\Phi(\beta 'X_i))+ \sum(1-y_i)(1-\Phi(\beta 'X_i))$$
# 
# This function is not solvable through direct calculations so it must be done iteratively. Below is the logisitic regression performed over 7 iterations before it converges.
# 
# # Estimation
# In this estimation, it is important to run something like a logistic on the entire dataset. The reason being is that they contain multiple ratings for the same players over multiple seasons. This causes interpretation issues as well as just being poor practice to double count. In an effort to eliminate the problem, the estimation is run on the seasons 2010 to 2016 seperately, and compared. The output below is reported in marginal effects as the beta estimates from a logistic indicate intensity. By converting the estimates to marginal effects using the [delta method](http://www.ucdenver.edu/academics/colleges/PublicHealth/resourcesfor/Faculty/perraillon/perraillonteaching/Documents/week%2013%20margins.pdf), percentage change in the probability of getting at least 1 win are reported by the coefficient.

# In[ ]:


#Logistic regression
df_16 = df[df['Season']==2016]
df_15 = df[df['Season']==2015]
df_14 = df[df['Season']==2014]
df_13 = df[df['Season']==2013]
df_12 = df[df['Season']==2012]
df_11 = df[df['Season']==2011]
df_10 = df[df['Season']==2010]
logit16 = logit(df_16['win'], df_16[['accu','power','sg','putt','score']]).fit()
logit15 = logit(df_15['win'], df_15[['accu','power','sg','putt','score']]).fit()
logit14 = logit(df_14['win'], df_14[['accu','power','sg','putt','score']]).fit()
logit13 = logit(df_13['win'], df_13[['accu','power','sg','putt','score']]).fit()
logit12 = logit(df_12['win'], df_12[['accu','power','sg','putt','score']]).fit()
logit11 = logit(df_11['win'], df_11[['accu','power','sg','putt','score']]).fit()
logit10 = logit(df_10['win'], df_10[['accu','power','sg','putt','score']]).fit()

print('2016 Season:\n', logit16.get_margeff(at='mean',method='dydx').summary(),
      '\n2015 Season:\n', logit15.get_margeff(at='mean',method='dydx').summary(),
      '\n\n2014 Season:\n', logit14.get_margeff(at='mean',method='dydx').summary(),
      '\n\n2013 Season:\n', logit13.get_margeff(at='mean',method='dydx').summary(),
      '\n\n2012 Season:\n', logit12.get_margeff(at='mean',method='dydx').summary(),
      '\n\n2011 Season:\n', logit11.get_margeff(at='mean',method='dydx').summary(),
      '\n\n2010 Season:\n', logit10.get_margeff(at='mean',method='dydx').summary()
)


# For each logit, the correct classification percentage and iteration is reported in the optimazation output. The correct classification is the percentage of accurate predictions by the estimated model. These correct predictions are fairly low, in the 30% range. This means that the beta coefficients are only correctly determining whether a player win's or not 35% of the time. This low accuracy rating hints at our explanatory variables not being accurate predictors of winning.
# 
# ## Marginal Effects Interpretation
# The follow is an example on how to interprete the marginal effects coefficients for the 2016 season.
# * If a player increased their ACCURACY rating by .1, they are expected to **increase** their probability of winning at least one match **by 0.4%**, though this is highly insignificant and can be ignored.
# * If a player increased their POWER rating by .1, they are expected to **increase** their probability of winning at least one match **by 1.93%**
# * If a player increased their SHORT GAME rating by .1, they are expected to **increase** their probability of winning at least one match **by 3.1%**, though this is highly insignificant and can be ignored.
# * If a player increased their PUTTING rating by .1, they are expected to **decrease** their probability of winning at least one match **by 10.23%**
# * If a player increased their SCORE rating by .1, they are expected to **increase** their probability of winning at least one match **by 3.1%**, though this is highly insignificant and can be ignored.
# 
# ## Comparisons Across Seasons
# When comparing coefficient signs and statistical significances across the 6 seasons, there is a lot of inconsistency. In addition, (not reported in the output above), the pseudo R-squared for these models is extremely small. So what does this mean?
# # Key Takeaways
# * Player ratings are extremely poor predictors of match winning
# * Player ratings are quite homogeneous, and may not offer enough variation to accurately model against
# * It is likely that higher resolution data on match performance is a more accurate predictor
