#!/usr/bin/env python
# coding: utf-8

# # *INTRODUCTION* 
# The 2018-2019 NBA season was like a dream come true for Raptors fans. Admittedly, I wasn't there for the full ride of those 24 grueling seasons, but I was gripping my seat during every minute of those nail-biting playoffs series and the NBA Finals (a series no one thought we would win). The 4-bounce game-winning Game 7 buzzer beater (FIRST IN HISTORY!) against the 76ers will forever be a part of the collective Toronto sports culture. Those series are what got me to truly follow basketball, and 6 months later I started my journey navigating the world of machine learning, going through the motions of Coursera and tutorials. Now, a couple of months down the line, I thought about a simple prediction I could make using my newfound knowledge. The NBA MVP award instantly sprung to mind; the target value was immediately obvious: points won (and perhaps award share. As with most sports, there is a wealth of historical data available, enabling quick and efficient model creation. For the purposes of this notebook, I will be using danchyy's [MVP Votings Throughout History dataset] [1].
# ![IS THIS THE DAGGER?][2]
# > &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**_"IS THIS THE DAGGER?"_**
# 
# [1]: https://www.kaggle.com/danchyy/nba-mvp-votings-through-history 
# [2]: https://static01.nyt.com/images/2019/05/14/sports/13shot4-sub/merlin_154781583_59ee5f0b-d267-40bf-a76e-7fd127c9adb1-articleLarge.jpg?quality=75&auto=webp&disable=upscale

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting correlation
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
mvp_votings = pd.read_csv("../input/nba-mvp-votings-through-history/mvp_votings.csv")
test_data = pd.read_csv("../input/nba-mvp-votings-through-history/test_data.csv")


# In[ ]:


mvp_votings.head()


# ## Who's On This List?
# Coming as no surprise, the list of players included on the MVP votings list includes some of the consensus top NBA players of all time. In fact, any player who has won the MVP award has been enshrined in the Naismith Memorial Hall of Fame. Just to illustrate, on this list we have:
# * Michael Jordan
# * LeBron James
# * Wilt Chamberlain
# * Kareem Abdul-Jabbar
# * Magic Johnson
# * Julius Erving
# * Larry Bird
# * Charles Barkley
# * Hakeem Olajuwon 
# * Kobe Bryant 
# * Shaquille O' Neal
# * Stephen Curry 
# * Kevin Durant
# * Tim Duncan
# * And many, many more...
# [1]: https://a.espncdn.com/photo/2009/0528/nba_jabbarhook01_412.jpg
# [2]: https://www.si.com/.image/c_fit%2Ccs_srgb%2Cfl_progressive%2Cq_auto:good%2Cw_620/MTY4MTg2MDIyNzgyODM4MDMz/1988-michael-jordan-001238167_0jpg.jpg
# [3]: https://lasentinel.net/wp-content/uploads/sites/5/2019/01/AP_18326142644065.jpg

# In[ ]:


test_data.head()


# ## Who are we predicting for?
# Obviously, the 2018-2019 season has been over for quite a while now and the MVP was Giannis Antetokounmpo. However, at the time that this dataset was published, the season was still on-going. The leading contenders for the award were Harden and Giannis, but the test data list contains a few more players. There's Kawhi, Curry, Lillard, Jokic, Gobert, Westbrook, PG and Embiid, just to name a few.  

# In[ ]:


print(mvp_votings.columns)
print(test_data.columns)


# ## The Statistics
# As for the data being used for predictions, it consists of basic statistics (points/assists/rebounds/steals/blocks/minutes per game, field goals made, field goal attempts, win percentage), accuracy metrics (true shooting percentage, field goal percentage, 3-point percentage, free throw shooting) and advanced metrics (**B**ox **P**lus/**M**inus, **W**in **S**hares, both per 48 and overall, usage percentage). 
# 
# The advanced metrics are mostly derived via equations from the basic metrics, so it's not ideal to examine the relationship between the 2. As the point of this notebook is to predict the next MVP and not analyze trends between statistics, relationships between these metrics will be left unexplored. Of utmost importance is the relationship between each variable and the points won/award share.

# # Basic Statistics

# In[ ]:


#The ever-important scoring metric: points per game
plt.plot(mvp_votings['pts_per_g'], mvp_votings['points_won'], 'bo')
plt.ylabel('MVP Votes Won')
plt.xlabel('Points Per Game')


# Due to the nature of MVP voting, only a **handful of players** actually receive more than 200 points per voting season. Even with all the noise present in the graph, it is easy to see that higher scoring generally translates to higher point totals. Curry's unanimous MVP season, for example, is the highest point on the graph, wherein he averaged 30.1 points. In general, it seems as if a PPG value from 27 to 32 is the "sweet spot" for voters. Historical scoring performances may involve a high count of free throw attempts per game, which may damage a player's narrative.

# In[ ]:


plt.plot(mvp_votings['ast_per_g'], mvp_votings['points_won'], 'bo')
plt.ylabel('MVP Votes Won')
plt.xlabel('Assists Per Game')


# **Surprisingly, assists per game do not seem to have any sort of overall trend in relation to MVP voting.** Players at both ends of the assist spectrum have received high vote counts. In reality, this graph demonstrates that averaging more than 10 assists per game is a relative rarity even for MVP-caliber players. The last season in which the MVP winner averaged 10+ assists per game was in 2016-2017, when Russell Westbrook **averaged a triple double for an entire season.**  

# In[ ]:


plt.plot(mvp_votings['trb_per_g'], mvp_votings['points_won'], 'bo')
plt.ylabel('MVP Votes Won')
plt.xlabel('Total Rebounds Per Game')


# **Total rebounds have even less of a trend than assists, and the distribution is much more tightly packed.** It doesn't seem as if total rebounding has strong predictive power for MVP voting. Even further, rebounding anomalies such as Dennis Rodman (whose monstrous 18.7, 17.3 and 16.8 RPG seasons are shown) saw abysmal voting in their favour (26, 1 and 9 points).

# In[ ]:


plt.plot(mvp_votings['stl_per_g'], mvp_votings['points_won'], 'bo')
plt.ylabel('MVP Votes Won')
plt.xlabel('Steals Per Game')


# **As for steals per game, there seems to be a linear trend.** It appears to form stacks, due to the narrow range recorded for steals per game. There seems to be a cluster of players receiving 500-700 points between 1.5 to 2.0 steals per game, which means it could serve as an indicator of points earned better than rebounds and assists.   

# In[ ]:


plt.plot(mvp_votings['blk_per_g'], mvp_votings['points_won'], 'bo')
plt.ylabel('MVP Votes Won')
plt.xlabel('Blocks Per Game')


# **As always, defense doesn't seem to factor in as much as it should, overshadowed by players who can score-at-will.** That is not to say, however, that there are no players with high blocks per game who received the MVP award, given players like Shaquille O' Neal (3.0 blocks per game, 1999-2000). Most players are clustered in the 0-1 range, which indicates that it may not be a great predictor of MVP voting. 

# In[ ]:


plt.plot(mvp_votings['win_pct'], mvp_votings['points_won'], 'bo')
plt.ylabel('MVP Votes Won')
plt.xlabel('Win Percentage')


# Unsurprisingly, win percentage has the strongest trend of any "basic" metric. Obviously, the Most Valuable Player will make sure their team wins, at least as much as humanly possible. Russell Westbrook is the only MVP whose team ended the season with less than 50 wins (< 61%). It is probably one of the most important features. 

# # Conclusions: Basic Metrics
# On a per-statistic basis, it may not seem like any of them except points per game and win percentage are accurate indicators of voting performance. However, all of them combined will serve to paint a much better picture, because just scoring is not enough to become the Most Valuable Player. It's necessary that a candidate be somewhat average in every basic statistic, making up for their lacunas in other ways. In an era when teams as a whole, for example, average 4.7-5.1 blocks per game, the MVP doesn't need to average 60% of the team's blocks. 
# # Attempts, Accuracy and Advanced Metrics

# In[ ]:


plt.figure(1)
plt.plot(mvp_votings['fga'], mvp_votings['points_won'], 'bo')
plt.ylabel('MVP Votes Won')
plt.xlabel('Field Goal Attempts')
plt.figure(2)
plt.plot(mvp_votings['fg3a'], mvp_votings['points_won'], 'bo', label = '2')
plt.ylabel('MVP Votes Won')
plt.xlabel('3-Point Field Goal Attempts')
plt.figure(3)
plt.plot(mvp_votings['fta'], mvp_votings['points_won'], 'bo')
plt.ylabel('MVP Votes Won')
plt.xlabel('Free Throw Attempts')


# In[ ]:


plt.figure(1)
plt.plot(mvp_votings['fg_pct'], mvp_votings['points_won'], 'bo')
plt.ylabel('MVP Votes Won')
plt.xlabel('Field Goal Percentage')
plt.figure(2)
plt.plot(mvp_votings['fg3_pct'], mvp_votings['points_won'], 'bo', label = '2')
plt.ylabel('MVP Votes Won')
plt.xlabel('3-Point Field Goal Percentage')
plt.figure(3)
plt.plot(mvp_votings['ft_pct'], mvp_votings['points_won'], 'bo')
plt.ylabel('MVP Votes Won')
plt.xlabel('Free Throw Percentage')


# ### Are attempts or accuracy percentages better predictors? 
# The appearances of the graphs (attempts vs. percentage) does not noticeably differ, and the strongest trend seems to be seen in the graphs for free throws (both attempts and percentage). Efficiency, however, does appear to be a significant cut-off factor for MVP winners. In the field goal percentage graph, the majority of players receiving more than 800 points in voting have a field goal percentage of at least 50%. The 3 pointer does not see the same trend, mostly due to the fact that the *3-Point Revolution* is very recent, being brought about by the outrageous shooting performances of Steph Curry. Free throw attempts see a wide variation, but a free throw percentage of at least 75% seems to be the minimum for those receiving the most points

# In[ ]:


#Data exploration, what correlatess best with the target (votes)
plt.plot(mvp_votings['per'], mvp_votings['points_won'], 'bo')
plt.ylabel('MVP Votes Won')
plt.xlabel('Player Efficiency Rating')


# In[ ]:


plt.plot(mvp_votings['ws'], mvp_votings['points_won'], 'bo')
plt.ylabel('MVP Votes Won')
plt.xlabel('Win Shares')


# In[ ]:


plt.plot(mvp_votings['ws_per_48'], mvp_votings['points_won'], 'bo')
plt.ylabel('MVP Votes Won')
plt.xlabel('Win Shares Per 48')


# In[ ]:


plt.plot(mvp_votings['usg_pct'], mvp_votings['points_won'], 'bo')
plt.ylabel('MVP Votes Won')
plt.xlabel('Usage Percentage')


# Surprisingly enough, the advanced metrics seem to have the strongest trend in relation to points won. Player Efficiency Rating and Win Shares / 48 have the strongest trend, while usage percentage is still a mix of values. Given the fact that win shares per 48 has a stronger trend, regular win shares will be disregarded when considering predictive features. For the most part, almost every other feature considered could be indicative of points won, and the incorporation of PER and WS/48 introduces redundancy (they are calculated based on the basic metrics).  

# In[ ]:


y = mvp_votings.points_won
y_2 = mvp_votings.award_share
feature_names = ['fga', 'fg3a', 'fta', 'per', 'ts_pct', 'usg_pct', 'bpm', 'win_pct', 'g', 'mp_per_g', 'pts_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'fg_pct', 'fg3_pct', 'ft_pct', 'ws_per_48']
X = mvp_votings[feature_names]


# In[ ]:


def mse_random_forest(estimators, data_X, data_y, val_X, val_y):
    test_model = RandomForestRegressor(n_estimators = estimators, random_state = 1)
    test_model.fit(data_X, data_y)
    predictions_y = test_model.predict(val_X)
    return mean_squared_error(val_y, predictions_y)

def mse_multi_layer_perceptron(hidden_layer_size_test, alpha_given, data_X, data_y, val_X, val_y):
    test_model = MLPRegressor(solver = 'lbfgs', hidden_layer_sizes = hidden_layer_size_test, alpha = alpha_given, random_state = 1) 
    #The LBFGS optimizer is being used due to dataset size
    test_model.fit(data_X, data_y)
    predictions_y = test_model.predict(val_X)
    return mean_squared_error(val_y, predictions_y)

def mse_sgd(iter_test, alpha_given, data_X, data_y, val_X, val_y):
    test_model = SGDRegressor(max_iter = iter_test, alpha = alpha_given)
    test_model.fit(data_X, data_y)
    predictions_y = test_model.predict(val_X)
    return mean_squared_error(val_y, predictions_y)


# In[ ]:


#Simply testing as many regression models as possible, both for award share and points won
#This code cell is dedicated to predicting POINTS WON, and nothing else. The next cell will deal with award share. 
from sklearn import svm #Support vector machine
from sklearn.linear_model import SGDRegressor #Stochastic Gradient Descent 
from sklearn.linear_model import Ridge #Ridge regression
from sklearn.ensemble import RandomForestRegressor #Random forest, w/ multiple decision trees
from sklearn.neural_network import MLPRegressor #Multi-layer perceptron network
from sklearn.metrics import mean_squared_error #Metric used for selecting best model
from sklearn.model_selection import train_test_split #Utility function for creating training and validation data
from sklearn import preprocessing #Scaling
#Most of these models benefit strongly from scaling, to reduce data input to have a mean of 0 and variance 1
X_scaled = preprocessing.scale(X)
#Data is split here, for training and cross-validation sets
train_X, val_X, train_y, val_y = train_test_split(X_scaled, y, random_state = 0)
train_X2, val_X2, train_y2, val_y2 = train_test_split(X_scaled, y_2, random_state = 1)
#Testing for which combination of characteristics helps produce the lowest validation error
#RANDOM FOREST REGRESSION
estimator_count = [50, 100, 200, 500, 1000]
print("RANDOM FOREST OPTIONS")
for estimator in estimator_count:
    print("Estimator count:", estimator, "| Mean Squared Error:", mse_random_forest(estimator, train_X, train_y, val_X, val_y))
mvp_model = RandomForestRegressor(n_estimators = 500, random_state = 1)#500 was found to be the best estimator count, a tad high but it should serve its purpose
print('-----------------------------------------------------------------------------------------------------------')
#MULTI-LAYER PERCEPTRON
hidden_layer_sizes_test = [50, 100, 150, 200, 250]
alpha = [0.0001, 0.0003, 0.001, 0.003]
print("MULTI LAYER PERCEPTRON OPTIONS")
for layer_size in hidden_layer_sizes_test:
    for alp_test in alpha:
        print("Layer size: ", layer_size, "| Alpha: ", alp_test, "| Mean Squared Error: ", mse_multi_layer_perceptron(layer_size, alp_test, train_X, train_y, val_X, val_y))
mvp_modelMLP = MLPRegressor(solver = 'lbfgs', hidden_layer_sizes = 150, alpha = 0.003, random_state = 1) #Minimum error was found with a layer size of 150 and alpha of 0.03
print('-----------------------------------------------------------------------------------------------------------')
#STOCHASTIC GRADIENT DESCENT
print("STOCHASTIC GRADIENT DESCENT OPTIONS")
max_iters = [500, 1000, 1500, 2000, 2500]
for iter_test in max_iters:
    for alp_test in alpha:
        print("Maximum iterations: ", iter_test, "| Alpha: ", alp_test, "| Mean Squared Error: ", mse_sgd(iter_test, alp_test, train_X, train_y, val_X, val_y))
mvp_modelSGD = SGDRegressor(max_iter = 2500, alpha = 0.0003) #Maximum iterations of 2500, alpha 3 times default
#Ridge Regression
mvp_modelRidge = Ridge()
#Support Vector Regression 
#SVR and Ridge will be left with no options tweaked, simply to keep 2 models default
mvp_modelSVR = svm.SVR()
#Model fitting
mvp_model.fit(train_X, train_y)
mvp_modelMLP.fit(train_X, train_y)
mvp_modelSGD.fit(train_X, train_y)
mvp_modelRidge.fit(train_X, train_y)
mvp_modelSVR.fit(train_X, train_y)


# In[ ]:


#TO-DO LIST
#Graphing to see whether or not it matches the original trendline
#Analyzing why each player got such a high vote-count
#SHAD Values to assess the main predictors of MVP voting as evaluated by models
#Final predictions for the 18-19 season
test_data['fg3_pct'] = test_data['fg3_pct'].fillna(value = 0)
mvp_preds = mvp_model.predict(preprocessing.scale(test_data[feature_names]))
mvp_predsMLP = mvp_modelMLP.predict(preprocessing.scale(test_data[feature_names]))
mvp_predsSGD = mvp_modelSGD.predict(preprocessing.scale(test_data[feature_names]))
mvp_predsRidge = mvp_modelRidge.predict(preprocessing.scale(test_data[feature_names]))
mvp_predsSVR = mvp_modelSVR.predict(preprocessing.scale(test_data[feature_names]))
test_data['Predicted MVP Voting Random Forest'] = mvp_preds
test_data['Predicted MVP Voting MLP'] = mvp_predsMLP
test_data['Predicted MVP Voting SGD'] = mvp_predsSGD
test_data['Predicted MVP Voting Ridge'] = mvp_predsRidge
test_data['Predicted MVP Voting SVR'] = mvp_predsSVR
print(test_data[['player', 'Predicted MVP Voting Random Forest', 'Predicted MVP Voting MLP', 'Predicted MVP Voting SGD']].sort_values('Predicted MVP Voting Random Forest', ascending = False))
print(test_data[['player', 'Predicted MVP Voting Ridge', 'Predicted MVP Voting SVR']].sort_values('Predicted MVP Voting Ridge', ascending = False))


# In[ ]:


#AWARD SHARE PREDICTIONS
print("RANDOM FOREST OPTIONS")
for estimator in estimator_count:
    print("Estimator count:", estimator, "| Mean Squared Error:", mse_random_forest(estimator, train_X2, train_y2, val_X2, val_y2))
award_model = RandomForestRegressor(n_estimators = 1000, random_state = 1)#500 was found to be the best estimator count, a tad high but it should serve its purpose
print('-----------------------------------------------------------------------------------------------------------')
#MULTI-LAYER PERCEPTRON
print("MULTI LAYER PERCEPTRON OPTIONS")
for layer_size in hidden_layer_sizes_test:
    for alp_test in alpha:
        print("Layer size: ", layer_size, "| Alpha: ", alp_test, "| Mean Squared Error: ", mse_multi_layer_perceptron(layer_size, alp_test, train_X2, train_y2, val_X2, val_y2))
award_modelMLP = MLPRegressor(solver = 'lbfgs', hidden_layer_sizes = 150, alpha = 0.0001, random_state = 1) #Minimum error was found with a layer size of 150 and alpha of 0.03
print('-----------------------------------------------------------------------------------------------------------')
#STOCHASTIC GRADIENT DESCENT
for iter_test in max_iters:
    for alp_test in alpha:
        print("Maximum iterations: ", iter_test, "| Alpha: ", alp_test, "| Mean Squared Error: ", mse_sgd(iter_test, alp_test, train_X2, train_y2, val_X2, val_y2))
award_modelSGD = SGDRegressor(max_iter = 1000, alpha = 0.003) #Maximum iterations of 2500, alpha 3 times default
#Ridge Regression
award_modelRidge = Ridge()
#Support Vector Regression 
#SVR and Ridge will be left with no options tweaked, simply to keep 2 models default
award_modelSVR = svm.SVR()
#Model fitting
award_model.fit(train_X2, train_y2)
award_modelMLP.fit(train_X2, train_y2)
award_modelSGD.fit(train_X2, train_y2)
award_modelRidge.fit(train_X2, train_y2)
award_modelSVR.fit(train_X2, train_y2)


# In[ ]:


award_preds = award_model.predict(preprocessing.scale(test_data[feature_names]))
award_predsMLP = award_modelMLP.predict(preprocessing.scale(test_data[feature_names]))
award_predsSGD = award_modelSGD.predict(preprocessing.scale(test_data[feature_names]))
award_predsRidge = award_modelRidge.predict(preprocessing.scale(test_data[feature_names]))
award_predsSVR = award_modelSVR.predict(preprocessing.scale(test_data[feature_names]))
test_data['Predicted Award Share Random Forest'] = award_preds
test_data['Predicted Award Share MLP'] = award_predsMLP
test_data['Predicted Award Share SGD'] = award_predsSGD
test_data['Predicted Award Share Ridge'] = award_predsRidge
test_data['Predicted Award Share SVR'] = award_predsSVR
print(test_data[['player', 'Predicted Award Share Random Forest', 'Predicted Award Share MLP', 'Predicted Award Share SGD']].sort_values('Predicted Award Share Random Forest', ascending = False))
print(test_data[['player', 'Predicted Award Share Ridge', 'Predicted Award Share SVR']].sort_values('Predicted Award Share SVR', ascending = False))


# # Results
# Knowing that Giannis won with a point total of 941.0 and award share of 93.2%, the models that performed the best were Random Forest (points won, predicted 1025.0 points) and SVR (award share, predicted 94.1%). Interestingly, Kawhi ranked higher than Harden from the random forest model but universally lower everywhere else. Some models even generated negative values, which may be due to some idiosyncracies of awards voting. I'll be using these fitted models again to try and predict the 2020 MVP, but the season has unfortunately been suspended due to COVID-19 and there are no indications as to when it will resume. 
# 
# Thank you for reading!
