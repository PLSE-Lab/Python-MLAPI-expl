#!/usr/bin/env python
# coding: utf-8

# **About Me**
# 
# I publish kaggle notebooks to outline my learning journey in Data Science. This is the third notebook that I have pusblished. after the last two kaggle challenges, I decided to try my hands on a challenge that coincides with my interests. And I honestly had alot of fun working on this. I am new to Data Science and aspiring to be make a career switch into the industry. However, I have been a Dota  player for far longer than I can remember.
# 
# While working on this challenge, it occurred to me the importance of Domain Knowledge for a Data Scientist. Due to my understanding about the game mechanics, I am able to quickly identify relevant features, create important features, eradicate outliers and interpret my findings meaningfully.
# 
# 
# I have structured this notebook into 8 essential steps:
# 1. Importing libraries and dataset
# 2. Performing data cleaning on trainset
# 3. Feature engineering
# 4. Feature selection
# 5. Dealing with outliers
# 6. Model selection and evaluation
# 7. Performing data cleaning on testset
# 8. Prediction and submission
# 
# 
# 
# Let's dive in!
# 
# **Step 1: Importing libraries and dataset**

# In[ ]:


# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing the models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score


# In[ ]:


# Importing datasets
train_X = pd.read_csv('../input/mlcourse-dota2-win-prediction/train_features.csv', index_col = 0)
train_y = pd.read_csv('../input/mlcourse-dota2-win-prediction/train_targets.csv', index_col = 0)
test_X = pd.read_csv('../input/mlcourse-dota2-win-prediction/test_features.csv', index_col = 0)
traindf = pd.merge(train_X, train_y, left_index = True, right_index = True)

traindf.head()


# In[ ]:


traindf.dtypes


# **Step 2: Performing data cleaning on trainset**
# 
# As we can see above, not every feature is numerical. As we know, almost all data science models work better with numerical data. So let's identify which are the non-numeric features.

# In[ ]:


traindf.select_dtypes(exclude='number')


# We can see that our target feature 'radiant_win' is a boolean, whereas 'next_roshan_team' is a string/object. We can easily convert radiant_win into a binary column made up of 1 and 0. And then we'll split 'next_roshan_team' into 2 binary columns - one for each team.

# In[ ]:


# Convert radiant_win into numerical data
traindf['radiant_win'] = (traindf['radiant_win'])*1 
traindf['radiant_win'].dtypes

# Convert next_roshan_team into numerical data
traindf['next_roshan_team_radiant'] = (traindf['next_roshan_team'] == 'Radiant')*1
traindf['next_roshan_team_dire'] = (traindf['next_roshan_team'] == 'Dire')*1
traindf = traindf.drop('next_roshan_team', axis=1)

# Checking Dtype again
tempdf = traindf.filter(['radiant_win','next_roshan_team_radiant','next_roshan_team_dire'],axis=1)
tempdf.dtypes


# **Step 3: Feature engineering**
# 
# 
# Instinctively, there are two things that jumps at me from this dataset.
# 
# Firstly, total team metrics should be more imporant than individual player's metric. This is because Dota2 is a team game where members play different roles, however the teams need to work together in order to play well.
# 
# Secondly, relative metrics are more important than absolute metrics. For example, instead of looking at how much gold each team has, we should look at the difference in gold between the teams.
# 
# In order to demonstrate the above two points, we'll go through a series of steps to create the following features:
# 1. variance in gold
# 2. variance in xp
# 3. variance in gold-to-xp ratio
# 4. variance in vision
# 5. variance in sentries planted
# 6. variance in towers killed
# 
# Point 1,2 and 5 are pretty self explanatory, we're just taking the variance between team gold, xp and number of sentry wards planted. The reason why we singled sentry wards is because in the recent versions of the game (observer wards are free while sentry wards cost 90 gold each, perhaps teams who are more willing to spend gold on sentry wards have clearer vision over their opponents).
# 
# With regards to point 3 (gold-to-xp ratio), we will compute the ratio by dividing total team gold by total team xp. The reason for this metric is because there is a distinction between gold and xp. While both features have high correlation (as we can see later), the ratio indicates the team's ability to 'last-hit' and 'grab bounties'. The more efficient the team is at these last 2 events, the higher the ratio.
# 
# With regards to point 4 (vision), we will add both observer wards and sentry wards planted and call them vision. The idea is the more wards being planted, the clearer vision the team will have over the map. This allows team to escape or set-up ganks effectively.
# 
# With regards to point 6 (towers killed), taking or having towers taken are key objectives that closely correlates to the stage of the game. The more towers killed, the closer the team is to taking down the throne. The number of towers taken indicates the amount of space created in the map. Space allows more effective manuveures and farm potential.
# 

# In[ ]:


# Creating Radiant Team columns
traindf['radiant_gold'] = traindf['r1_gold'] + traindf['r2_gold'] + traindf['r3_gold'] + traindf['r4_gold'] + traindf['r5_gold']
traindf['radiant_xp'] = traindf['r1_xp'] + traindf['r2_xp'] + traindf['r3_xp'] + traindf['r4_xp'] + traindf['r5_xp']
traindf['radiant_gold_xp_ratio'] = traindf['radiant_gold']/traindf['radiant_xp']
traindf['radiant_vision'] = traindf['r1_obs_placed'] + traindf['r2_obs_placed'] + traindf['r3_obs_placed'] + traindf['r4_obs_placed'] + traindf['r5_obs_placed'] + traindf['r1_sen_placed'] + traindf['r2_sen_placed'] + traindf['r3_sen_placed'] + traindf['r4_sen_placed'] + traindf['r5_sen_placed']
traindf['radiant_sen'] = traindf['r1_sen_placed'] + traindf['r2_sen_placed'] + traindf['r3_sen_placed'] + traindf['r4_sen_placed'] + traindf['r5_sen_placed']
traindf['radiant_towers_killed'] = traindf['r1_towers_killed'] + traindf['r2_towers_killed'] + traindf['r3_towers_killed'] + traindf['r4_towers_killed'] + traindf['r5_towers_killed']
traindf['radiant_stun'] = traindf['r1_stuns'] + traindf['r2_stuns'] + traindf['r3_stuns'] + traindf['r4_stuns'] + traindf['r5_stuns']


# In[ ]:


# Creating Dire Team columns
traindf['dire_gold'] = traindf['d1_gold'] + traindf['d2_gold'] + traindf['d3_gold'] + traindf['d4_gold'] + traindf['d5_gold']
traindf['dire_xp'] = traindf['d1_xp'] + traindf['d2_xp'] + traindf['d3_xp'] + traindf['d4_xp'] + traindf['d5_xp']
traindf['dire_gold_xp_ratio'] = traindf['dire_gold']/traindf['dire_xp']
traindf['dire_vision'] = traindf['d1_obs_placed'] + traindf['d2_obs_placed'] + traindf['d3_obs_placed'] + traindf['d4_obs_placed'] + traindf['d5_obs_placed'] + traindf['d1_sen_placed'] + traindf['d2_sen_placed'] + traindf['d3_sen_placed'] + traindf['d4_sen_placed'] + traindf['d5_sen_placed']
traindf['dire_sen'] = traindf['d1_sen_placed'] + traindf['d2_sen_placed'] + traindf['d3_sen_placed'] + traindf['d4_sen_placed'] + traindf['d5_sen_placed']
traindf['dire_towers_killed'] = traindf['d1_towers_killed'] + traindf['d2_towers_killed'] + traindf['d3_towers_killed'] + traindf['d4_towers_killed'] + traindf['d5_towers_killed']
traindf['dire_stun'] = traindf['d1_stuns'] + traindf['d2_stuns'] + traindf['d3_stuns'] + traindf['d4_stuns'] + traindf['d5_stuns']


# In[ ]:


# Creating Team Variance Columns (From Radiance POV)
traindf['var_gold'] = traindf['radiant_gold'] - traindf['dire_gold']
traindf['var_xp'] = traindf['radiant_xp'] - traindf['dire_xp']
traindf['var_gold_xp_ratio'] = traindf['radiant_gold_xp_ratio']/traindf['dire_gold_xp_ratio']
traindf['var_vision'] = traindf['radiant_vision'] - traindf['dire_vision']
traindf['var_sen'] = traindf['radiant_sen'] - traindf['dire_sen']
traindf['var_towers_killed'] = traindf['radiant_towers_killed'] - traindf['dire_towers_killed']
traindf = traindf.replace([np.inf, -np.inf], np.nan).fillna(0)


# **Step 4: Feature selection**
# 
# Now that we have created the features we find to be relevant, let's perform some data exploration to find out the correlations between these features. However as there are more than 250 features within our dataframe, let's just pick the top 10 features that correlates to our target variable - 'radiant_win'.

# In[ ]:


# Print top 10 correlated features to 'radiant_win'
cor = traindf.corr()
cor_target = abs(cor['radiant_win'])
cor_target = cor_target.sort_values(ascending=False)
print(cor_target.head(10))

# Plot top 10 correlated features into heatmap
selected_col = cor_target.head(10).index 
tempdf = traindf[selected_col]
f, ax = plt.subplots(figsize=(12, 9))
cor2 = tempdf.corr()
sns.heatmap(cor2, vmax=.8, annot=True, square=True, cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5 )
plt.show()


# Let's look at the correlation heatmap and try to interpret the results.
# 0. radiant_win: Obviously we'll disregard this correlation
# 
# 1. var_gold: When a team has gold advantage, it likely means that their heroes can be equiped with more or better items and be more effective in team fights and taking towers.
# 
# 2. var_xp: When a team has xp advantage, it likely means that their heroes are of higher levels and therefore have stronger stats and skill levels - also causing them to be more effective in team fights and taking towers.
# 
# 3. next_roshan_team_dire: Taking down Roshan means a key player within the team is now in possession of "Aegis of the Immortal". In addition to granting an extra life to the key player, it also effectively changes the strategy of the team's gameplay. Having "Aegis" usually means the team's ability to take down a critical tower (e.g. tier-3 towers) or win a critical fight that could result in a sharp turning point for the game.
# 
# 4. var_towers_killed: The more tower advantage a team has, the closer the team is to taking down the throne. It also means that the team likely has more space compared to the other team.
# 
# 5. next_roshan_team_radiant: Similar to point 3, but from the viewpoint of radiant.
# 
# 6. dires_towers_killed: In absolute numbers, dires towers need to be killed in succession before reaching the final throne. Therefore, we can see why there is a relatively high correlation. However, as mentioned before, relative metrics should take precedence over absolute metrics - and the relative metrics in point 4 should present a more accurate image of the game's state.
# 
# I have chosen to ignore the last 3 features. As discussed previously, we assume that team metrics play a more important role compared to individual metrics. In addition, the position of individual players are extremely fluid. A possible reason why we're seeing a relatively high correlation, is probably because players tend to get closer to the opponent's throne when they're pushing and winning the game, and vice versa. However this is highly related to the number of towers killed, as a sequence of towers need to be taken before reaching the final throne. Therefore sporadic positioning of individual players may not present new information to us, and can therefore be disregarded.
# 
# 
# And now that we have interpreted the results, the next step will be to decide which features we should include in our predictive model. It is safe to say, based on above interpretations, that only point 1 to 5 should be considered in the inclusion of the models. However, point 3 and 5, while a useful feature, can only be obtained in retrospect after the game end. Therefore, at the point of making game predictions, this information will not be available to us, and cannot be included in our model. And that leaves us with 3 features: "var_gold", "var_xp" and "var_towers_killed". 
# 
# 
# **Addressing possibility of comeback with "time" information**
# 
# It is at this point of the notebook, that I must make an argument for the inclusion of "game_time_x"(train set) and "game_time"(test set). Even though this metric does not show high correlation to "radiant_win". To understand the reason, we must first talk about the possibilities of "comebacks". In dota games, it is not uncommon to see the seemingly losing team turn the tides in the final hours and make a win in later parts of the game - this is what's known as "comeback".
# 
# If you're a dota fan, you'll know how Team Liquid and Miracle's Arc Warden manage an epic comeback despite Team VG's 43k gold advantage. While this comeback was epic and will probably be remembered for the remaining history of dota2, it was an extremely unlikely scenario - thats what made it so epic :P
# 
# This is because gold and xp advantages carries more weight in the later parts of the game. Huge advantages in late games makes the possibility of comeback even slimmer, and inversely advantages in early games are being downplayed proportionately. Therefore, I would argue the case for a "time" component to be included in our prediction model - as it moderates the significance of Gold, XP and Tower advantages at different stages of the game.

# In[ ]:


# Keeping only relevant columns
cols_to_keep = ['var_xp','var_gold','var_towers_killed','game_time_x']
traindf_X = traindf.filter(cols_to_keep,axis=1)
traindf_y = traindf[['radiant_win']]

traindf_X.head()


# In[ ]:


traindf_y.head()


# **Step 5: Dealing with outliers**
# 
# Now that the features have been selected, the final step would be to weed out outliers. One way to do it is by looking at a scatterplots among these features.

# In[ ]:


# Comparing scatterplots among key features
sns.set()
sns.pairplot(traindf[cols_to_keep], height = 2.5)
plt.show();


# It is obvious at first glance that entries with "game_time_x == 0" are the outliers. Not sure why it happened, but one thing is for certain, it's impossible for teams to gain such level of gold and experience before the game begins - therefore the conclusion is that "game_time_x" was not accurately recorded for these entries. As we're looking at only about 20 entries among the dataset of more than 30,000 entries, we can choose to remove them as part of the trainset.

# In[ ]:


# Dropping outliers where game_time_x == 0
traindf = traindf_X
traindf['radiant_win'] = traindf_y
traindf = traindf[traindf.game_time_x > 0]
traindf_y = traindf[['radiant_win']]
traindf_X = traindf.drop(['radiant_win'],axis=1)


# **Step 6: Model selection and evaluation**
# 
# For this challenge, I have decided on using Logistic Regression. Instinctively, when "Gold, XP, Tower" advantages crosses a threshold value, the possibility of "comeback" happening gets close to zero. Therefore, the shape of the points should coincide with the shape of our typical Logistic Regression model.
# 
# As we're using regression to solve for a classification problem, the ROC_AUC will be an appropriate scoring method of choice. To learn more about this, you can visit: https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
# 
# Using the Logistic Regression seems to yield encouraging results on the testset. Cross Validation shows a mean score of ~0.81. It is a good score, as any higher, we may run into the risk of overfitting.

# In[ ]:


# Modelling with Logistics Regression
model = LogisticRegression(solver='lbfgs')

# calcuate ROC-AUC for each split
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=8)
cv_score_mean = (cross_val_score(model, traindf_X, traindf_y.values.ravel(), cv=cv, scoring='roc_auc')).mean()
print(cv_score_mean)


# **Step 7: Perform data cleaning on testset**
# 
# Now all that's left is to process out testset the same method we did for our trainset.

# In[ ]:


# prep test_X data
test_X['radiant_gold'] = test_X['r1_gold'] + test_X['r2_gold'] + test_X['r3_gold'] + test_X['r4_gold'] + test_X['r5_gold']
test_X['radiant_xp'] = test_X['r1_xp'] + test_X['r2_xp'] + test_X['r3_xp'] + test_X['r4_xp'] + test_X['r5_xp']
test_X['radiant_towers_killed'] = test_X['r1_towers_killed'] + test_X['r2_towers_killed'] + test_X['r3_towers_killed'] + test_X['r4_towers_killed'] + test_X['r5_towers_killed']
test_X['dire_gold'] = test_X['d1_gold'] + test_X['d2_gold'] + test_X['d3_gold'] + test_X['d4_gold'] + test_X['d5_gold']
test_X['dire_xp'] = test_X['d1_xp'] + test_X['d2_xp'] + test_X['d3_xp'] + test_X['d4_xp'] + test_X['d5_xp']
test_X['dire_towers_killed'] = test_X['d1_towers_killed'] + test_X['d2_towers_killed'] + test_X['d3_towers_killed'] + test_X['d4_towers_killed'] + test_X['d5_towers_killed']
test_X['var_gold'] = test_X['radiant_gold'] - test_X['dire_gold']
test_X['var_xp'] = test_X['radiant_xp'] - test_X['dire_xp']
test_X['var_towers_killed'] = test_X['radiant_towers_killed'] - test_X['dire_towers_killed']

# replace game_time_x with game_time for test_X
cols_to_keep = ['var_xp','var_gold','var_towers_killed','game_time']
test_X = test_X.filter(cols_to_keep,axis=1)


# **Step 8: Prediction and submission**

# In[ ]:


# Make predictions and create submission file
model.fit(traindf_X, traindf_y.values.ravel())
submission = pd.DataFrame()
submission['match_id_hash'] = test_X.index
submission['radiant_win_prob']= model.predict_proba(test_X)[:, 1]
submission.to_csv('submission.csv', index=False)

