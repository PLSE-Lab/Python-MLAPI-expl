#!/usr/bin/env python
# coding: utf-8

# ![Logo](https://upload.wikimedia.org/wikipedia/en/thumb/b/b5/League_of_Legends_logo_2019.png/220px-League_of_Legends_logo_2019.png)
# 
# ## Introduction
# League of Legends (LoL) is a multiplayer online battle arena game developed and published by Riot Games for Microsoft Windows and macOS. Inspired by Warcraft III: The Frozen Throne mod Defense of the Ancients, the game follows a freemium model and is supported by microtransactions. In League of Legends, players assume the role of a "champion" with unique abilities and battle against a team of other players or computer-controlled champions. There are always two teams, the blue ones, which start at the bottom of the map and the red ones, which start at the top of the map. The goal is usually to destroy the opposing team Nexus, a structure that lies at the heart of a base protected by defensive structures, such as towers and inhibitors. To increase the strength of the champions, certain bosses can be defeated, such as dragon, baron, and herald. Each League of Legends match is discrete, with all champions starting relatively weak but increasing in strength by accumulating items and experience throughout the game. Champions span a variety of roles and blend a mixture of fantasy tropes. 
# 
# ![Map](https://gamehag.com/img/uploaded/Lci2gbpdudF4PppLvaGAEizERRhVdi.jpg)
# 
# League of Legends has an active and widespread competitive scene, which is commonly described as the preeminent global eSport and a major factor towards the industry's legitimisation. In North America and Europe, Riot Games organises the League Championship Series (LCS), located in Los Angeles and the League of Legends European Championship (LEC), located in Berlin, respectively, each of which consists of 10 professional teams. Similar regional competitions exist in China (LPL), South Korea (LCK), and various other regions. These regional competitions culminate with the annual World Championship. 
# 
# 

# In[ ]:


# Import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from pandas import DataFrame
from subprocess import check_output
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ## The Dataset
# 
# We will analyze the dataset available in Kaggle, found [here](https://www.kaggle.com/chuckephron/leagueoflegends), which according to the [author](https://www.kaggle.com/chuckephron), is a dataset with League of Legends competitive matches between 2015-2018. The matches include the NALCS, EULCS, LCK, LMS, and CBLoL leagues as well as the World Championship and Mid-Season Invitational tournaments.
# 
# The dataset in question is quite extensive and complete, but luckily, very well documented. The file `_columns.csv` seeks to clarify the meaning of each column of the data, while, of course, each line is a specific item. From these columns presented, we will mention here which ones will interest us for the analysis of the matches:
# 
# - `blueTeamTag` Blue Team's tag name (ex. Cloud9 is C9)
# - `bResult` Result of the match for Blue Team 1 is a win, 0 is a loss 
# - `goldblue`	Blue Team's total gold value by minute
# - `bKills`	List of Blue Team's kills - [Time in minutes, Victim, Killer, Assist1, Assist2, Assist3, Assist4, x_pos, y_pos]
# - `bTowers`	List of minutes that Blue Team destroyed a tower and Tower Location
# - `bInhibs`	List of minutes that Blue Team destroyed an inhibitor and Location
# - `bDragons`	List of minutes that Blue Team killed a dragon
# - `bBarons`	List of minutes that Blue Team killed a baron
# - `bHeralds`	List of minutes that Blue Team killed a rift herald 
# - `redTeamTag` Red Team's tag name (ex. Cloud9 is C9)
# - `bResult` Result of the match for Red Team 1 is a win, 0 is a loss 
# - `goldblue`	Red Team's total gold value by minute
# - `bKills`	List of Red Team's kills - [Time in minutes, Victim, Killer, Assist1, Assist2, Assist3, Assist4, x_pos, y_pos]
# - `bTowers`	List of minutes that Red Team destroyed a tower and Tower location
# - `bInhibs`	List of minutes that Red Team destroyed an inhibitor and location
# - `bDragons`	List of minutes that Red Team killed a dragon
# - `bBarons`	List of minutes that Red Team killed a baron
# - `bHeralds`	List of minutes that Red Team killed a rift herald 

# In[ ]:


# Data info
df_columns = pd.read_csv('../input/leagueoflegends/_columns.csv',sep=',')
df_original = pd.read_csv('../input/leagueoflegends/LeagueofLegends.csv',sep=',')

df_original[['bResult','goldblue','bKills','bTowers', 'bInhibs', 'bDragons', 'bBarons', 'bHeralds']].head(3)


# In[ ]:


#Look the information of the dataframe
df_original.info()
df = df_original.copy(deep=True)


# ## Manipulating the Dataset
# 
# 1. Note that the type of most of the columns we are going to use are as objects, despite being filled with integers. Thus, the conversion of these columns is necessary.
# 2. In addition, it is noted that certain columns are filled with 'lists' (pseudo lists), characterizing the value for each minute of the match, this should also be changed so that the data is more easily manipulated.
# 3. In this case we are only interested in the matches that Cloud9 (C9) participated in, so we will do a filter on the dataframe to collect only the lines that have C9 in any of the name tags.
# 4. We will also create a new column called `winner`, which will make interpreting who won the match more practical than analyzing the `bResult` and `rResult` columns. The `winner` column will have the number 1 if the blue team wins the match and the number 2 if the red team wins.

# In[ ]:


# Transform all the columns containing pseudo lists to real lists

df['goldblue'] = df['goldblue'].apply(literal_eval)
df['bKills'] = df['bKills'].apply(literal_eval)
df['bTowers'] = df['bTowers'].apply(literal_eval)
df['bInhibs'] = df['bInhibs'].apply(literal_eval)
df['bDragons'] = df['bDragons'].apply(literal_eval)
df['bBarons'] = df['bBarons'].apply(literal_eval)
df['bHeralds'] = df['bHeralds'].apply(literal_eval)

df['goldred'] = df['goldred'].apply(literal_eval)
df['rKills'] = df['rKills'].apply(literal_eval)
df['rTowers'] = df['rTowers'].apply(literal_eval)
df['rInhibs'] = df['rInhibs'].apply(literal_eval)
df['rDragons'] = df['rDragons'].apply(literal_eval)
df['rBarons'] = df['rBarons'].apply(literal_eval)
df['rHeralds'] = df['rHeralds'].apply(literal_eval)


# In[ ]:


# Capturing only the information that interests us from the data lists

data = pd.DataFrame()

data['blue_tag'] = df['blueTeamTag']
data['blue_result'] = df['bResult']
data['blue_end_gold'] = df['goldblue'].apply(max)
data['blue_kills'] = df['bKills'].apply(len)
data['blue_towers'] = df['bTowers'].apply(len)
data['blue_inhibs'] = df['bInhibs'].apply(len)
data['blue_dragons'] = df['bDragons'].apply(len)
data['blue_barons'] = df['bBarons'].apply(len)
data['blue_heralds'] = df['bHeralds'].apply(len)

data['red_tag'] = df['redTeamTag']
data['red_result'] = df['rResult']
data['red_end_gold'] = df['goldred'].apply(max)
data['red_kills'] = df['rKills'].apply(len)
data['red_towers'] = df['rTowers'].apply(len)
data['red_inhibs'] = df['rInhibs'].apply(len)
data['red_dragons'] = df['rDragons'].apply(len)
data['red_barons'] = df['rBarons'].apply(len)
data['red_heralds'] = df['rHeralds'].apply(len)

data = data[(data['blue_tag'] == 'C9') | (data['red_tag'] == 'C9')]
data = data.reset_index(drop=True)

data['winner'] = np.where(data['blue_result'] == 1, 1, 2)


data


# ## Explore the Dataset
# 
# A good indicator of how well your team is doing in a match is how many objectives your team has taken. These objectvies are:
# 
# - Gold: Gold is earned by killing monsters, players and constructions, and must be spent on items to strengthen and assist your champion.
# - Player kills: Killing players provides extra cash in addition to slowing your competitor's progress.
# - Tower kills: Towers are placed in each lane of the map and must be taken in order for your team to advance.
# - Inhibitor kills: Inhibitors are placed in each team base, and must be taken in order for your team to take the inner towers and eventually win the game.
# - Dragon kills: The Dragon is a powerful neutral monster that gives you and your teammates buffs and gold, which help establish a competitive edge over the other team.
# - Baron kills: The Baron is the most powerful monster on the map that gives your team a very competitve edge over the other team.
# - Herald kills: The Rift Herald is the third most powerful neutral monster that gives you and your teammates buffs and gold, which help establish a competitive edge over the other team.
# 
# We will use these as our source of data when predicting the outcome of each match.

# In[ ]:


fig = plt.figure(figsize=(12,12))

sns.set_style('darkgrid')
sns.heatmap(data[['blue_end_gold','blue_kills', 'blue_towers', 'blue_inhibs', 'blue_dragons', 'blue_barons', 'blue_heralds',
                  'red_end_gold','red_kills','red_towers','red_inhibs', 'red_dragons', 'red_barons', 'red_heralds', 'winner']].corr(), annot=True, square=True, cmap='coolwarm')


# 
# 
# From this heat map, we can see that the correlation for each objective is:
# 
# - Towers Kills: **~.87** (Strong correlation)
# - Inhibitors Kills: **~.71** (Strong/Moderate correlation)
# - Enemies Kills: ~.69 (Strong/Moderate correlation)
# - Barons Kills: ~.60 (Moderate correlation)
# - Dragons Kills: ~.53 (Moderate correlation)
# - Gold Earn: ~.32 (Weak correlation)
# - Heralds Kills: ~.15 (Weak correlation)
# 

# ## Split the Data
# 
# We want to split our data into a set that we train the model on and a set we test the model with. Using Scikit-learn, we can split our data so that we train on the majority (66.6%) and test the rest (33.3%) to see how we do.
# 

# In[ ]:


X = data[['blue_end_gold','blue_kills', 'blue_towers', 'blue_inhibs', 'blue_dragons', 'blue_barons', 'blue_heralds',
          'red_end_gold','red_kills','red_towers','red_inhibs', 'red_dragons', 'red_barons', 'red_heralds']]
y = data['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ## Create and Train the Model
# 
# With our data split, we can create a logistic regression model and fit it to our training data
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)


# ## Evaluate the Created Model
# 
# Now that we have a model, we can see how well we predicted the outcomes of the matches

# In[ ]:


cr = classification_report(y_test, predictions)
print('Classification Report : \n', cr)

acc = round(logmodel.score(X_test, y_test) * 100, 2)
print("Accuracy of Logistic Regression: " + str(acc) + "%")

cm = confusion_matrix(y_test,predictions)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=['2 win', '1 win'], yticklabels=['2 win', '1 win'],);


# We can see that with this configuration, only in one of the 101 cases used for testing, the prediction made was wrong (99.01% of accuracy).

# ## Testing the Predictor
# 
# Let is simulate the test of the first row of the `data` dataset, in a **TSM x C9** match, where C9 lose the match, trying to predict who will win the match from the data provided.

# In[ ]:


# Predict the TSM x C9 match
x1 = [[62729, 16, 9, 2, 1, 0, 0,
       56672, 9, 4, 0, 3, 1, 0]]

pred = logmodel.predict_proba(x1).reshape(-1,1)

win = round(logmodel.predict(x1)[0], 2)
print("Winner is :", win)

fir_prob = round(pred[0][0] * 100, 2)
sec_prob = round(pred[1][0] * 100, 2) 
print("First team (blue) win probability is: " + str(fir_prob) + "%")
print("Second team (red) win probability is: " + str(sec_prob) + "%")


# ## Conclusion
# 
# We were able to draw some conclusions from the presented study:
# 
# 1. Through the heatmap, it is evident that taking down the largest number of towers is extremely important, even more than the number of inhibitors. Furthermore, in certain situations where the player is in doubt as to whether it is worth sacrificing once to destroy a tower, it is apparently very beneficial.
# 2. The prediction of who won the match from the factors used may seem silly once the match is over. But the same prediction can become more interesting in real time, by providing players with information about what the priorities should be during the game to reverse a game in which they are being defeated, for example.
# 3. Although Logistic Regression is a relatively simple method, its accuracy reaches 100%, depending on the `random_state` used (=102, for example). However, in order to become more didactic, a `random_state` was purposely defined that would generate less than 100% accuracy, making the confusion matrix more understandable.
