#!/usr/bin/env python
# coding: utf-8

# **Hi guys, this notebook is the first one I publish on Kaggle. I tried to build some kind of case-study based on football analysis.**
# 
# **Please share your thoughts on it so I can improve my work! Thanks :)**

# ### Here is the thing
# 
# You're a data analyst for a Premier League mid-table club. After a deceiving 2016/2017 season finished below the middle of the table and the retirement of the club's lifelong striker, the team needs more than ever a new forward in order to grab one of the few European tickets.
# 
# Our club usually plays in a 4-4-1-1 defensive formation, quite popular in the league. This new striker will be surounded by a good offensive mildfielder with nice passing ability and two wingers on their favorite foot that enjoy throwing crosses. 
# 
# Based on this statement, the coach gave you the following requirements :
# 
# - **poacher** --> this striker doesn't have to be a playmaker as he will have quality teammates around him to do the job, but must be super effective in front of the target 
# - **aerial threat** --> our guy needs to be solid in the air to compete with the physical PL defenders and take benefits from our wingers' crossing quality
# - **counterattacker** --> the perfect candidate for the role should be performing well in counter-attack, as most of the chances he will get will come from this type of plays
# - **clear headed** --> due to the playstyle of our team, based on counter-attacks, our number of chances will be scarced, so we need someone who will not waste opportunities with shots that have no chance of going in
# - **experienced** --> this forward needs to have at least 5 years of experience at the highest level, in one of the Top 5 European Leagues

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#hide the warnings
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


data = pd.read_csv('/kaggle/input/football-events/events.csv')


# ### Data exploration

# The dataset we have is perfect for our mission. It sums-up the events from all the games of the 5 European major leagues between the 2011-2012 and the 2016-2017 season.
# 
# Let's discover our dataset and the explanatory file attached.

# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


txt = open("/kaggle/input/football-events/dictionary.txt","r")


# In[ ]:


txt.read()


# ### Reading of the encoding
# 
# **Type of event** 
# 
# Shot --> event_type = 1
# 
# 
# **Side**
# 
# Home --> side = 1,
# Away --> side = 2
# 
# **shot_place** is quite obvious
# 
# **Shot outcome**  
# 
# On target --> shot_outcome = 1,
# Off target --> shot_outcome = 2,
# Blocked (goalkeeper and defender?) --> shot_outcome = 3,
# Hit the bar (+post?) --> shot_outcome = 4
# 
# 
# **Position**
# 
# location feature, 18 values + 1 (not recorded)
# 
# **Body Part**
# 
# Right foot --> bodypart = 1,
# Left foot --> bodypart = 2,
# Head --> bodypart = 3
# 
# **Goal**
# 
# Goal --> is_goal = 1,
# No goal --> is_goal = 0
# 
# **Fast Break**
# 
# Action from counter attack --> fast_break = 1,
# Possession phase --> fast_break = 0

# ### "Old-shool' indicators

# To pick a good striker, we have a lot of indicators that are easy to get. One that we often see when we talk about forwards is the number of goals they score within a given period of time, or their **goal/shot ratio**. Let's get this latter metric.

# In[ ]:


#as we only focus on shots for this analysis, let's create another dataframe that only keeps the shot events
data_shot = data[data.event_type == 1]


# In[ ]:


messi = (data_shot.player == 'lionel messi')
ronaldo = (data_shot.player == 'cristiano ronaldo')


# In[ ]:


nb_shot_messi = data_shot.id_odsp[messi].count()
nb_goal_messi = data_shot.id_odsp[messi][data_shot.is_goal == 1].count()
ratio_messi = nb_goal_messi / nb_shot_messi

nb_shot_ronaldo = data_shot.id_odsp[ronaldo].count()
nb_goal_ronaldo = data_shot.id_odsp[ronaldo][data_shot.is_goal == 1].count()
ratio_ronaldo = nb_goal_ronaldo / nb_shot_ronaldo

print('Number of goals for Messi : ', nb_goal_messi)
print('Goal/shot ratio for Messi : ', ratio_messi)
print('Number of goals for Ronaldo : ', nb_goal_ronaldo)
print('Goal/shot ratio for Ronaldo : ', ratio_ronaldo)


# We can notice here that Messi has a higher goal/shot ratio than Cristiano. This is a cool indicator that can be useful, but it doesn't tell the exact truth : is Messi better than Ronaldo in front of the goal? This kind of indicator isn't enough to say so.
# 
# In fact, this ratio lacks a decisive element : the difficulty of the shots taken. We can pretend that Messi has good teammates that do all the job for him and allow him to only score taps-in, while Ronaldo has to take harder shots, and thus has a lower ratio of goals/shot.
# 
# That's why we need another way of evaluating the performances of a striker : the **Expected Goals** (xG) are the perfect indicator for that.
# 
# The objective of the xG is to give a value to each shot, between 0 and 1. With a value of 0, the shot has almost no chance to go into the net, while with a value of 1 the shot is unmissable. To give you an idea, the xG value of a penalty is 0.77.

# ### Let's build our xG model

# To perform our analysis and determine who's the best candidate for the role in our club, we are going to build our own xG model based on the dataset we have.  

# ### Data cleaning

# First of all, we saw that some shot locations are not recorded (location == 19). Let's check if there is a lot of these N/A values or not.

# In[ ]:


print ('Number of shots not located : ', data_shot.is_goal[data.location == 19].count()) 
print ('Split by goal or no goal : ', data_shot.is_goal[data.location == 19].value_counts()) 
print('      ')
print('Number of shot recorded', data_shot.is_goal.count())


# We are going to drop those N/A values for the shot location, as they represent less than 6% of the goals. Unfortunately almost all of them are goals, but we should have enough data. 
# 
# Of course, at the end some players will see their number of goals decresead, but we judge it acceptable.

# In[ ]:


data_shot = data_shot[data_shot.location != 19]


# In[ ]:


data_shot.count()


# In the features that are interesting for us, we have no NaN or undefined values, so we can keep on going.

# ### X/Y variables definition and train/test split

# Here we pick our explanatory variables. The output, obviously, will be if the shot ends in the back of the net or not.

# In[ ]:


X = data_shot[['time', 'side', 'bodypart', 'location', 'situation', 'assist_method', 'fast_break']]
y = data_shot['is_goal']


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, stratify = y)


# ### Try-on different models
# 
# Quick overview of some other classification models, to see if they can perform better than our K-nn

# In[ ]:


from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    KNeighborsClassifier(),
    LinearSVC()]


for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
print("="*30)


# It seems that the Gradient Boosting is the best technique in our case.
# 
# We should try to tune the hyperparameters to improve its accuracy, but as I'm not super comfortable with Gradient Boosting tuning for now, it won't be in this version :)

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
print('XGBoost model precision on test dataset : ', model.score(X_test, y_test) * 100)


# We get an accuracy around 91% for our model. But is that good or bad? We need to establish some baselines to judge. 
# 
# The most relevant in our case will be to predict the most frequent occurence (no goal), as our dataset is unbalanced. 

# In[ ]:


data_shot.is_goal.value_counts()


# In[ ]:


from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_test, y_test)
score = dummy_clf.score(X_test, y_test)
print('most frequent precision : ', score * 100, '%')


# In fact, the 91% accuracy is not that good, as we can reach a 89.9% accuracy if we say that all the shots resulted in no goal.
# 
# But we will have to keep this model for now, as we don't really have options to improve it.

# ### Give a value to each shot

# The purpose of our model is to give a value to each shot. To do so, we are going to use the probability of a given shot to turn into a goal. 
# 
# This can be found by using the predict_proba method on our dataset.

# In[ ]:


# add the probability to get 1 for every row as a new column
probas = model.predict_proba(X)
data_shot['xgoalpercent'] = probas[:,1] 


# In[ ]:


print('Maximum xG value for a shot : ', probas[:,1].max())
print('Minimum xG value for a shot : ', probas[:,1].min())


# ### Findings

# Let's try our xG model on the same players as in the beginning, our Argentinian and Portuguese superstars :

# In[ ]:


nb_shot = data_shot.id_odsp[messi].count()
print('Lionel Messi :')
print('Number of shots for : ', nb_shot)

print('Expected goals for : ', data_shot[messi]['xgoalpercent'].sum(axis = 0))

nb_goal = data_shot.id_odsp[data_shot.is_goal == 1][messi].count()
print('Number of goals for : ', nb_goal)
print('Difference between goals and xG : ', nb_goal - data_shot[messi]['xgoalpercent'].sum(axis = 0))
print('xG/shots :', data_shot[messi]['xgoalpercent'].sum(axis = 0) / nb_shot)

print('    ')

nb_shot = data_shot.id_odsp[ronaldo].count()
print('Cristiano Ronaldo :')
print('Number of shots : ', nb_shot)

print('Expected goals : ', data_shot[ronaldo]['xgoalpercent'].sum(axis = 0))

nb_goal = data_shot.id_odsp[data_shot.is_goal == 1][ronaldo].count()
print('Number of goals : ', nb_goal)
print('Difference between goals and xG : ', nb_goal - data_shot[messi]['xgoalpercent'].sum(axis = 0))
print('xG/shots :', data_shot[messi]['xgoalpercent'].sum(axis = 0) / nb_shot)


# We now have a much better comparison basis.
# 
# Based on the chances Messi had, an average player would have scored 131 goals, while the Argentinian scored 191. For Ronaldo, while an average player would have scored 151, the Portuguese scored 184.
# 
# Both players are over-performing, but Messi much more than Ronaldo : the difference is greater for Messi, eventhough he had less xG. 
# 
# **This is a solid argument if you want to proove that Messi is better than Ronaldo in front of the goal.**

# ### Comparison table

# In[ ]:


# we get the list of all the players, unique to make sure they only appear 1 time
list_of_players = data_shot.player.unique()
print(list_of_players)


# In[ ]:


comparison = pd.DataFrame(columns=['player', 'nb_shots', 'expected_goals', 'goals_scored', 'xg_dif',                                    'nb_headers', 'expected_head_goals', 'head_goals_scored', 'head_xg_dif',                                    'xg_per_shot', 'pct_goals_counter'])


# In[ ]:


for i in range(list_of_players.size):
    #get the player name
    player = list_of_players[i]
    #xg
    nb_shot = data_shot.id_odsp[data_shot.player == player].count()
    xg = data_shot[data_shot.player == player]['xgoalpercent'].sum(axis = 0)
    nb_goal = data_shot.id_odsp[data_shot.is_goal == 1][data_shot.player == player].count()
    xg_dif = nb_goal - xg
    #xg for headers
    nb_head = data_shot.id_odsp[data_shot.player == player][data_shot.bodypart == 3].count()
    xg_head = data_shot[data_shot.player == player][data_shot.bodypart == 3]['xgoalpercent'].sum(axis = 0)
    nb_goal_head = data_shot.id_odsp[data_shot.is_goal == 1][data_shot.bodypart == 3][data_shot.player == player].count()
    head_xg_dif = nb_goal_head - xg_head
    #xg/shot
    xg_per_shot = xg / nb_shot
    #counter-attacks
    nb_counter_goals = data_shot.id_odsp[data_shot.is_goal == 1][data_shot.fast_break == 1][data_shot.player == player].count()  
    pct_count_goals = nb_counter_goals / nb_goal * 100
    #append the player's row in the comparison dataframe
    comparison = comparison.append({'player' : player, 'nb_shots' : nb_shot, 'expected_goals' : xg, 'goals_scored' : nb_goal, 'xg_dif' : xg_dif,                                    'nb_headers' : nb_head, 'expected_head_goals' : xg_head, 'head_goals_scored' : nb_goal_head, 'head_xg_dif' : head_xg_dif,                                    'xg_per_shot' : xg_per_shot, 'pct_goals_counter' : pct_count_goals} , ignore_index=True)


# ### Indicators

# In[ ]:


comparison.columns


# In[ ]:


selection = comparison[comparison.xg_dif >= 5][comparison.goals_scored >= 10][comparison.head_goals_scored >= 5][comparison.head_xg_dif >= 2][comparison.xg_per_shot >= 0.1][comparison.pct_goals_counter >= 10]
selection.sort_values(by='xg_dif', ascending=False)


# **Cool !** We now have a table that contains every player that has shot between 2011-2012 and 2016-2017, with adjustable parameters according to our coach's requirements : 
# 
# - **poacher** --> the xG difference tells us which players perform better than average based on the situations they had
# - **aerial threat** --> the xG difference in the air gives us the same information but only for the headers, to spot the players over-performing in the air
# - **counterattacker** --> the % of goals on counter-attacks highlights the players that are used to counter-attacking playstyles
# - **clear headed** --> the xG per shot indicator shows which players take the shots that have the best chances to go in, and on the other side which ones tend to waste opportunities which shots that are difficult to convert
# - **experienced** --> as our dataset registers matches from 2011 to 2017, the number of goals scored can be used as a filter for experience, and at the same time filter out players that would match the other criterias but would not be relevant for our classification

# ### So tell us, who should we recruit?
# 
# Some important information are missing here, like the market value of the players. Moreover, data isn't a magical tool : we couldn't pick a player just based on these numbers, as a ton of other parameters need to be taken into account in real life when you recruit a player (age, contract length, salary, does the player want to join the club, languages spoken, personality,...) as well as other statistical parameters, and of course different scouting sessions in real life with professional scouts' opinion. 
# 
# Nevertheless, this could be a cool baseline for a Director of Football to easily and quickly filter the huge amount of players available to select which ones to scout in prority, or confirm an opinion on a player with facts given by the numbers.

# **I hope you enjoyed reading my work, once again do not hesitate to share your thoughts so I can make it better!**

# ![](https://ronaldo.com/wp-content/uploads/2019/09/GettyImages-1159265415.jpg)

# A ressource that helped me : https://www.kaggle.com/jeffd23/10-classifier-showdown-in-scikit-learn to build the quick classificators showdown
# 
# I also noticed afterwards a similar work on the same dataset, with some cool ideas to apply : https://www.kaggle.com/gabrielmanfredi/expected-goals-player-analysis#1.-xG-Model
