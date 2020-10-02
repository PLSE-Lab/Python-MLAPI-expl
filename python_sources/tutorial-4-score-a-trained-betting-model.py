#!/usr/bin/env python
# coding: utf-8

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


pd.options.mode.chained_assignment = None


# # Purpose
# 
# In a [previous tutorial](https://www.kaggle.com/mdabbert/tutorial-2-train-a-model-to-make-bet-predictions) I showed how to create a model that could predict profitable UFC bets by using `predict_proba()` and gambling odds for the fight.  Now I will show a way to evaluate how this model is performing.  It isn't as simple as seeing how many winning fighters it predicts.  It comes down to how much profit it sees from its predicted bets.
# 
# You may find you train models that predict a lot of favorites to win that has a much higher accuracy than models that predict a lot of underdogs to win.  But it could still be possible that the less accurate model is the more profitable one.  I will walk through how to set up an evaluation that will give a model a score based off it its profit.

# # 1. Prep the Train and Test Set

# In[ ]:


#Load the matches that have already occurred 
df = pd.read_csv("/kaggle/input/ultimate-ufc-dataset/ufc-master.csv")

#Let's put all the labels in a dataframe
df['label'] = ''
#If the winner is not Red or Blue we can remove it.
mask = df['Winner'] == 'Red'
df['label'][mask] = 0
mask = df['Winner'] == 'Blue'
df['label'][mask] = 1

#df["Winner"] = df["Winner"].astype('category')
df = df[(df['Winner'] == 'Blue') | (df['Winner'] == 'Red')]


#Make sure lable is numeric
df['label'] = pd.to_numeric(df['label'], errors='coerce')

#Let's fix the date
df['date'] = pd.to_datetime(df['date'])

#Create a label df:
label_df = df['label']

#Let's create an odds df too:
odds_df = df[['R_odds', 'B_odds']]

#Split the test set.  We are always(?) going to use the last 200 matches as the test set, so we don't want those around
#as we pick models

df_train = df[200:]
odds_train = odds_df[200:]
label_train = label_df[200:]

df_test = df[:200]
odds_test = odds_df[:200]
label_test = label_df[:200]

print(len(df_test))
print(len(odds_test))
print(len(label_test))

print(len(df_train))
print(len(odds_train))
print(len(label_train))


# # 2. Pick a Model and Features

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
#Pick a model
my_model = DecisionTreeClassifier(max_depth=5)

#Pick some features
#I would not recommend placing bets based off of these features...
my_features = ['R_odds', 'B_Stance']


# # 3. Create Some Helper Functions

# In[ ]:


#Input: American Odds, and Probability of a Winning Bet
#Output: Bet EV based on a $100 bet
def get_bet_ev(odds, prob):
    if odds>0:
        return ((odds * prob) - (100 * (1-prob)) )
    else:
        return ((100 / abs(odds))*100*prob - (100 * (1-prob)))


# In[ ]:


#Input: American Odds
#Output: Profit on a successful bet
def get_bet_return(odds):
    if odds>0:
        return odds
    else:
        return (100 / abs(odds))*100


# In[ ]:


#Takes a prepared df and returns stats about how the model did.
#This df must be prepared using the evaluate_model() function

def get_ev_from_df(ev_df, print_stats = False):
    get_total = True #if this is False we would return profit per bet
    min_ev = 0 #If we wanted to have a minimum ev to make a bet other than 0...
    num_matches = 0
    num_bets = 0
    num_wins = 0
    num_losses= 0
    num_under= 0
    num_under_losses = 0
    num_under_wins = 0
    num_even = 0
    num_even_losses = 0
    num_even_wins = 0
    num_fav = 0
    num_fav_wins = 0
    num_fav_losses = 0
    profit = 0
    profit_per_bet = 0
    profit_per_match = 0    

    for index, row in ev_df.iterrows():
        num_matches = num_matches+1
        t1_bet_ev = get_bet_ev(row['t1_odds'], row['t1_prob'])
        #print(f"ODDS:{row['t1_odds']} PROB: {row['t1_prob']} EV: {t1_bet_ev}")
        t2_bet_ev = get_bet_ev(row['t2_odds'], row['t2_prob'])
        #print(f"ODDS:{row['t2_odds']} PROB: {row['t2_prob']} EV: {t2_bet_ev}")
        #print()
        
        t1_bet_return = get_bet_return(row['t1_odds'])
        t2_bet_return = get_bet_return(row['t2_odds'])
        
        
        if (t1_bet_ev > min_ev or t2_bet_ev > min_ev):
            num_bets = num_bets+1

            
        if t1_bet_ev > min_ev:
            if row['winner'] == 0:
                num_wins += 1
                profit = profit + t1_bet_return
                #print(t1_bet_return)
            elif row['winner'] == 1:
                num_losses += 1
                profit = profit - 100
            if (t1_bet_return > t2_bet_return):
                num_under += 1
                if row['winner'] == 0:
                    num_under_wins += 1
                elif row['winner'] == 1:
                    num_under_losses += 1
            elif (t1_bet_return < t2_bet_return):
                num_fav += 1
                if row['winner'] == 0:
                    num_fav_wins += 1
                elif row['winner'] == 1:
                    num_fav_losses += 1
            else:
                num_even += 1
                if row['winner'] == 0:
                    num_even_wins += 1
                elif row['winner'] == 1:
                    num_even_losses += 1

        if t2_bet_ev > min_ev:
            if row['winner'] == 1:
                num_wins += 1                    
                profit = profit + t2_bet_return
            elif row['winner'] == 0:
                num_losses += 1
                profit = profit - 100
            if (t2_bet_return > t1_bet_return):
                num_under += 1
                if row['winner'] == 1:
                    num_under_wins += 1
                elif row['winner'] == 0:
                    num_under_losses += 1
            elif (t2_bet_return < t1_bet_return):
                num_fav += 1
                if row['winner'] == 1:
                    num_fav_wins += 1
                elif row['winner'] == 0:
                    num_fav_losses += 1
            else:
                num_even += 1
                if row['winner'] == 1:
                    num_even_wins += 1
                elif row['winner'] == 0:
                    num_even_losses += 1
            
    if num_bets > 0:
        profit_per_bet = profit / num_bets
    else:
        profit_per_bet = 0
    if num_matches > 0:
        profit_per_match = profit / num_matches
    else:
        profit_per_match = 0
        
    if print_stats:
        print(f"""
          Number of matches: {num_matches}
          Number of bets: {num_bets}
          Number of winning bets: {num_wins}
          Number of losing bets: {num_losses}
          Number of underdog bets: {num_under}
          Number of underdog wins: {num_under_wins}
          Number of underdog losses: {num_under_losses}
          Number of Favorite bets: {num_fav}
          Number of favorite wins: {num_fav_wins}
          Number of favorite losses: {num_fav_losses}
          Number of even bets: {num_even}
          Number of even wins: {num_even_wins}
          Number of even losses: {num_even_losses}
          Profit: {profit}
          Profit per bet: {profit_per_bet}
          Profit per match: {profit_per_match}
          
          """)
    if (get_total):
        #print(f"# Matches: {num_matches}, # Bets: {num_bets} # Wins: {num_wins}")
        return(profit)
    else:
        return (profit_per_bet)


# In[ ]:


#This function will give us a score
#input_model: the model we want to train and use.  It must be able to call predict_proba()
#input_features: the list of features we want to use.
#train_df: The training set. Created above
#train_labels: the winners of the training set.  Created above
#train_odds: A list of red fighter and blue fighter odds.  Created above
#test_df: The test set. Created above
#test_labels: The winners of the test set. Created above
#test_odds: The odds of the test set fights.  Created above
#verbose: if verbose is true we print a bunch of stats.
def evaluate_model(input_model, input_features, train_df, train_labels, train_odds, test_df, test_labels,
                  test_odds, verbose=True):
    
    model_score = 0
    
    #Tutorial 1 will explain what is going on here.....
    df_train = train_df[input_features].copy()
    df_test = test_df[input_features].copy()
    df_train = df_train.dropna()
    df_test = df_test.dropna()
        
    df_train = pd.get_dummies(df_train)
    df_test = pd.get_dummies(df_test)
    df_train, df_test = df_train.align(df_test, join='left', axis=1)    #Ensures both sets are dummified the same
    df_test = df_test.fillna(0)
    
    labels_train = train_labels[train_labels.index.isin(df_train.index)]
    odds_train = train_odds[train_odds.index.isin(df_train.index)] 
    labels_test = test_labels[test_labels.index.isin(df_test.index)]
    odds_test = test_odds[test_odds.index.isin(df_test.index)]     

    #Quick shape check
    #display(df_train.shape)
    #display(labels_train.shape)
    #display(odds_train.shape)
    #display(df_test.shape)
    #display(labels_test.shape)
    #display(odds_test.shape)    

    input_model.fit(df_train, labels_train)

    
    
    probs = input_model.predict_proba(df_test)

    
    odds_test = np.array(odds_test)    
    
    prepped_test = list(zip(odds_test[:, -2], odds_test[:, -1], probs[:, 0], probs[:, 1], labels_test))
    #Prepped test now is a list of [Red Odds, Blue Odds, Red Prob of winning, Blue prob of winning, winner label]
    ev_prepped_df = pd.DataFrame(prepped_test, columns=['t1_odds', 't2_odds', 't1_prob', 't2_prob', 'winner'])
    display(ev_prepped_df)
    model_score = get_ev_from_df(ev_prepped_df, print_stats = True)

    return(model_score)


# # 4. Run it and see how the model and features perform

# In[ ]:


score = evaluate_model(my_model, my_features, df_train, label_train, odds_train, df_test, label_test,
                         odds_test, verbose = True)

print(f"Model: {my_model}")
print(f"Features: {my_features}")
print(f"The score of this model feature combo is: {score}")


# In[ ]:


#Let's try another one quick to see the difference.
from sklearn.linear_model import LogisticRegression
my_model_2 = LogisticRegression()
my_features_2 = ['B_ev', 'country']
score = evaluate_model(my_model_2, my_features_2, df_train, label_train, odds_train, df_test, label_test,
                         odds_test, verbose = True)

print(f"Model: {my_model_2}")
print(f"Features: {my_features_2}")
print(f"The score of this model feature combo is: {score}")


# # Discussion of Performance:
# 
# This naive model that is simply a decision tree that takes into consideration the red fighter's odds and the blue fighter's stance actually does pretty well.  Out of 200 fights it sees 142 of them as having a positive expected value.  It wins 81 of these 142 bets.  It wins 38 out of 84 underdog bets.  A good number.  Anything close to 50% will be profitable for that part of it.  For favorites it wins 39/52 bets.  For even fights it is 4/6.  
# 
# For 142 bets of 100 units each this model sees a total return of 1162.21 units.  A profit of 8.18 per bet or 5.81 per match.
# 
# I just chose these features and model type at random, but it would not be a bad starting point to develop an actual model.

# # Alternate Scoring Ideas
# 
# I have used total profit as how I tally the score but there are other possibilities that have their pros and cons.
# 
# * Profit per bet: 
#   - Pros: With limited funds profit per bet could be a better metric.
#   - Cons: Could skew towards models with only a few outlier wins.
#   
#   
# * (Profit) / (Total Possible Profit):
#   - Pros: A number no greater than 1 that could be used to compare models that were evaluated at different times.  Possibly more useful for visualization
#   - Cons: Models that were evaluated at different times still isn't an apples-to-apples comparison

# In[ ]:




