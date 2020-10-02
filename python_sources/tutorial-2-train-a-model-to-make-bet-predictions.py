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
# This kernal builds off of my previous kernal, [Tutorial: Train a Model to Predict a Future Event](https://www.kaggle.com/mdabbert/tutorial-train-a-model-to-predict-a-future-event)
# 
# That kernal showed you how to train a model using events that have already occurred to predict the winners of a future event.  This kernal will take the next step and use probabilities to predict profitable bets for future events.
# 
# ### What is the Difference?
# 
# Taking a look at the upcoming fights we have a match between Amanda Ribas and Paige VanZant.  Amanda Rebas has American betting odds of -835.  This means you would need to bet 835 units to receive a return of 100 units on a winning bet.  To determine what percentage of time Amanda Rebas must win for this to be a profitable bet we use the following formula:
# 
# `(Payout on 100 Unit Bet) * (Break Even win probability) - 100 * (1 - Break Even win probability) = 0`
# 
# `((100 / 835) * 100) * x - 100 * (1 - x) = 0`
# 
# `11.97 * x - 100 + 100*x = 0`
# 
# `111.97x = 100`
# 
# `x = .893`
# 
# Amanda Rebas must win at least 89.3% of the time for this bet to be profitable.
# (89.3% of the time you win 11.97 units.  10.7% of the time you lose 100 units. .893 * 11.97 - .107 * 100 ~= 0)
# 
# Amanda Rebas simply being predicted to win the fight isn't enough.  We need to know that she has around a 90% chance to win to want to bet her.  I'd go as far to say that any model that doesn't predict Amanda Rebas to be the winner based off of a 50% threshold is probably a flawed model.
# 
# 

# # Method of Acquiring and Implementing Probabilites
# 
# Luckily for us many machine learning models have probabilities built in.  We can easily use these features in models including LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, and GaussianNB.
# 
# To get the probabilities we will follow my [previous notebook](https://www.kaggle.com/mdabbert/tutorial-train-a-model-to-predict-a-future-event) until we get to where we grabbed the predictions.  In place of this we will grab the probabilities.  If you have any questions about what is going on in this code, please look at the previous notebook.
# 
# The one difference in the following cell of code is that we are creating a df called `odds_test`.  It is a collection of odds for the upcoming fights.  It will make cycling through the fights easier.

# In[ ]:


#Load the matches that have already occurred 
df = pd.read_csv("/kaggle/input/ultimate-ufc-dataset/ufc-master.csv")

#Load the upcoming matches
df_upcoming = pd.read_csv("/kaggle/input/ultimate-ufc-dataset/upcoming-event.csv")

#Get the number of upcoming fights
num_upcoming_fights = len(df_upcoming)
print(f"We are going to predict the winner of {num_upcoming_fights} fights.")

#Combine the upcoming fights to the previous fights so we can clean it all at the same time.
df_combined = df_upcoming.append(df)

#Let's put all the labels into a dataframe
df_combined['label'] = ''

#We need to convert 'Red' and 'Blue' to 0 and 1
mask = df_combined['Winner'] == 'Red'
df_combined['label'][mask] = 0
mask = df_combined['Winner'] == 'Blue'
df_combined['label'][mask] = 1

#Make sure label is numeric
df_combined['label'] = pd.to_numeric(df_combined['label'], errors='coerce')

#Make sure the date column is datetime
df_combined['date'] = pd.to_datetime(df['date'])

#Copy the labels to their own dataframe
label_df = df_combined['label']

#Split the train set from the test set

df_train = df_combined[num_upcoming_fights:]
label_train = label_df[num_upcoming_fights:]

df_test = df_combined[:num_upcoming_fights]
label_test = label_df[:num_upcoming_fights]


#Make sure the sizes are the same
print(len(df_test))
print(len(label_test))

print(len(df_train))
print(len(label_train))

from sklearn.tree import DecisionTreeClassifier
#Pick a model
my_model = DecisionTreeClassifier(max_depth=5)

#Pick some features
#I would not recommend placing bets based off of these features...
my_features = ['R_odds', 'B_Stance']

#Let's grab the names of the fighters for the upcoming event
#This will be useful to print predictions at the end.
fighters_test = df_test[['R_fighter', 'B_fighter']]
odds_test = df_test[['R_odds', 'B_odds']]


#Make dataframes that only contain the relevant features
df_train_prepped = df_train[my_features].copy()
df_test_prepped = df_test[my_features].copy()

#If we need to dummify the datasets do it now.  We need to be careful that the test set has all of the features
#that the training set does

df_train_prepped = pd.get_dummies(df_train_prepped)
df_test_prepped = pd.get_dummies(df_test_prepped)

#Ensure both sets are dummified the same
df_train_prepped, df_test_prepped = df_train_prepped.align(df_test_prepped, join='left', axis=1)    

#The new test set may have new new features after the above join.  Fill them with zeroes
df_test_prepped = df_test_prepped.fillna(0)

#Since we may have dropped some rows we need to drop the matching rows in the labels
label_train_prepped = label_train[label_train.index.isin(df_train_prepped.index)]
label_test_prepped = label_test[label_test.index.isin(df_test_prepped.index)]
fighters_test_prepped = fighters_test[fighters_test.index.isin(df_test_prepped.index)]
odds_test_prepped = odds_test[odds_test.index.isin(df_test_prepped.index)]


#Quick test that lengths match.
print(len(label_train_prepped))
print(len(df_train_prepped))
print(len(label_test_prepped))
print(len(df_test_prepped))
print(len(fighters_test_prepped))
print(len(odds_test_prepped))

my_model.fit(df_train_prepped, label_train_prepped)


# predict_proba will work for LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, and GaussianNB among others

# In[ ]:


# Now we deviate.  Instead of grabbing the predictions we will grab the probabilities

probs = my_model.predict_proba(df_test_prepped)


# In[ ]:


#What we have here is an array with the probability the RED will win or BLUE will win.  We can use this to calculate bets

probs


# In[ ]:


#Let's put the fighter names, odds, and probabilities together so we can cycle through them easily
fighters_array = fighters_test_prepped.to_numpy()
odds_array = odds_test_prepped.to_numpy()


# In[ ]:


probs_list = np.array(list(zip(fighters_array, odds_array, probs)))


# In[ ]:


probs_list


# # Calculate Profitable Bets
# 
# This fucntion will use the American odds and the probability of a fighter winning to return the Expected Value (EV) of a 100 unit bet on a given fighter.
# 
# It works like this:
# 
# If a fighter has positive odds:
# 
# EV = (`odds` * `probability a fighter wins`) - (`100` * (`1` - `probability a fighter wins`))
# 
# If a fighter has negative odds:
# 
# EV = (`100` / abs(`odds`)) * `100` * `probability a fighter wins` - (`100` * (`1` - `probability a fighter wins`))

# In[ ]:


def get_bet_ev(odds, prob):
    if odds>0:
        return ((odds * prob) - (100 * (1-prob)) )
    else:
        return ((100 / abs(odds))*100*prob - (100 * (1-prob)))


# In[ ]:


for p in probs_list:
    red_ev = get_bet_ev(p[1][0], p[2][0])
    blue_ev = get_bet_ev(p[1][1], p[2][1])
    
    print (p[0][0], "(RED) vs ", p[0][1], "(BLUE)")
    print(p[0][0], "has a", "%.2f" % (p[2][0]*100), "percent chance of winning.  His odds are", p[1][0], "This give him a single bet EV of", "%.2f" %red_ev)
    print(p[0][1], "has a", "%.2f" % (p[2][1]*100), "percent chance of winning.  His odds are", p[1][1], "This give him a single bet EV of", "%.2f" %blue_ev)
    if red_ev > 0:
        print("RED is a good bet")
    elif (blue_ev > 0):
        print("BLUE is a good bet")
    else:
        print("There is NO good bet")
    
    print()


# # Conclusion
# 
# There will never bet 2 good bets for a fight.  There will be times when there is no good bet.  It probably won't occur as much as in the model above.  The model used in the kernel is very basic relying mostly on the fighter odds so it returns a lot of predictions in this "no man's land"
# 
# I am still new to Machine Learning projects and only taught myself Python in November. Any suggestions would be welcome! I hope this was helpful.
# 
# If you're interested in playing around with this notebook you just need to change these two lines. See what kind of difference it makes in the final predictions:
# 
# `my_model = DecisionTreeClassifier(max_depth=5)`
# 
# `my_features = ['R_odds', 'B_Stance']`

# In[ ]:




