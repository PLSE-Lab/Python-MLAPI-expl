#!/usr/bin/env python
# coding: utf-8

# # Expected Goals Model & Player Analysis

# First of all, thanks to Alin Secareanu for this magnificent dataset. It includes events from more than 7,000 games from the top 5 European Leagues from 2011 to 2016. I've been trying to gain insights on football analytics for a while now but couldn't find a proper dataset, until now.
# 
# This report/notebook is structured as follows:
# 
# **1. xG Model (Expected Goals Model)**
# 
# 1.a) XGB Classifier
# 
# 1.b) Logistic Regression
# 
# 1.c) Neural Network (MLP)
# 
# 
# **2. Conclusions about xG Model**
# 
# 2.a) How good is our model?
# 
# 2.b) How could we improve our model?
# 
# 
# **3. Player Analysis**
# 
# 3.a) Which players are the best finishers?
# 
# 3.b) Which players have the most "expected goals"?
# 
# 3.c) Which players are the worst at deciding their shots?
# 
# 3.d) Which players are the best headers?
# 
# 3.e) Which players are the best at shooting with their left foot?
# 
# 3.f) Which players are the best at shooting with their right foot?
# 
# 3.g) Which are the best outside-the-box shooters?
# 
# 3.h) Which players make the best/most dangerous passes?
# 
# 3.i) Which players make the best/most dangerous crosses?
# 
# 3.j) Which players are the most unlucky when passing the ball?

# We'll start by importing all that we might use later.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot
import scipy as sp
from xgboost import XGBClassifier
import sklearn
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# Now we'll load the dataset and create a new one named "shots" that include only the rows from the original dataset that correspond to shots (event_type==1).

# In[ ]:


filename = 'events.csv'
events = pd.read_csv('../input/events.csv')
shots = events[(events.event_type==1)]


# # 1. xG Model

# Here we prepare the data for making it suitable as input for our xG Model. First I will talk a bit about what Expected Goals is.
# 
# Expected Goals Models are an attempt to quantify how likely it is that a certain shot results in a goal. With this metric, we can analyze what happened in a game other than how many goals each team scored. Since the game is won with goals, and goals come only from shots (except for the rare case of an own goal), then the xG metric only needs data from shots to be developed. In other words, anything else that may influence how many "expected goals" a team would score has to happen via having more shots in the first place. For example, one could argue that if team B receives 3 red cards, then team A is supposed to score more goals (an increase in expected goals). However, in reality, having 3 more players on the field will result in taking more (and possibly better) shots, and it is taking more shots that will result in an increase in expected goals. Therefore, shots and their information are the only thing we need to take into account!
# 
# In my opinion, an xG Model should not take into account specific characteristics or skills of the players who intervene in the event. I've read in some other places that xG models should account for this to be more accurate, but I strongly disagree. Yes, I know that if Messi is one-on-one with the goalkeeper the chances of it being a goal are higher than with any other player, or that if Ter Stegen is the goalkeeper in question then the probabilities of it being a goal may go down. But that is not the point of the metric. We are trying to standardize through thousands of datapoints how likely it is that any given player would score from a certain position in a certain situation. If the player is an extraordinary finisher, then he will probably score more goals than expected, and that's fine. If we start taking into account the skills of the players involved, it is my opinion that we would be taking one step too far towards the design of this metric and would become less meaningful.
# 
# After this (long) introduction, let's get to it. We will prepare X and Y sets. Y will simply include every shot in the database and whether it was a goal or not (1 or 0). It is our target variable. X will include all the relevant information about the shot that we have in our data. That would be:
# 
# . **location:** attacking half, defensive half, centre of the box, left wing, right wing, difficult angle and long range, difficult angle on the left, difficult angle on the right, left side of the box, left side of the six yard box, right side of the box, right side of the six yard box, very close range, penalty spot, outside the box, long range, more than 35 yards, more than 40 yards, not recorded.
# 
# . **bodypart:** right foot, left foot, head.
# 
# . **assist_method:** none, pass, cross, headed pass, through ball.
# 
# . **situation:** open play, set piece, corner, free kick.
# 
# . **fast_break:** 1 or 0, whether the shot comes from a fast break or not.
# 
# So, we have a lot of meaningful information about every shot. The location from which it was taken, which part of the body was used for shooting, how the shoot was made available (after a pass? a cross? etc.), and the situation or context in which the shoot occured (open play, corner, etc.)
# 
# Since all of these are categorical variables, we have to convert them to binary dummies (except for fast_break, which is already binary).

# In[ ]:


shots_prediction = shots.iloc[:,-6:]
dummies = pd.get_dummies(shots_prediction, columns=['location', 'bodypart','assist_method', 'situation'])
dummies.columns = ['is_goal', 'fast_break', 'loc_centre_box', 'loc_diff_angle_lr', 'diff_angle_left', 'diff_angle_right', 'left_side_box', 'left_side_6ybox', 'right_side_box', 'right_side_6ybox', 'close_range', 'penalty', 'outside_box', 'long_range', 'more_35y', 'more_40y', 'not_recorded', 'right_foot', 'left_foot', 'header', 'no_assist', 'assist_pass', 'assist_cross', 'assist_header', 'assist_through_ball', 'open_play', 'set_piece', 'corner', 'free_kick']
dummies.head()


# In[ ]:


X = dummies.iloc[:,1:]
y = dummies.iloc[:,0]
print(X.shape)
print(y.shape)


# So we have a total of 229,135 shots. For each shot, we have 28 different characteristics that describe it. All of these 28 characteristics are binary, so they just indicate Yes or No to a certain characteristics in the shot.
# 
# Now we will divide our X and y into two different sets for training and testing. I will use 65% of them for training our model and 35% por testing it. This is because we have enough datapoints (in my opinion). So 65% should be enough to train it properly, and I'd rather have as many shots left as possible for pure testing later. 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=1)


# ## 1.a) XGB Classifier
# 
# We will first train a Gradient Boosting Classifier, that usually achieves good results with little adjusting. I used GridSearchCV in order to find the best parameters (tried with max_depth=[3,5,7,9] and n_estimators=[100,500,1000]). I concluded that max_depth=5 and n_estimators=100 worked the best.
# 
# So we just call the model, feed it with our X_train and y_train, and make it predict our y_test based on our X_test.

# In[ ]:


classifier = XGBClassifier(objective='binary:logistic', max_depth=5, n_estimators=100)
classifier.fit(X_train, y_train)


# In[ ]:


accuracy = classifier.score(X_test, y_test)
y_pred = classifier.predict_proba(X_test)
predict = classifier.predict(X_test)
y_total = y_train.count()
y_positive = y_train.sum()
auc_roc = roc_auc_score(y_test, y_pred[:, 1])
print('The training set contains {} examples (shots) of which {} are positives (goals).'.format(y_total, y_positive))
print('The accuracy of classifying whether a shot is goal or not is {:.2f} %'.format(accuracy*100))
print('Our classifier obtains an AUC-ROC of {:.4f}.'.format(auc_roc))


# So, we can see that our xG model is able to correctly predict whether a shot is goal or not 91% of the time. Furthermore, we obtain a pretty good "ROC-Area Under the Curve" metric of 0.82. This looks very promising.
# 
# However, these two metrics are not the best for evaluating our model because they do not consider that our dataset is imbalanced. There are many more shots that do not end up being a goal than shots that do. So, for example, if we would simply predict that the shot will not be a goal each and every single time, we would already obtain a pretty high accuracy of 89%.
# 
# So, we need other metrics to really understand if our model is any good.
# 
# We will try with the AUC (Under the Curve) of Precision-Recall, and with Cohen's Kappa statistics. Both of these are appropiate for our case, since they do take into account the imbalance of our data.
# 
# 

# In[ ]:


auc_pr_baseline = y_positive / y_total
print('The baseline performance for AUC-PR is {:.2f}. This is the AUC-PR that what we would get by random guessing.'.format(auc_pr_baseline))

auc_pr = average_precision_score(y_test, y_pred[:, 1])
print('Our classifier obtains an AUC-PR of {:.4f}.'.format(auc_pr))
cohen_kappa = cohen_kappa_score(y_test,predict)
print('Our classifier obtains a Cohen Kappa of {:.4f}.'.format(cohen_kappa))


# We end the evaluation of the model with a confusion matrix and an additional stats report. 
# 

# In[ ]:


print('Confusion Matrix:')
print(confusion_matrix(y_test,predict))
print('Report:')
print(classification_report(y_test,predict))


# The confusion matrix summarizes all predictions. It tells us that, from all the shots that were not goal, our model correctly identified 70,820 as no-goals, and made a mistake in 6,265 cases in which it predicted that the shot would not be a goal, but it was. From the other column, we see that it correctly predicted 874 goals, but failed to predict 2239 succesful shots as goals.
# 
# From the report we can see the model has excellent numbers when it comes to predict class 0 (no-goal), but not that good for predicting class 1 (goals). With the latter, we have a precision of 0.72, and a recall of 0.26, resulting in an F1 of 0.39. These are decent numbers, but not really good.
# 
# Does this make sense? Of course it does. Predicting whether a shot will be goal is extremely more difficult than predicting it will not be a goal (well, predicting itself is not really hard, but predicting it and also being correct is). This is specially true if you have no idea who the player shooting the ball is or who the goalkeeper is, which is the situation in which the algorithm is in. It would be interesting to make a human vs. machine study on this. Are human experts much better in judging whether any given shot will become a goal than a model such as this one is?
# 
# Finally we'll make a table containing the information of whether shots were goal or not (0 or 1), the probability that the model assigned to that shot being a goal (from 0 to 1), and the difference between the two. This is just to take a look and visualize how our model has performed.

# In[ ]:


predictions = X_test.copy()
predictions['true_goals'] = y_test
predictions['expected_goals'] = y_pred[:,1]
predictions['difference'] = predictions['expected_goals'] - predictions['true_goals']
predictions = predictions.iloc[:,28:31]
predictions.head()


# ## 1.b) Logistic Regression

# In[ ]:


logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)


# In[ ]:


logistic_regression.score(X_train, y_train)
accuracy = logistic_regression.score(X_test, y_test)
y_pred = logistic_regression.predict_proba(X_test)
accuracy_logreg = logistic_regression.score(X_test, y_test)
y_pred_logreg = logistic_regression.predict_proba(X_test)
predict_logreg = logistic_regression.predict(X_test)
y_total_logreg = y_train.count()
y_positive_logreg = y_train.sum()
auc_roc_logreg = roc_auc_score(y_test, y_pred_logreg[:, 1])
print('The training set contains {} examples (shots) of which {} are positives (goals).'.format(y_total_logreg, y_positive_logreg))
print('The accuracy of classifying whether a shot is goal or not is {:.2f} %'.format(accuracy_logreg*100))
print('Our classifier obtains an AUC-ROC of {:.4f}.'.format(auc_roc_logreg))

auc_pr_baseline = y_positive / y_total
print('The baseline performance for AUC-PR is {:.4f}. This is the AUC-PR that what we would get by random guessing.'.format(auc_pr_baseline))
auc_pr_logreg = average_precision_score(y_test, y_pred_logreg[:, 1])
print('Our classifier obtains an AUC-PR of {:.4f}.'.format(auc_pr_logreg))
cohen_kappa_logreg = cohen_kappa_score(y_test,predict_logreg)
print('Our classifier obtains a Cohen Kappa of {:.4f}.'.format(cohen_kappa_logreg))


# In[ ]:


print('Confusion Matrix:')
print(confusion_matrix(y_test,predict_logreg))
print('Report:')
print(classification_report(y_test,predict_logreg))


# So, we pretty much get the same as with our XGBClassifier. No difference whatsoever between how the two models perform. 
# 
# In order to get something useful out of this, I will print the coefficients of the Logistic Regression, as they have interesting information. They show us how the model makes a prediction.
# 
# Below we have a list with every feature and its coefficient. Some insights that we can learn from this are that shoots that come from fast breaks are much more likely to be goals (very positive coefficient), shots that come from a through-ball pass are much more likely to be goals than those that come from any other kind of pass or cross (the coefficient for "through_ball" is positive while the other are negative), and so on.

# In[ ]:


coefficients = pd.Series(logistic_regression.coef_[0], X_train.columns)
print(coefficients)


# One thing to worry about is the high coefficient of the location='not_recorded'. Apparently there are shots for which the location has not been recorded, and for some reason those shots are incredibly likely to be goals, which makes things easier for the model. Maybe to better understand the model it could be interesting to remove those rows and check its performance. I have tried it and it results in a slight decrease in its performance (AUC-ROC=0.806, AUC-PR = 0.41). I will leave these rows anyway, as I will do players analysis later and I don't want valuable information about the goals of the players being erased.

# ## 1.c) Neural Network (MLP)

# I decided to try also a Neural Network to see if I can improve the performance of our xG model.
# As is usually the case with Neural Networks, it's difficult to understand what's really going on inside the model. I tried several different configurations of hyperparameters and concluded that the best was with ReLu as activation function and 4 layers of 28 neurons each (the same number as the number of features in our data). So this is the configuration I'll use.

# In[ ]:


mlp = MLPClassifier(random_state=0, hidden_layer_sizes=(28,28,28,28), max_iter=2000, activation='relu')
mlp.fit(X_train, y_train)


# In[ ]:


mlp.score(X_train, y_train)
mlp.score(X_test, y_test)
accuracy = mlp.score(X_test, y_test)
print('The accuracy of classifying whether a shot is goal or not is {:.2f} %.'.format(accuracy*100))
y_pred = mlp.predict_proba(X_test)
predict = mlp.predict(X_test)
y_total = y_train.count()
y_positive = y_train.sum()
print('The training set contains {} examples of which {} are positives.'.format(y_total, y_positive))
auc_roc = roc_auc_score(y_test, y_pred[:,1])
print('Our MLP classifier obtains an AUC-ROC of {:.4f}.'.format(auc_roc))
auc_pr_baseline = y_positive / y_total
print('The baseline performance for AUC-PR is {:.4f}. This is what we would get by random guessing'.format(auc_pr_baseline))
auc_pr = average_precision_score(y_test, y_pred[:,1])
print('Our MLP classifier obtains an AUC-PR of {:.4f}.'.format(auc_pr))
cohen_kappa = cohen_kappa_score(y_test,predict)
print('Our classifier obtains a Cohen Kappa of {:.4f}.'.format(cohen_kappa))
MSE = sklearn.metrics.mean_squared_error(y_test, y_pred[:,1])
print('Our MLP classifier obtains a Mean Squared Error (MSE) of {:.4f}.'.format(MSE))


# In[ ]:


print('Confusion Matrix:')
print(confusion_matrix(y_test,predict))
print('Report:')
print(classification_report(y_test,predict))


# Again, very similar numbers! There is a slight increase in Cohen's Kappa and in our Recall, so we will use the predictions of our MLP model for determining the final xG of each of the shots in the dataset.

# In[ ]:


predictions = X_test.copy()
predictions['true_goals'] = y_test
predictions['expected_goals'] = y_pred[:,1]
predictions['difference'] = predictions['expected_goals'] - predictions['true_goals']
predictions = predictions.iloc[:,28:31]


# # 2. Conclusions about xG Model
# 
# ## 2.a) How good is our model?

# It is hard to determine whether the model is good, as there is not much to compare with. I think that Cohen's Kappa and AUC PR are very good indicators to measure the performance of our model, but it's not easy to find these numbers for other different xG models in order to compare. Many report the aggregated R2 by season, but in my opinion that does not make a lot of sense. In Section #3 we will see that our model results in a correlation of 0.97 between the total expected goals and the actual goals by player, suggesting that the model may be sufficiently good, but this measure is also not ideal.
# 
# While searching for other xG models and how well they've done, to be able to compare how good this is, I found the following regarding AUC-ROC in other models (see [here](http://business-analytic.co.uk/blog/assessing-expected-goals-models-part-2-anatomy-of-a-big-chance/)):
# 
# . Standard Model: AUC-ROC = 79.8%
# . Big Chance Model: AUC-ROC = 75.1%
# . Standard+Big Chance: AUC-ROC = 82%
# . Standard+Defensive = AUC-ROC = 81.4%
# 
# So, finally we can compare our model with some others, at least on this statistic, which as I have mentioned before I don't think it's the best one to determine how well our model performs.
# 
# Our model has an AUC-ROC of 81.9%, suggesting it is pretty decent.
# 

# ## 2.b) How could we improve our xG model?

# I've tried different algorithms with different hyperparameters and the results did not change much.
# 
# But, what additional data could we add to improve our model? First, it would be great to have information about the defending team. How many defenders are between the goal and the player with the ball? How much defensive pressure is the shooting player withstanding? How quick is the whole play? How much time does the player have in order to shoot?
# I imagine the (quantified) answers to many of these questions would give the model interesting new information to base its predictions on.
# 
# Additionally, we have the "location" of the shot divided in 17 different categories. Even though it's pretty decent, it could be better to have exact x and y coordinates of where the player is. This could lead to more precision in the location of each shot, and therefore in the predictions of the model.

# # 3. Player Analysis

# We'll start by adding the information we have about expected goals to each shot in our original data. From there, we can extract many interesting metrics about players, all of these metrics being related in one way or another to the new xG predictions for each shot.

# In[ ]:


ypred2 = mlp.predict_proba(X_train)
predictions_train = X_train.copy()
predictions_train['true_goals'] = y_train
predictions_train['expected_goals'] = ypred2[:,1]
predictions_train['difference'] = predictions_train['expected_goals'] - predictions_train['true_goals']
predictions_train = predictions_train.iloc[:,28:31]
all_predictions = pd.concat([predictions, predictions_train], axis=0)
events2 = pd.concat([events, all_predictions], axis=1)
shots2 = events2[events2.event_type==1]


# ## 3.a) Which players are the best finishers?

# By looking at the difference between a player's number of goals and his number of expected goals, we can see who are the best at finishing plays.

# In[ ]:


xG_players = shots2[['player', 'event_type', 'true_goals', 'expected_goals', 'difference']].groupby('player').sum()
xG_players.columns = ['n_shots', 'goals_scored', 'expected_goals', 'difference']
xG_players[['goals_scored', 'expected_goals']].corr()


# As mentioned above, we see that the correlation between goals_scored and expected_goals is very high, which speaks well about our xG model.

# In[ ]:


xG_players.sort_values(['difference', 'goals_scored'])


# Not surprisingly, we can see that Messi is by far the best in this metric. According to the number and characteristics of all shoots he's taken, he was expected to score 148 goals, but instead he has scored 205. As expected, what we find among the top of the list are all world-class famous players.
# 
# On the other end, we find Jesus Navas, Mario Balotelli, or Adrian Mutu, among others, who have scored much less goals that what they should have. Looking at players with much fewer goals than expected goals is interesting. It kind of tells us that these players have missed too many chances and should improve their shooting, true. But it also means that they have been there creating much more chances than what the most widespread stats (like "goals) tell us. So, in that sense, they are underrated.

# ## 3.b) Which players have the most "expected goals"?

# For this we simply order our table by "expected_goals". It tell us which players should have scored the highest number of goals according to all the chances that they've had, as predicted by our model.

# In[ ]:


xG_players.sort_values(['expected_goals'], ascending=False)


# ## 3.c) Which players are the worst at deciding their shots?

# By calculating a "xG / #shots" ratio, we can determine which players make the worst shooting decisions. That is, that they consistently tend to take shots that are not likely to end up in goals. We will only include players who have taken a minimum of 100 shots, as otherwise the results would be filled with players who attempted only 1 or 2 (very bad) shots in the entire 5-year period.

# In[ ]:


xG_players['xG_per_shot_ratio'] = xG_players['expected_goals'] / xG_players['n_shots']
xG_players[xG_players.n_shots>100].sort_values(['xG_per_shot_ratio', 'goals_scored'])


# We can see that Tom Huddlestone appears to be the worst at deciding when to shoot, with an xG per shot of 0.035.
# 
# The cases of Gohhan Inler and Ruben Rochina are really interesting. They seem to take a lot of very unlikely shots, from long distance, but it also seems that they are actually good at it. Look at the difference between their actual goals and their expected goals: despite taking "bad" shots, they have scored more than they were expected to. So, in their case, making these kinds of shots might not be such a bad decision after all.
# 
# On the other hand, we have cases like Ivan Radovanovic. He also has taken a lot of shots from unlikely locations, and he does not seem to be good at it, since he only has 2 goals when his expected goals are 7.5. These shots may also come in situations after corners, in which players sometimes shoot from far away in any direction without any hope of making a goal, but instead for the ball to go out and reorganize the defense. So maybe these bad shots did make sense in a way and we should not be so cruel about it.
# 
# At the bottom at the list we find the players who tend to take shots that are likely to be goals, from inside the box or very close range. As expected, here we find mostly strikers and target-men, who tend to play near the opposition's goal.
# Diego Milito, Kevin Gameiro, and Carlos Bacca are among these players.

# ## 3.d) Which players are the best headers?

# In[ ]:


headers = events2[(events2.event_type==1) & (events2.bodypart==3)]
headers_players = headers[['player', 'event_type', 'true_goals', 'expected_goals', 'difference']].groupby('player').sum()
headers_players.columns = ['n_headers', 'goals_scored', 'expected_goals', 'difference']
headers_players.sort_values(['difference'])


# We see that Cristiano Ronaldo is the best header in the game, when comparing the number of goals he scored with his head with the number of goals he should have scored with our model.
# The other two in the top 3 are Mario Mandzukic and Fernando Llorente.
# 
# This is interesting because we can easily see how it confirms something we kind of know. These three guys are exceptional at heading the ball.

# ## 3.e) Which players are the best at shooting with their left foot?

# In[ ]:


left_foot = events2[(events2.event_type==1) & (events2.bodypart==2)]
left_foot_players = left_foot[['player', 'event_type', 'true_goals', 'expected_goals', 'difference']].groupby('player').sum()
left_foot_players.columns = ['n_left_foot_shots', 'goals_scored', 'expected_goals', 'difference']
left_foot_players.sort_values(['difference'])


# Guess who? Of course Messi is first on the list. He is the one who scored the most with his left foot when compared to what was expected according to the characteristics surrounding the shot. The list is complete with Antoine Griezmann, Iago Falque, and Arjen Robben. Not surprisingly, the top players are all left-footed.
# 
# I'm a bit surprised by not seeing Cristiano Ronaldo here, as I remember him scoring many times with his left foot. So I'll look him up individually.

# In[ ]:





# In[ ]:


left_foot_players.loc['cristiano ronaldo']


# He still has more left-footed goals than what was expected by our model, so we can confirm that he is very decent at shooting with his left. But he is not among the best either, as we saw from the previous list.

# ## 3.f) Which players are the best at shooting with their right foot?

# In[ ]:


right_foot = events2[(events2.event_type==1) & (events2.bodypart==1)]
right_foot_players = right_foot[['player', 'event_type', 'true_goals', 'expected_goals', 'difference']].groupby('player').sum()
right_foot_players.columns = ['n_right_foot_shots', 'goals_scored', 'expected_goals', 'difference']
right_foot_players.sort_values(['difference'])


# We see Luis Suarez at the top of the list. Messi is in the 11th position besides right foot being his weak foot and the model not accounting for that, which is pretty amazing (the model doesn't know whether a player is right or left footed, it just knows which foot was used for shooting).

# ## 3.g) Which are the best outside-the-box shooters?

# In[ ]:


outside_box = shots2[(shots2.location==15)]
outbox_players = outside_box[['player', 'event_type', 'true_goals', 'expected_goals', 'difference']].groupby('player').sum()
outbox_players.columns = ['n_outside_box_shots', 'goals_scored', 'expected_goals', 'difference']
outbox_players.sort_values(['difference'])


# We see Messi, Pogba, and Higuain as the best when it comes to shooting from outside the box. In the other end, we find players like Mario Balotelli and Alessandro Diamanti. I'm surprised to see Nainggolan here, as I recall him as an excellent shooter from long range, scoring many goals. But maybe that's just because he actually tried 280 shots, and I only remember the ones that were goals. For these cases, better rely on the numbers.

# ## 3.h) Which players make the best/most dangerous passes?

# #### By looking at the player who made the pass/through-ball that came prior to the shot, we can evaluate how much xG a player created from his passing.

# In[ ]:


passes_and_throughballs = pd.concat([shots2[shots2.assist_method==1], shots2[shots2.assist_method==4]])
assisting_players = passes_and_throughballs[['player2', 'event_type', 'true_goals', 'expected_goals', 'difference']].groupby('player2').sum()
assisting_players['xGoals_per_pass'] = assisting_players['expected_goals'] / assisting_players['event_type']
assisting_players.columns = ['n_passes', 'goals_scored_from_passes', 'xGoals_from_passes', 'difference', 'xGoals_per_pass']

assisting_players[assisting_players.n_passes > 100].sort_values(['xGoals_per_pass'], ascending=False)


# We can see Luis Suarez, Di Maria, or Gareth Bale among the players who make the most dangerous passes, as measured by the "expected goals" metric of the subsequent shot.

# ## 3.i) Which players make the best/most dangerous crosses?

# In[ ]:


crosses = shots2[shots2.assist_method==2]
crosses_players = shots2[['player2', 'event_type', 'true_goals', 'expected_goals', 'difference']].groupby('player2').sum()
crosses_players.columns = ['n_crosses', 'goals_scored_from_crosses', 'xGoals_from_crosses', 'difference']
crosses_players['xGoals_per_cross'] = crosses_players['xGoals_from_crosses'] / crosses_players['n_crosses']
crosses_players.columns = ['n_crosses', 'goals_scored_from_crosses', 'xGoals_from_crosses', 'difference', 'xGoals_per_cross']
crosses_players[crosses_players.n_crosses > 50].sort_values(['xGoals_per_cross'], ascending=False)


# Surprisingly, we see Luis Suarez as the player who tends to make the most dangerous crosses.

# ## 3.j) Which players are the most unlucky when passing the ball?

# We can also see who the most 'unlucky' players are when it comes to passing. Those who have the largest difference between the number of goals that was expected to result from their passes, and the number of goals that actually came from them. I say unlucky because, unlike the player who makes the shot, this other player has no responsibility in whether his team-mates were able to score after his great pass or not!
# 
# The first table is sorted by the most unlucky when it comes to passes and through balls, while the second table refers to crosses.

# In[ ]:


print('Passes and Through-Balls:')
assisting_players.sort_values(['difference'], ascending=False)


# In[ ]:


print('Crosses:')
crosses_players.sort_values(['difference'], ascending=False)


# We can see that the bottom of the lists is filled with players from the strongest teams. This makes sense, as their teammates are good and so they are expected to score the chances that are created to them.
# 
# At the top of the list we also see some famous 'unlucky' players though, such as Philippe Coutinho and Eden Hazard, who should have seen higher numbers in their "Assists" stats according to the chances they created for the team. This does not speak highly of their team-mates at Liverpool or Chelsea from 2011 to 2016.
