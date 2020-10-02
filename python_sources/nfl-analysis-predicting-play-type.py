#!/usr/bin/env python
# coding: utf-8

# <h1>NFL ANALYSIS - Series 2: Predicting Run or Pass</h1>
# <body>
#     <p>This Data origionates from the curtousy of Kaggle user Max Horowitz (@ https://www.kaggle.com/maxhorowitz). He has used nflscrapR to obtain the data and also represents https://www.cmusportsanalytics.com/. nflscrapeR is an R package engineered to scrape NFL data by researchers Maksim Horowitz, Ron Yurko, and Sam Ventura.</p>
# </body>
# 
# <p>**Series 2 - Predicting Run or Pass:** Knowing whether the next play is going to be a run or a pass is tremendous advantage to a defensive coordinator. In this portion of the series, I'm going to load, explore, clean, and model the data to try and predict the play type for the Cleveland Browns. Following this, I'm going to see if I can make any changes to tune our best model.</p>

# **This notebook's purpose is to explore NFL data from 2009-2017. The goal is to hopefully provide useful analysis for others to use or to provide useful code for others to learn from.**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Essentials: Data Cleansing and ETL
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerLine2D

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, auc # good for evaluation of binary classification problems
from sklearn.model_selection import train_test_split


# <h1>
# Importing Data
# </h1>

# In[ ]:


df = pd.read_csv('../input/NFL Play by Play 2009-2017 (v4).csv')
df.head(10)


# <h1>
# Exploratory Analysis - Predicting the Play Type
# </h1>

# <h3>
# Looking at the Data
# </h3>
# <body>
#     We can see that the data is particularily encompassing. 102 Columns of data is a lot more data than we'll need while building potential models. This is because more attributes can typically cause over-fitting for any method we decide to impliment.
#     </body>

# In[ ]:


df.info()


# In[ ]:


print("Rows: ",len(df))


# <h1>Analysis </h1>
# <body>
#     <p>To continue the project I want to look into play calling...</p>
# </body>

# In[ ]:


# take the dataframe for field goals above and shorten the scope of the columns to just FG information
plays = ['Date','GameID','qtr','time','yrdline100','PlayType','FieldGoalResult','FieldGoalDistance','posteam','DefensiveTeam','PosTeamScore','DefTeamScore','Season']
plays = df[plays]

# Filter out any random results of NA for the Play Type
plays = plays[plays.PlayType != 'NA']


# <h3>Taking a Closer Look</h3>

# <body>
#     <p>To get a better idea of how certain variables impact the offensive coordinators play calling, we need to dive into the data and see if we can find any interesting aspects that could be related to the play calls.</p>
# </body>

# In[ ]:


# take the dataframe for field goals above and shorten the scope of the columns to just FG information
play_attr = ['GameID','qtr','TimeSecs','yrdline100','ydstogo','Drive','down','PlayType','PassAttempt','RushAttempt','Yards.Gained','posteam','DefensiveTeam','PosTeamScore','DefTeamScore','Season']
plays = df[play_attr]

plays = plays[plays.PlayType.notna() & (plays.PlayType != 'No Play') & (plays.PlayType != 'Kickoff') & (plays.PlayType != 'Extra Point')]
plays=plays.rename(columns = {'posteam':'Team'})
plays.head(5)


# <h3>Pass and Rush Attempts League-Wide by Quarter of the Game:</h3>

# In[ ]:


# group by qtr: count
regulation_plays = plays[plays.qtr != 5]
ax = regulation_plays.groupby(['qtr'])['PassAttempt','RushAttempt'].sum().plot.bar(figsize=(20,9),color=['saddlebrown','orange'],rot=0,fontsize=16)
ax.set_title("Amount of Plays each Quarter - by Rush or Pass", fontsize=24)
ax.set_xlabel("Quarter", fontsize=18)
ax.set_ylabel("# of Plays", fontsize=14)
ax.set_alpha(0.8)

# set individual bar lables using above list
for i in ax.patches:
    # get_x: width; get_height: verticle
    ax.text(i.get_x()+.04, i.get_height()-1500, str(round((i.get_height()), 2)), fontsize=16, color='black',rotation=0)


# <body>
#     <p>As we can see, it seems that passing plays are more prominant in all categories. Recently, the play calling in the NFL has changed to a more pass-friendly offensive approach which has resulted in the value of high quality quarterbacks. The final goal here is to determine when teams are running the ball or passing the ball, so its helpful to get an idea of the aggregation first. One important distinction here is that the 2nd and 4th quarters see a spike in the amount of passing attempts but the rushing attempts remains fairly constant across the quarters. This is most likely due to teams trying to score before the possession changes in the 2nd half or to win at the end of the game due to the time-friendly rule that the clock stops after an incomplete pass and being able to throw the ball towards the sideline to stop the clock.
# </p>
# </body>

# <h3>Play Call by Down</h3>
# 

# In[ ]:


# group by qtr: count
plays_down = plays[plays.down <= 3]
ax = plays_down.groupby(['down'])['PassAttempt','RushAttempt'].sum().plot.bar(figsize=(20,9),color=['saddlebrown','orange'],rot=0,fontsize=16)
ax.set_title("Play Calling by Down - by Rush or Pass", fontsize=24)
ax.set_xlabel("Down", fontsize=18)
ax.set_ylabel("# of Plays", fontsize=14)
ax.set_alpha(0.8)

# set individual bar lables using above list
for i in ax.patches:
    # get_x: width; get_height: verticle
    ax.text(i.get_x()+.06, i.get_height()-2400, str(round((i.get_height()), 2)), fontsize=16, color='black',rotation=0)


# In[ ]:


plays_down = plays[(plays.down <= 3) & (plays.qtr < 5) & (plays.Team == 'CLE')]
ax = plays_down.groupby(['Season','GameID'])['PassAttempt','RushAttempt'].sum().plot.line(color=['saddlebrown','orange'],figsize=(20,9),rot=0,fontsize=16)
ax.set_title("Season Play Calling - Cleveland Browns", fontsize=24)
ax.set_ylabel("# Plays", fontsize=14)
ax.set_alpha(0.8)


# <h1>Prepreparing Data for Building Model</h1>
# <body>
#     <p>Any model you build is only as good as the parameters its built on. We need to tune the data to maximize the potential of the model to minizime the amount of loss. To do so, here's a list of parameters I'll be using based on different aspects of a football game that contribute to the decision to call a passing play vs a running play:</p>
#     <ul>
#         <li>**Target: PlayType**</li>
#         <li>**TimeSecs:** Time left in game in seconds. Coaches often try to control the clock based on the score.</li>
#         <li>**Score Differential:** This factors in with the time to signify coaches decisions to rush or pass to take advantage of the clock.</li>
#         <li>**qtr:** Quarter the play is in (1-5 where 5 = overtime).</li>
#         <li>**Mean Rushing Yards by qtr:** This will help the models distinguish the current rushing success. Coaches will often make decisions based on the relative advantages on the field.</li>
#         <li>**Offensive Tendencies:** The moving average of the amount of rushing or passing plays called by quarter.</li>
#         <li>**Timeouts Remaining (Pre-snap):** Important to know the timeouts remaining for the teams to know whether to run the ball and force timeouts or to pass to save timeouts.</li> 
#         <li>**Yardline100:** Yard line the ball is on relative to the goal line. (99-50 is own side of field/ 49-1 is opponent's side).</li>
#         <li>**PlayTimeDiff:** The time between the last snap and the current play.</li>
#         <li>**No_Score_Prob:** The probability of there being no score within the half.</li>
#         <li>**TimeUnder:** Amount of time left in the half in minutes.</li>
#         <li>**Win_Prob:** The probability of winning the game with respect to the possession team.</li>
#         <li>**Season:** the season that the game is in.</li>
#         <li>**Down and Yards to Go for 1st down:** The down of the play. Coaches usually distinguish decisions by the amount of yards to gain for a first down. I suspect this is a major attribute due to 3rd and long basically gaurenteeing a pass in tight score differentials.</li>
#     </ul>
#     <p>I'm going to break the information down for the rushing yards gained and the weighted averages for the count of the plays to prepare for joining to the data set. Following this, we can break the 'plays' DataFrame down and further determine clean the data to place in our models.</p>
# </body>
# 

# <body>
#     <p>First, we'll want to gather stats regarding the Mean yards gained for running or passing plays by the team with possession of the ball. This will assure that we can reference the teams' relative performance that game when making a decision. I chose the Mean and not the Median to account for 'big plays' that can affect a coaches optimism of certain play calls. Finding this value is important to factor in because it definitely weighs on a human decision. For instance, imagine telling Bill Belicheck to go for it on 4th and 6 at the 30 yard line when the offense has struggled to move the ball all game. He'd probably break the clipboard over your head!</p>
# </body>
# <h3>Offensive Averages by Quarter</h3>

# In[ ]:


# Get average results for offensive plays by game for model
# to preserve the dataframe's shape (with GameID being unique), I'm going to use a split-apply-merge strategy

# Split - from origional DF: Get 2 DF's for plays that are labeled Run or Pass
r_off_agg = df[(df.PlayType == 'Run')]
p_off_agg = df[(df.PlayType == 'Pass')|(df.PlayType == 'Sack')]

# Apply - groupby aggregation to find the Median yards by game, team, PlayType, and qtr
r_off_agg = r_off_agg.groupby(['GameID','qtr','posteam'])['Yards.Gained'].mean().reset_index()
p_off_agg = p_off_agg.groupby(['GameID','qtr','posteam'])['Yards.Gained'].mean().reset_index()

r_off_agg = r_off_agg.rename(columns={'Yards.Gained':'RushingMean'}) # Rename the columns for clarity
p_off_agg = p_off_agg.rename(columns={'Yards.Gained':'PassingMean'})

# Merge - Combine the Away and Home averages into one dataframe
off_agg = pd.merge(r_off_agg,
                 p_off_agg,
                 left_on=['GameID','qtr','posteam'],
                 right_on=['GameID','qtr','posteam'],
                 how='outer')

off_agg.head(8)


# <h3>Offensive Tendencies and Trends</h3>
# <p>Beyond the average rushing and passing stats by quarter, I believe something else that participates in the determination of the play being a rush or a pass can be described as the tendencies of a coach making the calls. I'm looking to answer the question of what has been done most recently and over time to know the proportions of running plays vs passing plays over a period of time.</p>

# In[ ]:


off_tendencies = df[df.PlayType.notna()&
              (df.PlayType != 'No Play')&
              (df.PlayType != 'Kickoff')&
              (df.PlayType != 'Extra Point')&
              (df.PlayType != 'End of Game')&
              (df.PlayType != 'Quarter End')&
              (df.PlayType != 'Half End')&
              (df.PlayType != 'Two Minute Warning')&
              (df.PlayType != 'Field Goal')&
              (df.PlayType != 'Punt') &
              (df.PlayAttempted == 1)]

# Moving average by team, quarter, and season. This is a rolling average to consider recent decisions to compensate for coaching changes
off_tendencies = off_tendencies.groupby(['GameID','posteam','Season','qtr'])['PassAttempt','RushAttempt'].sum().reset_index()
off_tendencies['PassingWA']=off_tendencies.groupby(['posteam','qtr','Season']).PassAttempt.apply(lambda x: x.shift().rolling(8,min_periods=1).mean().fillna(x))
off_tendencies['RushingWA']=off_tendencies.groupby(['posteam','qtr','Season']).RushAttempt.apply(lambda x: x.shift().rolling(8,min_periods=1).mean().fillna(x))
off_tendencies = off_tendencies.drop(columns=['PassAttempt', 'RushAttempt'])
off_tendencies[(off_tendencies.posteam == 'CLE')&(off_tendencies.qtr == 1)].head(20)


# <body>
#     <p>Now that we have a few more hyperparameters to contribute to our data modelling, we can move on and begin to break the data down to see if we can understand whether we can accurately predict the play calling of an NFL offensive coordinator. I'm going to take a look at my beloved Cleveland Browns because well - maybe theres a reason they cant win football games?</p>
# </body>

# <h1>Cleveland Browns</h1>

# In[ ]:


# to limit the data size, lets look at one team to begin
team = 'CLE'


# <p>To easily adjust the team I'm looking at, I'm using a team variable to easily change if needed.</p>
# 
# <h3>Joining our New Parameters</h3>
# <p>Below, I'm joining the previously calculated parameters to the main data set. I am filtering out plays that have play types that aren't of a run or pass type, so that means eliminating Special Teams plays and other tuples of unneeded data.</p>

# In[ ]:


# take the dataframe for plays above and define particular columns we want
play_attr = ['PlayAttempted','GameID','qtr','TimeSecs','yrdline100','ydstogo','Drive','down','PlayType','GoalToGo',
             'TimeUnder','PlayTimeDiff','PassAttempt','RushAttempt','posteam','DefensiveTeam','PosTeamScore',
             'DefTeamScore','Season','HomeTimeouts_Remaining_Pre','AwayTimeouts_Remaining_Pre','No_Score_Prob',
             'Opp_Field_Goal_Prob','Opp_Safety_Prob','Win_Prob','HomeTeam','ExpPts']
plays = df[play_attr]


# filter out the records that we wont use to predict run or pass
plays = plays[plays.PlayType.notna()&
              (plays.PlayType != 'No Play')&
              (plays.PlayType != 'Kickoff')&
              (plays.PlayType != 'Extra Point')&
              (plays.PlayType != 'End of Game')&
              (plays.PlayType != 'Quarter End')&
              (plays.PlayType != 'Half End')&
              (plays.PlayType != 'Two Minute Warning')&
              (plays.PlayType != 'Field Goal')&
              (plays.PlayType != 'Punt')]

# assure that there was a play attempted to filter out penalties before the play occured.
plays = plays[plays.PlayAttempted == 1]

# add data regarding offensive stats
plays = pd.merge(plays,
                off_agg,
                left_on=['GameID','qtr','posteam'],
                right_on=['GameID','qtr','posteam'],
                how='left')

# merge data for moving average play calling tendencies
plays = pd.merge(plays,
                off_tendencies,
                left_on=['GameID','qtr','posteam','Season'],
                right_on=['GameID','qtr','posteam','Season'],
                how='left')

plays=plays.rename(columns = {'posteam':'Team'})

# filter on just possessions by the cleveland browns (woof woof)
plays = plays[(plays['Team'] == team)]
plays.head(5)


# <h3>Changing/Adding Parameters</h3>
# <p>Here, I'm creating an attribute for the score differential (because the other varies and I'd rather just do it in one line), the current score boolean indicator to show whether the team is winning or losing, the Home team indicator to be 1 if the possession team is home or away, and changing the timeout parameters to reflect the defensive team and defensive team instead of the home and away teams.</p>

# In[ ]:


# get score difference for each play cleveland is in possession of the ball
plays['ScoreDiff'] = plays['PosTeamScore'] - plays['DefTeamScore']

# add column to show boolean indicator for whether the Browns are winning or losing (I expect a lot of 0's)
plays['CurrentScoreBool'] = plays.apply(lambda x: 1 if x.ScoreDiff > 0 else 0, axis=1)

# add column to show if the Brownies are playing at home
plays['Home'] = plays.apply(lambda x: 1 if x.HomeTeam == team else 0, axis=1)

# changing the timeouts attributes to reflect the posteam: CLE and the defensive teams remaining timeouts
plays['PosTO_PreSnap'] = plays.apply(lambda x: x.HomeTimeouts_Remaining_Pre if x.HomeTimeouts_Remaining_Pre == team else x.AwayTimeouts_Remaining_Pre, axis=1)
plays['DefTO_PreSnap'] = plays.apply(lambda x: x.HomeTimeouts_Remaining_Pre if x.HomeTimeouts_Remaining_Pre != team else x.AwayTimeouts_Remaining_Pre, axis=1)

# indicator for 2-minute situations
plays['TwoMinuteDrill'] = plays.apply(lambda x: 1 if (
    (((x.TimeSecs <= 0)&(x.TimeSecs >= 120))|((x.TimeSecs <= 1920)&(x.TimeSecs >= 1800)))&
    (x.CurrentScoreBool == 0)) else 0, axis=1)
                                      


# <h1>Data Cleansing</h1>
# <p>Taking a look at what we're up against with pandas .info()</p>

# In[ ]:


plays.info()


# <h3>Adjusting Feature Data Types</h3>
# <body>
#     <p>Now that our dataset has much of the desired information contained in it, we can look at the data types for each attribute and make adjustments as needed:</p>
# </body>

# In[ ]:


# need to clean float data and transfer to integer
plays.TimeSecs = plays.TimeSecs.fillna(0).astype(int)
plays.yrdline100 = plays.yrdline100.fillna(0).astype(int)
plays.down = plays.down.fillna(0).astype(int)
plays.PosTeamScore = plays.PosTeamScore.fillna(0).astype(int)
plays.DefTeamScore = plays.DefTeamScore.fillna(0).astype(int)
plays.RushingMean = plays.RushingMean.fillna(0).astype(int)
plays.PassingMean = plays.PassingMean.fillna(0).astype(int)
plays.ScoreDiff = plays.ScoreDiff.fillna(0).astype(int)
plays.PlayTimeDiff = plays.PlayTimeDiff.fillna(0).astype(int)
plays.GoalToGo = plays.GoalToGo.fillna(0).astype(int)

plays.RushingWA = plays.RushingWA.fillna(0).round(0).astype(int)
plays.PassingWA = plays.PassingWA.fillna(0).round(0).astype(int)


# <body>
#     <p>Now the float attributes are modified to int-type. Below, I'm going to need to change our target parameter to show the rushing attempt vs the pass attempts. I could also use the commented out lines, but after trying them both ways it really doesn't make a difference.</p>
#     <p>Also, I need to adjust the probability odds that are in the DataFrame. to do this, I'm using Pandas built-in functionality to bin the probabilities into 4 equal categories of integers.</p>
# </body>

# In[ ]:


# play type changed to integer using map - removing others
# PlayTypes = {"Run": 0, "QB Kneel": 0, "Pass": 1, "Sack": 1, "Spike": 1}
# cle.PlayType = cle.PlayType.map(PlayTypes)
# cle.PlayType = cle.PlayType.fillna(0)
# cle.PlayType = cle.PlayType.astype(int)
plays = plays[(plays.PassAttempt == 1)|(plays.RushAttempt == 1)]
plays['PlayType'] = plays.apply(lambda x: 1 if x.PassAttempt == 1 else 0, axis=1)
plays.PlayType = plays.PlayType.fillna(0).astype(int)


# changing float64 to float32
plays.No_Score_Prob = plays.No_Score_Prob.fillna(0).astype(np.float32)
plays.Opp_Field_Goal_Prob = plays.Opp_Field_Goal_Prob.fillna(0).astype(np.float32)
plays.Opp_Safety_Prob = plays.Opp_Safety_Prob.fillna(0).astype(np.float32)
plays.Win_Prob = plays.Win_Prob.fillna(0).astype(np.float32)
plays.ExpPts = plays.ExpPts.fillna(0).astype(np.float32)


plays.No_Score_Prob = pd.qcut(plays['No_Score_Prob'], 5, labels=False)
plays.Opp_Field_Goal_Prob = pd.qcut(plays['Opp_Field_Goal_Prob'], 5, labels=False)
plays.Opp_Safety_Prob = pd.qcut(plays['Opp_Safety_Prob'], 5, labels=False)
plays.Win_Prob = pd.qcut(plays['Win_Prob'], 5, labels=False)
plays.ExpPts = pd.qcut(plays['ExpPts'], 5, labels=False)


# <body>
#     <p>Finally, we can drop some columns that we know we will not want to use in our models:</p>
# </body>

# In[ ]:


# drop unneeded columns to begin to de-clutter the set
plays = plays[plays.down != 0]
plays = plays.drop(columns=['PlayAttempted','HomeTeam','Team','DefensiveTeam',
                        'HomeTimeouts_Remaining_Pre','AwayTimeouts_Remaining_Pre','RushAttempt','PassAttempt'])

plays = plays.rename(columns = {'Drive_x':'Drive'})
plays.head(5)


# <h1>Building Models and Making Predictions</h1>
# <p>First, we need to define our target variable, y as well as define our decision parameters, X.</p>

# In[ ]:


# Define our prediction data
plays_predictors = ['ydstogo','down','ScoreDiff','PosTO_PreSnap','No_Score_Prob','Drive','Season','TimeSecs','TimeUnder','PlayTimeDiff','Opp_Field_Goal_Prob']
X = plays[plays_predictors]

# Define the prediction target: PlayType
y = plays.PlayType


# <p>Using train_test_split, we can easily segment our dataset into training data and testing data. Using these values we can easily build our models and measure the model accuracy.</p>

# In[ ]:


# Split our data into training and test data for both our target and prediction data sets
# random state = 0 means we get same result everytime if we want ot change later
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)


# In[ ]:


# Decision Tree Classifier
desc_tree = DecisionTreeClassifier()
desc_tree.fit(train_X, train_y)

dt_predictions = desc_tree.predict(val_X)

false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, dt_predictions)
dt_roc_auc = auc(false_positive_rate, true_positive_rate)


# In[ ]:


# Random Forest Classification
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_X, train_y)

rf_predictions = random_forest.predict(val_X)

false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, rf_predictions)
rf_roc_auc = auc(false_positive_rate, true_positive_rate)


# In[ ]:


# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(train_X, train_y)

lr_predictions = log_reg.predict(val_X)

false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, lr_predictions)
lr_roc_auc = auc(false_positive_rate, true_positive_rate)


# In[ ]:


# K-Means Clustering
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(train_X, train_y)

knn_predictions = knn.predict(val_X)

false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, knn_predictions)
knn_roc_auc = auc(false_positive_rate, true_positive_rate)


# In[ ]:


# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(train_X, train_y)

gnb_predictions = gnb.predict(val_X)

false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, gnb_predictions)
gnb_roc_auc = auc(false_positive_rate, true_positive_rate)


# In[ ]:


gbc = GradientBoostingClassifier()
gbc.fit(train_X, train_y)

gbc_predictions = gbc.predict(val_X)

false_positive_rate, true_positive_rate, thresholds = roc_curve(val_y, gbc_predictions)
gbc_roc_auc = auc(false_positive_rate, true_positive_rate)


# <h3>Model Outcomes</h3>
# <p>Lets show each models' results measured using the AUC (Area Under Curve) method. I'm using this method becuase it suits binary decision models well, which is what our target value (run or pass) represents.</p>

# In[ ]:


results = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest', 'Logistic Regression', 'KNN',
              'Naive Bayes', 'Gradient Boosting Classifier'],
    'AUC': [dt_roc_auc, rf_roc_auc, lr_roc_auc, knn_roc_auc, gnb_roc_auc, gbc_roc_auc]})
result_df = results.sort_values(by='AUC', ascending=False)
result_df = result_df.set_index('AUC')
result_df.head(7)


# <p>We can see that **Gradient Boosting Classification** is the best model for predicting run or pass. Lets take this and dive further into the results to take a look at further tuning our gradient boosting classification model.</p>
# <h2>Feature Importance</h2>
# <p>Lets plot the influence each of our hyperparameters have on our predictions:</p>

# In[ ]:


importances = pd.DataFrame({'feature':train_X.columns,'importance':np.round(gbc.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.plot.bar(figsize=(20,9),rot=0)


# <h2>Gradient Boosting Classification Tuning</h2>

# <p>I'm going to reset the training and testing variables and follow a guide found at: https://maviator.github.io/2017/12/24/InDepth-Gradient-Boosting/ by Mohtadi Be Fraj to further investigate the parameters contained in Sci Kit Learn's Gradient Boosting Classifier. I highly recommend his reading his blog to see in-depth explanations to understand how various modeling techniques!</p>

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y,random_state = 0)

model = GradientBoostingClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# <h4>Learning Rate Tuning</h4>

# In[ ]:


# a couple learning rates to see how they affect the outcome of our model
learning_rates = [0.2, 0.175 ,0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.01]

train_results = []
test_results = []
train_results = []
test_results = []
for eta in learning_rates:
    model = GradientBoostingClassifier(learning_rate=eta)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)

    y_pred = model.predict(x_test)


    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(learning_rates, train_results, 'b', label="Train AUC")
line2, = plt.plot(learning_rates, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('learning rate')
plt.show()


# <body>We can see that 0.05 (possibly .045) is our data's best fit. Higher Learning rates will result in overfitting the data.</body>

# <h4>Estimators</h4>

# In[ ]:


# n_estimators to adjust to tune outcome
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]

train_results = []
test_results = []
for estimator in n_estimators:
    model = GradientBoostingClassifier(n_estimators=estimator)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)

    y_pred = model.predict(x_test)


    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(n_estimators, train_results, 'b', label="Train AUC")
line2, = plt.plot(n_estimators, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()


# <body>Above, we can see that past around 75 estimators, our model's accuracy is overfitting.</body>
# <h4>Maximum Depths</h4>

# In[ ]:


max_depths = np.linspace(1, 7, 7, endpoint=True)

train_results = []
test_results = []
for max_depth in max_depths:
    model = GradientBoostingClassifier(max_depth=max_depth)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)

    y_pred = model.predict(x_test)


    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()


# <body>Above, we can see that anything beyond a max depth of 2 results in overfitting.</body>
# <h4>Minimum Samples Split and Minimum Leaf Sample</h4>

# In[ ]:


min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

train_results = []
test_results = []
for min_samples_split in min_samples_splits:
    model = GradientBoostingClassifier(min_samples_split=min_samples_split)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)

    y_pred = model.predict(x_test)


    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train AUC")
line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('min samples split')
plt.show()


# In[ ]:


min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
    model = GradientBoostingClassifier(min_samples_leaf=min_samples_leaf)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)

    y_pred = model.predict(x_test)


    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(min_samples_leafs, train_results, 'b', label="Train AUC")
line2, = plt.plot(min_samples_leafs, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('min samples leaf')
plt.show()


# <body>In both above cases, requiring the model to consider all of the features at each node will result in overfitting.</body>
# <h4>Maximum Features</h4>

# In[ ]:


max_features = list(range(1,X.shape[1]))
train_results = []
test_results = []
for max_feature in max_features:
    model = GradientBoostingClassifier(max_features=max_feature)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)

    y_pred = model.predict(x_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(max_features, train_results, 'b', label="Train AUC")
line2, = plt.plot(max_features, test_results, 'r', label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('Maximum Features')
plt.show()


# <p>Above, we can see that the # of features maximizes the model's potential at about 12 Features, but the difference isn't that variable in this case as the number of features rises for the model's test predictions.</p>
