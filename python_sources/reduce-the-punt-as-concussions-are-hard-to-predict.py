#!/usr/bin/env python
# coding: utf-8

# **Reduce the Punt as Concussions are hard to predict**
# 
# This is mainly just the data manipulation the main bulk of the conclusions has been drawn in the PDF

# In[ ]:


#Set up area
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn import tree #tree search
from sklearn import svm #Support Vector Machine search
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB #Gaussian Naive Bayesian Classifier
from sklearn.metrics import accuracy_score #Accuracy against validation
import matplotlib.pyplot as plt # plotting
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve 
import numpy as np # linear algebra
import os # accessing directory structure
from sklearn import datasets
import re
import seaborn as sns
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


#Set up data frames
player_df = pd.read_csv ('../input/NFL-Punt-Analytics-Competition/player_punt_data.csv')
play_player_role_df = pd.read_csv ('../input/NFL-Punt-Analytics-Competition/play_player_role_data.csv')
video_review_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
play_info_df = pd.read_csv ('../input/NFL-Punt-Analytics-Competition/play_information.csv')
game_info_df = pd.read_csv ('../input/NFL-Punt-Analytics-Competition/game_data.csv')


# In[ ]:


# Merge player info
master_player_df = pd.merge(player_df, play_player_role_df,
                          how='inner',
                          on=['GSISID'])

# Megre game info
master_game_df = pd.merge(game_info_df, play_info_df,
                          how='inner',
                          on=['GameKey'])
# Combine play and player information into one super table
master_df = pd.merge(master_player_df, master_game_df,
                          how='inner',
                          on=['GameKey','PlayID'])

#Merge the punt injuries data into the table
master_injury_df = pd.merge(master_df, video_review_df,
                          how='outer',
                          on=['GameKey','PlayID','GSISID'])
master_injury_df.head()


# In[ ]:


# Remove data I consider uncritical for getting the results
drop = ['Season_Year_x','Season_Type_x','Game_Date_x','Game_Site','Home_Team','Visit_Team','Stadium','GameWeather','OutdoorWeather','Season_Year_y','Season_Type_y','Game_Date_y','Week_y','Play_Type']
master_injury_df.drop(columns=drop, inplace=True)
master_injury_df.head()


# In[ ]:


#Convert Testing columns to ints
# Convert timestamp to int
master_injury_df['Start_Time'] = master_injury_df['Start_Time'].str.replace(":","").values.astype(int)

#Convert Game_Clock to int
master_injury_df['Game_Clock'] = master_injury_df['Game_Clock'].str.replace(":","").values.astype(int)

#Convert Temp to int
master_injury_df['Temperature'] = master_injury_df['Temperature'].values.astype(int)

#Convert Positions to ints
master_injury_df.loc[master_injury_df.Position == 'SS' , 'Position'] = 1 
master_injury_df.loc[master_injury_df.Position == 'OLB' , 'Position'] = 2 
master_injury_df.loc[master_injury_df.Position == 'WR' , 'Position'] = 3
master_injury_df.loc[master_injury_df.Position == 'FS' , 'Position'] = 4 
master_injury_df.loc[master_injury_df.Position == 'CB' , 'Position'] = 5 
master_injury_df.loc[master_injury_df.Position == 'RB' , 'Position'] = 6 
master_injury_df.loc[master_injury_df.Position == 'NT' , 'Position'] = 7 
master_injury_df.loc[master_injury_df.Position == 'LS' , 'Position'] = 8 
master_injury_df.loc[master_injury_df.Position == 'ILB' , 'Position'] = 9 
master_injury_df.loc[master_injury_df.Position == 'DE' , 'Position'] = 10 
master_injury_df.loc[master_injury_df.Position == 'FB' , 'Position'] = 11
master_injury_df.loc[master_injury_df.Position == 'TE' , 'Position'] = 12 
master_injury_df.loc[master_injury_df.Position == 'DT' , 'Position'] = 13
master_injury_df.loc[master_injury_df.Position == 'MLB' , 'Position'] = 14
master_injury_df.loc[master_injury_df.Position == 'K' , 'Position'] = 15
master_injury_df.loc[master_injury_df.Position == 'P' , 'Position'] = 16
master_injury_df.loc[master_injury_df.Position == 'LB' , 'Position'] = 17
master_injury_df.loc[master_injury_df.Position == 'S' , 'Position'] = 18
master_injury_df.loc[master_injury_df.Position == 'C' , 'Position'] = 19
master_injury_df.loc[master_injury_df.Position == 'T' , 'Position'] = 20
master_injury_df.loc[master_injury_df.Position == 'QB' , 'Position'] = 21
master_injury_df.loc[master_injury_df.Position == 'DE' , 'Position'] = 22
master_injury_df.loc[master_injury_df.Position == 'DB' , 'Position'] = 23
master_injury_df.loc[master_injury_df.Position == 'G' , 'Position'] = 24

master_injury_df['Position'] = master_injury_df['Position'].values.astype(int)

#Convert Roles to ints
master_injury_df.loc[master_injury_df.Role == 'GL' , 'Role'] = 1 
master_injury_df.loc[master_injury_df.Role == 'GLi' , 'Role'] = 2 
master_injury_df.loc[master_injury_df.Role == 'GLo' , 'Role'] = 3
master_injury_df.loc[master_injury_df.Role == 'GR' , 'Role'] = 4 
master_injury_df.loc[master_injury_df.Role == 'GRi' , 'Role'] = 5 
master_injury_df.loc[master_injury_df.Role == 'GRo' , 'Role'] = 6 
master_injury_df.loc[master_injury_df.Role == 'P' , 'Role'] = 7 
master_injury_df.loc[master_injury_df.Role == 'PC' , 'Role'] = 8 
master_injury_df.loc[master_injury_df.Role == 'PDL1' , 'Role'] = 9 
master_injury_df.loc[master_injury_df.Role == 'PDL2' , 'Role'] = 10 
master_injury_df.loc[master_injury_df.Role == 'PDL3' , 'Role'] = 11
master_injury_df.loc[master_injury_df.Role == 'PDL4' , 'Role'] = 12 
master_injury_df.loc[master_injury_df.Role == 'PDL5' , 'Role'] = 13
master_injury_df.loc[master_injury_df.Role == 'PDL6' , 'Role'] = 14
master_injury_df.loc[master_injury_df.Role == 'PDM' , 'Role'] = 15
master_injury_df.loc[master_injury_df.Role == 'PDR1' , 'Role'] = 16
master_injury_df.loc[master_injury_df.Role == 'PDR2' , 'Role'] = 17
master_injury_df.loc[master_injury_df.Role == 'PDR3' , 'Role'] = 18
master_injury_df.loc[master_injury_df.Role == 'PDR4' , 'Role'] = 19
master_injury_df.loc[master_injury_df.Role == 'PDR5' , 'Role'] = 20
master_injury_df.loc[master_injury_df.Role == 'PDR6' , 'Role'] = 21
master_injury_df.loc[master_injury_df.Role == 'PFB' , 'Role'] = 22
master_injury_df.loc[master_injury_df.Role == 'PLG' , 'Role'] = 23
master_injury_df.loc[master_injury_df.Role == 'PLL' , 'Role'] = 24
master_injury_df.loc[master_injury_df.Role == 'PLL1' , 'Role'] = 25 
master_injury_df.loc[master_injury_df.Role == 'PLL2' , 'Role'] = 26
master_injury_df.loc[master_injury_df.Role == 'PLL3' , 'Role'] = 27
master_injury_df.loc[master_injury_df.Role == 'PLM' , 'Role'] = 28
master_injury_df.loc[master_injury_df.Role == 'PLM1' , 'Role'] = 29
master_injury_df.loc[master_injury_df.Role == 'PLR' , 'Role'] = 30
master_injury_df.loc[master_injury_df.Role == 'PLR1' , 'Role'] = 31
master_injury_df.loc[master_injury_df.Role == 'PLR2' , 'Role'] = 32
master_injury_df.loc[master_injury_df.Role == 'PLR3' , 'Role'] = 33 
master_injury_df.loc[master_injury_df.Role == 'PLS' , 'Role'] = 34
master_injury_df.loc[master_injury_df.Role == 'PLT' , 'Role'] = 35 
master_injury_df.loc[master_injury_df.Role == 'PLW' , 'Role'] = 36
master_injury_df.loc[master_injury_df.Role == 'PPL' , 'Role'] = 37
master_injury_df.loc[master_injury_df.Role == 'PPLi' , 'Role'] = 38
master_injury_df.loc[master_injury_df.Role == 'PPLo' , 'Role'] = 39
master_injury_df.loc[master_injury_df.Role == 'PPR' , 'Role'] = 40
master_injury_df.loc[master_injury_df.Role == 'PPRi' , 'Role'] = 41
master_injury_df.loc[master_injury_df.Role == 'PPRo' , 'Role'] = 42
master_injury_df.loc[master_injury_df.Role == 'PR' , 'Role'] = 43
master_injury_df.loc[master_injury_df.Role == 'PRG' , 'Role'] = 44
master_injury_df.loc[master_injury_df.Role == 'PRT' , 'Role'] = 45
master_injury_df.loc[master_injury_df.Role == 'PRW' , 'Role'] = 46
master_injury_df.loc[master_injury_df.Role == 'VL' , 'Role'] = 47
master_injury_df.loc[master_injury_df.Role == 'VLi' , 'Role'] = 48
master_injury_df.loc[master_injury_df.Role == 'VLo' , 'Role'] = 49
master_injury_df.loc[master_injury_df.Role == 'VR' , 'Role'] = 50
master_injury_df.loc[master_injury_df.Role == 'VRi' , 'Role'] = 51
master_injury_df.loc[master_injury_df.Role == 'VRo' , 'Role'] = 52


master_injury_df['Role'] = master_injury_df['Role'].values.astype(int)

#Game Day number
master_injury_df.loc[master_injury_df.Game_Day == 'Thursday' , 'Game_Day'] = 1 
master_injury_df.loc[master_injury_df.Game_Day == 'Friday' , 'Game_Day'] = 2 
master_injury_df.loc[master_injury_df.Game_Day == 'Saturday' , 'Game_Day'] = 3
master_injury_df.loc[master_injury_df.Game_Day == 'Sunday' , 'Game_Day'] = 4 
master_injury_df.loc[master_injury_df.Game_Day == 'Monday' , 'Game_Day'] = 5 
master_injury_df.loc[master_injury_df.Game_Day == 'Tuesday' , 'Game_Day'] = 6 
master_injury_df.loc[master_injury_df.Game_Day == 'Wednesday' , 'Game_Day'] = 7 

master_injury_df['Game_Day'] = master_injury_df['Game_Day'].values.astype(int)

#Name Field Types ints
master_injury_df.loc[master_injury_df.Turf == '', 'Turf'] = 1 
master_injury_df.loc[master_injury_df.Turf == 'A-Turf Titan', 'Turf'] = 2 
master_injury_df.loc[master_injury_df.Turf == 'Artifical', 'Turf'] = 3
master_injury_df.loc[master_injury_df.Turf == 'Artificial', 'Turf'] = 3
master_injury_df.loc[master_injury_df.Turf == 'AstroTurf GameDay Grass 3D' , 'Turf'] = 4 
master_injury_df.loc[master_injury_df.Turf == 'DD GrassMaster' , 'Turf'] = 5 
master_injury_df.loc[master_injury_df.Turf == 'Field turf' , 'Turf'] = 6 
master_injury_df.loc[master_injury_df.Turf == 'Field Turf' , 'Turf'] = 6 
master_injury_df.loc[master_injury_df.Turf == 'FieldTurf' , 'Turf'] = 6 
master_injury_df.loc[master_injury_df.Turf == 'FieldTurf 360' , 'Turf'] = 7 
master_injury_df.loc[master_injury_df.Turf == 'FieldTurf360' , 'Turf'] = 7 
master_injury_df.loc[master_injury_df.Turf == 'grass' , 'Turf'] = 8
master_injury_df.loc[master_injury_df.Turf == 'Grass' , 'Turf'] = 8 
master_injury_df.loc[master_injury_df.Turf == 'Natrual Grass' , 'Turf'] = 9 
master_injury_df.loc[master_injury_df.Turf == 'Natural Grass' , 'Turf'] = 9 
master_injury_df.loc[master_injury_df.Turf == 'Natural grass' , 'Turf'] = 9
master_injury_df.loc[master_injury_df.Turf == 'Naturall Grass' , 'Turf'] = 9
master_injury_df.loc[master_injury_df.Turf == 'Natural Grass ' , 'Turf'] = 9
master_injury_df.loc[master_injury_df.Turf == 'Natural' , 'Turf'] = 10 
master_injury_df.loc[master_injury_df.Turf == 'Synthetic' , 'Turf'] = 12 
master_injury_df.loc[master_injury_df.Turf == 'Turf' , 'Turf'] = 13
master_injury_df.loc[master_injury_df.Turf == 'UBU Speed Series S5-M' , 'Turf'] = 14
master_injury_df.loc[master_injury_df.Turf == 'UBU Speed Series-S5-M' , 'Turf'] = 14
master_injury_df.loc[master_injury_df.Turf == 'UBU Sports Speed S5-M' , 'Turf'] = 15
master_injury_df['Turf'] = pd.to_numeric(master_injury_df['Turf'], errors='0')
                     
master_injury_df['Turf'] = master_injury_df['Turf'].values.astype(int)




master_injury_df.head()


# In[ ]:


#Greate a colum to identify concussion plays, this is not a clean way
master_injury_df.loc[master_injury_df.Turnover_Related != 'No' , 'Concussion'] = 'No' 
master_injury_df.loc[master_injury_df.Turnover_Related == 'No', 'Concussion'] = 'Yes' 

master_injury_df.head()


# In[ ]:


train, test = train_test_split(master_injury_df, test_size=0.2)
testlen = len(test)
print(testlen)

trainlen = len(train)
print(trainlen)


# In[ ]:


#Set training data
Xtrain = train[['Start_Time', 'Week_x','Position','Role','Game_Day','Temperature','Game_Clock','Turf']].values
Ytrain = train['Concussion']

Xtest = test[['Start_Time', 'Week_x','Position','Role','Game_Day','Temperature','Game_Clock','Turf']].values
Ytest = test['Concussion']



# In[ ]:


#Set Classifiers
clfgnb = GaussianNB()
clfsvm = svm.SVC()
clftree = tree.DecisionTreeClassifier()
lr = LogisticRegression(solver='lbfgs')
svc = LinearSVC(C=1.0)
rfc = RandomForestClassifier(n_estimators=100)


# In[ ]:


plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
for clf, name in [
                  (clfgnb, 'Naive Bayes'),
                    (clfsvm, 'SVM'),
                    (clftree, 'Decision Tree'),
                    (lr, 'Logistic'),
                  (svc, 'Support Vector Classification'),
                  (rfc, 'Random Forest')]:
    clf.fit(Xtrain, Ytrain)
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(Xtest)[:, 1]
    else:  # use decision function
        prob_pos = clf.decision_function(Xtest)
        prob_pos =             (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value =         calibration_curve(Ytest, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
             histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
plt.show()


# **Classifier Conclusions**
# 
# My attempts to build a classifier shows its very hard to determine if someone will get a concussion from preplay information such as the person's position at the start of the play, field conditions or time of game.  Though using further analysis of next gen data you may be able to work out if someone is about to suffer a concussion this is to late to change the situation of a play in football.  My conclusion from this is one individual change to aspect of punts or playing conditions is unlikely to reduce the likelihood of concussion. Similar in the way that changing multiple factors of kickoff was required to reduce the rate of concussions. This makes it a challenge to break apart what aspects of the play you need to adjust and one I think is hard to meet. 
# 
# As I was unable to predict which plays were likely to result in concussion, it makes it very hard to target which aspects of the play I should alter. Therefore I believe rather than trying to make punts safer, the best way to reduce concussions from punts is to attempt to reduce the number of punts. I want to look at this below. 
# 

# In[ ]:


#Set the next gen stat files forked from https://www.kaggle.com/hallayang/nfl-punt-analytics-proposal

ngs_files = ['../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv',
             '../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk1-6.csv',
             '../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk7-12.csv',
             '../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv',
             '../input/NFL-Punt-Analytics-Competition/NGS-2016-post.csv',
             '../input/NFL-Punt-Analytics-Competition/NGS-2017-pre.csv',
             '../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk1-6.csv',
             '../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk7-12.csv',
             '../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk13-17.csv',
             '../input/NFL-Punt-Analytics-Competition/NGS-2017-post.csv']

puntfiles_df = pd.read_csv ('../input/punt-avg/NFLpunt.csv')
playfile_df = pd.read_csv ('../input/playsavg/plays.csv')


# In[ ]:


#Work out return length, fair catch and if there was an injury forked from https://www.kaggle.com/hallayang/nfl-punt-analytics-proposal

plays_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv')

def get_return_yards(s):
    m = re.search('for ([0-9]+) yards', s)
    if m:
        return int(m.group(1))
    elif re.search('for no gain', s):
        return 0
    else:
        return np.nan

plays_df['Return'] = plays_df['PlayDescription'].map(
        lambda x: get_return_yards(x))

video_review = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
video_review = video_review.rename(columns={'GSISID': 'InjuredGSISID'})

plays_df= plays_df.merge(video_review, how='left',
                         on=['Season_Year', 'GameKey', 'PlayID'])

plays_df['InjuryOnPlay'] = 0
plays_df.loc[plays_df['InjuredGSISID'].notnull(), 'InjuryOnPlay'] = 1

plays_df = plays_df[['Season_Year', 'GameKey', 'PlayID', 'Return', 'InjuryOnPlay']]

ngs_df = []
for filename in ngs_files:
    df = pd.read_csv(filename, parse_dates=['Time'])
    df = df.loc[df['Event'].isin(['fair_catch', 'punt_received'])]
    df = pd.concat([df, pd.get_dummies(df['Event'])], axis=1)
    df = df.groupby(['Season_Year', 'GameKey', 'PlayID'])[['fair_catch', 'punt_received']].max()
    ngs_df.append(df.reset_index())
ngs_df = pd.concat(ngs_df)

plays_df = plays_df.merge(ngs_df, on=['Season_Year', 'GameKey', 'PlayID'])

plays_df.head()


# [USA Today Reports](https://eu.usatoday.com/story/sports/nfl/2018/01/26/nfl-concussions-2017-season-study-history/1070344001/) that there was a total of 281 concussions total in the NFL in 2017
# 
# 
# [Fortune reports](http://fortune.com/2017/01/29/nfl-concussions-2016/) that there was a total of 244  concussions total in the NFL in 2016
# 
# [Avg Plays per game](https://www.teamrankings.com/nfl/stat/plays-per-game)

# In[ ]:


#Working out the concusion rate per 1000 non punt plays

Totconc = (281 + 244)

nonpuntconc = Totconc - len(video_review_df.index)

playfile_df['Average 16/17']= playfile_df['2017'] + playfile_df['2016']

avgtotplay = (playfile_df['Average 16/17'].sum())*64

injury_per_1000_non_punt = ((nonpuntconc/avgtotplay)*1000)


# In[ ]:


#Visulise concission rates forked from https://www.kaggle.com/hallayang/nfl-punt-analytics-proposal

injury_per_1000_fair_catch = 1000 * plays_df.loc[plays_df['fair_catch']==1,
                                          'InjuryOnPlay'].mean()
injury_per_1000_punt_received = 1000 * plays_df.loc[plays_df['punt_received']==1,
                                           'InjuryOnPlay'].mean()
fig = plt.figure()
ax = plt.subplot2grid((1, 1), (0, 0))
plt.bar([0, 1, 2], [injury_per_1000_fair_catch, injury_per_1000_punt_received,injury_per_1000_non_punt])
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Fair Catch', 'Punt Received','Non Punt Plays'])
plt.text(0, injury_per_1000_fair_catch+0.2, '{:.1f}'.format(injury_per_1000_fair_catch))
plt.text(1, injury_per_1000_punt_received+0.2, '{:.1f}'.format(injury_per_1000_punt_received))
plt.text(2, injury_per_1000_non_punt+0.2, '{:.1f}'.format(injury_per_1000_non_punt))
plt.title("Concussion Rate")
plt.ylabel("Injuries per 1000 Events")
sns.despine(top=True, right=True)
plt.show()


# 

# In[ ]:


#Visulise return lengths forked from https://www.kaggle.com/hallayang/nfl-punt-analytics-proposal


x_groups = ['0-3 yds', '3-5 yds', '5-7 yds', '7-9 yds',
            '9-12 yds', '12-15 yds', '15-20 yds', '20+ yds']
rec = plays_df.loc[(plays_df['punt_received']==1) 
                   &(plays_df['Return'].notnull())]

y_groups = [sum(rec['Return']<=3) / len(rec),
            sum((rec['Return']>3) & (rec['Return']<=5)) / len(rec),
            sum((rec['Return']>5) & (rec['Return']<=7)) / len(rec),
            sum((rec['Return']>7) & (rec['Return']<=9)) / len(rec),
            sum((rec['Return']>9) & (rec['Return']<=12)) / len(rec),
            sum((rec['Return']>12) & (rec['Return']<=15)) / len(rec),
            sum((rec['Return']>15) & (rec['Return']<=20))/ len(rec),
            sum(rec['Return']>20) / len(rec)]

y_bottoms = [0,
             sum(rec['Return']<=3) / len(rec),
             sum(rec['Return']<=5) / len(rec),
             sum(rec['Return']<=7) / len(rec),
             sum(rec['Return']<=9) / len(rec),
             sum(rec['Return']<=12) / len(rec),
             sum(rec['Return']<=15) / len(rec),
             sum(rec['Return']<=20) / len(rec)]

fig = plt.figure(figsize=(8.5,4.5))
ax = plt.subplot2grid((1, 1), (0, 0))
plt.bar(range(len(x_groups)), y_groups, bottom=y_bottoms)
ax.set_xticks(range(len(x_groups)))
ax.set_xticklabels(x_groups)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
for i in range(len(x_groups)):
    plt.text(i-0.2, y_bottoms[i]+y_groups[i]+0.02, '{:.0f}%'.format(100*y_groups[i]))
sns.despine(top=True, right=True)
plt.title("Distribution of Punt Returns by Length")
plt.show()


# 

# [Average Punt length of 2018](http://www.espn.com/nfl/statistics/team/_/stat/punting/sort/grossAvgPuntYards/seasontype/2)

# In[ ]:


#Work out average of average punt length of 2018 data source is http://www.espn.com/nfl/statistics/team/_/stat/punting/sort/grossAvgPuntYards/seasontype/2

(puntfiles_df['AVG']).plot.bar()
puntfiles_df['AVG'].mean()




# **Punt Conclusions**
# 
# From looking at punt concussion data returning the punt is clearly the most dangerous aspect of the play so if you can encourage punt plays where they are returned you will reduce concussion rates. As per the diagram above you can see that fair catches concussion rates align very closely with concussion rates of non punt plays. 
# 
# If strictly you wanted to increase the safety of the game whilst keeping it in the same balance you could simply offer additional 5yds from the spot of the fair catch. This would be in line with the average return distance and likely encourage fair catches. 
# 
# This addition of Yds for making no play though runs counter to a lot of other aims in football to make the game exciting. Therefore I would suggest the option of if a 4th down attempt is failed within a teams own half the ball is turned over 20 Yds further down field from the spot. This is significantly shorter than the average punt length. But would encourage teams to go for it on 4th down increasing excitement, without the increased risk of injuries punts bring.
# 
