#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

epl = pd.read_csv('../input/epl-results-19932018/EPL_Set.csv')

epl


# In[ ]:


epl_data = epl.dropna()

epl_data


# In[ ]:


epl_data.describe(include="all")


# In[ ]:


# Make varaible name shorter

epl = epl_data


# In[ ]:


epl_home = epl['HomeTeam']
epl_away = epl['AwayTeam']


# In[ ]:


'''
key = team_name 
value = total_games
'''
team_total_dic = {}
team_home_dic = {}
team_away_dic = {}


# In[ ]:


team_list = list(epl['HomeTeam'].unique())


# In[ ]:


'''

Total home games
Total away games

Total gmaes = Total home games + Total away games

'''

for team in team_list:
    home_games_cnt = 0

    for home in epl_home:
        if home == team:
            home_games_cnt += 1
    
    team_home_dic[team] = home_games_cnt
    
sorted(team_home_dic.items(), key=lambda team : team[1], reverse=True)


# In[ ]:


for team in team_list:
    away_games_cnt = 0

    for away in epl_away:
        if away == team:
            away_games_cnt += 1
    
    team_away_dic[team] = away_games_cnt
    
sorted(team_away_dic.items(), key=lambda team : team[1], reverse=True)


# In[ ]:


# team is key of dic
for team in team_home_dic:
    team_total_dic[team] = team_home_dic[team] + team_away_dic[team]


# team_total_dic = team_home_dic + team_away_dic

sorted(team_total_dic.items(), key=lambda team : team[1], reverse=True)


# In[ ]:


# Full time result

epl_ftr = epl['FTR']


# In[ ]:


# Get total wins each teams

team_win_dic = {}

for team in team_list:
    win_cnt = 0
    
    # The index had been cleared, so add 924 to access pre index
    for idx, ftr in enumerate(epl_ftr):
        if ftr == 'H' and epl['HomeTeam'][idx + 924] == team:
            win_cnt += 1
        elif ftr == 'A' and epl['AwayTeam'][idx + 924] == team:
            win_cnt += 1
               
    team_win_dic[team] = win_cnt

sorted(team_win_dic.items(), key=lambda team : team[1], reverse=True)


# In[ ]:


# Get each teams winning rate


total_win_rate = {}

for team in team_list:

    total_win_rate[team] = round((team_win_dic[team] / team_total_dic[team]) * 100, 2)

sorted(total_win_rate.items(), key=lambda team : team[1], reverse=True)


# In[ ]:


# Get home wins each teams

home_win_cnt = {}
home_win_rate = {}


for team in team_list:
    win_cnt = 0
    
    # The index had been cleared, so add 924 to access pre index
    for idx, ftr in enumerate(epl_ftr):
        if ftr == 'H' and epl['HomeTeam'][idx + 924] == team:
            win_cnt += 1
               
    home_win_cnt[team] = win_cnt



for team in team_list:

    home_win_rate[team] = round((home_win_cnt[team] / team_home_dic[team]) * 100, 2)

sorted(home_win_rate.items(), key=lambda team : team[1], reverse=True)


# In[ ]:


# Get away wins each teams

away_win_cnt = {}
away_win_rate = {}


for team in team_list:
    win_cnt = 0
    
    # The index had been cleared, so add 7582 to access pre index
    for idx, ftr in enumerate(epl_ftr):
        if ftr == 'A' and epl['AwayTeam'][idx + 924] == team:
            win_cnt += 1
               
    away_win_cnt[team] = win_cnt



for team in team_list:

    away_win_rate[team] = round((away_win_cnt[team] / team_away_dic[team]) * 100, 2)

sorted(away_win_rate.items(), key=lambda team : team[1], reverse=True)


# In[ ]:


# Is there any team have been more wins in away?

strong_away_team = False
print('Teams that strong in when away:')

for team in team_list:
    if away_win_rate[team] > home_win_rate[team]:
        print(team)
        strong_away_team = True
       

if strong_away_team == False:
    print("No such team")
# There isn`t...


# ## Data from 1995 to 2018 show no strong team on Away

# In[ ]:


# Have to preprocessing

# Conver float to int

epl.FTAG = epl.FTAG.astype(int)
epl.FTHG = epl.FTHG.astype(int)
epl.HTAG = epl.HTAG.astype(int)
epl.HTHG = epl.HTHG.astype(int)


# # Visualiztaion

# In[ ]:


# Visualization

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

epl.hist(figsize=(21.7, 13.27))


# ## There are more goals in team on home 

# In[ ]:


print('FTAG : ',epl['FTAG'].sum())
print('FTHG : ',epl['FTHG'].sum())
print('HTAG : ',epl['HTAG'].sum())
print('HTHG : ',epl['HTHG'].sum())


# In[ ]:


epl.boxplot(figsize=(8,8))


# In[ ]:


sns.boxplot(data=epl)


# ## In first half goal 
# 
# Home/Away 0 ~ 1
# 
# ## Goals on full time
# 
# Home team in 1 ~ 2
# 
# Away team in 0 ~ 2
# 
# 
# ## The max of the goals are in home team
# 
# 

# # So it is more advantages in home team

# In[ ]:


total_win_rate = pd.DataFrame.from_dict(total_win_rate, orient='index')

total_win_rate.columns = ['Win rate']


# In[ ]:


# Modify total_win_rate by descending in Win rate

total_win_rate_desc = total_win_rate.sort_values("Win rate", ascending=False)
total_win_rate_desc


# In[ ]:


# Divide teams by win rate 
# top, mid, low

top = total_win_rate_desc[:16]

mid = total_win_rate_desc[16:32]

low =total_win_rate_desc[32:48]


# In[ ]:


top.plot.bar()


# In[ ]:


mid.plot.bar()


# In[ ]:


low.plot.bar()


# In[ ]:


# Transpose for get easily team list

top_t = top.T
mid_t = mid.T
low_t = low.T


# In[ ]:


top_team_list = []

for team in top_t:
    top_team_list.append(team)
    
top_team_list


# In[ ]:


mid_team_list = []

for team in mid_t:
    mid_team_list.append(team)
    
mid_team_list


# In[ ]:


low_team_list = []

for team in low_t:
    low_team_list.append(team)
    
low_team_list


# # HTR, FTR ratio

# In[ ]:


epl['HTR'].value_counts(normalize=True)


# In[ ]:


epl['FTR'].value_counts(normalize=True)


# In[ ]:


epl['FTR'].value_counts(normalize=True)


# # First half and full time
# 
# - In the first half, draws were the most common, but overall results were the most likely to be won by the home team.

# In[ ]:


sns.countplot(y=epl['HTR'])


# In[ ]:


sns.countplot(y=epl['FTR'])


# In[ ]:


# Mean of HTHG and HTAG 

print(epl['HTHG'].groupby(epl['HTR']).mean(),'\n\n', epl['HTAG'].groupby(epl['HTR']).mean())


# In[ ]:


epl['HTHG'].groupby(epl['HTR']).mean().plot.bar()


# In[ ]:


epl['HTAG'].groupby(epl['HTR']).mean().plot.bar()


# In[ ]:


sns.countplot(epl['HTHG'], hue=epl['HTR'])


# In[ ]:


sns.countplot(epl['HTAG'], hue=epl['HTR'])


# In[ ]:


sns.countplot(epl['FTHG'], hue=epl['FTR'])


# In[ ]:


sns.countplot(epl['FTAG'], hue=epl['FTR'])


# # Preprocessing
# 
# Make epl_prep for prototype

# In[ ]:


# Drop the non valued columns

epl_prep = epl.drop(['Div'], axis=1)
epl_scatt = epl.drop(['Div'], axis=1)


# In[ ]:


from sklearn import preprocessing

le_hda = preprocessing.LabelEncoder()
le_hda.fit(epl_prep['FTR'])
le_hda_pred = le_hda.transform(epl_prep['FTR'])


# In[ ]:


epl_prep.insert(0, 'Predicted', le_hda_pred)


# In[ ]:


epl_prep


# In[ ]:


le_hda_htr = le_hda.transform(epl_prep['HTR'])
le_hda_ftr = le_hda.transform(epl_prep['FTR'])

epl_prep['HTR'] = le_hda_htr
epl_prep['FTR'] = le_hda_ftr


# In[ ]:


le_date = preprocessing.LabelEncoder()
le_date.fit(epl_prep['Date'])

le_date = le_date.transform(epl_prep['Date'])

epl_prep['Date'] = le_date


# In[ ]:


le_ssn = preprocessing.LabelEncoder()
le_ssn.fit(epl_prep['Season'])

le_ssn = le_ssn.transform(epl_prep['Season'])

epl_prep['Season'] = le_ssn


# In[ ]:


# Encoidng the team top, mid, low

'''

top_team_list 0
mid_team_list 1
low_team_list 2

'''

for team in epl_prep['HomeTeam']:
    for top in top_team_list:
        if team == top:
            epl_prep['HomeTeam'] = np.where(epl_prep.HomeTeam == top, 0, epl_prep.HomeTeam)

    for mid in mid_team_list:
        if team == mid:
            epl_prep['HomeTeam'] = np.where(epl_prep.HomeTeam == mid, 1, epl_prep.HomeTeam)
            
    for low in low_team_list:
        if team == low:
            epl_prep['HomeTeam'] = np.where(epl_prep.HomeTeam == low, 2, epl_prep.HomeTeam)




for team in epl_prep['AwayTeam']:
    for top in top_team_list:
        if team == top:
            epl_prep['AwayTeam'] = np.where(epl_prep.AwayTeam == top, 0, epl_prep.AwayTeam)

    for mid in mid_team_list:
        if team == mid:
            epl_prep['AwayTeam'] = np.where(epl_prep.AwayTeam == mid, 1, epl_prep.AwayTeam)
            
    for low in low_team_list:
        if team == low:
            epl_prep['AwayTeam'] = np.where(epl_prep.AwayTeam == low, 2, epl_prep.AwayTeam)


# In[ ]:


# Delet the FTR, its the same as 'Predicted' attribute

epl_prep = epl_prep.drop(['FTR'], axis=1)


# In[ ]:


# Date is too specific attribute so drop it

epl_prep = epl_prep.drop(['Date'], axis=1)


# In[ ]:


# FTHG and FTAG is not in frist half so drop it

epl_prep = epl_prep.drop(['FTHG'], axis=1)
epl_prep = epl_prep.drop(['FTAG'], axis=1)


# In[ ]:


# Categorized data (numerical)

epl_prep


# In[ ]:


from sklearn.model_selection import train_test_split

# Make train_test_split

epl_train = epl_prep.iloc[:, epl_prep.columns != 'Predicted']
epl_test = epl_prep.iloc[:, epl_prep.columns == 'Predicted']


X_train, X_test, y_train, y_test = train_test_split(epl_train,
                                                     epl_test,
                                                     test_size = 0.20, 
                                                     random_state = 42)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                   y_train,
                                                   test_size = 0.25, 
                                                   random_state = 42)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Validation check

rf_v = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=2,
                             min_samples_leaf=10,
                             max_features='auto',
                             oob_score=True,
                             random_state=42,
                             n_jobs=-1)

rf_v.fit(X_train, y_train.values.ravel())
print("OOB Score : %.4f" % rf_v.oob_score_)
score = rf_v.score(X_val, y_val)
print("Score : ", score)


# In[ ]:


# Test data check

rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=2,
                             min_samples_leaf=10,
                             max_features='auto',
                             oob_score=True,
                             random_state=42,
                             n_jobs=-1)

rf.fit(X_train, y_train.values.ravel())
print("OOB Score : %.4f" % rf.oob_score_)
score = rf.score(X_test, y_test)
print("Score : ", score)


# In[ ]:


# Get the best parameters

# -- This take some minutes, so run it if you want --

# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier


# param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12, 16], "n_estimators": [50, 100, 400, 700, 1000]}

# gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)

# gs = gs.fit(X_train, y_train.values.ravel())

# print(gs.best_score_)

# print(gs.best_params_)

# print(gs.cv_results_)


# -- This is the results --

# -> 0.648048048048048

# -> {'criterion': 'gini', 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 1000}

# -> {'mean_fit_time': array([0.32810688, 0.55880944, 2.31270464, 4.24339898, 5.48761559, ...


# In[ ]:


plt.scatter(epl_prep['HTHG'],
            
            epl_prep['HTAG'],
            
            alpha=0.42)

plt.xlabel('Half Time Home Goal', fontsize=14)
plt.ylabel('Half Time Away Goal', fontsize=14)
plt.legend()


# In[ ]:


sns.scatterplot(x='HTHG', 

                y='HTAG', 

                hue='HTR',
                
                s=90,

                style='FTR',

                data=epl_scatt)

plt.show()


# In[ ]:


sns.scatterplot(x='HTHG', 

                y='FTHG', 

                hue='HTR',
                
                s=90,

                style='FTR',

                data=epl_scatt)

plt.show()


# In[ ]:


sns.scatterplot(x='HomeTeam', 

                y='HTHG', 

                hue='HTR',
                
                s=90,

                style='FTR',

                data=epl_scatt)

sns.set(font_scale=0.4)
sns.set(rc={'figure.figsize':(1001.7,800.27)})


plt.show()


# In[ ]:


# Importance of attributes

epl_imp = pd.concat((pd.DataFrame(epl_prep.iloc[:, 1:].columns, columns = ['Attribute']), 
          pd.DataFrame(rf.feature_importances_, columns = ['Importance'])),    
          axis = 1).sort_values(by='Importance', ascending=False)

epl_imp


# In[ ]:


# Ascending importance

epl_imp.sort_values('Importance', ascending=True, inplace=True)


# In[ ]:


epl_imp.plot(kind='barh', x='Attribute', y='Importance', legend=False, figsize=(6, 10))

plt.title('Random forest feature importance', fontsize = 24)
plt.xlabel('')
plt.ylabel('')
plt.xticks([], [])
plt.yticks(fontsize=20)
plt.show()


# # Conclusions
# 
# Data about the winning team in the first half can be used to predict the game to some extent.
# 
# Can predict the game result by the first half, a little, but more attributes are needed.
