#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV


def get_SeasonCompactResults():
    df_season = pd.read_csv('../input/RegularSeasonDetailedResults.csv')
    df_tourney = pd.read_csv('../input/NCAATourneyDetailedResults.csv')
    df_all = df_season.append(df_tourney)
    return df_all

def get_mean_sum_per_team_per_year():
    z =  m.groupby(['Season','team_a'], as_index=False).agg(
                          ['mean','std'])
    z.columns = ["_".join(x) for x in z.columns.ravel()]
    z.reset_index(inplace=True)
    return z


def get_SeasonCompactResults():
    df_season = pd.read_csv('../input/RegularSeasonDetailedResults.csv')
    df_tourney = pd.read_csv('../input/NCAATourneyDetailedResults.csv')
    df_all = df_season.append(df_tourney)
    return df_all

def get_mean_sum_per_team_per_year(m):
        
        z =  m.groupby(['Season','team_a'], as_index=False).agg(
                              ['mean','std'])
        z.columns = ["_".join(x) for x in z.columns.ravel()]
        z.reset_index(inplace=True)
        z['next_year'] = z['Season']+1
        return z

def prepare_dataset():
    df = get_SeasonCompactResults()
    df=df[df['Season']>2013]
    df['season_day'] = (df['Season'].map(str) + df['DayNum'].map(str)).astype(int)

    x = pd.read_csv('../input/RegularSeasonDetailedResults.csv')
    x.columns

    wins = x[['Season','DayNum','WTeamID','WScore','LTeamID','LScore','WStl','LStl'
             ,'WBlk','LBlk','WAst','LAst','WDR','LDR','WTO','LTO']]
    wins.columns = ['Season','DayNum','team_a','score_a','team_b','score_b','stl_a','stl_b',
                   'blk_a','blk_b','ast_a','ast_b','dr_a','dr_b','to_a','to_b']
    wins['win']=1


    lose = x[['Season','DayNum','LTeamID','LScore','WTeamID','WScore','LStl','WStl'
             ,'LBlk','WBlk','LAst','WAst','LDR','WDR','LTO','WTO']]
    lose.columns = ['Season','DayNum','team_a','score_a','team_b','score_b','stl_a','stl_b',
                    'blk_a','blk_b','ast_a','ast_b','dr_a','dr_b','to_a','to_b']
    lose['win']=0

    m = wins.append(lose)
    z =get_mean_sum_per_team_per_year(m)
    del z['Season']
    # print z.columns
    sub_z_a = z.loc[:,z.columns.str.contains('_a')
                   |z.columns.str.contains('next_year')]

    m = m.merge(sub_z_a,how='inner',left_on=['Season','team_a'],right_on=['next_year','team_a'])
    m = m.merge(sub_z_a,how='inner',left_on=['Season','team_b'],right_on=['next_year','team_a'])
    del m['team_a_y']
    m.rename(columns={'team_a_x':'team_a'},inplace=True)
    return m,z


m,get_mean_sum_per_team_per_year = prepare_dataset()
m['season_day'] = (m['Season'].map(str) + m['DayNum'].map(str)).astype(int)

df_seed = pd.read_csv('../input/NCAATourneySeeds.csv')
df_seed['overal_seed'] = df_seed['Seed'].str.extract('(\d+)').astype(int)
#merge seeds
sas = m.merge(df_seed[['Season','TeamID','overal_seed']],how='left',left_on=['Season','team_a'],right_on=['Season','TeamID'])
sas['overal_seed_team_a'] = sas['overal_seed'].fillna(100).copy()
del sas['overal_seed']
del sas['TeamID']
sas = sas.merge(df_seed[['Season','TeamID','overal_seed']],how='left',left_on=['Season','team_b'],right_on=['Season','TeamID'])
sas['overal_seed_team_b'] = sas['overal_seed'].fillna(100).copy()
del sas['overal_seed']
del sas['TeamID']


z = pd.concat([sas, pd.get_dummies(sas['team_a'], prefix='is_team_a_')], axis=1)
z = pd.concat([z, pd.get_dummies(sas['team_b'], prefix='is_team_b_')], axis=1)




z['Season'] = np.log(z['Season'])
z['season_day'] = np.log(z['season_day'])
y = z['win']
x = z.loc[:,
          z.columns.str.startswith('is_')
          |z.columns.str.startswith('season_day') 
          |z.columns.str.startswith('_a_') 
           |z.columns.str.startswith('_b_') 
         |z.columns.str.startswith('overal_seed_team')
         ]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.1, random_state=1500)

clf = MLPClassifier(activation='logistic', max_iter=100, alpha=0.0000001,shuffle=True,
                     solver='adam', verbose=10,  random_state=21,tol=0.00000000001,warm_start=True,
                   learning_rate = 'adaptive')



# grid = {
#     'C': np.power(10.0, np.arange(-10, 10))
#      , 'solver': ['newton-cg']
# }
# clf = LogisticRegression(penalty='l2', random_state=777, max_iter=10000, tol=10)
# gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=fold)
# gs.fit(x, y)

# print ('gs.best_score_:', gs.best_score_)
    
    
# clf = LogisticRegression(max_iter=1000,penalty='l1')


clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)


# In[4]:


accuracy_score(y_test, y_pred)


# In[5]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[6]:


sns.heatmap(cm, center=True)
plt.show()


# In[51]:



def predict_game_winner(team_a,team_b,year,x_test_columns,clf,m):
    list = [team_a,team_b,year]
    l = pd.DataFrame([list])
    l.columns=['team_a','team_b','year']
    entry = pd.concat([l, pd.get_dummies(l['team_a'], prefix='is_team_a_')], axis=1)
    entry = pd.concat([entry, pd.get_dummies(l['team_b'], prefix='is_team_b_')], axis=1)
    entry = entry.merge(get_mean_sum_per_team_per_year,how='inner',left_on=['year','team_a'],right_on=['next_year','team_a'])
    list_of_missing = [item for item in x_test.columns if item not in entry.columns]
    df_nan = pd.DataFrame(0, index=[0], columns=list_of_missing)
    single_pred = pd.concat([entry,df_nan], axis=1)
    single_pred=single_pred[x_test.columns]

    x = single_pred.loc[:,single_pred.columns.str.startswith('is_')
          |single_pred.columns.str.startswith('season_day') 
          |single_pred.columns.str.startswith('_a_') 
           |single_pred.columns.str.startswith('_b_') 
         |single_pred.columns.str.startswith('overal_seed_team')]

    return clf.predict_proba(x)


# test single game
predict_game_winner(1314,1139,2017,x_test.columns,clf,m)



# In[53]:


df_sample_sub = pd.read_csv('../input/SampleSubmissionStage1.csv')
df_sample_sub['Season'] = df_sample_sub['ID'].apply(lambda x : int(x.split('_')[0]) )
df_sample_sub['team_a'] = df_sample_sub['ID'].apply(lambda x : int(x.split('_')[1]) )
df_sample_sub['team_b'] = df_sample_sub['ID'].apply(lambda x : int(x.split('_')[2]) )

for row in df_sample_sub.iterrows():
    team_a = row[1]['team_a']
    team_b = row[1]['team_b']
    season = row[1]['Season']
    res = predict_game_winner(team_a,
                              team_b,
                              2017,
                              x_test.columns
                              ,gs,
                             m)
    df_sample_sub.loc[(df_sample_sub['team_a']==team_a)&
                      (df_sample_sub['team_b']==team_b)&
                      (df_sample_sub['Season']==season),'Pred'] = res[0][0]
print 'done'


# In[54]:


# import pickle
# pickle.dump(clf, open('saved_NN_logistic_adam_adaptive', 'wb'))

# # some time later...

# # load the model from disk
# loaded_model = pickle.load(open('saved_NN_logistic_adam_adaptive', 'rb'))
# loaded_model.predict(x_test)

print df_sample_sub.columns
df_sample_sub[['ID','Pred']].to_csv('submission_file.csv',index=False)

