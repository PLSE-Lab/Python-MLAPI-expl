# Adapted from work by dune_dweller https://www.kaggle.com/dvasyukova/d/kaggle/meta-kaggle/scripty-medals/notebook


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# All Competitions
competitions = (pd.read_csv('../input/Competitions.csv')
                .rename(columns={'Id':'CompetitionId'}))
competitions = competitions[(competitions.UserRankMultiplier > 0)]

# Scriptprojects to link scripts to competitions
scriptprojects = (pd.read_csv('../input/ScriptProjects.csv')
                    .rename(columns={'Id':'ScriptProjectId'}))

# Evaluation algorithms
evaluationalgorithms = (pd.read_csv('../input/EvaluationAlgorithms.csv')
                          .rename(columns={'Id':'EvaluationAlgorithmId'}))
competitions = (competitions.merge(evaluationalgorithms[['IsMax','EvaluationAlgorithmId']],
                                   on='EvaluationAlgorithmId',how='left')
                            .set_index('CompetitionId'))

# Fill missing values for two competitions
competitions.loc[4488,'IsMax'] = True # Flavours of physics
competitions.loc[4704,'IsMax'] = False # Santa's Stolen Sleigh

# Teams
teams = (pd.read_csv('../input/Teams.csv')
         .rename(columns={'Id':'TeamId'}))
teams = teams[teams.CompetitionId.isin(competitions.index)]
teams['Score'] = teams.Score.astype(float)


competitions['Nteams'] = teams.groupby('CompetitionId').size()
t = competitions.Nteams
competitions.loc[t <= 99, 'Bronze'] = np.floor(0.4*t)
competitions.loc[t <= 99, 'Silver'] = np.floor(0.2*t)
competitions.loc[t <= 99, 'Gold'] = np.floor(0.1*t)

competitions.loc[(100<=t)&(t<=249),'Bronze'] = np.floor(0.4*t)
competitions.loc[(100<=t)&(t<=249),'Silver'] = np.floor(0.2*t)
competitions.loc[(100<=t)&(t<=249), 'Gold'] = 10

competitions.loc[(250<=t)&(t<=999),'Bronze'] = 100
competitions.loc[(250<=t)&(t<=999),'Silver'] = 50
competitions.loc[(250<=t)&(t<=999), 'Gold'] = 10 + np.floor(0.002*t)

competitions.loc[t >= 1000, 'Bronze'] = np.floor(0.1*t)
competitions.loc[t >= 1000, 'Silver'] = np.floor(0.05*t)
competitions.loc[t >= 1000, 'Gold'] = 10 + np.floor(0.002*t)


scriptycomps = competitions.sort_values(by='Nteams')
fig, ax = plt.subplots(figsize=(10,35))
h = np.arange(len(scriptycomps))
ax.barh(h, scriptycomps.Nteams,color='white')
ax.barh(h, scriptycomps.Bronze,color='#F0BA7C')
ax.barh(h, scriptycomps.Silver,color='#E9E9E9')
ax.barh(h, scriptycomps.Gold,color='#FFD448')
ax.set_yticks(h+0.4)
ax.set_yticklabels(scriptycomps.Title.values);
ax.set_ylabel('');
ax.set_xlim(0,1000)
ax.legend(['None','Bronze','Silver','Gold'],loc=4,fontsize='large');
ax.set_title('Medals by Competition Size');
ax.set_xlabel('Rank')
ax.set_ylim(0,h.max()+1);

fig.savefig('1-medals-by-teams.png')

scriptycomps = competitions.sort_values(by='DateEnabled')
fig, ax = plt.subplots(figsize=(10,35))
h = np.arange(len(scriptycomps))
ax.barh(h, scriptycomps.Nteams,color='white')
ax.barh(h, scriptycomps.Bronze,color='#F0BA7C')
ax.barh(h, scriptycomps.Silver,color='#E9E9E9')
ax.barh(h, scriptycomps.Gold,color='#FFD448')
ax.set_yticks(h+0.4)
ax.set_yticklabels(scriptycomps.Title.values);
ax.set_ylabel('');
ax.set_xlim(0,1000)
ax.legend(['None','Bronze','Silver','Gold'],loc=4,fontsize='large');
ax.set_title('Medals be Competition Date');
ax.set_xlabel('Rank')
ax.set_ylim(0,h.max()+1);

fig.savefig('2-medals-by-date.png')