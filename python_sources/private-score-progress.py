import os
import pandas as pd
import numpy as np
import itertools

import random

import matplotlib.pylab as plt
import matplotlib.colors as colors
import matplotlib 
matplotlib.style.use('ggplot')


###########################################################
# parameters of this script

#input_dir = 'data'
input_dir = '../input/'


competition_title = 'Africa Soil Property Prediction Challenge'

n_top = 3
vmin = 0.465
vmax = 0.52


###########################################################
# main

competitions = pd.read_csv(os.path.join(input_dir, 'Competitions.csv'))

c = competitions[competitions['Title'].str.contains(competition_title)].copy()
c['Deadline'] = np.array(c['Deadline'], dtype='datetime64[s]')
competition_id = c['Id'].iloc[0]

teams = pd.read_csv(os.path.join(input_dir, 'Teams.csv'))
teams.rename(columns={'Id':'TeamId'}, inplace=True)
teams = teams[teams['CompetitionId'] == competition_id]

submissions = pd.read_csv(os.path.join(input_dir, 'Submissions.csv'))

submissions.rename(columns={'Id':'SubmissionId'}, inplace=True)
#submissions = submissions[submissions['IsAfterDeadline'] == False] # this does not work?
submissions['DateSubmitted'] = np.array(submissions['DateSubmitted'], dtype='datetime64[s]')

submissions = submissions[submissions['DateSubmitted'] <= c['Deadline'].iloc[0]]

df = pd.merge(submissions, teams, how='inner', on='TeamId')
df = df[['TeamId', 'PublicScore', 'PrivateScore', 'Ranking', 'TeamName', 'IsSelected']].groupby('TeamId').agg({
    'Ranking': (lambda x: x.iloc[0]),
    'TeamName': (lambda x: x.iloc[0]),
    'IsSelected': (lambda x: x.iloc[0]),
    'PublicScore': np.min,
    'PrivateScore': np.min,
    })

df.sort('PublicScore', ascending=True, inplace=True)
df.reset_index(drop=False, inplace=True)
df['PublicRanking'] = df.index + 1
df.rename(columns={'Ranking': 'PrivateRanking'}, inplace=True)


df.sort('PrivateRanking', ascending=True, inplace=True)

df = df[(df['PrivateRanking'] <= n_top) | (df['PublicRanking'] <= n_top)]

df.reset_index(drop=True, inplace=True)

ax = None
cs = []
ys = []

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(1,1,1)

for i, r in df.iterrows():
    l = '{} ({} -> {})'.format(r['TeamName'], r['PublicRanking'], r['PrivateRanking'])

    ss = submissions[submissions['TeamId'] == r['TeamId']].copy()
    ss.sort('DateSubmitted', ascending=True, inplace=True)

    rv, = ax.plot(ss['DateSubmitted'], ss['PrivateScore'], label=l)
    c = rv.get_color()

    cs.append(c)
    ys.append(np.min(ss['PrivateScore']))
    
(xmin, xmax) = ax.get_xlim()

for y, c in zip(ys, cs):
    ax.hlines(y, xmin, xmax, color=c, linestyles='--')

plt.ylim([vmin, vmax])

plt.title('Private score progress @ {}'.format(competition_title))

plt.legend()

#plt.show()
plt.savefig('plot.png')