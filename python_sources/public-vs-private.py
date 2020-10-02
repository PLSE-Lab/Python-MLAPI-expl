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
vmin = 0.35
vmax = 0.70

random.seed(42) 
np.random.seed(0)


###########################################################
# main

color_names = [
    'red',
    'blue',
    'green',
    'cyan',
    'magenta',
    'yellow',
    ]

#color_names = [c for c in colors.cnames if ('dark' in c)]
#color_names = [c for c in colors.cnames if ('deep' in c)]
#color_names = [c for c in colors.cnames]
random.shuffle(color_names)


competitions = pd.read_csv(os.path.join(input_dir, 'Competitions.csv'))

c = competitions[competitions['Title'].str.contains(competition_title)]
competition_id = c['Id'].iloc[0]

teams = pd.read_csv(os.path.join(input_dir, 'Teams.csv'))
teams.rename(columns={'Id':'TeamId'}, inplace=True)
teams = teams[teams['CompetitionId'] == competition_id]

submissions = pd.read_csv(os.path.join(input_dir, 'Submissions.csv'))
submissions.rename(columns={'Id':'SubmissionId'}, inplace=True)

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
for i, r in df.iterrows():
    c = color_names[i % len(color_names)]
    l = '{} ({} -> {})'.format(r['TeamName'], r['PublicRanking'], r['PrivateRanking'])

    ss = submissions[submissions['TeamId'] == r['TeamId']]

    ss_n = ss[ss['IsSelected'] == 0] # not selected submissions 
    ss_s = ss[ss['IsSelected'] == 1] # selected submissions

    m_n = 'x'
    m_s = '.'

    a_n = 0.4
    a_s = 1.0

    s_n = 80
    s_s = 120

    fs = (10, 10)

    if i == 0:
        ax = ss_s.plot(kind='scatter', x='PublicScore', y='PrivateScore', color=c, label=l, alpha=a_s, marker=m_s, s=s_s, figsize=fs)
        ss_n.plot(kind='scatter', x='PublicScore', y='PrivateScore', color=c, label='', alpha=a_n, marker=m_n, s=s_n, ax=ax)
    else:
        if ss_s.shape[0] > 0:
            ss_s.plot(kind='scatter', x='PublicScore', y='PrivateScore', color=c, label=l, alpha=a_s, marker=m_s, s=s_s, ax=ax)
            ss_n.plot(kind='scatter', x='PublicScore', y='PrivateScore', color=c, label='', alpha=a_n, marker=m_n, s=s_n, ax=ax)
        else:
            ss_n.plot(kind='scatter', x='PublicScore', y='PrivateScore', color=c, label=l, alpha=a_n, marker=m_n, s=s_n, ax=ax)



plt.xlim([vmin, vmax])
plt.ylim([vmin, vmax])

plt.plot(np.linspace(vmin, vmax, 101), np.linspace(vmin, vmax, 101), c='Gray', alpha=0.20, linewidth=1)

plt.title('Public score vs. Private score @ {}'.format(competition_title))
#plt.show()
plt.savefig('plot.png')
