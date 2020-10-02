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

random.seed(42) 
np.random.seed(0)


###########################################################
# main

competitions = pd.read_csv(os.path.join(input_dir, 'Competitions.csv'))

c = competitions[competitions['Title'].str.contains('Otto')]
competition_id = c['Id'].iloc[0]

print('[1] When was the Competition started?')
print(c['DateEnabled'])
print('')

teams = pd.read_csv(os.path.join(input_dir, 'Teams.csv'))
teams.rename(columns={'Id':'TeamId'}, inplace=True)
teams = teams[teams['CompetitionId'] == competition_id]

submissions = pd.read_csv(os.path.join(input_dir, 'Submissions.csv'))
submissions.rename(columns={'Id':'SubmissionId'}, inplace=True)

df = pd.merge(submissions, teams, how='inner', on='TeamId')
df.sort('DateSubmitted', ascending=True, inplace=True)

df = df[df['PublicScore'] < 0.41]

df = df.groupby('TeamId').first().reset_index(drop=False)
df.sort('DateSubmitted', ascending=True, inplace=True)

print(df.head(40))
