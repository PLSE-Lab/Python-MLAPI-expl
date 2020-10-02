# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

df = pd.read_excel('/kaggle/input/Training.xlsx')

df['fatigue'] = 0
df['fitness'] = 0
df['form'] = df['fitness'] - df['fatigue']

time_fatigue = 7
time_fitness = 42
next_week = date.today() + timedelta(days=7)

df['Date'] = pd.to_datetime(df.Date)
df_1 = df.sort_values(by=['Date']).reset_index(drop=True)
#########################################################################
for i in range(1, len(df_1)):
    df_1.loc[i, 'fatigue'] = df_1.loc[i-1,'fatigue'] + ( (df_1.loc[i, 'TSS'] - df_1.loc[i-1, 'fatigue']) / time_fatigue)
    
for i in range(1, len(df_1)):
    df_1.loc[i, 'fitness'] = df_1.loc[i-1,'fitness'] + ( (df_1.loc[i, 'TSS'] - df_1.loc[i-1, 'fitness']) / time_fitness)
    
df_1['form'] = df_1['fitness'] - df_1['fatigue']
#########################################################################
print(df_1[-7:])

ax = sns.lineplot(x='Date', y='fatigue', data=df_1, color='red')
ax = sns.lineplot(x='Date', y='fitness', data=df_1, color = 'green')
ax = sns.lineplot(x='Date', y='form', data=df_1, color = 'blue')
ax = sns.scatterplot(x='Date', y='TSS', data=df_1, color = 'black')
ax = plt.xlim(pd.to_datetime('2020-01-01'), next_week)