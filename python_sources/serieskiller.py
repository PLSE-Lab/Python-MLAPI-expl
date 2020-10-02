# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df_basic = pd.read_csv('../input/database.csv')
df_Stranger = df_basic[df_basic['Relationship'] == 'Stranger']
df_others = df_basic[df_basic['Relationship'] != 'Stranger']

#Time & case number analysis
case_stranger = df_Stranger['Year'].value_counts().sort_index()
case_others = df_others['Year'].value_counts().sort_index()
fig, ax = plt.subplots()
ax.plot(case_stranger.index,case_stranger.values,'.-.',color='green',label = 'Case by stranger')
ax.plot(case_others.index,case_others.values,'.-.',color = 'orangered',label = 'Case by others')
ax.set_title('Case number between 1980 and 2014')
ax.set_ylabel('Case Number')
ax.legend(loc='upper right', shadow=True, fontsize='10')
print ('done')
plt.show()