# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter##


teams = pd.read_csv('../input/Teams.csv')

teams = teams[teams['yearID'] >= 1985]
teams = teams[['yearID', 'teamID', 'Rank', 'R', 'RA', 'G', 'W', 'H', 'BB', 'HBP', 'AB', 'SF', 'HR', '2B', '3B']]


teams = teams.set_index(['yearID','teamID'])

print(teams['W'][2001,'OAK'])

salaries = pd.read_csv('../input/Salaries.csv')
salaries_by_yearID_teamID = salaries.groupby(['yearID', 'teamID'])['salary'].sum()

salaries_by_yearID_teamID[2001, 'OAK']

teams = teams.join(salaries_by_yearID_teamID)

teams['salary'][2001, 'OAK']


plt.plot(teams['salary'][2001], teams['W'][2001])
plt.show()