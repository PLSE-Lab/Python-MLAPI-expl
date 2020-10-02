# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import matplotlib.pyplot as plt   # Import matplotlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input//all.csv') 

states = df.groupby('State').sum()
print(states[:5])

states['Num.Person.Per.HouseHolds'] = states['Persons']/ states['Number.of.households']
states['NumMalesHouseHolds'] = states['Males']/ states['Number.of.households']
states['NumFemalesHouseHolds'] = states['Females']/ states['Number.of.households']

numHousehold =  states[['Num.Person.Per.HouseHolds']]
numHousehold = numHousehold.sort_values(['Num.Person.Per.HouseHolds'],ascending=[1])
numHousehold.plot(kind  = 'bar')
plt.xlabel("State name", size = 20)
plt.ylabel("Num Person in households", size  = 20)
plt.savefig('output.png')