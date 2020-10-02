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

# This is to read csv file

df=pd.read_csv("../input/all.csv",sep=",")

states=df.groupby('State').sum()
states=states[['Drinking.water.facilities','Safe.Drinking.water']]
states=states.sort_values(['Drinking.water.facilities','Safe.Drinking.water'],ascending=[0,0])
print(states)
plt.figure(figsize = (2,5))
states.plot(kind='bar' , stacked =True)
plt.xlabel("States", size = 20)
plt.ylabel("Water", size  = 20)
plt.show()
plt.savefig('plot1.png', format='png')