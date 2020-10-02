#This script looks at the survival rate for those with the most common names

import numpy as np
import pandas as pd

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
dfTitanic = train

dfTitanic['FirstName'] = dfTitanic['Name'].str.extract('(Mr\. |Miss\. |Master. |Mrs\.[A-Za-z ]*\()([A-Za-z]*)')[1]

dfPassengersWithPopularNames = dfTitanic[dfTitanic['FirstName'].isin(dfTitanic['FirstName'].value_counts()[dfTitanic['FirstName'].value_counts() > 10].index)]

ax = (dfPassengersWithPopularNames.groupby('FirstName').Survived.sum()/dfPassengersWithPopularNames.groupby('FirstName').Survived.count()).order(ascending=False).plot(kind='bar',y='Survival rate',fontsize=8)

ax.set_ylabel("Survival Rate")
fig = ax.get_figure()
fig.savefig('figure.png')




