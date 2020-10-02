#This script looks at the survival rate for those with the most common names

import numpy as np
import pandas as pd

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
dfTitanic = train

#Extract the first name from passenger name
dfTitanic['FirstName'] = dfTitanic['Name'].str.extract('(Mr\. |Miss\. |Master. |Mrs\.[A-Za-z ]*\()([A-Za-z]*)')[1]

#pull out the passengers that have popular names (> 10 occurances)
dfPassengersWithPopularNames = dfTitanic[dfTitanic['FirstName'].isin(dfTitanic['FirstName'].value_counts()[dfTitanic['FirstName'].value_counts() > 10].index)]


test = dfPassengersWithPopularNames.groupby(['Sex', 'FirstName'])
print(test['Survived'].agg([np.mean, len]))




#calculate the surival rate by popular name
ax = (dfPassengersWithPopularNames.groupby('FirstName').Survived.sum()/dfPassengersWithPopularNames.groupby('FirstName').Survived.count()).order(ascending=False).plot(kind='barh',fontsize=15)

#set y axis label and save to png for display below
fig = ax.get_figure()
fig.savefig('figure.png')

print('Finished printing ... ')



