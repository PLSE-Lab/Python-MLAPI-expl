# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.import numpy as np
import pandas as pd
import matplotlib as pt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
countydata=pd.read_csv('./county_facts.csv')
#replace column values with real values
countydata= countydata[['fips','area_name','state_abbreviation','PST045214','AGE775214','RHI225214','RHI725214','RHI825214','EDU635213','EDU685213','INC110213','PVY020213','POP060210']]
countydata.rename(columns={'PST045214': 'Population', 'AGE775214': 'Age > 65','RHI225214':'Black','RHI725214':'Latino','RHI825214':'White','EDU635213':'HighSchool','EDU685213':'Bachelors','INC110213':'Median Household','PVY020213':'< Powerty level','POP060210':'Population PSM'}, inplace=True)
countydata.head()
results=pd.read_csv('./primary_results.csv')
results = results[results.party == "Democrat"]
results = results[(results.state != "Maine") & (results.state != "Massachusetts") & (results.state != "Vermont") & (results.state != "Illinois") ]
results = results[(results.candidate != ' Uncommitted') & (results.candidate != 'No Preference')]
results = results[(results.candidate == "Hillary Clinton") |(results.candidate == "Bernie Sanders") ]
demv=results
demv.head()
votesByState = [[candidate, state, party] for candidate in demv.candidate.unique() for state in demv.state.unique() for party in Dem.party.unique()]
for i in votesByState:
	i.append(demv[(demv.candidate == i[0]) & (demv.state == i[1])].votes.sum())
	i.append(i[3]*1.0/demv[demv.state == i[1]].votes.sum())
vbs = pd.DataFrame(votesByState, columns = ['candidate', 'state', 'party', 'votes','partyFrac'])
print(vbs)
fulldata=pd.merge(vbs,countydata,how='inner',left_on='state',right_on='area_name')
fulldata.drop('state_abbreviation',axis=1, inplace=True)
fulldata.head()
hillary=fulldata[(fulldata.candidate=='Hillary Clinton')]
hillary=hillary.reset_index();
hillary.drop('index',axis=1)
features=['Population', 'Age > 65','Black','Latino','White','HighSchool','Bachelors','Median Household','< Powerty level','Population PSM']
xfeatures=hillary[features]
yprediction=hillary.partyFrac
rf=RandomForestRegressor(n_estimators=1000)
rf.fit(xfeatures,yprediction)
penn=countydata[countydata.area_name=='Pennsylvania']
conn=countydata[countydata.area_name=='Connecticut']
mary=countydata[countydata.area_name=='Maryland']
ny=countydata[countydata.area_name=='New York']
delw=countydata[countydata.area_name=='Delaware']
ohio=countydata[countydata.area_name=='Ohio']
xtest1=penn[features]
ypred1=rf.predict(xtest1)
predict=np.array(ypred1);


xtest1=conn[features]
ypred1=rf.predict(xtest1)
predict=np.append(predict,ypred1[0])

xtest1=mary[features]
ypred1=rf.predict(xtest1)
predict=np.append(predict,ypred1[0])

xtest1=ny[features]
ypred1=rf.predict(xtest1)
predict=np.append(predict,ypred1[0])

xtest1=delw[features]
ypred1=rf.predict(xtest1)
predict=np.append(predict,ypred1[0])

xtest1=ohio[features]
ypred1=rf.predict(xtest1)
predict=np.append(predict,ypred1[0])
#predicted
predicted=pd.Series(predict,index=['Pennsylvania','Connecticut','Maryland','Delaware','New York','Ohio'])
predicted
#actual results from wikipedia
actual1 = np.array([.55,.51,.63,.60,.58,.56])
actual=pd.Series(actual1, index=['Pennsylvania','Connecticut','Maryland','Delaware','New York','Ohio'])
#calculating the error between the real and predicted values
error=np.sqrt(mean_squared_error(predicted,actual))
print('The error between actual and predicted is', error)
print('Actual values are ',actual)