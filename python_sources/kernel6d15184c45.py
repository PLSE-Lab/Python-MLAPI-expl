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

# Any results you write to the current directory are saved as output.


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


# %%
train_data = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv', na_filter=False, parse_dates=[4])
test_data = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv',na_filter=False)


# %%

#countries = train_data["Country_Region"].unique()
states = test_data.drop_duplicates(subset=["Province_State","Country_Region"])#.unique()
#states = test_data.drop_duplicates(subset=["Country_Region"])
#itr_list = pd.concat((train_data["Country_Region"], train_data["Province_State"]), axis=1, join='inner' )#.unique()
#print(states[['Province_State','Country_Region']])
count=0

tuned_parameters = {'max_depth': [4,8,12], 'learning_rate': [1e-1, 1e-2, 1e-3],

                    }
'''
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
'''
params = {'n_estimators':500, 'loss':'ls', 'min_samples_split':2}

for index,row in states.iterrows():
    dummy = row['Province_State'], row['Country_Region']
    train_x = train_data.loc[ (train_data['Country_Region']==dummy[1]) & (train_data['Province_State']==dummy[0])]['Date']
    test_x = test_data.loc[ (test_data['Country_Region']==dummy[1]) & (test_data['Province_State']==dummy[0])]['Date']
    test_id = test_data.loc[ (test_data['Country_Region']==dummy[1]) & (test_data['Province_State']==dummy[0])]['ForecastId']
    #dummy = row['Country_Region']
    print(dummy)
    #train_x = train_data.loc[ (train_data['Country_Region']==dummy) ]['Date']
    #test_x = test_data.loc[ (test_data['Country_Region']==dummy) ]['Date']
    #test_id = test_data.loc[ (test_data['Country_Region']==dummy) ]['ForecastId']
    le = LabelEncoder()
    le.fit(pd.concat([train_x,test_x], ignore_index=True))
    public_training_dates = train_x.loc[(train_x < '2020-04-01')]
    public_test_dates = test_x.loc[(test_x <'2020-04-16')]
    private_test_dates = test_x.loc[(test_x >= '2020-04-16')]
    le.transform(public_training_dates)

    public_training_fatal = train_data.loc[ (train_data['Country_Region']==dummy[1]) & (train_data['Province_State']==dummy[0]) & (train_data['Date']<'2020-04-01')]["Fatalities"].to_numpy()
    public_training_case = train_data.loc[ (train_data['Country_Region']==dummy[1]) & (train_data['Province_State']==dummy[0]) & (train_data['Date']<'2020-04-01')]["ConfirmedCases"].to_numpy()
    private_training_fatal = train_data.loc[ (train_data['Country_Region']==dummy[1]) & (train_data['Province_State']==dummy[0])]["Fatalities"].to_numpy()
    private_training_case = train_data.loc[ (train_data['Country_Region']==dummy[1]) & (train_data['Province_State']==dummy[0])]["ConfirmedCases"].to_numpy()
    
    #public_training_fatal = train_data.loc[ (train_data['Country_Region']==dummy) & (train_data['Date']<'2020-03-26')]["Fatalities"].to_numpy()
    #public_training_case = train_data.loc[ (train_data['Country_Region']==dummy)  & (train_data['Date']<'2020-03-26')]["ConfirmedCases"].to_numpy()
    #private_training_fatal = train_data.loc[ (train_data['Country_Region']==dummy) ]["Fatalities"].to_numpy()
    #private_training_case = train_data.loc[ (train_data['Country_Region']==dummy)]["ConfirmedCases"].to_numpy()

    '''
    public_split_svr_rbf_fatal = ensemble.GradientBoostingRegressor(**params) #SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1, max_iter=10000)
    public_split_svr_rbf_case = ensemble.GradientBoostingRegressor(**params) #SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1, max_iter=10000)
    private_split_svr_rbf_fatal = ensemble.GradientBoostingRegressor(**params) #SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1, max_iter=10000)
    private_split_svr_rbf_case = ensemble.GradientBoostingRegressor(**params) #SVR(kernel='poly', C=100, gamma=0.1, epsilon=.1, max_iter=10000)
    '''
    public_split_svr_rbf_fatal = GridSearchCV(ensemble.GradientBoostingRegressor(**params), tuned_parameters)
    public_split_svr_rbf_case = GridSearchCV(ensemble.GradientBoostingRegressor(**params), tuned_parameters)
    private_split_svr_rbf_fatal = GridSearchCV(ensemble.GradientBoostingRegressor(**params), tuned_parameters)
    private_split_svr_rbf_case = GridSearchCV(ensemble.GradientBoostingRegressor(**params), tuned_parameters)
    
    
    public_split_svr_rbf_fatal.fit(le.transform(public_training_dates).reshape(-1,1),public_training_fatal)
    public_split_svr_rbf_case.fit(le.transform(public_training_dates).reshape(-1,1),public_training_case)
    private_split_svr_rbf_fatal.fit(le.transform(train_x).reshape(-1,1),private_training_fatal)
    private_split_svr_rbf_case.fit(le.transform(train_x).reshape(-1,1),private_training_case)


    public_predictions_fatal = public_split_svr_rbf_fatal.predict(le.transform(public_test_dates).reshape(-1,1))
    public_predictions_case = public_split_svr_rbf_case.predict(le.transform(public_test_dates).reshape(-1,1))
    private_predictions_fatal = private_split_svr_rbf_fatal.predict(le.transform(private_test_dates).reshape(-1,1))
    private_predictions_case = private_split_svr_rbf_case.predict(le.transform(private_test_dates).reshape(-1,1))


    test_predictions_fatalities = np.concatenate((public_predictions_fatal, private_predictions_fatal))
    test_predictions = np.concatenate((public_predictions_case, private_predictions_case))


    output = pd.DataFrame({'ForecastId': test_id, 'ConfirmedCases': np.ceil(test_predictions), 'Fatalities': np.ceil(test_predictions_fatalities)})
    output.to_csv('submission.csv', index=False, mode='a', header= not(count))
    count+=1