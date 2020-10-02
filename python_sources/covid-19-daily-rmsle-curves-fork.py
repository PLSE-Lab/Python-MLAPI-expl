#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
pd.set_option('display.max_columns', 99)
pd.set_option('display.max_rows', 99)
import os
import numpy as np
from tqdm import tqdm
import datetime as dt


# In[ ]:


# NQ add
from collections import Counter


# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
sns.set_palette(sns.color_palette('tab20', 20))

import plotly.express as px
import plotly.graph_objects as go


# In[ ]:


def to_log(x):
    return np.log(x + 1)


def to_exp(x):
    return np.exp(x) - 1

start = dt.datetime.now()

lb_periods = {
    1: ('2020-03-26', '2020-04-23'),
    2: ('2020-04-02', '2020-04-30'),
    3: ('2020-04-09', '2020-05-07'),
    4: ('2020-04-16', '2020-05-14')
}


# In[ ]:


# NQ add
TOPN = 0
RANK = 10; pwr = 0.5


# In[ ]:


def get_competition_data(week, top = -1):
    train = pd.read_csv(f'../input/covid19-global-forecasting-week-{week}/train.csv')
    test = pd.read_csv(f'../input/covid19-global-forecasting-week-{week}/test.csv')
    
    if 'Province/State' in test.columns:
        test = test.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region'})
        train = train.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region'})

    # NQ add
    if top > 0:
        train = train.sort_values(['Date', 'Fatalities'], ascending=False).groupby('Date').head(top)
#         print(Counter(train.Country_Region))

    train['Location'] = train['Country_Region'] + '-' + train['Province_State'].fillna('')
    test['Location'] = test['Country_Region'] + '-' + test['Province_State'].fillna('')
    train = train[['Date', 'Location', 'ConfirmedCases', 'Fatalities']]
    
#     print(train[train.Date == train.Date.max()])
    
    return train, test


# In[ ]:


def get_submissions(week, raw = False):
    submission_path = f'../input/covid19-global-forecasting-submissions/week_{week}'
    submission_files = os.listdir(submission_path)
    submissions_list = []

    for f in tqdm(submission_files):
        submission = pd.read_csv(os.path.join(submission_path, f))
        submission.insert(0, 'SubmissionId', int(f[:-4]))
        submissions_list.append(submission)

    submissions = pd.concat(submissions_list, ignore_index=True, sort=False)
    
    submissions = submissions[['SubmissionId', 'ForecastId', 'ConfirmedCases', 'Fatalities']]
    
    submissions.ConfirmedCases = submissions.ConfirmedCases.clip(0, None)
    submissions.Fatalities = submissions.Fatalities.clip(0, None)
    
    _, test = get_competition_data(week)
    submissions = submissions.merge(test, on='ForecastId', how='left')
    
    submissions = submissions.loc[submissions.Date >= lb_periods[week][0]]
    
    ## **
    
    if raw:
        return submissions
    
    actual, _ = get_competition_data(week=4, top = TOPN)
#     print(actual.Date.unique())
    submissions = submissions.merge(actual, how='left', on=['Date', 'Location'], suffixes=['', 'Actual'])

    submissions = submissions[(submissions.ConfirmedCasesActual >= 0) & (submissions.FatalitiesActual >= 0)]
    
    
    
    #     print(submissions.Date.unique())
#     print(submissions)
    return submissions


# In[ ]:


actual, _ = get_competition_data(week=4, top = TOPN)

week = 4
submissions = get_submissions(week)


# submissions.head(RANK)
submissions.shape


# In[ ]:


def add_errors(submissions):
    submissions.loc[:,'FatalitiesSLE'] = (to_log(submissions.Fatalities) - to_log(submissions.FatalitiesActual)) ** 2
    submissions.loc[:,'ConfirmedCasesSLE'] = (to_log(submissions.ConfirmedCases) - to_log(submissions.ConfirmedCasesActual)) ** 2
    return submissions

def calculate_lb(submissions):
    lb = submissions[['SubmissionId', 'FatalitiesSLE', 'ConfirmedCasesSLE']].groupby('SubmissionId').mean().reset_index()
    lb.loc[:, 'FatalatiesRMSLE'] = np.sqrt(lb['FatalitiesSLE'])
    lb.loc[:, 'ConfirmedCasesRMSLE'] = np.sqrt(lb['ConfirmedCasesSLE'])
    lb.loc[:, 'RMSLE'] = (lb['FatalatiesRMSLE'] + lb['ConfirmedCasesRMSLE']) / 2.0
    lb = lb.sort_values(by='RMSLE')
    lb['Rank'] = np.arange(len(lb)) + 1
    return lb


# In[ ]:


submissions = add_errors(submissions)

lb = calculate_lb(submissions)
submissions = submissions.merge(lb[['SubmissionId', 'RMSLE', 'Rank']], on='SubmissionId')
# submissions.head(RANK)
# lb.head(RANK)


# In[ ]:


def get_ensemble(submissions, k=10):
    submissions['LogCC'] = to_log(submissions.ConfirmedCases)
    submissions['LogF'] = to_log(submissions.Fatalities)

    ensemble = submissions[submissions.Rank <= k].groupby(['Date', 'Location'])[['LogCC', 'LogF']].mean()
    ensemble['ConfirmedCases'] = to_exp(ensemble.LogCC)
    ensemble['Fatalities'] = to_exp(ensemble.LogF)
    ensemble = ensemble.reset_index()

    ensemble = ensemble.merge(actual, how='left', on=['Date', 'Location'], suffixes=['', 'Actual'])
    ensemble = add_errors(ensemble)
    return ensemble


# In[ ]:


def calculate_lb_and_ensemble(week, ranks=10):
    submissions = get_submissions(week)
#     print(submissions)
    submissions = add_errors(submissions)

    lb = calculate_lb(submissions)
    submissions = submissions.merge(lb[['SubmissionId', 'RMSLE', 'Rank']], on='SubmissionId')

    ens = get_ensemble(submissions, k=ranks)
    np.sqrt((ens.FatalitiesSLE.mean() + ens.ConfirmedCasesSLE.mean() ) / 2.0)

    daily_error = submissions[submissions.Rank <= ranks].groupby(['SubmissionId', 'Date']).mean().reset_index()
    daily_error['Daily RMSLE'] = np.sqrt(0.5 * daily_error.FatalitiesSLE + 0.5 * daily_error.ConfirmedCasesSLE)
    daily_error['LB Score'] = daily_error.RMSLE.round(5).astype(str)
    daily_error['Label'] = "#" + daily_error['Rank'].astype(int).map(str) + ": "                             + daily_error.SubmissionId.map(str)#                         + " @ "  + daily_error['RMSLE'].round(3).map(str)
    daily_error = daily_error.sort_values(['Rank', 'Date'], ascending=True)
    fig = px.line(daily_error, x='Date', y='Daily RMSLE', color='Label')
    fig.layout.showlegend = True
    _ = fig.update_layout(
        title_text=f'COVID-19 Daily Prediction Error (Week {week})'
    )

    return submissions, lb, ens, daily_error, fig
    


# In[ ]:


def get_lowvar(week, lb):
    finals = get_submissions(week, raw=True)
    finals = finals[finals.Date == finals.Date.max()]
    top_finals = finals.merge(lb[lb.Rank <= R][['SubmissionId', 'RMSLE', 'Rank']], on='SubmissionId')
    top_finals.ConfirmedCases = np.log(top_finals.ConfirmedCases+1)
    top_finals.Fatalities = np.log(top_finals.Fatalities+1)
    median_final = top_finals.groupby('Location').mean()
    top_finals = top_finals.merge(median_final[['ConfirmedCases', 'Fatalities']], on='Location', 
                                      suffixes = ('', '_median'))

    top_finals['CC_RMSLE'] = (top_finals.ConfirmedCases - top_finals.ConfirmedCases_median) ** 2
    top_finals['F_RMSLE'] = (top_finals.Fatalities - top_finals.Fatalities_median) ** 2
    top_finals['RMSLE'] = (top_finals['F_RMSLE'] + top_finals['CC_RMSLE'] ) / 2
    
    return top_finals


# In[ ]:


def clean_lb(lb, R):
    return lb.drop(columns = [c for c in lb.columns if 'sSLE' in c]).head(R).round(4)            .set_index('SubmissionId', drop=True)            [['Rank', 
                    'RMSLE', 'FatalatiesRMSLE', 'ConfirmedCasesRMSLE' ]]


# # Week 1 Top Submissions

# In[ ]:


week = 1
R = int(RANK * np.power(week, pwr))
submissions, lb, ens1, daily_error, fig = calculate_lb_and_ensemble(week, R)
clean_lb(lb, R)


# In[ ]:


fig.show()
# print(daily_error.Rank.unique())


# In[ ]:


top_finals = get_lowvar(1, lb)
print('Low Variance Models')
top_finals.groupby('SubmissionId').mean().sort_values('RMSLE')        [['RMSLE','Rank', 'CC_RMSLE', 'F_RMSLE']].head(10)


# In[ ]:





# # Week 2 Top Submissions

# In[ ]:


week = 2
R = int(RANK * np.power(week,pwr))

submissions, lb, ens2, daily_error, fig = calculate_lb_and_ensemble(week, R)
clean_lb(lb, R)


# In[ ]:


fig.show()


# In[ ]:


top_finals = get_lowvar(2, lb)
print('Low Variance Models')
top_finals.groupby('SubmissionId').mean().sort_values('RMSLE')        [['RMSLE','Rank', 'CC_RMSLE', 'F_RMSLE']].head(10)


# # Week 3 Top Submissions

# In[ ]:


week = 3
R = int(RANK * np.power(week,pwr))
submissions, lb, ens3, daily_error, fig = calculate_lb_and_ensemble(week, R)
clean_lb(lb, R)


# In[ ]:


fig.show()


# In[ ]:


top_finals = get_lowvar(3, lb)
print('Low Variance Models')
top_finals.groupby('SubmissionId').mean().sort_values('RMSLE')        [['RMSLE','Rank', 'CC_RMSLE', 'F_RMSLE']].head(10)


# # Week 4 Top Submissions

# In[ ]:


week = 4
R = int(RANK * np.power(week,pwr))
submissions, lb, ens4, daily_error, fig = calculate_lb_and_ensemble(week, R)
clean_lb(lb, R)


# In[ ]:


fig.show()


# In[ ]:





# In[ ]:


top_finals = get_lowvar(4, lb)
print('Low Variance Models')
top_finals.groupby('SubmissionId').mean().sort_values('RMSLE')        [['RMSLE','Rank', 'CC_RMSLE', 'F_RMSLE']].head(10)


# In[ ]:





# # Ensemble Performance

# In[ ]:


ens1['Week'] = 1
ens2['Week'] = 2
ens3['Week'] = 3
ens4['Week'] = 4

ens1['Days'] = (pd.to_datetime(ens1.Date) - pd.to_datetime(ens1.Date).min()).dt.days
ens2['Days'] = (pd.to_datetime(ens2.Date) - pd.to_datetime(ens2.Date).min()).dt.days
ens3['Days'] = (pd.to_datetime(ens3.Date) - pd.to_datetime(ens3.Date).min()).dt.days
ens4['Days'] = (pd.to_datetime(ens4.Date) - pd.to_datetime(ens4.Date).min()).dt.days
ensembles = pd.concat([ens1, ens2, ens3, ens4])

daily_error = ensembles.groupby(['Week', 'Date']).mean().reset_index()
daily_error['Daily RMSLE'] = np.sqrt(0.5 * daily_error.FatalitiesSLE + 0.5 * daily_error.ConfirmedCasesSLE)

fig = px.line(daily_error, x='Date', y='Daily RMSLE', color='Week')
_ = fig.update_layout(
    title_text=f'COVID-19 Ensemble Daily Prediction Error'
)
fig.show()


# In[ ]:


fig = px.line(daily_error, x='Days', y='Daily RMSLE', color='Week')
_ = fig.update_layout(
    title_text=f'COVID-19 Ensemble Daily Prediction Error'
)
fig.show()


# In[ ]:


NDP = 10


# In[ ]:


'Difficult Locations: Fatalities'
ensembles[ensembles.Week >= 3]    .groupby(['Week', 'Location']).mean().reset_index().sort_values(by='FatalitiesSLE', ascending=False)                 .head(NDP).drop(columns=['LogCC', 'LogF']).round(1)
'Easiest Locations: Fatalities'
ensembles[ensembles.Week >= 3]    .groupby(['Week', 'Location']).mean().reset_index().sort_values(by='FatalitiesSLE', ascending=False)            .dropna().tail(NDP).drop(columns=['LogCC', 'LogF']).round(1)


# In[ ]:


'Difficult Locations: Confirmed Cases'
ensembles[ensembles.Week >= 3]    .groupby(['Week', 'Location']).mean().reset_index().sort_values(by='ConfirmedCasesSLE', ascending=False)        .head(NDP).drop(columns=['LogCC', 'LogF']).round(1)
'Easiest Locations: Confirmed Cases'
ensembles[ensembles.Week >= 3]    .groupby(['Week', 'Location']).mean().reset_index().sort_values(by='ConfirmedCasesSLE', ascending=False)            .dropna().tail(NDP).drop(columns=['LogCC', 'LogF']).round(1)


# In[ ]:


end = dt.datetime.now()
print('Finished', end, (end - start).seconds, 's')

