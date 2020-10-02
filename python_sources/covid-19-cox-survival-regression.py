#!/usr/bin/env python
# coding: utf-8

# # Purpose:
# I wanted take this opportunity and apply one of the most famous survival analysis methods, called Cox Regression on this dataset. I am sure many of you are familiar with this one.
# We are dealing with right censored data, (left censored as well, but we could make the assumption that time 0 is the confirmed date), so survival analysis seemed to me really applicable in this case. We will build and evaluate the predictive power of our model, as well as extract some key results
# 
# I would like to thank all involved into gathering this dataset. Additionally the author of the Lifelines python package, for implementing all sorts of Survival Analysis algorithms. I am not a data scientist myself, but I am actively using these survival techniques in the consumer lending industry.
# 

# In[ ]:


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


# For this exercise, we will work with Patient Level data.

# In[ ]:


data = pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')


# Let's install the lifelines library

# In[ ]:


get_ipython().system(' pip install lifelines')


# In[ ]:


from datetime import datetime
from lifelines import CoxPHFitter
from lifelines.utils.sklearn_adapter import sklearn_adapter
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import re
import math
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
pd.set_option('mode.chained_assignment', None)


# # Building the dataset

# In[ ]:


df = data
df['start_date'] = df.confirmed_date
df['start_date'] = pd.to_datetime(df['start_date'])
df['deceased_date'] = pd.to_datetime(df['deceased_date'])
df['released_date'] = pd.to_datetime(df['released_date'])
# The assumption here is that if the patient is not deceased or recovered, he is censored, meaning we do not know the outcome yet.
df['survived_date'] = df.deceased_date.fillna(df.released_date).fillna(datetime.today())

df = df.drop(df[(df.state == 'deceased') & (df.deceased_date.isna())].index)
df = df.drop(df[(df.state == 'released') & (df.released_date.isna())].index)

df['died'] = df.deceased_date.notnull().astype(int)
df['recovered'] = df.released_date.notnull().astype(int)
df['survived'] = (df.survived_date - df.start_date).dt.days

df['is_male'] = (df.sex == 'male').astype(int)
df['age_temp_a'] = df.age.apply(lambda x: int(re.findall("\d+", str(x))[0]) / 10 if pd.notnull(x) else None)
df['age_temp_b'] = df.birth_year.apply(lambda x: np.floor((2020 - x) / 10) if x != None else None)
df['age_decade'] = df.age_temp_a.fillna(df.age_temp_b)
df = df.drop(df[df.age_decade.isna()].index)
df = df[['patient_id', 'age_decade', 'is_male', 'died', 'recovered', 'survived']]
df = df.drop(df[df.isnull().any(axis = 1)].index)
df = df.drop(df[df.patient_id.duplicated()].index)
df.head()


# # Survival Analysis - Cox Regression
# I would not like to bore you with the details of survival analysis. These concepts are much better explained in the original [lifelines](https://lifelines.readthedocs.io/en/latest/) documentation. The only thing that I want to describe is how I handle the events:
# 
# * died : If died = 1 the patient is deceased
# * recovered : If recovered = 1 the patient has recovered
# * survived: Number of days that the patient has "survived". This is not to be confused with the recovery of a patient. It rather means how many days a patient survived while being observed. Survival in this case means not death and not recovery. Consequently, if the patient is not deceased or has he not recovered, the patient is censored, meaning we do not know the outcome yet.
# 
# At this point I would like to state that we are working within a "competing risks" setting. In effect, recovery from the virus is a competing risk of death and vice versa. Let's move on and split our dataset into train and test sets

# In[ ]:


train, test = train_test_split(df, test_size = 0.3)
X_train_death = train[['died', 'age_decade', 'is_male']]
y_train_death = train['survived']
X_test_death = test[['died', 'age_decade', 'is_male']]
y_test_death = test['survived']

X_train_recovery = train[['recovered', 'age_decade', 'is_male']]
y_train_recovery = train['survived']
X_test_recovery = test[['recovered', 'age_decade', 'is_male']]
y_test_recovery = test['survived']
print('train size : {}, test_size : {}'.format(train.shape[0], test.shape[0]))


# In[ ]:


def cox_ph_fitter(X_train, y_train, X_test, y_test, event):
    CoxRegression = sklearn_adapter(CoxPHFitter, event_col = event)
    cph = CoxRegression()
    cph.fit(X_train, y_train)
    cph.lifelines_model.print_summary()
    print('---')
    print('Test Score = {}'.format(cph.score(X_test, y_test)))
    return cph.lifelines_model


# # Hazard and Baseline Hazard
# 
# In Survival Analysis, Hazard is a fucntion of time 
# 
# $h(t) =  \lim_{\delta t \rightarrow 0 } \; \frac{Pr( t \le T \le t + \delta t  |  T > t)}{\delta t}$
# 
# and in effect it models the probability of experiencing an event *T* given that the subject has survived more than *t*. In Cox Regression particularly, the hazard of a subject is described by the following function
# 
# $\underbrace{h(t | x)}_{\text{hazard}} = \overbrace{b_0(t)}^{\text{baseline hazard}} \underbrace{\exp \overbrace{\left(\sum_{i=1}^n b_i (x_i - \overline{x_i})\right)}^{\text{log-partial hazard}}}_ {\text{partial hazard}}$
# 
# where $b_0(t)$ is the baseline hazard and which is scaled up or down based on some regressors X
# 
# In this exercise I am using the hazard concept as the conditional probability of experiencing an event (death or recovery) given the patient's survival until a specific day. The features I am using for predicting the hazard are the age decade of the patient and the gender.
# 
# But since we are operating in a competing risks setting, we will have to fit 2 different models. The first one outputs the baseline hazard for the death event, while the second one will output the baseline hazard of the recovery event. An important distinction that we need to make is that we censor competing events at the time of the occurence. Let's say for example that we are modeling the event of death. A patient who recovered 14 days after he was diagnosed, will be censored at 14 days of survival (the "survived" column in the dataset will have the value 14). The opposite will happen for a patient who died, when we will be modeling the event of recovery.
# This approach is the "cause specific" hazard approach, in contrast with "subdistribution" hazard approach, where in the latter, subjects that experience a competing risk remain in the risk set. [An interesting read](https://statisticalhorizons.com/for-causal-analysis-of-competing-risks) for someone who wants to become more familiar with the topic

# In[ ]:


cph_death = cox_ph_fitter(X_train_death, y_train_death, X_test_death, y_test_death,  'died')


# Let's speak about the summary of the model after fitting is completed. Our concordance index is ~0.9, meaning that the feautres selected have good ranking power. If we revisit the Hazard function in Cox Regression, we will see that both coefficients of our futures are positive. In effect, this means that if we increase the age decade we increase the hazard of dying. Similarly men have a higher risk of dying than women.

# In[ ]:


cph_recovery = cox_ph_fitter(X_train_recovery, y_train_recovery, X_test_recovery, y_test_recovery, 'recovered')


# Above we modeled the recovery event. Obvously, the younger you are, the higher the "hazard" of being recovered. Similarly, women have higher chances of being recovered than men.

# In[ ]:


base_case_death = pd.Series({'age_decade' : 0, 'is_male' : 0})
cummulative_baseline_hazard_death = cph_death.predict_cumulative_hazard(base_case_death)
baseline_hazard_death = cummulative_baseline_hazard_death.diff().fillna(cummulative_baseline_hazard_death)

base_case_recovery = pd.Series({'age_decade' : 0, 'is_male' : 0})
cummulative_baseline_hazard_recovery = cph_recovery.predict_cumulative_hazard(base_case_recovery)
baseline_hazard_recovery = cummulative_baseline_hazard_recovery.diff().fillna(cummulative_baseline_hazard_recovery)

baseline_hazards = pd.merge(baseline_hazard_death[0:], baseline_hazard_recovery[0:], left_index=True, right_index=True)
baseline_hazards = baseline_hazards.rename(columns = {'0_x' : 'baseline_hazard_death', '0_y' : 'baseline_hazard_recovery'})
baseline_hazards.head()


# Because we will manually predict the hazard of each patient and because lifelines outputs the baseline hazard of a patient with average features ( see [this issue](https://github.com/CamDavidsonPilon/lifelines/issues/543) raised) we will costruct the baseline hazards ourseves, by encoding a base case where all X's are set to 0

# In[ ]:


test['key'] = 1
baseline_hazards['key'] = 1

#cross join of patient in the test set with the baseline hazards
test_lines = pd.merge(test, baseline_hazards[:30], on = 'key')

#simple row number, indicating the number of days since confirmed.
test_lines['days'] = test_lines.groupby('patient_id').cumcount()

#according to the definition of the hazard function in cox regression, we calculate the hazard for each patient individually
#in effect this is the conditional probability of death and recovery for each patient, given the patient has survived so far
test_lines['p_death_given_s'] = test_lines.baseline_hazard_death * np.exp(np.dot(test_lines[['age_decade', 'is_male']], cph_death.params_))
test_lines['p_recovery_given_s'] = test_lines.baseline_hazard_recovery * np.exp(np.dot(test_lines[['age_decade', 'is_male']], cph_recovery.params_))

test_lines['survive_death'] = 1-test_lines.p_death_given_s
test_lines['survive_recovery'] = 1-test_lines.p_recovery_given_s

#probability of survival defined as if the patient did not die and did not recover so far 
test_lines['p_survive'] = test_lines.groupby('patient_id').survive_death.cumprod() * test_lines.groupby('patient_id').survive_recovery.cumprod()

#unconditional probability of death and recovery. 
test_lines['p_death'] = test_lines.p_death_given_s * test_lines.groupby('patient_id').p_survive.shift().fillna(1)
test_lines['p_recovery'] = test_lines.p_recovery_given_s * test_lines.groupby('patient_id').p_survive.shift().fillna(1)

#observed deaths and recoveries
test_lines['recovered'] = ((test_lines.survived == test_lines.days) & (test_lines.recovered == 1)).astype(int)
test_lines['died'] = ((test_lines.survived == test_lines.days) & (test_lines.died == 1)).astype(int)


# Let's aggregate and visualize the results

# In[ ]:


agg = test_lines.groupby('days').agg({
    'died' : np.sum
    ,'recovered' : np.sum
    ,'p_death' : np.sum
    ,'p_recovery' : np.sum
    ,'patient_id' : pd.Series.nunique
}).reset_index()
agg['death_rate'] = agg.died.cumsum() / agg.patient_id
agg['death_rate_pred'] = agg.p_death.cumsum() / agg.patient_id
agg['recovery_rate'] = agg.recovered.cumsum() / agg.patient_id
agg['recovery_rate_pred'] = agg.p_recovery.cumsum() / agg.patient_id


# In[ ]:


sns.lineplot(agg.days, agg.death_rate_pred, label = 'predictions')
sns.lineplot(agg.days, agg.death_rate, label = 'actuals')


# As you can see, the model seems to predict pretty well. Maybe the specific random fold of the data into train and test is such, that favors prediction. What will follow is a "manual" cross validation, where we will iterate with different folds and average the erros.
# But before going into that let's observe the produced chart. According to the model, if 100 new patients enter the hospital, by 10 days ~1.5% of the patients will be deceased and by 30 days ~2%

# In[ ]:


sns.lineplot(agg.days, agg.recovery_rate_pred, label = 'predictions')
sns.lineplot(agg.days, agg.recovery_rate, label = 'actuals')


# Similarly, the model predicts 40% recovery rate by the 30th after being diagnosed.

# # Cross Validation
# Since there is not a big amount of data so far, the split into train and test could make the results fluctuate a lot. For that reason I proceeded in coding a "manual" cross validation and I will take average error of each iteration as a final metric for the model

# In[ ]:


def cox_ph_fitter(X_train, y_train, X_test, y_test, event):
    CoxRegression = sklearn_adapter(CoxPHFitter, event_col = event)
    cph = CoxRegression()
    cph.fit(X_train, y_train)
    #cph.lifelines_model.print_summary()
    #print('---')
    #print('Test Score = {}'.format(cph.score(X_test, y_test)))
    return cph.lifelines_model

aggregations = pd.DataFrame(columns = ['iteration','days', 'death_rate', 'death_rate_pred', 'recovery_rate', 'recovery_rate_pred'])

for iteration in range (1,21):
    train, test = train_test_split(df, test_size = 0.3)
    X_train_death = train[['died', 'age_decade', 'is_male']]
    y_train_death = train['survived']
    X_test_death = test[['died', 'age_decade', 'is_male']]
    y_test_death = test['survived']

    X_train_recovery = train[['recovered', 'age_decade', 'is_male']]
    y_train_recovery = train['survived']
    X_test_recovery = test[['recovered', 'age_decade', 'is_male']]
    y_test_recovery = test['survived']

    cph_death = cox_ph_fitter(X_train_death, y_train_death, X_test_death, y_test_death,  'died')
    cph_recovery = cox_ph_fitter(X_train_recovery, y_train_recovery, X_test_recovery, y_test_recovery, 'recovered')

    base_case_death = pd.Series({'age_decade' : 0, 'is_male' : 0})
    cummulative_baseline_hazard_death = cph_death.predict_cumulative_hazard(base_case_death)
    baseline_hazard_death = cummulative_baseline_hazard_death.diff().fillna(cummulative_baseline_hazard_death)

    base_case_recovery = pd.Series({'age_decade' : 0, 'is_male' : 0})
    cummulative_baseline_hazard_recovery = cph_recovery.predict_cumulative_hazard(base_case_recovery)
    baseline_hazard_recovery = cummulative_baseline_hazard_recovery.diff().fillna(cummulative_baseline_hazard_recovery)

    baseline_hazards = pd.merge(baseline_hazard_death[0:], baseline_hazard_recovery[0:], left_index=True, right_index=True)
    baseline_hazards = baseline_hazards.rename(columns = {'0_x' : 'baseline_hazard_death', '0_y' : 'baseline_hazard_recovery'})

    test['key'] = 1
    baseline_hazards['key'] = 1

    #cross join of patient in the test set with the baseline hazards
    test_lines = pd.merge(test, baseline_hazards[:30], on = 'key')

    #simple row number, indicating the number of days since confirmed.
    test_lines['days'] = test_lines.groupby('patient_id').cumcount()

    #according to the definition of the hazard function in cox regression, we calculate the hazard for each patient individually
    #in effect this is the conditional probability of death and recovery for each patient, given the patient has survived so far
    test_lines['p_death_given_s'] = test_lines.baseline_hazard_death * np.exp(np.dot(test_lines[['age_decade', 'is_male']], cph_death.params_))
    test_lines['p_recovery_given_s'] = test_lines.baseline_hazard_recovery * np.exp(np.dot(test_lines[['age_decade', 'is_male']], cph_recovery.params_))

    test_lines['survive_death'] = 1-test_lines.p_death_given_s
    test_lines['survive_recovery'] = 1-test_lines.p_recovery_given_s

    #probability of survival defined as if the patient did not die and did not recover so far 
    test_lines['p_survive'] = test_lines.groupby('patient_id').survive_death.cumprod() * test_lines.groupby('patient_id').survive_recovery.cumprod()

    #unconditional probability of death and recovery. 
    test_lines['p_death'] = test_lines.p_death_given_s * test_lines.groupby('patient_id').p_survive.shift().fillna(1)
    test_lines['p_recovery'] = test_lines.p_recovery_given_s * test_lines.groupby('patient_id').p_survive.shift().fillna(1)

    #observed deaths and recoveries
    test_lines['recovered'] = ((test_lines.survived == test_lines.days) & (test_lines.recovered == 1)).astype(int)
    test_lines['died'] = ((test_lines.survived == test_lines.days) & (test_lines.died == 1)).astype(int)

    agg = test_lines.groupby('days').agg({
        'died' : np.sum
        ,'recovered' : np.sum
        ,'p_death' : np.sum
        ,'p_recovery' : np.sum
        ,'patient_id' : pd.Series.nunique
    }).reset_index()
    agg['death_rate'] = agg.died.cumsum() / agg.patient_id
    agg['death_rate_pred'] = agg.p_death.cumsum() / agg.patient_id
    agg['recovery_rate'] = agg.recovered.cumsum() / agg.patient_id
    agg['recovery_rate_pred'] = agg.p_recovery.cumsum() / agg.patient_id
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    sns.lineplot(agg.days, agg.death_rate_pred, label = 'predictions', ax = ax1)
    sns.lineplot(agg.days, agg.death_rate, label = 'actuals', ax = ax1)

    sns.lineplot(agg.days, agg.recovery_rate_pred, label = 'predictions', ax = ax2)
    sns.lineplot(agg.days, agg.recovery_rate, label = 'actuals', ax = ax2)
    fig.suptitle('Iteration : {}'.format(iteration) )
    
    agg['iteration'] = iteration
    agg_to_append = agg[['iteration', 'days', 'death_rate', 'death_rate_pred', 'recovery_rate', 'recovery_rate_pred']]
    aggregations = aggregations.append(agg_to_append, ignore_index = True)
    


# Let's compute our errors and visualize them

# In[ ]:


aggregations['squared_death_diff'] = (aggregations.death_rate - aggregations.death_rate_pred) ** 2
aggregations['squared_recovery_diff'] = (aggregations.recovery_rate - aggregations.recovery_rate_pred) ** 2
errors = aggregations.groupby('days').agg({
    'squared_death_diff' : np.sum
    ,'squared_recovery_diff' : np.sum
    ,'iteration' : np.size
}).reset_index()
errors['RMSD_death_rate'] = np.sqrt(errors.squared_death_diff / errors.iteration)
errors['RMSD_recovery_rate'] = np.sqrt(errors.squared_recovery_diff / errors.iteration)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
sns.lineplot(errors.days, errors.RMSD_death_rate, ax = ax1)
sns.lineplot(errors.days, errors.RMSD_recovery_rate, ax = ax2)


# Our Root Mean Square Deviation when predicting death rates at 30 days is 0.6%. We need to be cautious about this number, since it depends on the magnitude of the number of deceased patients. It is not with certainty that if the death rate was higher, the RMSD of the death rate would still be 0.6%. Similarly our RMSD for the recovery rate is 10% at 30 days. For sure, the further we look into the future, the less accurate our results become.

# # Thank you
# Once more, I would like to thank all involved into gathering this dataset. The descipline required for this effort really amazes me and I could not feel anything less than gratitude. You really enabled lots of researchers to proceed in complex pieces of analysis. I wish all of you to keep on "surviving" and even better in our context "recovering"!
# 
# Ilias Katsabalos
# 
# Strategic Analyst

# In[ ]:




