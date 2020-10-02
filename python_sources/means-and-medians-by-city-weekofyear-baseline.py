import numpy as np 
import pandas as pd 

train = pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv')
test = pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv')

train['total_cases'] = pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv', usecols=[3]).values
train.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)

means = train.groupby(['city','weekofyear'])['total_cases'].mean().reset_index()
medians = train.groupby(['city','weekofyear'])['total_cases'].median().reset_index()

test = pd.merge(test, means, how='left', on=['city','weekofyear'])
test = pd.merge(test, medians, how='left', on=['city','weekofyear'])

test['total_cases'] = (test['total_cases_y'] + test['total_cases_x'])/2
test.interpolate(method='linear', limit_direction='forward', axis=0, inplace=True)

test['total_cases'] = np.round(test['total_cases'],0)
test[['city','year','weekofyear','total_cases']].to_csv('submission_baseline.csv', float_format='%.0f', index=False)
