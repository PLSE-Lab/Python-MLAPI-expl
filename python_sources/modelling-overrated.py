import pandas as pd
X_submit = pd.read_csv('../input/ArmutSampleSubmission.csv')
X_train = pd.read_csv('../input/armut_challenge_training.csv', parse_dates=['createdate'],index_col=0)
X_train = X_train[X_train.userid.isin(X_submit.userid.unique())].drop(columns=['createdate'])
X_train['count_per_userid'] = X_train.groupby(['userid','serviceid'])['serviceid'].transform('count')
X_train = X_train.sort_values(by=['count_per_userid','userid','serviceid'],ascending=False)
X_train = X_train.groupby(['userid'], as_index=False)['serviceid'].first()
X_train.to_csv('submission.csv', index=False)