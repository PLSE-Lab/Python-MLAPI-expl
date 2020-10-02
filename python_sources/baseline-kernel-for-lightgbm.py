import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

path = '../'

# True for validation or False to generate predictions
is_valid = False

# validation for the deezer competition
#based in the last song the user listened
def split_validation(df):
	df = df.sort_values(by=["user_id", "ts_listen"])
	df = df.reset_index()
	val_indexes = df.groupby('user_id')['index'].max()
	df_train = df[~df['index'].isin(val_indexes)]
	df_valid = df[df['index'].isin(val_indexes)]
	del df_train['index'], df_valid['index']
	return df_train, df_valid

#parameters for LightGBM 
def train_lgb(dtrain, val_sets, n_round):
	params = {   
		'boosting_type': 'gbdt',
		'objective': 'binary',
		'metric': 'auc',
		'learning_rate': 0.1,
		'num_leaves': 50,
		'min_data_in_leaf': 10, 
		'max_depth': 10,
		'feature_fraction': 1,
		'bagging_freq': 1,
		'bagging_fraction': 1,
		'lambda_l1': 0.1,
		'random_state': 123,
		'verbosity': -1}

	model = lgb.train(params,
					dtrain,
					num_boost_round = n_round,
					valid_sets = val_sets,
					verbose_eval=10,
					early_stopping_rounds = 50)
	return model

train = pd.read_csv(path+'input/train.csv')
test = pd.read_csv(path+'input/test.csv')

if (is_valid == True):
	train, valid = split_validation(train)

	y_train = train['is_listened']
	y_valid = valid['is_listened']
	del train['is_listened'], valid['is_listened']

	d_train = lgb.Dataset(train, y_train)
	d_valid = lgb.Dataset(valid, y_valid)

	model = train_lgb(d_train, val_sets=[d_train, d_valid], n_round=10000)
else:
	y_train = train['is_listened']
	del train['is_listened']

	features = train.columns
	d_train = lgb.Dataset(train, y_train)

	model = train_lgb(d_train, val_sets=[d_train], n_round=20)
	preds = model.predict(test[features])
	sub = pd.DataFrame({'sample_id': test['sample_id'], 'is_listened': preds})
	sub.to_csv('sub_001.csv', index=False)