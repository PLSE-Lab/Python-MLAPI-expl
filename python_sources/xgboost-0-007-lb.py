import kagglegym
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import math


# Function XGBOOST ########################################################
def xgb_obj_custom_r(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_mean = np.mean(y_true)
    y_median = np.median(y_true)
    c1 = y_true
    #c1 = y_true - y_mean
    #c1 = y_true - y_median
    grad = 2*(y_pred-y_true)/(c1**2)
    hess = 2/(c1**2)
    return grad, hess


def xgb_eval_custom_r(y_pred, dtrain):
    y_true = dtrain.get_label()
    ybar = np.sum(y_true)/len(y_true)
    ssres = np.sum((y_true - y_pred) ** 2)
    sstot = np.sum((y_true - ybar)**2)
    r2 = 1 - ssres/sstot
    error = np.sign(r2) * np.absolute(r2)**0.5
    return 'error', error


# Main #####################################################################
# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

df = observation.train
df = df[~(df['y'] == 0)]

low_y_cut = -0.086093
high_y_cut = 0.093497
df = df[(df['y'] < high_y_cut) & (df['y'] > low_y_cut)]

#cols_to_use = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']
#cols_to_use = ['technical_20','technical_40']
cols_to_use = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19', 'technical_40']

test_size = 0.25
end_train = round(df.shape[0] * test_size)
train = df[0:end_train]
valid = df[end_train:df.shape[0]]

# train, valid = train_test_split(df, test_size=test_size, random_state=2017)

# Convert to XGB format
to_drop = ['timestamp', 'y']
train = train[np.abs(train['y']) < 0.018976588919758796]
train_xgb = xgb.DMatrix(data=train.drop(to_drop, axis=1)[cols_to_use],
                        label=train['y'])

valid_xgb = xgb.DMatrix(data=valid.drop(to_drop, axis=1)[cols_to_use],
                        label=valid['y'])

params = {
    'objective': 'reg:linear'
    ,'eta': 0.1
    ,'max_depth': 3
    , 'subsample': 0.9
    #, 'colsample_bytree': 1
    ,'min_child_weight': 2**11
    #,'gamma': 100
    , 'seed': 10
}

evallist = [(train_xgb, 'train'), (valid_xgb, 'valid')]

model = xgb.train(params.items()
                  , dtrain=train_xgb
                  , num_boost_round=100000
                  , evals=evallist
                  , early_stopping_rounds=20
                  , maximize=True
                  , verbose_eval=10
                  , feval=xgb_eval_custom_r
                  )


while True:
    test = observation.features
    test_xgb = xgb.DMatrix(data=test.drop(['id', 'timestamp'], axis=1)[cols_to_use])
    test_y = model.predict(test_xgb, ntree_limit=model.best_ntree_limit)
    observation.target['y'] = test_y
    #observation.target['y'] = observation.target.apply(get_weighted_y, axis = 1)
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break