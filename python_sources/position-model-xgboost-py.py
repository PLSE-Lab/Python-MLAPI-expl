import numpy as np
import pandas as pd

import xgboost as xgb

data = []
labels = []

#df = pd.read_csv('../input/beacon_readings.csv')
#data = df.iloc[:, 0:3]
#labels = df.iloc[:, 3]

def get_train_data():
    with open("../input/beacon_readings.csv", "r") as fp:
        emp_data = fp.readlines()
        X = []
        Y = []
        cnt = 1
        for line in emp_data:
            line = line.strip()
            arr = line.split(',')
            Y.append( float(arr[3]) )
            X.append([ float(arr[0]), float(arr[1]), float(arr[2]) ])
            cnt += 1
            #X.append(int(arr[0]))
        return X, Y

data, labels = get_train_data()

data = np.array(data)
labels = np.array(labels)

#data = np.random.rand(5,10) # 5 entities, each contains 10 features
#labels = np.random.randint(2, size=5) # binary target

T_train_xgb = xgb.DMatrix(data, label=labels)

#params = {"objective": "reg:linear", "booster":"gblinear"}
params = {'bst:max_depth':6, 'bst:eta':1, 'gamma':15, 'silent':1, 'booster':'gblinear', 'objective':'reg:linear' }
params['nthread'] = 4
params['eval_metric'] = 'mae'
#params['updater'] = 'grow_gpu'

evallist  = [(T_train_xgb,'eval'), (T_train_xgb,'train')]
num_round = 100

gbm = xgb.train(params, T_train_xgb, num_round, evallist)


Y_pred = gbm.predict(xgb.DMatrix(data, labels))

for i in range(len(Y_pred)):
    print ("Predicted: " + str(Y_pred[i]) + " Target: " + str(labels[i]) + " Difference: " + str(Y_pred[i] - labels[i]))