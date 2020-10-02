# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.metrics import r2_score
import tflearn
import tensorflow as tf
import seaborn
import warnings
warnings.filterwarnings('ignore')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')
labels = train['SalePrice']

test = pd.read_csv('../input/test.csv')

data = pd.concat([train,test], ignore_index=True)
data = data.drop('SalePrice',1)
ids = test['Id']

# labels_n1 = labels
# labels_n1 = labels_n1.reshape(-1,1)

tf.reset_default_graph()
r2 = tflearn.R2()
net = tflearn.input_data(shape=[None, train.shape[1]])
net = tflearn.fully_connected(net, 30, activation='linear')
net = tflearn.fully_connected(net, 10, activation = 'linear')
net = tflearn.fully_connected(net, 1, activation = 'linear')
sgd = tflearn.SGD(learning_rate=0.1, lr_decay = 0.01, decay_step=100)
net = tflearn.regression(net, optimizer=sgd, loss='mean_square', metric=r2)
model = tflearn.DNN(net)

model.fit(train, labels, show_metric=True, validation_set=0.2, shuffle = True, n_epoch=50)

preds_DNN = model.predict(test)
preds_DNN = np.exp(preds_DNN)
preds_DNN = preds_DNN.reshape(-1,)

output = pd.DataFrame({"id":ids, "SalePrice": preds_DNN})
output.to_csv('output2.csv', index=False)



