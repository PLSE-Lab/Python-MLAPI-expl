# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split

tf.set_random_seed(42)
np.random.seed(42)

COLS = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
FEATURES = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]
LABEL = "Class"

df = pd.read_csv('../input/creditcard.csv', skiprows=1, skip_blank_lines=1, skipinitialspace=1, names=COLS)


train, test = train_test_split(df, test_size=0.002)

print('Number of samples for training: {}' .format(len(train)))
print('Number of samples for testing: {}' .format(len(test)))

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

regressor = tf.estimator.DNNClassifier(feature_columns=feature_cols, 
                                       hidden_units=[10, 10],
                                       model_dir='./model/creditcard', n_classes=2)


def get_input_fn(dataset, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(num_epochs=num_epochs, 
                                               x=pd.DataFrame({k: dataset[k].values for k in FEATURES}),
                                               y = pd.Series(dataset[LABEL].values),
                                               shuffle=shuffle)
    
regressor.train(input_fn=get_input_fn(train), steps=1000)

#evaluate
ev = regressor.evaluate(input_fn=get_input_fn(test, num_epochs=1, shuffle=False))
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))
print ('Accuray achieved: {}'.format(ev['accuracy']))

result = regressor.predict(input_fn=get_input_fn(test, num_epochs=1, shuffle=False))
got = [d['classes'].astype(int)[0] for d in list(result)]
expected = np.array(test['Class'].tolist())

acc = pd.DataFrame(list(zip(expected, got)), columns=['expected', 'got'])
total_right = sum(acc['expected'] == acc['got'])
print('Accuracy: {}'.format(total_right/len(acc.index)))

