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

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2
from keras.backend.tensorflow_backend import set_session
from keras.utils import np_utils
import keras

import tensorflow as tf
import pandas as pd
import numpy as np

np.random.seed(0)



data = pd.read_csv('../input/creditcard.csv')

# Preprocessing Amount
amt_scale = StandardScaler()
data['NormAmount'] =  amt_scale.fit_transform(data['Amount'].values.reshape(-1, 1))

# Split Train and Test Data
X = data.drop(['Time', 'Amount', 'Class'], axis=1).as_matrix()
Y = data['Class'].as_matrix()

# Standardization
scale_x = StandardScaler()
X = scale_x.fit_transform(X)

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=1)

fraud_test_y = test_y == 1
fraud_test_x = test_x[fraud_test_y]
fraud_test_y = test_y[fraud_test_y]

train_category_y = np_utils.to_categorical(train_y)
test_category_y = np_utils.to_categorical(test_y)