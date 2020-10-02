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



## load data
df = pd.read_csv('../input/voice.csv')


maleDataNum = df.loc[df.label=='male'].shape
femaleDataNum = df.loc[df.label=='female'].shape
print(maleDataNum, femaleDataNum)


## Prepare data

from sklearn.preprocessing import LabelEncoder

genderEncoder = LabelEncoder()
label = df.loc[:, 'label']
label = genderEncoder.fit_transform(label)

size, = label.shape
Y = np.zeros((size, 2), dtype=int)
Y[:, 0] = label
Y[:, 1] = 1 - label


from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
data = df.iloc[:, :-2]

data = StandardScaler().fit_transform(data)
data = Normalizer().fit_transform(data)

trainX, testX, trainY, testY = train_test_split(data, Y, test_size=0.33)


## Build tf graph of softmax algorithm

_, featureCnt = data.shape

Xp = tf.placeholder(tf.float32, [None, featureCnt])
Yp = tf.placeholder(tf.int32, [None, 2])

W = tf.Variable(tf.zeros([featureCnt, 2]))
b = tf.Variable([0.001, 0.0002])

Y_= tf.matmul(Xp, W) + b
crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_, labels=Yp))

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(crossEntropy)

##accuracy
correntPredict = tf.equal(tf.arg_max(Y_, 1), tf.arg_max(Yp, 1))
accuracy = tf.reduce_mean(tf.cast(correntPredict, tf.float32))


### Run a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(8000):    
    sess.run(optimizer, feed_dict={Xp:trainX, Yp:trainY})
    
    #if i % 4000 == 0:
    #    ce = sess.run(crossEntropy, feed_dict={Xp:trainX, Yp:trainY})
    #    ratio = sess.run(accuracy, feed_dict={Xp:trainX, Yp:trainY})
    #    print({'crossEntropy':ce, 'accuracy': ratio})


testRatio = sess.run(accuracy, feed_dict={Xp:testX, Yp:testY})
print('test accuracy:', testRatio)