# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from __future__ import division, print_function, unicode_literals
import numpy as np
import scipy.io 
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from functools import partial
from scipy import signal
from datetime import datetime
import os
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE, BorderlineSMOTE,  SVMSMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler, NearMiss, CondensedNearestNeighbour, EditedNearestNeighbours, TomekLinks, NeighbourhoodCleaningRule




def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

os.chdir('../input')
data= scipy.io.loadmat('../input/Tex.mat')

reset_graph()

    
features=data['Feat_All']
#XX=features[:,0:149];
labels=data['Lbls']
n_outputs = 2
#features=np.transpose(features) 
labels=np.reshape(labels,(labels.shape[0],))

Xtrain, Xtest, ytrain, ytest = train_test_split(features,labels,test_size=0.2)
yy=ytrain
XX=Xtrain

learning_rate = 0.00001
n_epochs = 1000
n_inputs=XX.shape[1]  # No of features
scale = 0.001  

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
# training_ph = tf.placeholder_with_default(False, shape=(), name='training')


he_init = tf.variance_scaling_initializer()
my_dense_layer = partial(
    tf.layers.dense, activation=tf.nn.relu,kernel_initializer=he_init,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale))

with tf.name_scope("dnn"):
    hidden1 = my_dense_layer(X, 100, name="hidden1")
    hidden2 = my_dense_layer(hidden1, 100, name="hidden2")
    hidden3 = my_dense_layer(hidden2, 100, name="hidden3")
    hidden4 = my_dense_layer(hidden3, 100, name="hidden4")
    hidden5 = my_dense_layer(hidden4, 100, name="hidden5")
    #hidden6 = my_dense_layer(hidden5, 100, name="hidden6")
    #hidden7 = my_dense_layer(hidden6, 100, name="hidden7")
   # hidden8= my_dense_layer(hidden7, 100, name="hidden8")
   # hidden9 = my_dense_layer(hidden8, 100, name="hidden9")
    #hidden10 = my_dense_layer(hidden9, 100, name="hidden10")
    prediction = my_dense_layer(hidden5, n_outputs, name="outputs",activation="sigmoid")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=prediction)
    baseloss = tf.reduce_mean(xentropy, name="baseloss")
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([baseloss] + reg_losses, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(prediction, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    try:
        saver.restore(sess,"./model.ckpt")
    except:
        init.run()
    print("start training")
    cvB = StratifiedKFold(n_splits=2)
    Melanoma_Data=XX[yy==1];
    Melanoma_lbl=yy[yy==1]
    Bengin_Data=XX[yy==0];
    Bengin_lbl=yy[yy==0]
    del features
    del labels
    # This for loop to split the train data to avoid unbalnce in the train data
    for trainB, testB in cvB.split(Bengin_Data, Bengin_lbl):
        BX_train=Bengin_Data[trainB]
        By_train=Bengin_lbl[trainB]
        BX_test=Bengin_Data[testB]
        By_test=Bengin_lbl[testB]
        features=np.concatenate([BX_test,Melanoma_Data], axis=0)
        labels=np.concatenate([By_test,Melanoma_lbl], axis=0)
        for epoch in range(n_epochs):
            X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=0.2)
            sess.run(training_op, feed_dict={X: X_train, y: y_train})
            acc_train =  accuracy.eval(feed_dict={X: X_train, y: y_train})
            acc_test  =  accuracy.eval(feed_dict={X: X_test, y: y_test})
            print(" in ",epoch, "Train accuracy:","%.2f"%acc_train,"test accuracy:","%.2f"%acc_test)
        save_path = saver.save(sess, "./model.ckpt")
    sess.run(training_op, feed_dict={X: Xtest, y: ytest})
    Acc_tst =  accuracy.eval(feed_dict={X: Xtest, y: ytest})
    pred_test  =  prediction.eval(feed_dict={X: Xtest, y: ytest})
    print(pred_test.shape)
    print(pred_test)
    print("Testing accuracy:","%.2f"%Acc_tst)
    
        
        
        
