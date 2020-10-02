# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import tensorflow as tf
import sklearn.model_selection as ms
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from skimage.exposure import equalize_adapthist

tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()

train_data =pd.read_csv("../input/train.csv")
test_data =pd.read_csv("../input/test.csv")

labels = train_data.label.values
train_data.drop("label", axis=1, inplace=True)
# inverting the images to make the background black
#
features = np.invert(train_data.astype(np.int32))
prd_features = np.invert(test_data.astype(np.int32))
#applying adaptive equalizer
#
#features = equalize_adapthist(features.astype(np.int32)).astype(np.float32)
#prd_features = equalize_adapthist(prd_features.astype(np.int32)).astype(np.float32)

scaler = MinMaxScaler()
features = scaler.fit_transform(features).astype(np.float32)
prd_features = scaler.fit_transform(prd_features).astype(np.float32)

xtrain, xtest, ytrain, ytest = ms.train_test_split(features, labels, test_size=0.25, random_state=42)

#self explainatory, worth to metion that change of dropout trate to 0.9 
#after achieveing 0.98 accuracy
         
def model_fn(mode, features, labels):
    features = tf.reshape(features, [-1, 28, 28, 1])
    
    network = tf.layers.conv2d(inputs=features,
                           filters=32,
                           kernel_size=5,
                           padding='SAME', activation=tf.nn.tanh)
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)
    
    network = tf.layers.conv2d(inputs=network,
                           filters=64,
                           kernel_size=5,
                           padding='SAME', activation=tf.nn.elu)
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)
    
    network_flat = tf.reshape(network, [-1, 7*7*64])
    
    dense_nw = tf.layers.dense(inputs=network_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense_nw, rate=0.75, 
                                        training=(mode==tf.estimator.ModeKeys.TRAIN))
    
    y_logits = tf.layers.dense(inputs=dropout, units=10,
                                  activation=None)
                                  
    train_op=None
    loss = None
    global_step = tf.train.get_global_step()
    eval_metric_ops=None
    
    if (mode == tf.estimator.ModeKeys.EVAL or
             mode == tf.estimator.ModeKeys.TRAIN):
        labels = tf.one_hot(labels, 10)
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(
                    onehot_labels=labels, logits=y_logits)) + tf.losses.get_regularization_loss()
        eval_metric_ops = {"accuracy":tf.metrics.accuracy(tf.argmax(labels,1),
                                    tf.argmax(y_logits,1)),
                                "precision":tf.metrics.precision(tf.argmax(labels,1),
                                    tf.argmax(y_logits,1)),
                                "recall":tf.metrics.recall(tf.argmax(labels,1),
                                    tf.argmax(y_logits,1))}
    
    if (mode == tf.estimator.ModeKeys.TRAIN):
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss,
                                        global_step = global_step)
    
    predictions = {"classes": tf.argmax(
                        input=y_logits, axis=1),
                        "probabilities": tf.nn.softmax(y_logits)}
    
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                        loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)
                        
classifier = tf.estimator.Estimator(model_fn, model_dir='./model_dir')

for i in range(50):

    classifier.train(input_fn=tf.estimator.inputs.numpy_input_fn(xtrain, ytrain,
                batch_size=50, num_epochs=5, shuffle=True), steps= 3000)
    #evaluating test images                
    classifier.evaluate(input_fn = tf.estimator.inputs.numpy_input_fn(
                xtest, ytest, batch_size=50, shuffle=True), steps= 200)
    #evaluating train images                
    classifier.evaluate(input_fn = tf.estimator.inputs.numpy_input_fn(
                xtrain, ytrain, batch_size=50, shuffle=True), steps= 200)
            
pred = classifier.predict(input_fn = tf.estimator.inputs.numpy_input_fn(
                prd_features, batch_size=50, shuffle=False))
predictions=list()
for i, prd in enumerate(pred):
    predictions.append(prd["classes"])
ids= [i+1 for i in range(28000)]
sub_df = pd.DataFrame({"ImageId":ids, "Label":predictions})
sub_df.to_csv("kaggle_kernel_sub.csv", index=False)
