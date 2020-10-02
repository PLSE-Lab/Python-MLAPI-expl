# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from scipy import misc
import os


def normalizeSize(img, size_x, size_y):
    return misc.imresize(img,(size_x, size_y))
    
def cnn_model_fn(features, labels, mode):

  features = tf.convert_to_tensor(features)
  labels = tf.convert_to_tensor(labels)
  input_layer = tf.reshape(features, [-1, 224, 224, 1])

  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],strides=2)

  conv3 = tf.layers.separable_conv2d(
      inputs=pool1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  conv4 = tf.layers.separable_conv2d(
      inputs=conv3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

  conv5 = tf.layers.separable_conv2d(
      inputs=pool2,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  batch1 = tf.layers.batch_normalization(inputs=conv5)

  conv6 = tf.layers.separable_conv2d(
      inputs=batch1,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  batch2 = tf.layers.batch_normalization(inputs=conv6)

  conv7 = tf.layers.separable_conv2d(
      inputs=batch2,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  pool3 = tf.layers.max_pooling2d(inputs=conv7,pool_size=[2,2],strides=2)

  pool_flat = tf.layers.flatten(pool3)
  dense = tf.layers.dense(inputs=pool_flat, units=256, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

tf.logging.set_verbosity(tf.logging.INFO)
filenamesNormal = os.listdir("../input/chest_xray/chest_xray/train/NORMAL")
filenamesPneumonia = os.listdir("../input/chest_xray/chest_xray/train/PNEUMONIA")
filenamesNormalEval = os.listdir("../input/chest_xray/chest_xray/test/NORMAL")
filenamesPneumoniaEval = os.listdir("../input/chest_xray/chest_xray/test/PNEUMONIA")
if '.DS_Store' in filenamesNormal:
    filenamesNormal.remove('.DS_Store')
if '.DS_Store' in filenamesNormalEval:
    filenamesNormalEval.remove('.DS_Store')
if '.DS_Store' in filenamesPneumonia:
    filenamesPneumonia.remove('.DS_Store')
if '.DS_Store' in filenamesPneumoniaEval:
    filenamesPneumoniaEval.remove('.DS_Store')
negativeExamples = []
positiveExamples = []
negativeExamplesEval = []
positiveExamplesEval = []
for i in range(len(filenamesNormal)):
    negativeExamples.append(misc.imread("../input/chest_xray/chest_xray/train/NORMAL/" + filenamesNormal[i],flatten=True))
    negativeExamples[i] = normalizeSize(negativeExamples[i],224,224)
for i in range(len(filenamesPneumonia)):
    positiveExamples.append(misc.imread("../input/chest_xray/chest_xray/train/PNEUMONIA/" + filenamesPneumonia[i],flatten=True))
    positiveExamples[i] = normalizeSize(positiveExamples[i], 224, 224)
for i in range(len(filenamesNormalEval)):
    negativeExamplesEval.append(misc.imread("../input/chest_xray/chest_xray/test/NORMAL/" + filenamesNormalEval[i],flatten=True))
    negativeExamplesEval[i] = normalizeSize(negativeExamplesEval[i], 224, 224)
for i in range(len(filenamesPneumoniaEval)):
    positiveExamplesEval.append(misc.imread("../input/chest_xray/chest_xray/test/PNEUMONIA/" + filenamesPneumoniaEval[i],flatten=True))
    positiveExamplesEval[i] = normalizeSize(positiveExamplesEval[i], 224, 224)

X = negativeExamples + positiveExamples
X_eval = negativeExamplesEval + positiveExamplesEval
y = [0 for i in negativeExamples] + [1 for i in positiveExamples]
y_eval = [0 for i in negativeExamplesEval] + [1 for i in positiveExamplesEval]
X = np.asarray(X,dtype="float32")
X_eval = np.asarray(X_eval,dtype="float32")
y = np.asarray(y)
y_eval = np.asarray(y_eval)


estimator = tf.estimator.Estimator(model_fn=cnn_model_fn)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=X,
    y=y,
    batch_size=60,
    num_epochs=None,
    shuffle=True)
estimator.train(
     input_fn=train_input_fn,
     steps=1000)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=X_eval,
    y=y_eval,
    num_epochs=1,
    shuffle=False)
eval_results = estimator.evaluate(input_fn=eval_input_fn)
print(eval_results)
