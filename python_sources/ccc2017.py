import numpy as np
import pandas as pd
import tensorflow as tf
import pydicom
import random
import os
from scipy import misc

annots = pd.read_csv('../input/CrowdsCureCancer2017Annotations.csv')
num_records = annots.shape[0]

def extractImageSample(img, x_start, y_start, x_end, y_end):
    pixarr = pydicom.dcmread(img).pixel_array
    return pixarr[x_start:x_end+1,y_start:y_end+1]

def normalizeSize(img, size_x, size_y):
    return misc.imresize(img,(size_x, size_y))

def extractFromRawData(number):
    #Not checked
    extractedData = []
    j = 0
    while j < number:
        i = random.randint(0,num_records-1)
        pth = "../input/annotated_dicoms/{}/{}/{}".format(annots['patientID'][i], annots['seriesUID'][i],
                                                                         annots['sliceIndex'][i])
        start_x = int(annots['start_x'][i])
        end_x = int(annots['end_x'][i])
        start_y = int(annots['start_y'][i])
        end_y = int(annots['end_y'][i])
        if start_x > end_x:
            start_x, end_x = end_x, start_x
        if start_y > end_y:
            start_y, end_y = end_y, start_y
        #Hmm, still the same thing
        if abs(start_x - end_x) < 5 or abs(start_y - end_y) < 5:
            continue
        extractedData.append(extractImageSample(pth, start_x, start_y, end_x, end_y))
        j += 1
    return extractedData

def generateFromRawData(number, use_distribution, mean_x=None, mean_y=None, std_x=None, std_y=None, size_x=50, size_y=50):
    #Not checked
    #Might yeild non-informative examples
    #Might actually randomly extract positive example
    generatedData = []
    for i in range(number):
        rand_slice = random.randint(0, num_records - 1)
        rand_shape = (0,0)
        #Hmm, see the problem?
        if not use_distribution:
            rand_shape = (size_x,size_y)
        else:
            while rand_shape[0] < 5 or rand_shape[1] < 5:
                rand_shape = (int(np.random.normal(mean_x, std_x)), int(np.random.normal(mean_y, std_y)))
        start_x = random.randint(150, 400 - rand_shape[0])
        start_y = random.randint(150, 400 - rand_shape[1])
        end_x = start_x + rand_shape[0]
        end_y = start_y + rand_shape[1]
        pth = "../input/annotated_dicoms/{}/{}/{}".format(annots['patientID'][rand_slice],
                                                                         annots['seriesUID'][rand_slice],
                                                                         annots['sliceIndex'][rand_slice])
        generatedData.append(extractImageSample(pth, start_x, start_y, end_x, end_y))
    return generatedData

def calculateMeanSize(samples):
    #Not checked
    mean_x = np.mean([sample.shape[0] for sample in samples])
    mean_y = np.mean([sample.shape[1] for sample in samples])
    std_x = np.std([sample.shape[0] for sample in samples])
    std_y = np.std([sample.shape[1] for sample in samples])
    return mean_x,mean_y,std_x,std_y

import tensorflow as tf

def cnn_model_fn(features, labels, mode):

  features = tf.convert_to_tensor(features)
  labels = tf.convert_to_tensor(labels)
  input_layer = tf.reshape(features, [-1, 50, 50, 1])

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

    
positiveExamples = extractFromRawData(1000)
mean_x,mean_y,std_x,std_y = calculateMeanSize(positiveExamples)
negativeExamples = generateFromRawData(1000,True,mean_x,mean_y,std_x,std_y)

positiveExamplesEval = extractFromRawData(1000)
mean_x,mean_y,std_x,std_y = calculateMeanSize(positiveExamples)
negativeExamplesEval = generateFromRawData(1000,True,mean_x,mean_y,std_x,std_y)

for i in range(len(positiveExamples)):
    positiveExamples[i] = normalizeSize(positiveExamples[i],50,50)
for i in range(len(negativeExamples)):
    negativeExamples[i] = normalizeSize(negativeExamples[i],50,50)

for i in range(len(positiveExamplesEval)):
    positiveExamplesEval[i] = normalizeSize(positiveExamplesEval[i],50,50)
for i in range(len(negativeExamplesEval)):
    negativeExamplesEval[i] = normalizeSize(negativeExamplesEval[i],50,50)

X = positiveExamples + negativeExamples
X = np.asarray(X,dtype='float32')
y = [1 for x in positiveExamples] +  [0 for x in negativeExamples]
y = np.asarray(y)

XEval = positiveExamplesEval + negativeExamplesEval
XEval = np.asarray(XEval,dtype='float32')
yEval = [1 for x in positiveExamplesEval] +  [0 for x in negativeExamplesEval]
yEval = np.asarray(yEval)



estimator = tf.estimator.Estimator(model_fn=cnn_model_fn)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=X,
    y=y,
    batch_size=60,
    num_epochs=None,
    shuffle=True)
estimator.train(
     input_fn=train_input_fn,
     steps=10000)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=XEval,
    y=yEval,
    num_epochs=1,
    shuffle=False)
eval_results = estimator.evaluate(input_fn=eval_input_fn)
print(eval_results)
