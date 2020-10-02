#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from ipywidgets import  interact, interactive, fixed, interact_manual,FloatSlider


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


y_train = train.loc[:, train.columns == 'label'].values.flatten()
x_train = train.loc[:, train.columns != 'label'].values
    


# In[ ]:


# create a boolean matrix of the correct answers
y_train_labels = [[(value == i) * 1 for i in range(0,10)] for value in y_train]


# ## Building Model

# In[ ]:


BANDWIDTH = 0.2


# In[ ]:


def _pattern(input,name,feature_count,h):
    with tf.variable_scope(name) as scope:
        bias = tf.get_variable('bias',[feature_count, 1],initializer=tf.constant_initializer(0),dtype=tf.float32)
        bandwidth = tf.constant(1.0/(h * feature_count),dtype=tf.float32)
        offset = tf.add(input, tf.transpose(bias))
        mapping_layer = tf.map_fn(lambda x: (gaussian_tf(x)/h),offset)
        summing_layer = tf.reduce_sum(mapping_layer,axis=1)
        return tf.multiply(summing_layer,bandwidth)


# In[ ]:



def model(features, labels, mode):

    conv1 = tf.layers.conv2d(
          inputs=tf.reshape(features["x"], [-1, 28, 28, 1]),
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])


    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    result = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=result, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(result, name="softmax_tensor")
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=labels))

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=tf.argmax(labels, 1), predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# # Training

# In[ ]:


classifier = tf.estimator.Estimator(
    model_fn=model, model_dir="./pnn_feedfoward_model")


# In[ ]:


# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)


# In[ ]:


# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": x_train.astype(np.float32)},
    y=np.array(y_train_labels).astype(np.float32),
    batch_size=2000,
    num_epochs=None,
    shuffle=True)

classifier.train(
    input_fn=train_input_fn,
    steps=100,
    hooks=[logging_hook])


# In[ ]:




# Evaluate the model and print results
# eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"x": x_test.astype(np.float32)},
#     y=np.array(y_test_labels).astype(np.float32),
#     num_epochs=1,
#     shuffle=False)
# eval_results = classifier.evaluate(input_fn=eval_input_fn)
# print(eval_results)


# In[ ]:


test = pd.read_csv('../input/test.csv').values


# In[ ]:


eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test.astype(np.float32)},
    num_epochs=1,
    shuffle=False)
result = pd.DataFrame([ {'ImageId':index + 1,'Label':i['classes']} for index,i in enumerate(classifier.predict(input_fn=eval_input_fn))])
# result.to_csv('results.csv')
result.to_csv('submission.csv', index=False)


# ## Sources
# 
# https://www.sciencedirect.com/science/article/pii/089360809090049Q
# 
# https://stats.stackexchange.com/questions/244012/can-you-explain-parzen-window-kernel-density-estimation-in-laymans-terms

# In[ ]:




