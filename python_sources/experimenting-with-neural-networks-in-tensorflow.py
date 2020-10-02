#!/usr/bin/env python
# coding: utf-8

# This notebook is a complete end to end introduction on creating a multilayer perceptron in tensorflow.
# 
# The basic outline is as follows:
# 
#  1. **Data Ingestion**
#  2. **Data Visualization**
#  3. **Preprocessing Data**
#  4. **Designing the tf graph**
#  5. **Executing the tf graph**
#  6. **Result Visualizations**

# In[ ]:


# boilerplate code
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
import pandas as pd
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML

import tensorflow as tf
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Ingestion
# 
# Let us fetch and explore the data before getting into training the neural network

# In[ ]:


df = pd.read_csv('../input/voice.csv')
df.head()


# Now let us perform some sanity check on the data to make sure there are no empty fields which we might need to impute further.

# In[ ]:


np.where(pd.isnull(df))


# Awesome. We don't have any null's in the dataset. One less thing to worry about. Now let us check how the labels are distributed.

# In[ ]:


print("Number of male: {}".format(df[df.label == 'male'].shape[0]))
print("Number of female: {}".format(df[df.label == 'female'].shape[0]))


# # Data Visualization
# 
# **Pearson Correlation Heatmap**
# 
# Let us perform the pearson correlation heatmap to figure out how features correlate to one another
# 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df.iloc[:,:-1].astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# While looking at the plot, we can figure out some interesting correlations. If you look at `meanfreq` vs `centroid` their correlation is maximum possible value of 1. Same is the case with `maxdom` and `dfrange`. So essentially we could filter out these features and still get an equivalent performance as they aren't adding any new information. 
# 
# *TODO*: refactor the notebook to filter out these columns easily

# # Preprocessing Data
# 
# There are multiple steps we will take in this section to transform the original data into format which we can easily plug inside tensorflow's tensors.
# 
# The basic steps are:
# * Creating 1-hot vector from the original labels
# * Normalize each feature
# * Create train/test/validation datasets

# In[ ]:


def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)


# Since the labels aren't numbers, let us first convert them into numerical categories and then subsequently convert it to a one-hot vector

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label=df.iloc[:,-1]

# Encode label category
# male -> 1
# female -> 0

gender_encoder = LabelEncoder()
label = gender_encoder.fit_transform(label)
label = convertToOneHot(label, 2)


# Let us look at the shape of the labels to confirm it is now a 1-hot vector

# In[ ]:


label.dtype


# In[ ]:


data = df.iloc[:,:-1]


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)


# Next step is to transform the dataset into train/test/validation datasets. sklearn doesn't directly give 3 way split so we do a 2 way split twice 

# In[ ]:


from sklearn.cross_validation import train_test_split
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(data,label,test_size=0.3)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=0.6)


# In[ ]:


x_train, x_test, y_train, y_test, x_validation, y_validation = np.array(x_train,dtype='float32'), np.array(x_test,dtype='float32'),np.array(y_train,dtype='float32'),np.array(y_test,dtype='float32'), np.array(x_validation, dtype='float32'), np.array(y_validation, dtype='float32')


# Let us look at the data first and make sure everything looks good

# In[ ]:


def print_shape(data_set, name):
    print("shape of {} is {} and datatype is {}".format(name, data_set.shape, data_set.dtype))


# In[ ]:


print_shape(x_train, "x_train")
print_shape(y_train, "y_train")
print_shape(x_test, "x_test")
print_shape(y_test, "y_test")
print_shape(x_validation, "x_validation")
print_shape(y_validation, "y_validation")


# # Design the tf graph
# 
# Now we are all set to get started with creating the tensorflow graph. It is important to understand that after this step there is no actual computation being done by tensorflow. It just creates a lazy graph according to the nodes we create in the `multilayer_perceptron` method. 

# In[ ]:


num_features = 20
num_hidden_layers = 2
mlp_layer_sizes = [20, 15, 10, 2]
weights = {}
biases = {}
num_features = mlp_layer_sizes[0]
n_hidden_1 = mlp_layer_sizes[1]  # 1st layer number of features
n_hidden_2 = mlp_layer_sizes[2]  # 2nd layer number of features
n_classes = mlp_layer_sizes[-1]
weights = {
    'h1': tf.get_variable("W1", shape=[num_features, n_hidden_1],
                          initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
    'h2': tf.get_variable("W2", shape=[n_hidden_1, n_hidden_2],
                          initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
    'out': tf.get_variable("W3", shape=[n_hidden_2, n_classes],
                           initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name="B1"),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name="B2"),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), name="B3")
}
# tf Graph input
x = tf.placeholder("float", [None, mlp_layer_sizes[0]], name="x")
y_ = tf.placeholder("float", [None, mlp_layer_sizes[-1]], name="y_")


# In[ ]:


def multilayer_perceptron(x, num_hidden_layers, weights,
                          biases, activation="relu", keep_prob=1.0, multiplication_factor=1.0):
    out_layer = None
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases[
                     'b1'], name="layer_1")
    layer_1 = activate(layer_1, activation, tf_name="layer_1")
    layer_1 = tf.scalar_mul(multiplication_factor, layer_1)
    layer_1 = tf.nn.dropout(
        layer_1, tf.constant(keep_prob), name="layer_1")
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases[
                     'b2'], name="layer_2")
    layer_2 = activate(layer_2, activation, tf_name="layer_2")
    layer_2 = tf.scalar_mul(multiplication_factor, layer_2)
    layer_2 = tf.nn.dropout(layer_2, tf.constant(keep_prob), name="layer_2")
    # Output layer
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'], name="out_layer")
    return out_layer

def activate(linear, activation, tf_name):
    if activation == 'sigmoid':
        return tf.nn.sigmoid(linear, name=tf_name)
    elif activation == 'softmax':
        return tf.nn.softmax(linear, name=tf_name)
    elif activation == 'linear':
        return linear
    elif activation == 'tanh':
        return tf.nn.tanh(linear, name=tf_name)
    elif activation == 'relu':
        return tf.nn.relu(linear, name=tf_name)


def loss_function(predictions, labels, loss, tf_name):
    if loss == 'cross-entropy':
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=predictions, labels=labels), name=tf_name)
    elif loss == 'rmse':
        return tf.reduce_mean(tf.square(predictions - labels), name=tf_name)


# In[ ]:


# construct the model using the above defined helper function
train_prediction = multilayer_perceptron(x, num_hidden_layers, weights, biases,
                                                    activation='relu', keep_prob=0.5)

validation_prediction = activate(multilayer_perceptron(x, num_hidden_layers, weights, biases,
                                                    activation='relu', multiplication_factor=0.5), 'softmax', tf_name="validation_prediction")


# In this section we define the optimizer and loss functions. We also define our notion of correct and incorrect prediction and what we mean by accuracy. This is the metric we will track later when we actually execute the graph

# In[ ]:


# constants
BATCH_SIZE = 100
DISPLAY_STEP = 10

# Define loss and optimizer
global_step = tf.Variable(0, name="global_step", trainable=False)
loss_function = loss_function(
        train_prediction, y_, 'cross-entropy', tf_name="loss_function")
learning_rate = tf.train.exponential_decay(
        0.0010000000474974513, global_step, 500, 0.98, name="learning_rate")

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.8999999761581421, beta2=0.9990000128746033,
                                       name="optimizer").minimize(loss_function, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(validation_prediction, 1), tf.argmax(y_, 1), name="correct_prediction")

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")


# # Execute tf graph
# 
# Now that the graph is set up, we need to initialize it and run it

# In[ ]:


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


# In[ ]:


num_epochs = 10000
prev_accuracy = 0.0
unoptimized_count = 0
final_epoch_count = 0

train_size = x_train.shape[0]

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(num_epochs):
    avg_accuracy = 0.0
    total_batch = int(train_size // BATCH_SIZE)
    # Loop over all batches
    for step in range(total_batch):
        offset = (step * BATCH_SIZE) % train_size
        batch_data = x_train[offset:(offset + BATCH_SIZE), :]
        batch_labels = y_train[offset:(offset + BATCH_SIZE)]
        _, a, current_step = sess.run([optimizer, accuracy, global_step], feed_dict={x: batch_data, y_: batch_labels})
        # Compute average loss & accuracy
        avg_accuracy += a / total_batch
    validation_accuracy = sess.run([accuracy], feed_dict={x: x_validation, y_: y_validation})
    current_step = tf.train.global_step(sess, global_step)
    if epoch % DISPLAY_STEP == 0:
        print("Epoch:{} training_accuracy={}".format(epoch + 1, avg_accuracy))
        print("Epoch:{} validation_accuracy={}".format(epoch + 1, validation_accuracy))
    if (avg_accuracy - prev_accuracy) < 0.01:
        unoptimized_count += 1
    else:
        unoptimized_count = 0
        prev_accuracy = avg_accuracy
    if unoptimized_count > 50:
        final_epoch_count = epoch + 1
        break
if final_epoch_count == 0:
    final_epoch_count = range(num_epochs)
validation_accuracy, validation_pred = sess.run([accuracy, validation_prediction],
                                      feed_dict={x: x_validation, y_: y_validation})  
test_accuracy, test_pred = sess.run([accuracy, validation_prediction],
                                feed_dict={x: x_test, y_: y_test})
print("Finally, validation_accuracy={}, test_accuracy={}".format(validation_accuracy, test_accuracy))


# # Result Visualizations
# 
# Now that we have our required predictions on test dataset, we can compute the precision, recall and all sorts of other metrics whichever suits our required problem statement.

# In[ ]:


from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

precision, recall, thresholds = precision_recall_curve(
            y_test[:, 1], test_pred[:, 1])

plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall')

