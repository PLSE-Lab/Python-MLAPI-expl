# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import time as time
import sklearn.preprocessing as sklrn
t0 = time.time()

testfile = "../input/test.csv"
trainfile = "../input/train.csv"

df_init = pd.read_csv(trainfile)
df_test_init = pd.read_csv(testfile)

train_label_dataset = df_init['label'].values ## 42000x1
train_dataset = df_init.drop('label',axis=1).values ## 42000 x 784 when using full MNIST dataset.

test_dataset = df_test_init.values

num_labels = 10
image_size = 28
################ NEED TO CHANGE VARIABLE NAMES TO MATCH DIRECTORY. 


def reformat(labels):
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return labels

## Need to normalize data to having zero mean and unit variance, and reformat labels to be a matrix, 
## not a vector. 
train_dataset = sklrn.scale(train_dataset,axis=0)
test_dataset = sklrn.scale(test_dataset,axis=0)

train_labels = reformat(train_label_dataset);
print(np.shape(train_labels))


def write_to_csv(prediction):
    mx = np.argmax(prediction,1)
    if np.shape(np.shape(mx))[0] == 1:
        mx = mx[:,np.newaxis]
    else:
        pass
    output_lst = []
    length_of_prediction = np.shape(mx)[0]
    for i in range(0,length_of_prediction):
        imgid = i+1
        output_lst.append({'ImageId':imgid,'Label':mx[i][0]})
    
    df_output = pd.DataFrame(output_lst)
    df_output.to_csv("results.csv",header=True,index=False)

def accuracy(predictions, labels):
    return ((100.0 * np.sum(np.argmax(predictions,1) == np.argmax(labels,1))) / predictions.shape[0])


batch_size = 128
## Choose hidden layer sizes:
nhid1 = 250 # hidden 1
nhid2 = 250 # hidden 2
nhid3 = 250 # hidden 3
nhid4 = 250 # hidden 4

## Build tensorflow graph, i.e. build the structure of the neural network
## and build what computations will be done in that structure.
graph = tf.Graph()
with graph.as_default():
    
    ## Input data and define constants and placeholders
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size,image_size*image_size))
    tf_train_label = tf.placeholder(tf.float32, shape=(batch_size,num_labels))
    tf_test_dataset = tf.constant(test_dataset,tf.float32)
    
    
    ## Define variables.
    theta1 = tf.Variable(
        tf.truncated_normal([image_size*image_size,nhid1]))
    theta1_bias = tf.Variable(tf.zeros([nhid1]))
    #hid1
    theta2 = tf.Variable(
        tf.truncated_normal([nhid1,nhid2]))
    theta2_bias = tf.Variable(tf.zeros([nhid2]))
    #hid2
    theta3 = tf.Variable(
        tf.truncated_normal([nhid2,nhid3]))
    theta3_bias = tf.Variable(tf.zeros([nhid3]))
    #hid3
    theta4 = tf.Variable(
        tf.truncated_normal([nhid3,nhid4]))
    theta4_bias = tf.Variable(tf.zeros([nhid4]))
    #hid4
    theta5 = tf.Variable(
        tf.truncated_normal([nhid4,num_labels]))
    theta5_bias = tf.Variable(tf.zeros([num_labels]))
    #out
    
    ## Define/Set up computations that will be performed.
    hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset,theta1) + theta1_bias)
    hidden2 = tf.nn.relu(tf.matmul(hidden1,theta2) + theta2_bias)
    hidden3 = tf.nn.relu(tf.matmul(hidden2,theta3) + theta3_bias)
    hidden4 = tf.nn.relu(tf.matmul(hidden3,theta4) + theta4_bias)
    logits = tf.matmul(hidden4,theta5) + theta5_bias
    loss =    tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_label, logits = logits))
    
    ## Optimization.
    optimizer = tf.train.GradientDescentOptimizer(0.00005).minimize(loss)
    
    ## Predictions for training, validation, and test datasets
    train_prediction = tf.nn.softmax(logits)
    
    ## In order to compute test_prediction
    hidden1_test = (tf.nn.relu(tf.matmul(tf_test_dataset,theta1) + theta1_bias))
    hidden2_test = (tf.nn.relu(tf.matmul(hidden1_test,theta2) + theta2_bias))
    hidden3_test = (tf.nn.relu(tf.matmul(hidden2_test,theta3) + theta3_bias))
    hidden4_test = (tf.nn.relu(tf.matmul(hidden3_test,theta4) + theta4_bias))
    test_prediction = tf.nn.softmax(tf.matmul(hidden4_test,theta5) + theta5_bias)
        
###############        
        
num_steps = 4001
#num_steps = 801
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_label : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      
  preds = test_prediction.eval();
  #print("Test accuracy: %.1f%%" % accuracy(preds, test_labels))
  #print(test_prediction.eval()[0:10])
write_to_csv(preds)

               
print(time.time()-t0)
