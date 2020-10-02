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

import csv
import tensorflow as tf
import random
import pandas as pd
import numpy as np
from time import time

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


from tensorflow.contrib import layers
from tensorflow.contrib import learn

df = pd.read_csv('../input/diabetes.csv', index_col = False)


X = df[["Pregnancies", 'Glucose','BloodPressure','SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction','Age']]

y = df[['Outcome']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)




clf= LogisticRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print('LogReg accuracy: ', accuracy)


df['NotDiabetes'] = 1-df['Outcome']
y = df[['Outcome', 'NotDiabetes']]




X.to_csv('X.csv', index= False, header=False)
y.to_csv('y.csv', index= False, header=False)


data = pd.read_csv('X.csv', index_col = False)#.replace(0, np.nan, inplace = True)
labels = pd.read_csv('y.csv', index_col=False)





X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=0)

learning_rate = tf.train.exponential_decay(learning_rate=0.002,
  global_step= 1,
  decay_steps=X_train.shape[0],
  decay_rate= 0.95,
  staircase=True)
training_epochs = 1000
batch_size = 100
display_step = 100

n_hidden_1 = 700 
n_hidden_2 = 700 
n_input = X_train.shape[1]
n_classes = y_train.shape[1]
dropout = 0.5



x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float")
keep_prob = tf.placeholder(tf.float32)



def neural_network(x, weights, biases,dropout):
    # Hidden layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
 
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    out_layer = tf.nn.dropout(out_layer, dropout)
    return out_layer



weights = {
'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev = 0.0001)),
'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev = 0.0001)),
'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes], stddev = 0.0001))
}
biases = {
'b1': tf.Variable(tf.random_normal([n_hidden_1])),
'b2': tf.Variable(tf.random_normal([n_hidden_2])),
'out': tf.Variable(tf.random_normal([n_classes]))
}


pred = neural_network(x, weights, biases, keep_prob)

# cross_entropy = -tf.reduce_sum(pred*tf.log(y + 1e-10))
# cost = tf.reduce_mean(cross_entropy)
cost =  tf.nn.l2_loss(tf.nn.softmax_cross_entropy_with_logits(pred, y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

    
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X_train)/batch_size)

        X_batches = np.array_split(X_train, total_batch)
        Y_batches = np.array_split(y_train, total_batch)


        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]

            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("epoch:", '%d' % (epoch+1), "cost=", "{:.3f}".format(avg_cost))
            # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # print(accuracy.eval({x: X_test, y: y_test, keep_prob: 1}))


    # Test model

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: X_test, y: y_test, keep_prob: 1}))


