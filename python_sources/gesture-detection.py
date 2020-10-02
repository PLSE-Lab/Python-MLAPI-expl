'''importing the required libraries'''
import pandas as pd
import tensorflow as tf
import numpy as np

train_data = pd.read_csv('../input/gesture-detection/train.csv')
test_data = pd.read_csv("../input/gesture-detection/test.csv")
test = test_data
X = train_data

'''calculating z score values and removing outliers and seperating input and output from data'''
from scipy import stats
z = np.abs(stats.zscore(X))
X = X[(z < 10).all(axis=1)]
y = X['class']
X = X.drop(['class'], axis= 1)


'''standardising the test and train input data'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
test = sc_X.transform(test)

'''applying pca for dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(0.95)
X = pca.fit_transform(X)
test = pca.transform(test)'''

'''splitting the data into train and cross validation'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.14, random_state = 0)

y_test = pd.get_dummies(y_test)
y_train = pd.get_dummies(y_train)
y_test = y_test.as_matrix()
y_train = y_train.as_matrix()
y_test = y_test.astype(np.float64)
y_train = y_train.astype(np.float64)

'''declaring placeholders for inputs'''
x=tf.placeholder(tf.float64, [None, None])
y_true = tf.placeholder(tf.float64, [None, None])
keep_prob = tf.placeholder(tf.float64)
epsilon = 1e-3

'''intializing all the weights and biases in the neural network
     its a feedforward neural network with 3 hidden layers each layer containg 128 hidden units  '''
weights = tf.Variable(tf.cast(tf.random.normal([9, 100])/np.sqrt(31), tf.float64))
scale1 = tf.Variable(tf.cast(tf.ones([100]), tf.float64))
beta1 = tf.Variable(tf.cast(tf.zeros([100]), tf.float64))
weights1 = tf.Variable(tf.cast(tf.random.normal([100, 100])/np.sqrt(100), tf.float64))
scale2 = tf.Variable(tf.cast(tf.ones([100]), tf.float64))
beta2 = tf.Variable(tf.cast(tf.zeros([100]), tf.float64))
weights2 =  tf.Variable(tf.cast(tf.random.normal([100, 100])/np.sqrt(100), tf.float64))
scale3 = tf.Variable(tf.cast(tf.ones([100]), tf.float64))
beta3 = tf.Variable(tf.cast(tf.zeros([100]), tf.float64))
weights3 =  tf.Variable(tf.cast(tf.random.normal([100, 6])/np.sqrt(100), tf.float64))
biases3 = tf.Variable(tf.cast(tf.random.normal([6]), tf.float64))

'''first layer'''
logits1 = tf.matmul(x, weights)
'''implementing batch normalization'''
batch_mean1, batch_var1 = tf.nn.moments(logits1,[0])
BN1 = tf.nn.batch_normalization(logits1,batch_mean1,batch_var1,beta1,scale1,epsilon)
y_pred1 = tf.nn.tanh(BN1)
'''implementing dropout regularization'''
y1_dropout = tf.nn.dropout(y_pred1, keep_prob)
 
'''second layer'''
logits2 = tf.matmul(y_pred1, weights1)
batch_mean2, batch_var2 = tf.nn.moments(logits2,[0])
BN2 = tf.nn.batch_normalization(logits2,batch_mean2,batch_var2,beta2,scale2,epsilon)
y_pred2 = tf.nn.tanh(BN2)
y2_dropout = tf.nn.dropout(y_pred2, keep_prob)

'''third layer''' 
logits3 = tf.matmul(y_pred2, weights2)
batch_mean3, batch_var3 = tf.nn.moments(logits3,[0])
BN3 = tf.nn.batch_normalization(logits3,batch_mean3,batch_var3,beta3,scale3,epsilon)
y_pred3 = tf.nn.tanh(BN3)
y2_dropout = tf.nn.dropout(y_pred2, keep_prob) 

'''output unit'''
y_pred4 = tf.matmul(y_pred3 ,weights3) + biases3

y_pred = tf.nn.softmax(y_pred4)
y_pred_cls = tf.argmax(y_pred, dimension=1)

'''declaring cost function'''

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred4, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

'''reducing cost using adam optimizer'''
optimizer = tf.train.AdamOptimizer(learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-08).minimize(cost)

'''initializing all the variables and running the optimizer to minimize the cost'''
session = tf.Session()
session.run(tf.initialize_all_variables())
def optimize(num_iterations):
    for i in range(num_iterations):
        feed_dict_train = {x: X_train, y_true: y_train, keep_prob: 0.6}
        session.run(optimizer, feed_dict=feed_dict_train)

'''function to calculate accuracy on both test and cross validation set'''
y_true_cls = tf.argmax(y_true, dimension=1)                 
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy2 = tf.losses.mean_squared_error(y_pred_cls, y_true_cls)
feed_dict_test = {x: X_test, y_true: y_test, keep_prob: 0.6}
feed_dict_train = {x: X_train, y_true: y_train, keep_prob: 0.6}
def print_accuracy():
    acc = session.run(accuracy, feed_dict = feed_dict_test)
    print("accuracy on test set: {}".format(acc))
    acc = session.run(accuracy, feed_dict = feed_dict_train)
    print("accuracy on train set: {}".format(acc))
    acc2 = session.run(accuracy2, feed_dict = feed_dict_test)
    print("mse accuracy of test is: {}".format(acc2))
        
def submission():
    submission = pd.DataFrame()
    submission['time'] = test_data['time']
    submission['class'] = y_pred_cls.eval(session=session, feed_dict = {x: test})
    submission.to_csv('submission215s.csv', index=False)
    
