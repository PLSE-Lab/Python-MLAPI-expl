#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# 
# In this notebook, the human activity recognition dataset having records of acceleration and angular velocity measurements from different physical aspects in all three spatial dimensions (X, Y, Z) is used to train a machine and predict the activity from one of the six activities performed. 
# 
# To start with, let's do some exploratory analysis in hope of understanding various measures and their effect on the activities.

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# load data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print('Train Data', train.shape,'\n', train.columns)
print('\nTest Data', test.shape)


# ### Exploratory Analysis
# The data has 7352 observations with 563 variables with the first few columns representing the mean and standard deviations of body accelerations in 3 spatial dimensions (X, Y, Z). The last two columns are "subject" and "Acitivity" which represent the subject that the observation is taken from and the corresponding activity respectively. Let's see what activities have been recorded in this data.

# In[ ]:


print('Train labels', train['Activity'].unique(), '\nTest Labels', test['Activity'].unique())


# We have 6 activities, 3 passive (laying, standing and sitting) and 3 active (walking, walking_downstairs, walking_upstairs) which involve walking. So, each observation in the dataset represent one of the six activities whose features are recorded in the 561 variables. Our goal would be trian a machine to predict one of the six activities given a feature set of these 561 variables.
# 
# Let's check how many observations are recorded by each subject.

# In[ ]:


pd.crosstab(train.subject, train.Activity)


# It is good that the data is almost evenly distributed for all the activities among all the subjects. Let's pick subject 15 and compare the activities with the first three variables - mean body acceleration in 3 spatial dimensions.

# In[ ]:


sub15 = train.loc[train['subject']==15]


# In[ ]:


fig = plt.figure(figsize=(32,24))
ax1 = fig.add_subplot(221)
ax1 = sb.stripplot(x='Activity', y=sub15.iloc[:,0], data=sub15, jitter=True)
ax2 = fig.add_subplot(222)
ax2 = sb.stripplot(x='Activity', y=sub15.iloc[:,1], data=sub15, jitter=True)
plt.show()


# So, the mean body acceleration is more variable for walking activities than for passive ones especially in the X direction. Let's create a dendrogram and see if we can discover any structure with mean body acceleration.

# In[ ]:


sb.clustermap(sub15.iloc[:,[0,1,2]], col_cluster=False)


# Even though we see some dark spots in the X and Z directions (possibly from the walking activities), the bulk of the map is pretty homogenous and does not help much. Perhaps other attributes like maximum or minimum acceleration might give us a better insight than the average.
# 
# Plotting maximum acceleration with activity.

# In[ ]:


fig = plt.figure(figsize=(32,24))
ax1 = fig.add_subplot(221)
ax1 = sb.stripplot(x='Activity', y='tBodyAcc-max()-X', data=sub15, jitter=True)
ax2 = fig.add_subplot(222)
ax2 = sb.stripplot(x='Activity', y='tBodyAcc-max()-Y', data=sub15, jitter=True)
plt.show()


# That's interesting! Passive activities fall mostly below the active ones. It actually makes sense that maximum acceleration is higher during the walking activities. Let's again plot the cluster map but this time with maximum acceleration. Notice the walkdown acitivity is above all others in the X-direction recording values between 0.5 and 0.8.

# In[ ]:


sb.clustermap(sub15[['tBodyAcc-max()-X', 'tBodyAcc-max()-Y', 'tBodyAcc-max()-Z']], col_cluster=False)


# We can now see the difference in the distribution between the active and passive activities with the walkdown activity (values between 0.5 and 0.8) clearly distinct from all others especially in the X-direction. The passive activities are indistinguishable and present no clear pattern in any direction (X, Y, Z).
# 
# ### Clustering using KMeans
# Now, let us cluster the entire data using KMeans algorithm.

# In[ ]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6).fit(sub15.iloc[:,:-2])
clust = pd.crosstab(kmeans.labels_, sub15['Activity'])
clust


# Upon clustering using kmeans, all the walking activities seem to separate out while the passive ones are still mixed. All three Laying, Sitting and Standing are distributed in two different clusters. 

# In[ ]:


kmeans.cluster_centers_.shape


# In[ ]:


plt.plot(kmeans.cluster_centers_[np.asscalar(clust[clust.WALKING_DOWNSTAIRS!=0].index),:100], "o")


# Here is a plot of first 100 columns for the cluster mapped to 'WALKING_DOWNSTAIRS' acitivity. The column 40 and a few around 50 seems to be the dominant columns for this cluster center. Let's see what these are.

# In[ ]:


print(sub15.columns[[40, 49, 50, 51]])


# So, the different aspects of gravity have the most effect on walking down activity which makes perfect sense. 
# 
# Even though we could not find a clear pattern with any of the passive activities from the analysis so far, it can be inferred that the sensory measurements seem pretty good in order to train a machine and make predictions on new examples.

# ## Training
# In this section, I am going to train a model in TensorFlow using the train set and predict the activity using the test set.

# In[ ]:


import tensorflow as tf

# load train and test data
num_labels = 6
train_x = np.asarray(train.iloc[:,:-2])
train_y = np.asarray(train.iloc[:,562])
act = np.unique(train_y)
for i in np.arange(num_labels):
    np.put(train_y, np.where(train_y==act[i]), i)
train_y = np.eye(num_labels)[train_y.astype('int')] # one-hot encoding

test_x = np.asarray(test.iloc[:,:-2])
test_y = np.asarray(test.iloc[:,562])
for i in np.arange(num_labels):
    np.put(test_y, np.where(test_y==act[i]), i)
test_y = np.eye(num_labels)[test_y.astype('int')]

# shuffle the data
seed = 456
np.random.seed(seed)
np.random.shuffle(train_x)
np.random.seed(seed)
np.random.shuffle(train_y)
np.random.seed(seed)
np.random.shuffle(test_x)
np.random.seed(seed)
np.random.shuffle(test_y)


# ### Softmax Classifier
# 
# Training a softmax classifier and using gradient descent to optimize the weights.

# In[ ]:


# place holder variable for x with number of features - 561
x = tf.placeholder('float', [None, 561], name='x')
# place holder variable for y with the number of activities - 6
y = tf.placeholder('float', [None, 6], name='y')
# softmax model
def train_softmax(x):
    W = tf.Variable(tf.zeros([561, 6]), name='weights')
    b = tf.Variable(tf.zeros([6]), name='bias')
    lr = 0.25
    prediction = tf.nn.softmax(tf.matmul(x, W) + b, name='op_predict')
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for epoch in range(1000):
        loss = 0
        _, c = sess.run([optimizer, cost], feed_dict = {x: train_x, y: train_y})
        loss += c
        if (epoch % 100 == 0 and epoch != 0):
            print('Epoch', epoch, 'completed out of', 1000, 'Training loss:', loss)
    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='op_accuracy')
    
    print('Train set Accuracy:', sess.run(accuracy, feed_dict = {x: train_x, y: train_y}))
    print('Test set Accuracy:', sess.run(accuracy, feed_dict = {x: test_x, y: test_y}))


# In[ ]:


train_softmax(x)


# After training for 1000 iterations, we got a test accuracy of approximately 93% from softmax classifier

# ### Neural Network Classifier
# 
# Training using a simple artificial neural network with one hidden layer

# In[ ]:


n_nodes_input = 561 # number of input features
n_nodes_hl = 30     # number of units in hidden layer
n_classes = 6       # number of activities
x = tf.placeholder('float', [None, 561])
y = tf.placeholder('float')


# In[ ]:


def neural_network_model(data):
    # define weights and biases for all each layer
    hidden_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_input, n_nodes_hl], stddev=0.3)),
                      'biases':tf.Variable(tf.constant(0.1, shape=[n_nodes_hl]))}
    output_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl, n_classes], stddev=0.3)),
                    'biases':tf.Variable(tf.constant(0.1, shape=[n_classes]))}
    # feed forward and activations
    l1 = tf.add(tf.matmul(data, hidden_layer['weights']), hidden_layer['biases'])
    l1 = tf.nn.sigmoid(l1)
    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']
    
    return output


# In[ ]:


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=prediction))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for epoch in range(1000):
        loss = 0
        _, c = sess.run([optimizer, cost], feed_dict = {x: train_x, y: train_y})
        loss += c
        if (epoch % 100 == 0 and epoch != 0):
            print('Epoch', epoch, 'completed out of', 1000, 'Training loss:', loss)
    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='op_accuracy')
    
    print('Train set Accuracy:', sess.run(accuracy, feed_dict = {x: train_x, y: train_y}))
    print('Test set Accuracy:', sess.run(accuracy, feed_dict = {x: test_x, y: test_y}))


# In[ ]:


train_neural_network(x)


# Training using the neural network gave us a test accuracy of about 95%. Notice that I did not specify any learning rate here and used just the default from tensorflow. 
