#!/usr/bin/env python
# coding: utf-8

# **Import Dependencies and predict Satisfaction using Deep Neural Network Tensorflow with Softmax Cross Entropy**

# Below summary for the whole dataset in 2D space after apply Standard Scaler scikit into the dataset

# #### Predict Job Satisfaction using Deep Neural Network

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="whitegrid", palette="muted")
current_palette = sns.color_palette()

xrange = range


# In[ ]:


dataset = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
age = dataset.ix[:, 0]
attrition = dataset.ix[:, 1]

dataset.ix[:, 0] = attrition
dataset.ix[:, 1] = age

dataset = dataset.rename(columns = {'Age': 'Attrition', 'Attrition': 'Age'})
dataset.head()


# In[ ]:


from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = dataset.ix[:, 1:]
Y = dataset.ix[:, :1]

labels = np.unique(Y.values).tolist()

Y.ix[:, 0] = LabelEncoder().fit_transform(Y.ix[:, 0])

for i in xrange(X.shape[1]):
    if str(type(X.ix[0, i])).find('str') > 0:
        X.ix[:, i] = LabelEncoder().fit_transform(X.ix[:, i])

X, _, Y, _ = train_test_split(X, Y, test_size=0.5)

X.ix[:, :] = StandardScaler().fit_transform(X.ix[:, :])

data_visual = TSNE(n_components = 2).fit_transform(X.values)

X.head()


# In[ ]:


for i, _ in enumerate(np.unique(Y.values)):
    plt.scatter(data_visual[Y.values[:, 0] == i, 0], data_visual[Y.values[:, 0] == i, 1], color = current_palette[i], label = labels[i])
    
plt.legend()
plt.show()


# In[ ]:


dataset_copy = dataset[['Department', 'TotalWorkingYears', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole']]

dataset_copy.head()


# In[ ]:


dataset_copy = dataset_copy.ix[:200, :]

plt.figure(figsize=(30,10))

department = pd.melt(dataset_copy, "Department", var_name="Working Variables")

swarm_plot = sns.swarmplot(x="Working Variables", y="value", hue="Department", data=department)
plt.show()


# In[ ]:


fig = plt.figure(figsize=(20,10))

head_subplot = np.unique(dataset.ix[:, 4].values)

head_int_label = LabelEncoder().fit_transform(dataset['Department'])

unique_head_int_label = np.unique(head_int_label)

rows = ['DailyRate', 'EnvironmentSatisfaction', 'JobSatisfaction', 'MonthlyIncome', 'RelationshipSatisfaction']

Y = dataset['Age'].ix[:].values

labelset = LabelEncoder().fit_transform(dataset['Attrition'])

labels = dataset['Attrition'].unique()

num = 1

for i in xrange(len(head_subplot)):
    for k in xrange(len(rows)):
        plt.subplot(len(head_subplot), len(rows), num)

        X = dataset[rows[k]].ix[:].values

        X = X[head_int_label == unique_head_int_label[i]]
        
        Y_in = Y[head_int_label == unique_head_int_label[i]]
        
        labelset_filter = labelset[head_int_label == unique_head_int_label[i]]
        
        for no, text in enumerate(labels):
            plt.scatter(X[labelset_filter == no], Y_in[labelset_filter == no], color = current_palette[no],
                        label = labels[no])
        plt.title(head_subplot[i])
        plt.ylabel('Age')
        plt.xlabel(rows[k])
        plt.legend()
        
        num += 1
        
fig.tight_layout()        
plt.show()     
    


# In[ ]:


dataset = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

del dataset['Over18']
del dataset['OverTime']

# 17th column and above
X = dataset.ix[:, 17:].values

# change marriage status into int
X[:, 0] = LabelEncoder().fit_transform(X[:, 0])

Y = dataset['JobSatisfaction'].ix[:].values

def one_hot_label(x):
    data = np.zeros((x.shape[0], np.unique(x).shape[0]), dtype = np.float32)
    
    for i in xrange(x.shape[0]):
        data[0, x[i] - 1] = 1.0
        
    return data

Y = one_hot_label(Y)

X = StandardScaler().fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)


# In[ ]:


import tensorflow as tf

epoch = 100
learning_rate = 0.01
delta_penalty = 0.005
prob_dropout = 0.5

first_layer = 64
second_layer = 128
third_layer = 64

x_placeholder = tf.placeholder('float', [None, X.shape[1]])
y_placeholder = tf.placeholder('float', [None, Y.shape[1]])

weights = {
    'first_weight' : tf.Variable(tf.random_normal([X.shape[1], first_layer])),
    'second_weight' : tf.Variable(tf.random_normal([first_layer, second_layer])),
    'third_weight' : tf.Variable(tf.random_normal([second_layer, third_layer])),
    'fourth_weight' : tf.Variable(tf.random_normal([third_layer, Y.shape[1]])),
}

biases = {
    'first_bias' : tf.Variable(tf.random_normal([first_layer])),
    'second_bias' : tf.Variable(tf.random_normal([second_layer])),
    'third_bias' : tf.Variable(tf.random_normal([third_layer])),
    'fourth_bias' : tf.Variable(tf.random_normal([Y.shape[1]])),
}

first_layer = tf.nn.relu(tf.add(tf.matmul(x_placeholder, weights['first_weight']), biases['first_bias']))
first_layer = tf.nn.dropout(first_layer, prob_dropout)

second_layer = tf.nn.relu(tf.add(tf.matmul(first_layer, weights['second_weight']), biases['second_bias']))
second_layer = tf.nn.dropout(second_layer, prob_dropout)

third_layer = tf.nn.relu(tf.add(tf.matmul(second_layer, weights['third_weight']), biases['third_bias']))
third_layer = tf.nn.dropout(third_layer, prob_dropout)

fourth_layer = tf.add(tf.matmul(third_layer, weights['fourth_weight']), biases['fourth_bias'])

regularizers =  sum(map(lambda x: tf.nn.l2_loss(x), [value for _, value in weights.items()]))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fourth_layer, labels = y_placeholder)) + (delta_penalty * regularizers)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(fourth_layer, 1), tf.argmax(y_placeholder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[ ]:


import time

sess = tf.InteractiveSession()
    
sess.run(tf.global_variables_initializer())

for i in xrange(epoch):
    
    last_time = time.time()
    
    acc,_, lost = sess.run([accuracy, optimizer, loss], feed_dict = {x_placeholder : x_train, y_placeholder : y_train})
    
    print ("epoch: ", i + 1, ", loss: ", lost, ", seconds per epoch: ", time.time() - last_time)
    print ("total accuracy: ", acc)
    

