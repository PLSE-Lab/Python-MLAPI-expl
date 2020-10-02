#!/usr/bin/env python
# coding: utf-8

# # Content
# 1. Introduction
# 2. Dataset description
# 3. Extreme Learning Machine algorithm theory
# 4. Advantages of using ELM over a traditional neural networks algorithm
# 5. Extreme Learning Machine algorithm implementation
# 6. Accuracy Evaluation
# 7. Conclusion
# ***

# ## 1. Introduction
# [**Artificial neural networks**](https://en.wikipedia.org/wiki/Artificial_neural_network) (ANNs) or connectionist systems are computing systems inspired by the biological neural networks. Such systems "learn" to perform tasks by considering examples, generally without being programmed with any task-specific rules.
# 
# In general, an artificial neural network is composed of a series of interconnected **layers** and a [**machine learning algorithm**](https://en.wikipedia.org/wiki/Machine_learning). A layer is composed of computational units, [**perceptrons**](https://en.wikipedia.org/wiki/Perceptron), which process data. In order to function, a neural network needs a database to work with. It computates the data from the database in each layer. After one layer computates the data, the computated data is sent to the next layer. Most of the ANN arhitectures have a final layer which gives the **network answer**.
# ***

# ## 2. Dataset description
# In this example, it has been used the [**Breast Cancer Wisconsin (Prognostic) Data Set**](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Prognostic%29). Each record represents follow-up data for one breast cancer case. These are consecutive patients seen by Dr. Wolberg since 1984, and include only those cases exhibiting invasive breast cancer and no evidence of distant metastases at the time of diagnosis. 
# 
# Below we can see the database shape and the class distribution, along with a statistical analysis about the database.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

csv_dataset = pd.read_csv("../input/bcwp_1995.csv")
csv_dataset.loc[:,['class']].plot()
print("Database shape rows: %s columns:%s \n" % np.shape(csv_dataset))
print(csv_dataset.describe())
plt.show()


# ***

# ## 3. Extreme Learning Machine algorithm theory
# In theory, the [**Extreme Learning Machine algorithm (ELM)**](https://en.wikipedia.org/wiki/Extreme_learning_machine) tends to provide good performance at extremely fast learning speed. Unlike most conventional NN learning algorithms, the ELM does not use a gradient-based technique. With this method, all the parameters are tuned once. This algorithm does NOT need iterative training. 
# 
# ELM algorithm has, by far, the easyest implementation of all. Tho its implementation is easy, the algorithm has great results with minimum computational time.
# 
# We have the following training set: 
# ![training set](https://image.ibb.co/mBRFiJ/train_x.png)
# where N represents the lines of the training set, and X_i is a values array from the database. X_i shape length is equal to the database colums number without the class column.
# We have the following set of classes:
# ![set of classes](https://image.ibb.co/iv5Ecd/test_t.png)
# where T_i is a binary array which contains the class for X_i entry from the training set.
# 
# **ELM implementation steps:**
# 1. Generate the weights matrix for the input layer;
# ![weight matrix](https://image.ibb.co/hw1HVy/w_matrix.png)
# 2. Calculate the hidden layer output matrix. After that, we need to activate the output matrix. We can choose any activation function that we desire;
# ![output matrix](https://image.ibb.co/cFsrqy/h.png)
# 3. Calculate the [**Moore-Penrose pseudoinverse**](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)
# ![pseudoinverse moore-penrose](https://image.ibb.co/fLE4Ay/mp.png)
# 4.  Calculate the output weight matrix beta;
# ![beta](https://image.ibb.co/eP5cVy/beta.png)
# 5.  We repeat step 2 for the testing dataset, creating a new H matrix. After that, we create the result matrix called O. We use the already known beta matrix.
# ![o](https://image.ibb.co/fZrnxd/o.png)
# 6. Use [**Soft Max algorithm**](https://en.wikipedia.org/wiki/Softmax_function) to transform O matrix. After that, compare the O matrix with the T matrix using the [**Winner Takes All algorithm**](https://en.wikipedia.org/wiki/Winner-take-all_(computing)). 
# ***

# ## 4. Advantages of using ELM over a traditional neural networks algorithm
# 
# Traditional algorithms use gradient-based learning techniques. This makes them slow when facing large databases. ELM does not use gradient-based techinques. This makes it run much faster than its competitors. Another advantage that the ELM has over traditional algorithms, is that all the parameters are tuned **ONCE**. ELM does **not need** iterative training. 
# ***

# ## 5. Extreme Learning Machine algorithm implementation

# In[ ]:


def create_one_hot_encoding(classes, shape):
    one_hot_encoding = np.zeros(shape)
    for i in range(0, len(one_hot_encoding)):
        one_hot_encoding[i][int(classes[i])] = 1
    return one_hot_encoding


# In[ ]:


def train(weights, x, y):
    h = x.dot(weights)
    h = np.maximum(h, 0, h)
    return np.linalg.pinv(h).dot(y)


# In[ ]:


def soft_max(layer):
    soft_max_output_layer = np.zeros(len(layer))
    for i in range(0, len(layer)):
        numitor = 0
        for j in range(0, len(layer)):
            numitor += np.exp(layer[j] - np.max(layer))
        soft_max_output_layer[i] = np.exp(layer[i] - np.max(layer)) / numitor
    return soft_max_output_layer

def matrix_soft_max(matrix_):
    soft_max_matrix = []
    for i in range(0, len(matrix_)):
        soft_max_matrix.append(soft_max(matrix_[i]))
    return soft_max_matrix


# In[ ]:


def check_network_power(o, o_real):
    count = 0
    for i in range(0, len(o)):
        count += 1 if np.argmax(o[i]) == np.argmax(o_real[i]) else 0
    return count


# In[ ]:


def test(weights, beta, x, y):
    h = x.dot(weights)
    h = np.maximum(h, 0, h)  # ReLU
    o = matrix_soft_max(h.dot(beta))
    return check_network_power(o, y) / len(y)


# After we have all the methods requiered for the algorithm, we can make an accuracy test.

# In[ ]:


class_column = 0
test_size = 0.1
db = csv_dataset.iloc[:, :].values.astype(np.float)
np.random.shuffle(db)
y = db[:, class_column]
y -= np.min(y)
output_layer_perceptron_count = len(np.unique(y))
y = create_one_hot_encoding(y, (len(y), len(np.unique(y))))
x = np.delete(db, [class_column], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
hidden_layer_perceptron_count = len(y_test)
x = preprocessing.normalize(x)
weights = np.random.random((len(x[0]), hidden_layer_perceptron_count))
beta = train(weights, x_train, y_train)
print("Accuracy = %s." % test(weights, beta, x_test, y_test))


# ***

# ## 6. Accuracy evaluation
# As we can see above, the accuracy can go over average. The ELM algorithm has very high statistical power. The accuracy can depend on:
# - the activation function that we use;
# - if the data is normalised or not;
# - the nature of the attributes;
# - the overall database distribution.
# ***

# ## 7. Conclusion 
# Extreme Learning Machine algorithm is one of the most efficient machine learning algorithms in the neural networks world. It works on very large datasets. Because of the non-iterative training all the parameters are tuned once. This results in a high training speed. Its implementation is easy to understand, and it can be used to solve complex problems. Don't forget to test the algorithm on multiple databases. Test the database on more activation functions. In this example we used [**ReLU**](https://en.wikipedia.org/wiki/Rectifier_(neural_networks) activation function because it is easy to implement.
# 
# For more tutorials check out my [**kernels**](https://www.kaggle.com/andreicosma/kernels) page.
# 
# Don't let failure stop you from achieving greatness.
# 
# References: Extreme learning machine: Theory and applications Guang-Bin Huang, Qin-Yu Zhu, Chee-Kheong Siew.
