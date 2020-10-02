#!/usr/bin/env python
# coding: utf-8

# ### Declaring the environment

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading training data

# In[ ]:


# Load the data - the train.csv and test.csv is in the directory ./data
def read_data(file_name):
    data = pd.read_csv('./data/' + file_name)
    return data


# In[ ]:


data = read_data('./train.csv')
data.head()


# In[ ]:


data.describe()


# ### Viewing data with piechart, analyzing and cleaning the data 
# 
# The analyzing of data is inspired by ramanshah: https://github.com/ramansah/kaggle-titanic/blob/master/Analysis.ipynb

# In[ ]:


def analyze_age(data):
    age = data['Age']
    print("# of rows with age as null: ", age.isna().sum())
    print("age.min", age.min())
    print("age.max", age.max())
    print("age.mean", age.mean())


# In[ ]:


analyze_age(data)


# In[ ]:


def prepare_age(data):
    age = data['Age']
    
    # Mean age
    mean_age = age.mean()
    
    # Variance age
    var_age = age.var()

    # Fill the age that is missing i.e. has NaN value
    age.fillna(mean_age, inplace=True)
    
    # Normalize age
    age = age - mean_age
    age = age / var_age

    # print("Age is: " +  str(age))
    return age.as_matrix()
    


# In[ ]:


age = prepare_age(data)
print("shape of age matrix: " + str(age.shape))
# print(str(age))


# In[ ]:


def analyze_pclass(data):
    fig = plt.figure(figsize=(10,10))
    i = 1
    for pclass in data['Pclass'].unique():
        fig.add_subplot(3, 3, i)
        plt.title('Pclass: {}'.format(pclass))
        labels = '0', '1'
        colors = 'lightskyblue', 'green'
        data.Survived[data['Pclass'] == pclass].value_counts().plot(kind='pie', labels=labels, colors=colors)
        i += 1
    


# In[ ]:


analyze_pclass(data)


# In[ ]:


def prepare_pclass(data):
    pclass = data['Pclass']
    
    mean_pclass = pclass.mean()
    var_pclass = pclass.var()
    pclass = pclass - mean_pclass
    pclass = pclass / var_pclass
    return pclass.as_matrix()
    


# In[ ]:


pclass = prepare_pclass(data)
print("shape of pclass matrix: " + str(pclass.shape))
# print(str(pclass))


# In[ ]:


def analyze_name(data):
    data['Name'] = data['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
    #titles = data['Name'].unique()
    #print(titles)

    fig = plt.figure(figsize=(14,10))
    i = 1
    for title in data['Name'].unique():
        fig.add_subplot(3, 6, i)
        plt.title('Title: {}'.format(title))
        labels = '0', '1'
        colors = 'lightskyblue', 'green'
        data.Survived[data['Name'] == title].value_counts().plot(kind='pie', labels=labels, colors=colors)
        i += 1


# In[ ]:


#analyze_name(data)


# In[ ]:


# Not using the "title" in the feature list. However, it is possible to extract information about the title
# and use it as one of the features. 
def prepare_name(data):
    
    data['Name'] = data['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())

    name = data['Name']
    
    replacement = {
        'Mr': 0,
        'Mrs': 1,
        'Miss': 2,
        'Master': 3,
        'Don': 4, 
        'Rev': 5, 
        'Dr': 6,
        'Mme': 7,
        'Ms': 8, 
        'Major': 9, 
        'Lady': 10,
        'Sir': 11,
        'Mile': 12, 
        'Col': 13,
        'Capt': 14,
        'the Countess': 15,
        'jonkheer':16,
        '': 17
    }

    #name = data['Name'].apply(lambda x: replacement.get(x))
    #titles = name.map(lambda x: x.split(',')[1].split('.')[0].strip())
    titles = name.apply(lambda x: replacement.get(x))

    #titles = data['Name'].unique()
    return titles.as_matrix()


# In[ ]:


name = prepare_name(data)
print("shape of name matrix: " + str(name.shape))
# print(str(name))


# In[ ]:


def analyze_sex(data):
    fig = plt.figure(figsize=(10,10))
    i = 1
    for sex in data['Sex'].unique():
        fig.add_subplot(3, 3, i)
        plt.title('Sex: {}'.format(sex))
        labels = '0', '1'
        colors = 'lightskyblue', 'green'
        data.Survived[data['Sex'] == sex].value_counts().plot(kind='pie', labels=labels, colors=colors)
        i += 1


# In[ ]:


analyze_sex(data)


# In[ ]:


def prepare_sex(data):
    sex = data['Sex']
    
    # Printing the column Sex
    # print(sex.unique())
    
    # Replace male with 0, female with 1 
    sex.replace("male", 0, inplace=True)
    sex.replace("female", 1, inplace=True)
    
    return sex.as_matrix()


# In[ ]:


sex = prepare_sex(data)
print("shape of sex matrix: " + str(sex.shape))
# print(str(sex))


# In[ ]:


def prepare_sibsp(data):
    sibsp = data['SibSp']
        
    mean_sibsp = sibsp.mean()
    var_sibsp = sibsp.var()
    sibsp = sibsp - mean_sibsp
    sibsp = sibsp / var_sibsp
        
    return sibsp.as_matrix()


# In[ ]:


sibsp = prepare_sibsp(data)
print("shape of sibsp matrix: " + str(sibsp.shape))
# print(str(sibsp))


# In[ ]:


def prepare_parch(data):
    parch = data['Parch']
    # print("unique values of parch: " , parch.unique())
    
    mean_parch = parch.mean()
    var_parch = parch.var()
    parch = parch - mean_parch
    parch = parch / var_parch
    
    return parch.as_matrix()


# In[ ]:


parch = prepare_parch(data)
print("shape of parch: ", parch.shape)
# print(str(parch))


# In[ ]:


def prepare_fare(data):
    fare = data['Fare']
    #unique_fare = fare.unique()
    #print("unique fare size: ", unique_fare)
    #print("check fare is null: " + str(fare.isnull()))
    
    mean_fare = fare.mean()
    var_fare = fare.var()
    fare = fare - mean_fare
    fare = fare / var_fare
    return fare.as_matrix()
      


# In[ ]:


fare = prepare_fare(data)
print("shape of fare: ", fare.shape)
# print(str(fare))


# In[ ]:


def analyze_cabin(data):
    cabin = data['Cabin']
    # print(cabin.unique())
    # print(cabin.isnull().sum())
    cabin.fillna('U', inplace=True)
    data['Cabin'] = data['Cabin'].apply(lambda x: x[0])
    
    fig = plt.figure(figsize=(10,10))    
    i = 1
    for cabin in data['Cabin'].unique():
        fig.add_subplot(3, 3, i)
        plt.title('Cabin: {}'.format(cabin))
        labels = '0', '1'
        colors = 'lightskyblue', 'green'
        data.Survived[data['Cabin'] == cabin].value_counts().plot(kind='pie', labels=labels, colors=colors)
        i += 1
    return cabin


# In[ ]:


cabin = analyze_cabin(data)


# In[ ]:


def prepare_cabin(data):
    cabin = data['Cabin']
    cabin.fillna('U', inplace=True)
    
    # print(cabin)
    
    data['Cabin'] = data['Cabin'].apply(lambda x: x[0])
    
    # print(cabin.unique())
    
    replacement = {
        'U': 0,
        'C': 1,
        'E': 2,
        'G': 3,
        'D': 4, 
        'A': 5,
        'B': 6,
        'F': 7,
        'T': 8
    }
    
    cabin = cabin.apply(lambda x: replacement.get(x))
    
    mean_cabin = cabin.mean()
    var_cabin = cabin.var()
    cabin = cabin - mean_cabin
    cabin = cabin / var_cabin
    return cabin.as_matrix()    


# In[ ]:


cabin = prepare_cabin(data)
print("shape of cabin: ", cabin.shape)
# print(str(cabin))


# In[ ]:


def prepare_embarked(data):
    embarked = data['Embarked']
    
    # print("unique values: ", embarked.unique())
    # print(embarked)
    
    embarked.replace('S', 0, inplace=True)
    embarked.replace('C', 1, inplace=True)
    embarked.replace('Q', 2, inplace=True)
    embarked.fillna(3, inplace=True)
    
    # print(embarked)
    mean_embarked = embarked.mean()
    var_embarked = embarked.var()
    embarked = embarked - mean_embarked
    embarked = embarked / var_embarked
    
    return embarked.as_matrix()
    


# In[ ]:


embarked = prepare_embarked(data)
print("shape of embarked: ", embarked.shape)
# print(str(embarked))


# ### Normalize Features

# In[ ]:


def normalize_features(data):
    pclass = prepare_pclass(data)
    # print("plcass.shape: ", pclass.shape) 
    
    #name = prepare_name(data)
    #print("name.shape: ", name.shape)
    #print(str(name))
    
    sex = prepare_sex(data)
    # print("sex.shape: ", sex.shape)
    
    age = prepare_age(data)
    # print("age.shape: ", age.shape) 
    
    sibsp = prepare_sibsp(data)
    # print("sibsp.shape: ", sibsp.shape)
    
    parch = prepare_parch(data)
    #print("parch.shape: ", parch.shape)
    #print(str(parch))
    
    fare = prepare_fare(data)
    # print("fare.shape: ", fare.shape)
    # print(str(fare))
    
    cabin = prepare_cabin(data)
    # print("cabin.shape: ", cabin.shape)
    # print(str(cabin))
    
    
    embarked = prepare_embarked(data)
    # print("embarked.shape: ", embarked.shape)
    # print(str(embarked))
      
    #X = np.column_stack((pclass, name, sex, age, sibsp, parch, fare, cabin, embarked))
    X = np.column_stack((pclass, sex, age, sibsp, parch, fare, cabin, embarked))

    return X


# In[ ]:


# Checking to make sure no more missing values
display(data.head())
for feature in data:
    print(feature, data[feature].isnull().sum())


# ### Preparing Training & Test data

# In[ ]:


def prepare_training_data(filename):
    data = read_data(filename)
    X_train = normalize_features(data)   
    Y_train = np.reshape(data['Survived'].as_matrix(), (X_train.shape[0],1))
    
    #print("shape of X_train: ", X_train.shape)
    #print("shape of Y_train: ", Y_train.shape)
    
    return X_train.T, Y_train.T


# In[ ]:


#prepare_training_data('train.csv')


# In[ ]:


def prepare_test_data(filename):
    data = read_data(filename)
    X_test = normalize_features(data)
    #print("shape of X_test:", X_test.shape)    
    return X_test.T


# In[ ]:


#prepare_test_data('test.csv')


# ### Model Architecture
# 
#     4 layers NN
#     Input Features => 8 {plcass, sex, age, sibsp, parch, fare, cabin, embarked }
#     1st hidden layer => 20 nodes, ReLu activation
#     2nd hidden layer => 10 nodes, ReLu activation 
#     3rd hidden layer => 5 nodes, ReLu activation 
#     4th layer (output layer) => 1 node, Sigmoid activation 
#     
#     Xavier initialization 
#     AdamOptimization 
#     Learning rate = 0.0001
#     epoch = 6000
#     single batch, no regularization  
# 

# ### Create placeholders

# In[ ]:


def create_placeholders(n_x, n_y):
    X = tf.placeholder(dtype=tf.float32, shape=([n_x, None]), name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=([n_y, None]), name="Y")
    return X, Y


# ### Initialize parameters

# In[ ]:


def initialize_parameters():
    """
    Initialize parameters to build a 6 layered NN with following shapes
        W1 : [20, 8]
        b1 : [20, 1]
        W2 : [10, 20]
        b2 : [10, 1]
        W3 : [5, 10]
        b3 : [5, 1]
        W4 : [1, 5]
        b4 : [1, 1]

    Returns: 
    paramters -- a dictionary of tensors contaiing W1, b1, W2, b2, W3, b3, W4, b4,
    """
    
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [20, 8], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [20, 1], initializer = tf.zeros_initializer())
    
    W2 = tf.get_variable("W2", [10, 20], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [10, 1], initializer = tf.zeros_initializer())

    W3 = tf.get_variable("W3", [5, 10], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [5, 1], initializer = tf.zeros_initializer())
    
    W4 = tf.get_variable("W4", [1, 5], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b4 = tf.get_variable("b6", [1, 1], initializer = tf.zeros_initializer())
 
    parameters = { "W1": W1,
                   "b1": b1, 
                   "W2": W2, 
                   "b2": b2,
                   "W3": W3, 
                   "b3": b3,
                   "W4": W4,
                   "b4": b4
                 }
    return parameters


# In[ ]:


tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("W3 = " + str(parameters["W3"]))
    print("b3 = " + str(parameters["b3"]))
    print("W4 = " + str(parameters["W4"]))
    print("b4 = " + str(parameters["b4"]))
            
    


# ### Forward propogation

# In[ ]:


def forward_propagation(X, parameters):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    
    Z4 = tf.add(tf.matmul(W4, A3), b4)    
    return Z4


# ### Compute cost

# In[ ]:


def compute_cost(Z, Y): 
    
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


# ### Plot cost

# In[ ]:


def plot_cost(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations per tens')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()


# ### Train the model and make predictions for the test cases

# In[ ]:


def train_model_and_predict(X_train, Y_train, X_test, learning_rate, num_epochs):
    
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    
    costs = []
      
    X, Y = create_placeholders(n_x, n_y)
    
    parameters = initialize_parameters()
    #print("Parameters:" + str(parameters))
    
    # Forward propagation
    print("Shape of X: ", X.shape)
    Z4 = forward_propagation(X, parameters)
    cost = compute_cost(Z4, Y)
    
    # Backward propagation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)        
        for epoch in range(1, num_epochs):
            
            _, calculated_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})

            # Print cost after every epoch
            if epoch % 100 == 0:
                print("cost after epoch %i: %f" % (epoch, calculated_cost))
            if epoch % 50 == 0:
                costs.append(calculated_cost)
            
        plot_cost(costs, learning_rate)
        
        # Saving the parameeters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        
        Y_predict_prob = tf.cast(tf.nn.sigmoid(Z4), dtype=tf.float32)
        Y_predict_class = tf.cast(tf.greater(Y_predict_prob, 0.5), 'float')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_predict_class, Y ), 'float')) 
        
        prediction_train = tf.argmax(tf.nn.sigmoid(Z4))
        sess.run([prediction_train], feed_dict={X:X_train, Y:Y_train})
        print("Train Accuracy: ", accuracy.eval({X:X_train, Y:Y_train}))
        
        prediction_test = sess.run(Y_predict_class, feed_dict={X:X_test})
        # print("Predicted: : " + str(prediction_test))
        
        return prediction_test
        


# * ### Invoke the model 

# In[ ]:


ops.reset_default_graph()
from subprocess import check_output

# Clean training set
X_train, Y_train = prepare_training_data('train.csv')

# Clean test set
X_test = prepare_test_data('test.csv')

# train and predict
prediction_test = train_model_and_predict(X_train=X_train, 
                                          Y_train=Y_train, 
                                          X_test=X_test,
                                          learning_rate=0.0001, 
                                          num_epochs=6000)

# transform the values to binary
prediction_test = np.where(prediction_test < 1, 0, 1)
#print("prediction_test: " + str(prediction_test))

data = read_data('test.csv')
#display(data.head())
submission = pd.DataFrame({
    "PassengerId": data["PassengerId"],
    "Survived": prediction_test.reshape(X_test.shape[1])
})
submission[:10]
submission.to_csv('submission.csv', index=False)
#print(str(submission[:100]))

print(check_output(["ls", "."]).decode("utf8"))

