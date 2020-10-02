#!/usr/bin/env python
# coding: utf-8

# 

# **Welcome**
# 
# **Logistic Regression** is one of the best known algorithms in machine learning. It is available in **scikit - learn library**. However, it is always better to know how the algorithm actually works rather than relying only on the libraries. In libraries, one might not get an intuitive sense of how the algorithm works in the back end. Understanding the code from scratch would always be a good approach rather than just using the libraries. 
# So, I've coded the entire part without using scikit - learn library for logistic regression.I tried to simplify and use various functions (with the help of **Andrew Ng's Deep Learning specialization**) that are actually the ones that are used in scikit - learn in the back end. 
# I've coded this from scratch and the model was able to achieve** 88 percent accuracy** on the testing set that is considered to be quite good for a model coded from stratch. 
# Let us now start our journey to understand how this algorithm actually works. Before we dive into the algorithm, it is always good to **visualize the data** at hand and see if there is any relationship between various parameters. 
# 
# **Part - I**
# 
# This deals with the visualization of various input parameters and understand the data thoroughly. In addition to this, it is also important to discard some other parameters that are not worth much consideration. 
# 
# **Part - II**
# 
# This deals with the definition of various helper functions that would be later used by the main function at the end of the code. These helper functions would make the execution much simpler and divide the work flow. Therefore, just by looking at the function, one could get to know it's functionality. 
# 
# **Part - III** 
# 
# In this part of the code, we would define the main function that would take various arguments (hyperparameters) used to train the model. The body of this main function contains all the helper functions that we have already defined. 
# 
# **Part - IV**
# 
# This is the last part of the code. Here, we would define actually run the whole model with **various hyperparameters** and measure the training and the test accuracy. It turns out that the **training accuracy is 85%** (approx) while the** test accuracy is 88%**. We can see clearly from the accuracy in the test set that the model is not suffering from either overfitting (variance) or underfitting (bias) to a large extent. 
# 
# This is the outline of the entire code. Feel free to comment in the end if there is any doubt or query. I'd be glad to help!!

# In[ ]:


import pandas as pd                  #importing the library for data manipulation and storage 
import numpy as np                   #importing the library for scientific computation
import seaborn as sns                #importing the library used for interactive plots
import matplotlib.pyplot as plt      #importing the library used for plots (not high end)
from sklearn.metrics import classification_report, confusion_matrix   #importing sub components from sklearn library
from sklearn.model_selection import train_test_split   #importing train_test_split which we would use later  


# In[ ]:


df = pd.read_csv('../input/heart.csv')   #reading the csv from the directory and storing the values in df


# In[ ]:


df.head()   #having a look at the first 5 rows of the dataframe


# In[ ]:


df['target'].value_counts()     #counting the number of target variables (the number of diseased vs non-diseased heart people)


# In[ ]:


plt.scatter(x = 'chol', y = 'trestbps', color = 'green', data = df)  #using scatter plot to see the relationship between cholosterol and trestbps
plt.xlabel('cholestrol')    #giving the label to the x-axis                
plt.ylabel('trestbps')      #giving the label to the y-axis
plt.title('Cholestrol Vs Trestbps')  #giving a name to the title


# In[ ]:


df.columns                #having a look at various input features or columns 


# In[ ]:


plt.scatter(x = 'chol', y = 'age', color = 'orange', data = df)   #using scatter plot, noticing the relationship between cholestrol and age
plt.xlabel('Cholestrol')        #labeling the x-axis 
plt.ylabel('Age')               #labeling the y-axis
plt.title('Cholestrol vs Age')  #giving the title to the graph


# In[ ]:


sns.countplot(df['sex'], palette = ("RdYlGn"))           #counting the number of male and female candidates in the dataset


# In[ ]:


sns.distplot(df['age'])    #having a look at the distribution of males and females in the plot


# In[ ]:


sns.boxplot(x = 'age', palette = "BuGn", data = df)  #using box plot to see how the age is distributed


# In[ ]:


sns.jointplot(x = 'age', y = 'oldpeak', kind = 'kde', color = 'Gold', data = df)


# In[ ]:


sns.jointplot(x = 'cp', y = 'age', kind = 'kde', color ='Green', data = df)


# In[ ]:


df.head(1)         #having a look at the dataset again


# In[ ]:


X = df.drop(['target'], axis = 1)      #here, we would take all the columns except 'target' as input vector
y = df['target']                       #here, we are taking the output as the 'target' column in our dataset
ynewtest = y
xnewtest = X
y = y[:, np.newaxis]                   #converting the output to an array 
print('The shape of the input is {}'.format(X.shape))     #printing the shape of the input
print('The shape of the output is {}'.format(y.shape))    #printing the shape of the output


# Below, we would be dividing the giving input and output values into 2 sets namely the training set and the test set. We use the training set to train the model and tune the hyperparameters so that it would be efficient in the test set. In the test set, we would be measuring the accuracy of the trained model.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 101)


# In[ ]:


print('The shape of the input training set is {}'.format(X_train.shape))
print('The shape of the output training set is {}'.format(y_train.shape))
print('The shape of the input testing set is {}'.format(X_test.shape))
print('The shape of the output testing set is {}'.format(y_test.shape))


# In[ ]:


#We are initially defining the sigmoid function that could be used later
def sigmoid(z):
    
    s = 1 / (1 + np.exp(-z))
    
    return s


# In[ ]:


#This is a function that is used to initialize the weights with 0 and biases also with 0
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


# In[ ]:


#this network ensures that there is a forward propagation and at the same time, returns the cost
def propagate(w, b, X, y):
    
    m = X.shape[0]
    A = sigmoid(np.dot(X, w) + b)
    cost = -(1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A)) #computing the cost function or the error function
    dw = (1 / m) * np.dot(X.T, (A - y))   #this is derivative of the cost function with respect to w
    db = (1 / m) * np.sum(A - y)          #this is the derivative of the cost function with respect to b
    grads = {'dw': dw, 'db': db}          #these values are stored in a dictionary so as to access them later
    return grads, cost 


# In[ ]:


#We are trying to get the parameters w and b after modifying them using the knowledge of the cost function
def optimize(w, b, X, y, num_iterations, learning_rate, print_cost = False):
    costs = []                    #This is an empty list created so that it stores all the values later
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, y)       #we are calling the previously defined function 
        dw = grads['dw']                          #we are accessing the derivatives of cost with respect to w
        db = grads['db']                          #we are accessing the derivatives of cost with respect to b
        w = w - learning_rate * dw                #we are modifying the parameter w so that the cost would reduce in the long run
        b = b - learning_rate * db                #we are modifying the parameter b so that the cost would reduce in the long run
        np.squeeze(cost)
        if i % 100 == 0:
            costs.append(cost)                    #we are giving all the cost values to the empty list that was created initially
        if print_cost and i % 1000 == 0:
            print("cost after iteration {}: {}".format(i, cost))
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    params = {'w': w, 'b': db}                    #we are storing this value in the dictionary so that it could be accessed later
    grads = {'dw': dw, 'db': db}                  #we are storing these valeus in the dictionary so that they could be accessed later
    return params, grads, costs


# In[ ]:


#This is a function that gives 1 if the activation is greater that 0.5 and 0 otherwise
def predict(w, b, X):
    m = X.shape[0]
    y_prediction = np.zeros((m, 1))
    A = sigmoid(np.dot(X, w) + b)
    for i in range(A.shape[0]):
        if (A[i, 0] <= 0.5):
            y_prediction[i, 0] = 0
        else:
            y_prediction[i, 0] = 1
            
    return y_prediction


# In[ ]:


def model(X_train, X_test, y_train, y_test, num_iterations, learning_rate, print_cost = True):
    w, b = initialize_with_zeros(X.shape[1])
    parameters, grads, costs = optimize(w, b, X, y, num_iterations, learning_rate, print_cost = True)
    w = parameters["w"]
    b = parameters["b"]
    y_prediction_test = predict(w, b, X_test)
    y_prediction_train = predict(w, b, X_train)
    
    print('train accuracy: {}'.format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print('test accuracy: {}'.format("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100)))
    
    d = {"costs": costs,
         "y_prediction_test": y_prediction_test, 
         "y_prediction_train" : y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


# This is the final part of the code. Here, you would be testing the model on the test and the training set and pay attention to the accuracy. You would notice that the model was able to produce an accuracy of 85% on the training set and 88 percent on the test set. The graph simply indicates that with the number of iterations on the x-axis, there is a decrease in the cost (error) of the output in its prediction. 

# In[ ]:


d = model(X_train, X_test, y_train, y_test, num_iterations = 100000, learning_rate = 0.00015, print_cost = True)


# In[ ]:


df


# In[ ]:


xpred = xnewtest
ypred = ynewtest
i = 300         #play around with this number to access each row in the training and test set and check the accuracy
xnewpred = xpred.iloc[i]
ynewpred = ypred.iloc[i]
print('The input values of the features are:')
print(xnewpred)
print('The actual output whether a person has a heart disease or not is:')
print(float(ynewpred))
xnewpred = xnewpred[:, np.newaxis]
xnewpred = xnewpred.T
ynew = predict(d["w"], d["b"], xnewpred)
print('The output of the predicted value is:')
print(ynew[0][0])


# Now that you have executed the code upto this point, it is now time to put the training into practice. You may consider a patient's case where all the input parameters are known as shown in the dataset. Then you can use the **'predict'** function that is present in one of the cells above. You make take the result in some variable and then print it. In order to do the steps above, it is important that you run the code from the beginning so that there wouldn't be some bad errors. 
# 
# For those who are new to kaggle who want to run the code, they may click on the blue box in the top right corner called **'Fork'**. This will create a virtual environment for the code to run. Once you click on fork, you would be in the virtual environment with all the code. You just need to select the beginning cell (whether it be markdown or code cell) and press 'Shift + Enter' to run the code from the start. At the end of the code, you may also test your own case and check the outcome. 
# 
# If you have any questions, feel free to post them in the **comment section** below. I'll try to ensure that they are solved in a very short span of time. 
# 
# If you like the code, you may **up vote**. You can do this by clicking the button that is on the top right corner. It has a box with a small arrow on top. 
# Hope you enjoyed looking at the code and understanding it. See you!!!!

# In[ ]:




