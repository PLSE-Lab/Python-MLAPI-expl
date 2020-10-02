#!/usr/bin/env python
# coding: utf-8

# **Titanic Neuron Network on Numpy ONLY ** 
# 
# In this kernel, I would like to show a basic solution to the famous Kaggle Titanic Challenge, by using hand-coded Neuron Network, which will use only numpy to run. Of course, libraries like PyTorch or Keras can be extremely useful, as they provide fast and convienient way to construct a ML model, however, I think it is extremely vital to be able to code those networks by yourself and understand the mathematical principals behind them to be able to fully use their potential. If you are only interested in the network itself, you can skip straight to the Neuron Network chapter. Additional chapters of this kernel are for people new to Kaggle or Python and are showing how to import and preprocess data and also how to create a submission.

# **1. Data Import and Preprocessing**
# 
# First we need to load the data from .csv file. I will create a dictionary filled with numpy vector containing the data just for the sake of convienince.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Data import from the csv file to a dictionary with numpy arrays
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
#Shuffle dataset
train_data = train_data.sample(frac=1)

columns = dict()
for column in train_data.columns:
    columns[column] = np.array(train_data[column]).reshape(-1,1)


# To help the network with handling the data, we need to first preprocess it. Lets define functions that will be later used for this purpose:

# In[ ]:


from scipy import stats #Needed for convienient mode function

def binarize(vector):
    #Get rid of nan values and replace them with the most common value in the whole feature
    for i in range(len(vector)):
        try: 
            if np.isnan(np.array(vector[i],dtype=np.float64)): vector[i] = stats.mode(vector).mode
        except: 
            pass
    
    #Create new matrix storing the binary endcoded vector
    binarized_matrix = np.zeros(shape=(vector.shape[0],len(np.unique(vector))))
    
    for no,value in enumerate(np.unique(vector)):
        for entry in range(vector.shape[0]):
            if vector[entry] == value: binarized_matrix[entry,no] = 1
        
    #Return binarized matrix without the last column, to avoid the curse of dimensionality
    return binarized_matrix[:,:len(np.unique(vector))-1]
    
def standarize(vector,mean,std):
    #Get rid of nan values and replace them with mean
    vector[np.isnan(vector)==True] = np.nanmean(vector)
    
    #Subtract mean and divide by variance
    vector -= mean
    vector /= std
    
    return vector


# Now lets apply those function to our data, and then concatenate it to one matrix X containing everything that is going to be treated as an input to our network. Then we will divide X into train set, cross validation set and test set.

# In[ ]:


#Continuous features are Age and Fare, all of them should be standarized
Age_mean = np.nanmean(columns['Age'])
Age_std = np.nanstd(columns['Age'])
Fare_mean = np.nanmean(columns['Fare'])
Fare_std = np.nanstd(columns['Fare'])
columns['Age'] = standarize(columns['Age'],Age_mean,Age_std)
columns['Fare'] = standarize(columns['Fare'],Fare_mean,Fare_std)

#Discrete features are: Pclass, Sex, SibSp, Parch,Embarked, out of which Pclass, Sex and Embarked should be binarized
columns['Pclass'] = binarize(columns['Pclass'])
columns['Sex'] = binarize(columns['Sex'])
columns['Embarked'] = binarize(columns['Embarked'])

#All of the features mentioned above shall be concatenated into one array:
X = columns['Age']
for column in {'Embarked','Fare','Sex','SibSp','Parch','Pclass'}:
    X = np.concatenate((X,columns[column]),axis=1)

Y = columns['Survived']

#Divide into subsets:
m = X.shape[0]
X_train = X[:(int)(m*0.6),:]
X_cv = X[(int)(m*0.6):(int)(m*0.8),:]
X_test = X[(int)(m*0.8):,:]

Y_train = Y[:(int)(m*0.6),:]
Y_cv = Y[(int)(m*0.6):(int)(m*0.8),:]
Y_test = Y[(int)(m*0.8):,:]


# **2. Coding the Neuron Network**
# 
# For our neuron network we are gonna need two activation function, sigmoid and relu. Lets first define them as well as their gradients that will later be usefull for backpropagation:

# In[ ]:


def relu(x):
    x = np.copy(x)
    x[x<0] = 0
    return x

def relu_backward(x):
    x = np.copy(x)
    x[x<0] = 0
    x[x>=0] = 1
    return x

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_backward(x):
    return sigmoid(x)*(1-sigmoid(x))


# I will now define the neuron network class. While creating an object of this class, user has to specify the number of input features, size of the hidden layer, as well as the size of output. The class will have the forward() function performing forward propagation, backward() function for backward prop. and storing gradients in internal variables w1_grad and w2_grad, compute_cost() to compute the current cost of the model, as well as update_weights() which updates weights based on the functions stored in the network's memory.
# 
# My proposed architecture of the network is linear -> relu -> linear -> sigmoid, if you want you can modify my code and change the network's architecture.
# 

# In[ ]:


EPSILON = 1e-10 #To avoid dividing by zero and taking logs from negative numbers

#Function for random initialization of weights:
def initialize_weights(m,n):
    w = np.random.randn(m,n)
    return w

class Network:
    def __init__(self,in_size,hidden_size,out_size):
        
        self.in_1 = None #Input to first layer
        self.w1 = initialize_weights(in_size + 1,hidden_size) #Weights of first layer (+1 because of bias unit)
        self.w1_grad = np.zeros_like(self.w1) #Gradients of w1
        self.z_1 = None #Output of first layer
        self.out_1 = None #Output of first layer after activation
        
        self.in_2 = None #Input to second layer
        self.w2 = initialize_weights(hidden_size + 1,out_size) #Weights of second layer (+1 because of bias unit)
        self.w2_grad = np.zeros_like(self.w2) #Gradients of w2
        self.z_2 = None #Output of second layer
        self.out_2 = None #Output of second layer after activation
    
  
    def forward(self,X):
        
        #First layer--------------------------
        self.in_1 = np.concatenate((np.ones(shape=(X.shape[0],1)),X),axis=1) #Add bias unit
        
        self.z_1 = np.dot(self.in_1,self.w1) #matmul
        
        self.out_1 = relu(self.z_1) #relu
        #--------------------------------------
        
        #Second layer--------------------------
        self.in_2 = np.concatenate((np.ones(shape=(self.out_1.shape[0],1)),self.out_1),axis=1) #Add bias unit
        
        self.z_2 = np.dot(self.in_2,self.w2) #matmul
        
        self.out_2 = sigmoid(self.z_2) #sigmoid
        
        #--------------------------------------
        
        return self.out_2

    def compute_cost(self,h,Y):
        #Compute cost based on cross entropy
        
        loss = - (Y * np.log(h+EPSILON) +  (1-Y)*np.log(1-h+EPSILON))/Y.shape[0]
        
        return sum(loss)
    
    def backward(self,h,Y):
        #Backprop through cost:
        delta = - (Y * 1/(h+EPSILON) - (1-Y) * 1/(1-h+EPSILON))/Y.shape[0]
        
        #Second layer----------------------------------------------------
        delta = sigmoid_backward(self.z_2)*delta #Backprop through sigmoid
        
        self.w2_grad = np.dot(self.in_2.T,delta) #Backprop through matmul
        
        delta = np.dot(delta,self.w2.T)[:,1:] #Backprop the gradients to next layer
        #----------------------------------------------------------------
        
        
        #First layer-----------------------------------------------------
        delta = relu_backward(self.z_1)*delta #Backprop through relu
        
        self.w1_grad = np.dot(self.in_1.T,delta) #Backprop through matmul
        
        
        #----------------------------------------------------------------
    
    def update_weights(self,learning_rate):
        self.w1 -= learning_rate * self.w1_grad
        self.w2 -= learning_rate * self.w2_grad


# **3. Fitting the model**
# 
# Now lets try to create and train a model based on the Network class created earlier. There are 9 input features and we want to have 1 output; we also need to choose the size of the hidden layer. There is a very important trade-off here: if we add more neurons the network will be larger, slower and more prone to overfitting, but it will also be able to find more correlations. Here, we have very little features and also not many examples, so the network performance is highly dependent on the initial values of weights, which are chosen randomly. We need to have enough layers so that the weights have a chance to initialize in beneficial ways for at least some of the neurons. I found out that the size of 18 (number of features twice) does the job quite well.

# In[ ]:


model = Network(9,18,1)
cost = []
cost_cv = []


# I will now proceed to the training process. We will need a loop that excecutes forward and then backward propagation. This loop shall also store the cost history of training and cross validation set. I found out that the model fits the data quite good after 30K iterations with learning rate (lr) equal to 1e-3, which is being decreased to 3e-4 after 15K iterations have been executed. 

# In[ ]:


lr = 1e-3 #Learning rate
maxit = 30000 #Number of iterations

for it in range(maxit):
    h = model.forward(X_train)
    model.backward(h,Y_train)
    model.update_weights(lr)
    
    cost.append(model.compute_cost(h,Y_train))
    cost_cv.append(model.compute_cost(model.forward(X_cv),Y_cv))
    
    #Decrease learning rate after 15K iterations
    if it==15000: lr /=3


# Lets now look at what happend to the cost during the learning process:

# In[ ]:


import matplotlib.pyplot as plt #Needed for plotting

plt.plot(cost)
plt.plot(cost_cv)
plt.show()


# As we see, the cost on both training and cv set has been significantly decreased. As I have chosen the network's architecture and training parameters based on the cv set, lets now see how network behaves on a completely unseen data. To estimate that I will measure its prediction's accuracy:

# In[ ]:


TRESHOLD = 0.5
def accuracy(h,Y):
    r = np.zeros_like(h)
    r[h>TRESHOLD] = 1
    return sum(r==Y)/Y.shape[0]

print('Accuracy on the test set is equal to: ',accuracy(model.forward(X_test),Y_test))


# **4.Making a submission**
# 
# As we already have a model that can make quite a good predictions on an unseen example, lets now submit predictions of the unlabelled data. 

# In[ ]:


#Importing and preprocessing data in the same way as before
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_columns = dict()
for column in test_data.columns:
    test_columns[column] = np.array(test_data[column]).reshape(-1,1)

#Continuos data need to be standarized using the same mean and std as before
test_columns['Age'] = standarize(test_columns['Age'],Age_mean,Age_std) 
test_columns['Fare'] = standarize(test_columns['Fare'],Fare_mean,Fare_std)

test_columns['Pclass'] = binarize(test_columns['Pclass'])
test_columns['Sex'] = binarize(test_columns['Sex'])
test_columns['Embarked'] = binarize(test_columns['Embarked'])

test = test_columns['Age']
for column in {'Embarked','Fare','Sex','SibSp','Parch','Pclass'}:
    test = np.concatenate((test,test_columns[column]),axis=1)


# Now to make a prediction:

# In[ ]:


predicted_scores = model.forward(test)

predicted_labels = np.zeros_like(predicted_scores)
predicted_labels[predicted_scores>TRESHOLD] = 1


# Now we just need to convert the data to a suitable format:

# In[ ]:


index = np.arange(test.shape[0]) + 892
submission = np.concatenate((index.reshape(-1,1),predicted_labels),axis=1)
submission = np.array(submission,dtype=np.int32)


# And save it to .csv file using pandas:

# In[ ]:


pd.DataFrame(submission,columns=['PassengerId','Survived']).to_csv('Submission.csv',index=False)

