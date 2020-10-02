#!/usr/bin/env python
# coding: utf-8

# # Introduction Artificial Neural Networks using PyTorch
# 
# *15 November 2018*  
# 
# #### ***[Soumya Ranjan Behera](https://www.linkedin.com/in/soumya044)***
# 
# ### In this Kernel I have demonstrated a beginner's approach of implementing an  Artificial Neural Network (ANN) using [PyTorch](https://pytorch.org/) to classify the digits into their respective categories which have scored **97.7 %** Accuracy in the Digit Recognizer Competition.
# 
# ### Goals of this Kenel:  
# * To demonstrate a new Deep Learning framework **[PyTorch](https://pytorch.org/)**
# * To provide a basic implementation of Artificial Neural Network (ANN)
# * A beginner friendly kernel to show a procedure to compete in Kaggle Digit Recognizer Competition

# ## What is PyTorch?
# > A NEW WAY FROM RESEARCH TO PRODUCTION
# 
# PyTorch is an open source deep learning platform that provides a seamless path from research prototyping to production deployment.

# ## Key Features Of PyTorch
# * HYBRID FRONT-END
# * DISTRIBUTED TRAINING
# * PYTHON-FIRST
# * ADVANCED TOOLS & LIBRARIES
# * NATIVE ONNX SUPPORT  (Open Neural Network Exchange Support) 
# * VERY FAST PRODUCTION READY DEPLOYMENT WITH C++
# 
# Reference: [click here](https://pytorch.org/features)

# ## Dataset Overview
#  ``` 
#  ../input/
#      |_ train.csv  
#      |_ test.csv  
#      |_ sample_submission.csv```  
#      
# * ```train.csv``` contains 42k samples of 28x28 digit images with their labels
# * ```test.csv``` contains 28k samples of 28x28 digit images without labels
# * We've to predict the labels for the ```test``` samples and submit them in a csv file as shown  in ```sample_submission.csv```
# 

# **Note:** In this kernel I have provided only the implementation of ANN using PyTorch. For basics of ANN or PyTorch please refer to this free course [Intro to Deep Learning using PyTorch](https://in.udacity.com/course/deep-learning-pytorch--ud188) 

# # 1. Prepare our Data
# 
# ### Import Numpy, Pandas and Matplotlib

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Visualization
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ### Import Training data as Numpy array

# In[ ]:


dataset = pd.read_csv('../input/train.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values


# ### Let's see the distribution of data

# In[ ]:


import seaborn as sns
sns.countplot(y)


# Since all our target classes are well-balanced we can move to our next step.

# ### Visualization of our Data (Sample Images)

# In[ ]:


# Let's see some sample images
fig = plt.figure(figsize=(25,4))
fig.subplots_adjust(hspace=0.5)
for i,index in enumerate(np.random.randint(0,100,10)):
    ax = fig.add_subplot(2,5,i+1)
    ax.imshow(X[index].reshape(28,28), cmap='gray')
    ax.set_title("Label= {}".format(y[index]), fontsize = 20)
    ax.axis('off')
plt.show()


# ### Check for Null Values

# In[ ]:


# Check IF some Feature variables are NaN
np.unique(np.isnan(X))[0]


# In[ ]:


# Check IF some Target Variables are NaN
np.unique(np.isnan(y))[0]


# Since no NULL or NaN values are there, we're good to go!

# ### Splitting Dataset into Training set and Test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)


# ### Normalization
# The goal of normalization is to change the values of numeric columns in the dataset to use a common scale, without distorting differences in the ranges of values or losing information. Here we're dividing all the pixel values with 255.0 to make all pixel values lie in between 0 and 1.

# In[ ]:


# Normalization
X_train = X_train / 255.0
X_test = X_test / 255.0


# # 2. Building our ANN using PyTorch

# ### Import required libraries

# In[ ]:


import torch
import torch.utils.data
from torch.autograd import Variable


# ### Convert Numpy Arrays to Tensors 
# Here we have to convert our train and test data sets (which are in numpy array format) to tensors. Because PyTorch only takes tensors as input.

# In[ ]:


'''Create tensors for our train and test set. 
As you remember we need variable to accumulate gradients. 
Therefore first we create tensor, then we will create variable '''
# Numpy to Tensor Conversion (Train Set)
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)

# Numpy to Tensor Conversion (Train Set)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)


# ### Build Train and Test Data loaders

# In[ ]:


# Make torch datasets from train and test sets
train = torch.utils.data.TensorDataset(X_train,y_train)
test = torch.utils.data.TensorDataset(X_test,y_test)

# Create train and test data loaders
train_loader = torch.utils.data.DataLoader(train, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = 64, shuffle = True)


# ### Architect our Neural Network

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F


# Define any desired architecture with feed-forward function

# In[ ]:


class ANN(nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(ANN, self).__init__()
    
        # Input Layer (784) -> 784
        self.fc1 = nn.Linear(input_dim, 784)
        # 784 -> 128
        self.fc2 = nn.Linear(784, 128)
        # 128 -> 128
        self.fc3 = nn.Linear(128, 128)
        # 128 -> 64
        self.fc4 = nn.Linear(128, 64)
        # 64 -> 64
        self.fc5 = nn.Linear(64, 64)
        # 64 -> 32
        self.fc6 = nn.Linear(64, 32)
        # 32 -> 32
        self.fc7 = nn.Linear(32, 32)
        # 32 -> output layer(10)
        self.output_layer = nn.Linear(32,10)
        # Dropout Layer (20%) to reduce overfitting
        self.dropout = nn.Dropout(0.2)
    
    # Feed Forward Function
    def forward(self, x):
        
        # flatten image input
        x = x.view(-1, 28 * 28)
        
        # Add ReLU activation function to each layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # Add dropout layer
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        # Don't add any ReLU activation function to Last Output Layer
        x = self.output_layer(x)
        
        # Return the created model
        return x


# ### Create the Model

# In[ ]:


# Create the Neural Network Model
model = ANN(input_dim = 784, output_dim = 10)
# Print its architecture
print(model)


# ### Define Loss Function and Optimizer
# Here we've used Cross Entropy Loss Function and SGD optimizer. Feel free to experiment with other hyperparameters.

# In[ ]:


import torch.optim as optim
# specify loss function
loss_fn = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay= 1e-6, momentum = 0.9,nesterov = True)


# # 3. Train our Model with simultaneous Validation

# In[ ]:


# Define epochs (between 20-50)
epochs = 30

# initialize tracker for minimum validation loss
valid_loss_min = np.Inf # set initial "min" to infinity

# Some lists to keep track of loss and accuracy during each epoch
epoch_list = []
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []



# Start epochs
for epoch in range(epochs):
    # monitor training loss
    train_loss = 0.0
    val_loss = 0.0
    
    ###################
    # train the model #
    ###################
    # Set the training mode ON -> Activate Dropout Layers
    model.train() # prepare model for training
    # Calculate Accuracy         
    correct = 0
    total = 0
    
    # Load Train Images with Labels(Targets)
    for data, target in train_loader:
        
        # Convert our images and labels to Variables to accumulate Gradients
        data = Variable(data).float()
        target = Variable(target).type(torch.LongTensor)
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        
        # Calculate Training Accuracy 
        predicted = torch.max(output.data, 1)[1]        
        # Total number of labels
        total += len(target)
        # Total correct predictions
        correct += (predicted == target).sum()
        
        # calculate the loss
        loss = loss_fn(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
    
    # calculate average training loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)
    
    # Avg Accuracy
    accuracy = 100 * correct / float(total)
    
    # Put them in their list
    train_acc_list.append(accuracy)
    train_loss_list.append(train_loss)
    
        
    # Implement Validation like K-fold Cross-validation 
    # Set Evaluation Mode ON -> Turn Off Dropout
    model.eval() # Required for Evaluation/Test

    # Calculate Test/Validation Accuracy         
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:

            # Convert our images and labels to Variables to accumulate Gradients
            data = Variable(data).float()
            target = Variable(target).type(torch.LongTensor)

            # Predict Output
            output = model(data)

            # Calculate Loss
            loss = loss_fn(output, target)
            val_loss += loss.item()*data.size(0)
            # Get predictions from the maximum value
            predicted = torch.max(output.data, 1)[1]

            # Total number of labels
            total += len(target)

            # Total correct predictions
            correct += (predicted == target).sum()
    
    # calculate average training loss and accuracy over an epoch
    val_loss = val_loss/len(test_loader.dataset)
    accuracy = 100 * correct/ float(total)
    
    # Put them in their list
    val_acc_list.append(accuracy)
    val_loss_list.append(val_loss)
    
    # Print the Epoch and Training Loss Details with Validation Accuracy   
    print('Epoch: {} \tTraining Loss: {:.4f}\t Val. acc: {:.2f}%'.format(
        epoch+1, 
        train_loss,
        accuracy
        ))
    # save model if validation loss has decreased
    if val_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        val_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = val_loss
    # Move to next epoch
    epoch_list.append(epoch + 1)


# ## Load the Model with the Lowest Validation Loss

# In[ ]:


model.load_state_dict(torch.load('model.pt'))


# # 4.  Performance Evaluation

# ### Visualize Training Stats

# **Average Loss VS Number of Epochs Graph**

# In[ ]:


plt.plot(epoch_list,train_loss_list)
plt.plot(val_loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Number of Epochs")
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


# **Avg. Accuracy VS Number of Epochs Graph**

# In[ ]:


plt.plot(epoch_list,train_acc_list)
plt.plot(val_acc_list)
plt.xlabel("Epochs")
plt.ylabel("Training Accuracy")
plt.title("Accuracy vs Number of Epochs")
plt.legend(['Train', 'Test'], loc='best')
plt.show()


# ### Validation/Test Accuracy
# Here we're taking average of last 10 validation accuracies

# In[ ]:


val_acc = sum(val_acc_list[20:]).item()/10
print("Test Accuracy of model = {} %".format(val_acc))


# # 5. Prepare Final Submission

# ### Import test data

# In[ ]:


kaggle_test_set = pd.read_csv('../input/test.csv')

# Convert it to numpy array and Normalize it
kaggle_test_set = kaggle_test_set.values/255.0


# ### Convert our Test Data to Tensor Variable

# In[ ]:


kaggle_test_set = Variable(torch.from_numpy(kaggle_test_set)).float()


# ### Predict Labels/Targets for Test Images

# In[ ]:


# Predicted Labels will be stored here
results = []

# Set Evaluation Mode ON -> Turn Off Dropout
model.eval() # Required for Evaluation/Test

with torch.no_grad():
    for image in kaggle_test_set:
        output = model(image)
        pred = torch.max(output.data, 1)[1]
        results.append(pred[0].numpy())


# In[ ]:


# Convert List to Numpy Array
results = np.array(results)


# ### Visualiza some Test Images and their Predicted Labels

# In[ ]:


# Plot using Matplotlib
fig = plt.figure(figsize=(25,4))
fig.subplots_adjust(hspace=0.5)
for i,index in enumerate(np.random.randint(0,100,10)):
    ax = fig.add_subplot(2,5,i+1)
    ax.imshow(kaggle_test_set[index].reshape(28,28), cmap='gray')
    ax.set_title("Label= {}".format(results[index]), fontsize = 20)
    ax.axis('off')
plt.show()


# **From the visualization we can see that our model performs really Good !!!**

# ### Let's make our Final Submission CSV file

# **Convert our Results Numpy Array to Pandas Series**

# In[ ]:


results = pd.Series(results,name="Label")


# **Add an ' ImageId ' Column and Save as CSV file**

# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)


# **Just Check our format to ensure correctness**

# In[ ]:


submission.head()


# # Thank You  
# 
# If you liked this kernel please **Upvote**. Don't forget to drop a comment or suggestion.  
# 
# ### *Soumya Ranjan Behera*
# Let's stay Connected! [LinkedIn](https://www.linkedin.com/in/soumya044)  
# 
# **Happy Coding !**
