#!/usr/bin/env python
# coding: utf-8

# # **Neural Network MNIST Using Pytorch** <br>
# TeYang, Lau<br>
# Created: 12/7/2020<br>
# Last update: 12/7/2020<br>
# 
# <img src = 'https://miro.medium.com/max/6400/1*LLVL8xUiUOBE8WHgzAuY-Q.png' width="900">
# <br>
# 
# This kernel was created by following the 2nd week of the Pytorch course [here](https://www.youtube.com/watch?v=4ZZrP68yXCI&t=6728s) by JovianML and freeCodeCamp.<br>
# 
# This is my first intro to **Pytorch** and here I will be applying it onto the MNIST dataset using **Logistic Regression** and **Neural Network**.
# PyTorch is python native, and integrates easily with other python packages, which makes this a simple choice for researchers. Many researchers use Pytorch because the API is intuitive and easier to learn, and get into experimentation quickly, rather than reading through documentation.
# 
# 
# 
# The process is as follows:
# 1. [Data Loading and Structure](#Data_loading_structure)
# 2. [Preparing Train, Validation & Test Data](#Preparing_data)<br><br>
# **Logistic Regression**
# 3. [Model](#Model)
# 4. [Evaluation Metric and Loss Function](#Loss)
# 5. [Creating Model Class](#ModelClass)
# 6. [Train and Evaluate Model](#Train_model)    
# 7. [Accuracy and Loss Plots](#Accuracy_loss_plots)
# 8. [Predicting on Test Set](#Predict_test)
# 9. [Model Evaluation Metrics](#Evaluation_metrics)
# 10. [Plot Predictions against Actual Labels](#Plot_predictions)<br><br>
# **Neural Network**
# 11. [Creating Model Class](#ModelClassNN)
# 12. [Set Up GPU](#GPU)
# 13. [Train and Evaluate Model](#Train_modelNN) 
# 14. [Accuracy and Loss Plots](#Accuracy_loss_plotsNN)
# 15. [Predicting on Test Set](#Predict_testNN)

# <a id='Data_loading_structure'></a>
# ## **1. Data Loading and Structure** ##
# 
# We start by loading the dependencies and data, and exploring the dataset to look at its structure. We the print some images to get a hang of it.

# In[ ]:


import numpy as np 
import pandas as pd 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from shutil import copyfile


# In[ ]:


# download MNIST data
dataset = MNIST(root='data/', download=True)


# PyTorch datasets allow us to specify one or more transformation functions which are applied to the images as they are loaded. torchvision.transforms contains many such predefined functions, and we'll use the ToTensor transform to convert images into PyTorch tensors.

# In[ ]:


# Get the train and test set

dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())

test_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor())
print(len(dataset), len(test_dataset))


# In[ ]:


# shape of the images

img_tensor, label = dataset[0]
print(img_tensor.shape, label)


# Here we can see that the train set consists of 60k images, and the test set 10k. The images are also of size 28x28 pixels, with only 1 channel (greyscale). Let's plot an example!

# In[ ]:


# plot image

plt.imshow(img_tensor[0,:,:], cmap='gray')
print('Label:', label)


# <a id='Preparing_data'></a>
# ## **2. Preparing Train, Validation & Test Data** ##
# 
# Now it's time to prepare our training, validation and testing dataset. We do this using the *random_split* function in torch. Here we split the train dataset into 50k train and 10k validation. The train dataset will be used for training the logistic regression model while the validation dataset will be used to evaluate its performance. Hyperparameters can be tuned to improve the performance on the validation set.
# 
# 1. **Training set** - used to train the model i.e. compute the loss and adjust the weights of the model using gradient descent.
# 2. **Validation set** - used to evaluate the model while training, adjust hyperparameters (learning rate etc.) and pick the best version of the model.
# 3. **Test set** - used to compare different models, or different types of modeling approaches, and report the final accuracy of the model.

# In[ ]:


train_ds, val_ds = random_split(dataset, [50000, 10000])
len(train_ds), len(val_ds)


# Here we select a batch size of 128 to perform mini_batch gradient descent. This is a hyperparameter that can be tuned. The batch size means that the 50k train images will be divided into batches of 128 images and gradient descent will be performed on each of this 128 images in one epoch (1 runthrough of the whole data).

# In[ ]:


batch_size = 128

# shuffle so that batches in each epoch are different, and this randomization helps generalize and speed up training
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
# val is only used for evaluating the model, so no need to shuffle
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)


# Let's visualize a batch of data in a grid using the make_grid function from torchvision. We'll also use the .permute method on the tensor to move the channels to the last dimension, as expected by matplotlib.

# In[ ]:


for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break


# # Logistic Regression
# 
# <a id='Model'></a>
# ## **3. Model** ##
# 
# Now that we have prepared our data loaders, we can define our model.
# 
# A logistic regression model is almost identical to a linear regression model i.e. there are weights and bias matrices, and the output is obtained using simple matrix operations (pred = x @ w.t() + b).
# 
# We can use nn.Linear to create the model instead of defining and initializing the matrices manually.
# 
# Since nn.Linear expects the each training example to be a vector, each 1x28x28 image tensor needs to be flattened out into a vector of size 784 (28*28), before being passed into the model.
# 
# The output for each image is vector of size 10, with each element of the vector signifying the probability a particular target label (i.e. 0 to 9). The predicted label for an image is simply the one with the highest probability.

# In[ ]:


input_size = 28*28 # 784 weights to train, 1 for each pixel
num_classes = 10 # 10 outputs, 10 biases

# Logistic regression model
model = nn.Linear(input_size, num_classes)


# Let's look at the weights and biases. These are random initialized values.

# In[ ]:


print(model.weight.shape)
model.weight


# In[ ]:


print(model.bias.shape)
model.bias


# Here we create a class to initialize a model, and to rescale/flatten the input image as our image is not of shape (1, 28, 28), and we need it to be a flat vector.

# In[ ]:


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    # reshape/flatten input tensor when it is passed to model
    def forward(self, xb):
        xb = xb.reshape(-1, 784) # -1 so that it will work for different batch sizes
        out = self.linear(xb)
        return out

model = MnistModel()


# Inside the __init__ constructor method, we instantiate the weights and biases using nn.Linear. And inside the forward method, which is invoked when we pass a batch of inputs to the model, we flatten out the input tensor, and then pass it into self.linear.
# 
# xb.reshape(-1, 28*28) indicates to PyTorch that we want a view of the xb tensor with two dimensions, where the length along the 2nd dimension is 28*28 (i.e. 784). One argument to .reshape can be set to -1 (in this case the first dimension), to let PyTorch figure it out automatically based on the shape of the original tensor.
# 
# Note that the model no longer has .weight and .bias attributes (as they are now inside the .linear attribute), but it does have a .parameters method which returns a list containing the weights and bias, and can be used by a PyTorch optimizer.

# In[ ]:


print(model.linear.weight.shape, model.linear.bias.shape)
list(model.parameters())


# Let's test if our model works by inputting the first batch of data.

# In[ ]:


# check that model class works

for images, labels in train_loader:
    print('images.shape:', images.shape)
    outputs = model(images)
    break
    
print('outputs.shape:', outputs.shape)
print('Sample outputs:\n', outputs[:2].data)


# First batch of output has 128 (the batch size) * 10 (10 classes) shape. However, each of the 10 values are not probabilities that sum to 1. To do that, we will need to use the softmax function. Softmax = exp(`yi`)/sum(exp(`e^yi`))

# <img src = 'https://i.imgur.com/EAh9jLN.png' width="400">

# ## Softmax Function
# 
# Here we see that by applying the softmax function, the probabilities of all 10 classes now adds up to 1. Although it seems daunting at first, the softmax is actually quite easy to understand and implement.

# In[ ]:


prob = torch.exp(outputs[0])/torch.sum(torch.exp(outputs[0]))
prob


# In[ ]:


torch.sum(prob)


# The softmax function is included in the `torch.nn.functional` package, and requires us to specify a dimension along which the softmax must be applied.

# In[ ]:


probs = F.softmax(outputs, dim=1) # output shape is (128, 10), apply softmax to 10 class dim

# Look at sample probabilities
print('Sample probabilities:\n', probs[:2].data)

# Add up the probabilities of an output row
print('Sum:', torch.sum(probs[0]).item())


# Finally, we can determine the predicted label for each image by simply choosing the index of the element with the highest probability in each output row. This is done using `torch.max`, which returns the largest element and the index of the largest element along a particular dimension of a tensor.

# In[ ]:


max_probs, preds = torch.max(probs, dim=1) # probs shape is (128, 10), apply max to 10 class dim; max returns largest element and index of it
print(preds)
print(max_probs)


# <a id='Loss'></a>
# ## **4. Evaluation Metric and Loss Function** ##
# 
# Here we define a function to calculate the accuracy of our predictions.

# In[ ]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1) # probs shape is (128, 10), apply max to 10 class dim; max returns largest element and index of it
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds


# While the accuracy is a great way for us (humans) to evaluate the model, it can't be used as a loss function for optimizing our model using gradient descent, for the following reasons:
# 
# 1. It's not a differentiable function. `torch.max` and `==` are both non-continuous and non-differentiable operations, so we can't use the accuracy for computing gradients w.r.t the weights and biases.
# 
# 2. It doesn't take into account the actual probabilities predicted by the model, so it can't provide sufficient feedback for incremental improvements. 
# 
# Due to these reasons, accuracy is a great **evaluation metric** for classification, but not a good loss function. A commonly used loss function for classification problems is the **cross entropy**, which has the following formula:
# 
# ![cross-entropy](https://i.imgur.com/VDRDl1D.png)
# 
# While it looks complicated, it's actually quite simple:
# 
# * For each output row, pick the predicted probability for the correct label. E.g. if the predicted probabilities for an image are `[0.1, 0.3, 0.2, ...]` and the correct label is `1`, we pick the corresponding element `0.3` and ignore the rest.
# 
# * Then, take the [logarithm](https://en.wikipedia.org/wiki/Logarithm) of the picked probability. If the probability is high i.e. close to 1, then its logarithm is a very small negative value, close to 0. And if the probability is low (close to 0), then the logarithm is a very large negative value. We also multiply the result by -1, which results is a large postive value of the loss for poor predictions.
# 
# * Finally, take the average of the cross entropy across all the output rows to get the overall loss for a batch of data.
# 
# Unlike accuracy, cross-entropy is a continuous and differentiable function that also provides good feedback for incremental improvements in the model (a slightly higher probability for the correct label leads to a lower loss). This makes it a good choice for the loss function. 
# 
# As you might expect, PyTorch provides an efficient and tensor-friendly implementation of cross entropy as part of the `torch.nn.functional` package. Moreover, it also performs softmax internally, so we can directly pass in the outputs of the model without converting them into probabilities.

# In[ ]:


loss_fn = F.cross_entropy


# In[ ]:


# Loss of current batch of data

loss = loss_fn(outputs, labels) # pass outputs instead of preds as cross_entropy will apply softmax, labels will be converted to one-hot encoded vectors
print(loss)


# <a id='ModelClass'></a>
# ## **5. Creating Model Class** ##
# 
# Some parts of the training loop are specific to the specific problem we're solving (e.g. loss function, metrics etc.) whereas others are generic and can be applied to any deep learning problem. Let's impelment the problem-specific parts within our `MnistModel` class:

# In[ ]:


class MnistModel(nn.Module):
    # this is the constructor, which creates an object of class MnistModel when called
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    # reshape/flatten input tensor when it is passed to model
    def forward(self, xb):
        xb = xb.reshape(-1, 784) # -1 so that it will work for different batch sizes
        out = self.linear(xb)
        return out
    
    # this is for loading the batch of train image and outputting its loss, accuracy & predictions
    def training_step(self, batch):
        images,labels = batch
        out = self(images)                            # generate predictions
        loss = F.cross_entropy(out, labels)           # compute loss
        acc,preds = accuracy(out, labels)             # calculate accuracy
        return {'train_loss': loss, 'train_acc':acc}
       
    # this is for computing the train average loss and acc for each epoch
    def train_epoch_end(self, outputs):
        batch_losses = [x['train_loss'] for x in outputs]   # get all the batches loss
        epoch_loss = torch.stack(batch_losses).mean()       # combine losses
        batch_accs = [x['train_acc'] for x in outputs]      # get all the batches acc
        epoch_acc = torch.stack(batch_accs).mean()          # combine accuracies
        return {'train_loss': epoch_loss.item(), 'train_acc': epoch_acc.item()}
    
    # this is for loading the batch of val/test image and outputting its loss, accuracy, predictions & labels
    def validation_step(self, batch):
        images,labels = batch
        out = self(images)                       # generate predictions
        loss = F.cross_entropy(out, labels)      # compute loss
        acc,preds = accuracy(out, labels)        # calculate accuracy and get predictions
        return {'val_loss': loss, 'val_acc':acc, 'preds':preds, 'labels':labels}
    
    # this is for computing the validation average loss and acc for each epoch
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]     # get all the batches loss
        epoch_loss = torch.stack(batch_losses).mean()       # combine losses
        batch_accs = [x['val_acc'] for x in outputs]        # get all the batches acc
        epoch_acc = torch.stack(batch_accs).mean()          # combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    # this is for printing out the results after each epoch
    def epoch_end(self, epoch, train_result, val_result):
        print('Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format(epoch+1, train_result['train_loss'], train_result['train_acc'], val_result['val_loss'], val_result['val_acc']))
    
    # this is for using on the test set, it outputs the average loss and acc, and outputs the predictions
    def test_prediction(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()                           # combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()                              # combine accuracies
        batch_preds = [pred for x in outputs for pred in x['preds'].tolist()]   # combine predictions
        batch_labels = [lab for x in outputs for lab in x['labels'].tolist()]   # combine labels
        return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item(), 'test_preds': batch_preds, 'test_labels': batch_labels}       
        


# Note that `validation_step` is used for both val and test.
# 
# Next, we will define an `evaluate` function, which will perform the validation phase, a `fit` function which will perform the entire training process, and a `test_predict` function, which will use the trained model weights to evaluate the test set and return the predictions, acc, and loss.

# In[ ]:


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader] # perform val for each batch
    return model.validation_epoch_end(outputs)                       # get the results for each epoch 

def fit(model, train_loader, val_loader, epochs, lr, opt_func=torch.optim.SGD):
    history = {}
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        
        # Training phase
        train_outputs = []
        for batch in train_loader:
            outputs = model.training_step(batch)              # compute loss and accuracy
            loss = outputs['train_loss']                      # get loss
            train_outputs.append(outputs)
            loss.backward()                                   # compute gradients
            optimizer.step()                                  # update weights 
            optimizer.zero_grad()                             # reset gradients to zero
        train_results = model.train_epoch_end(train_outputs)  # get the train average loss and acc for each epoch
            
        # Validation phase
        val_results = evaluate(model, val_loader)
        
        # print results
        model.epoch_end(epoch, train_results, val_results)
                
        # save results to dictionary
        to_add = {'train_loss': train_results['train_loss'], 'train_acc': train_results['train_acc'],
                 'val_loss': val_results['val_loss'], 'val_acc': val_results['val_acc']}
        for key,val in to_add.items():
            if key in history:
                history[key].append(val)
            else:
                history[key] = [val]
                
    return history


def test_predict(model, test_loader):
    outputs = [model.validation_step(batch) for batch in test_loader] # perform testing for each batch
    results = model.test_prediction(outputs)                          # get the results
    print('test_loss: {:.4f}, test_acc: {:.4f}'.format(results['test_loss'], results['test_acc']))
    return results['test_preds'], results['test_labels']


# <a id='Train_model'></a>
# ## **6. Train and Evaluate Model** ##
# 
# It's time to train and evaluate our model on the entire train and validation sets.
# 
# Configurations like `batch size`, `learning rate` etc. need to picked in advance while training machine learning models, and are called **hyperparameters**. Picking the right hyperparameters is critical for training an accurate model within a reasonable amount of time, and is an active area of research and experimentation. Feel free to try different learning rates and see how it affects the training process.

# In[ ]:


# Hyperparameters
lr = 0.001
num_epochs = 10

model = MnistModel()  
history = fit(model, train_loader, val_loader, num_epochs, lr)


# That's a great result! With just 10 epochs of training, our model has reached an accuracy of over 80% on the validation set.

# <a id='Accuracy_loss_plots'></a>
# ## **7. Accuracy and Loss Plots** ##

# In[ ]:


# Plot Accuracy and Loss 
epochs=10

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,epochs+1))
ax1.plot(epoch_list, history['train_acc'], label='Train Accuracy')
ax1.plot(epoch_list, history['val_acc'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, epochs+1, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history['train_loss'], label='Train Loss')
ax2.plot(epoch_list, history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epochs+1, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")


# It's quite clear from the above picture that the model probably won't cross the accuracy threshold of 90% even after training for a very long time. One possible reason for this is that the learning rate might be too high. It's possible that the model's paramaters are "bouncing" around the optimal set of parameters that have the lowest loss. We can try reducing the learning rate and training for a few more epochs to see if it helps.
# 
# The more likely reason is that **the model just isn't powerful enough**. If you remember our initial hypothesis, we have assumed that the output (in this case the class probabilities) is a **linear function** of the input (pixel intensities), obtained by perfoming a matrix multiplication with the weights matrix and adding the bias. This is a fairly weak assumption, as there may not actually exist a linear relationship between the pixel intensities in an image and the digit it represents. While it works reasonably well for a simple dataset like MNIST (getting us to >80% accuracy), we need more sophisticated models (e.g., **Neural Networks**) that can capture non-linear relationships between image pixels and labels for complex tasks like recognizing everyday objects, animals etc. 
# 

# <a id='Predict_test'></a>
# ## **8. Predicting on Test Set** ##
# 
# It's time to test our model on the test set and see how well it performs on data that **it has not seen before**.

# In[ ]:


test_loader = DataLoader(test_dataset, batch_size=256)
preds,labels = test_predict(model, test_loader)


# We see that it performs about the same as the validation set. We expect this to be similar to the accuracy/loss on the validation set. If not, we might need a better validation set that has similar data and distribution as the test set (which often comes from real world data).
# 
# Next, we can output individual sample prediction and its associated label:

# In[ ]:


img_num = 100
img_tensor, label = test_dataset[img_num]
plt.imshow(img_tensor[0,:,:], cmap='gray')
print('Label:', label, 'Prediction:', preds[img_num])


# <a id='Evaluation_metrics'></a>
# ## **9. Model Evaluation Metrics** ##
# 
# Next, we get a measure of how well our model is performing by evaluating several metrics of the predictions against the actual target_labels. It seems to have a good balance between precision and recall.

# In[ ]:


# Evaluate Model Performance

# copy .py file into the working directory (make sure it has .py suffix)
copyfile(src = "../input/model-evaluation-utils/model_evaluation_utils.py", dst = "../working/model_evaluation_utils.py")

from model_evaluation_utils import get_metrics

get_metrics(true_labels=labels,
            predicted_labels=preds)


# <a id='Plot_predictions'></a>
# ## **10. Plot Predictions against Actual Labels** ##
# 
# Here, we plot out a random subset of the test dataset and see how well it performs on them. Accuracy is pretty good, with some occasional mistakes on nubmers that look similar. These are expected from a linear model.

# In[ ]:


idxs = torch.randint(0, len(test_dataset)+1, (10,)).data # select random test images indices

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30,12))
for c,i in enumerate(idxs):
    img_tensor, label = test_dataset[i]
    ax[c//5][c%5].imshow(img_tensor[0,:,:], cmap='gray')
    ax[c//5][c%5].set_title('Label: {}, Prediction: {}'.format(label, preds[i]), fontsize=25)
    ax[c//5][c%5].axis('off')


# # Neural Network
# 
# <a id='ModelClassNN'></a>
# ## **11. Creating Model Class** ##
# 
# To improve upon [logistic regression](https://jvn.io/aakashns/a1b40b04f5174a18bd05b17e3dffb0f0), we'll create a **neural network** with one **hidden layer**. Here's what this means:
# 
# * Instead of using a single `nn.Linear` object to transform a batch of inputs (pixel intensities) into a batch of outputs (class probabilities), we'll use two `nn.Linear` objects. Each of these is called a layer in the network. 
# 
# * The first layer (also known as the hidden layer) will transform the input matrix of shape `batch_size x 784` into an intermediate output matrix of shape `batch_size x hidden_size`, where `hidden_size` is a preconfigured parameter (e.g. 32 or 64).
# 
# * The intermediate outputs are then passed into a non-linear *activation function*, which operates on individual elements of the output matrix.
# 
# * The result of the activation function, which is also of size `batch_size x hidden_size`, is passed into the second layer (also knowns as the output layer), which transforms it into a matrix of size `batch_size x 10`, identical to the output of the logistic regression model.
# 
# Introducing a hidden layer and an activation function allows the model to learn more complex, multi-layered and non-linear relationships between the inputs and the targets. Here's what it looks like visually:
# 
# ![](https://i.imgur.com/vDOGEkG.png)
# 
# The activation function we'll use here is called a **Rectified Linear Unit** or **ReLU**, and it has a really simple formula: `relu(x) = max(0,x)` i.e. if an element is negative, we replace it by 0, otherwise we leave it unchanged.
# 
# <img src = 'https://i.ytimg.com/vi/DDBk3ZFNtJc/maxresdefault.jpg' width="600">
# 
# <br><br>
# To define the model, we extend the `nn.Module` class, just as we did with logistic regression.

# In[ ]:


class MnistModel_NN(nn.Module):
    # this is the constructor, which creates an object of class MnistModel_NN when called
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(input_size, hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, num_classes)
    
    # reshape/flatten input tensor when it is passed to model
    def forward(self, xb):
        xb = xb.reshape(-1, 784) # -1 so that it will work for different batch sizes
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        return out
    
    # this is for loading the batch of train image and outputting its loss, accuracy & predictions
    def training_step(self, batch):
        images,labels = batch
        out = self(images)                            # generate predictions
        loss = F.cross_entropy(out, labels)           # compute loss
        acc,preds = accuracy(out, labels)             # calculate accuracy
        return {'train_loss': loss, 'train_acc':acc}
       
    # this is for computing the train average loss and acc for each epoch
    def train_epoch_end(self, outputs):
        batch_losses = [x['train_loss'] for x in outputs]   # get all the batches loss
        epoch_loss = torch.stack(batch_losses).mean()       # combine losses
        batch_accs = [x['train_acc'] for x in outputs]      # get all the batches acc
        epoch_acc = torch.stack(batch_accs).mean()          # combine accuracies
        return {'train_loss': epoch_loss.item(), 'train_acc': epoch_acc.item()}
    
    # this is for loading the batch of val/test image and outputting its loss, accuracy, predictions & labels
    def validation_step(self, batch):
        images,labels = batch
        out = self(images)                       # generate predictions
        loss = F.cross_entropy(out, labels)      # compute loss
        acc,preds = accuracy(out, labels)        # calculate accuracy and get predictions
        return {'val_loss': loss.detach(), 'val_acc':acc, 'preds':preds, 'labels':labels} # detach extracts only the needed number, or other numbers will crowd memory
    
    # this is for computing the validation average loss and acc for each epoch
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]     # get all the batches loss
        epoch_loss = torch.stack(batch_losses).mean()       # combine losses
        batch_accs = [x['val_acc'] for x in outputs]        # get all the batches acc
        epoch_acc = torch.stack(batch_accs).mean()          # combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    # this is for printing out the results after each epoch
    def epoch_end(self, epoch, train_result, val_result):
        print('Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format(epoch+1, train_result['train_loss'], train_result['train_acc'], val_result['val_loss'], val_result['val_acc']))
    
    # this is for using on the test set, it outputs the average loss and acc, and outputs the predictions
    def test_prediction(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()                           # combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()                              # combine accuracies
        batch_preds = [pred for x in outputs for pred in x['preds'].tolist()]   # combine predictions
        batch_labels = [lab for x in outputs for lab in x['labels'].tolist()]   # combine labels
        return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item(), 'test_preds': batch_preds, 'test_labels': batch_labels}       
        


# <a id='GPU'></a>
# ## **12. Set Up GPU** ##

# In[ ]:


torch.cuda.is_available()


# In[ ]:


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
device = get_default_device()
device


# In[ ]:


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# In[ ]:


for images, labels in train_loader:
    print(images.shape)
    images = to_device(images, device)
    print(images.device)
    break


# Here, we define a `DeviceDataLoader` class to wrap our existing data loaders and move data to the selected device, as batches are accessed. Interestingly, we don't need to extend an existing class to create a PyTorch dataloader. All we need is an `__iter__` method to retrieve batches of data, and an `__len__` method to get the number of batches.

# In[ ]:


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device) # yield will stop here, perform other steps, and the resumes to the next loop/batch

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[ ]:


train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)


# Tensors that have been moved to the GPU's RAM have a device property which includes the word cuda. Let's verify this by looking at a batch of data from valid_dl.

# In[ ]:


for xb, yb in val_loader:
    print('xb.device:', xb.device)
    print('yb:', yb)
    break


# Before we train the model, we need to ensure that the data and the model's parameters (weights and biases) are on the same device (CPU or GPU). We can reuse the to_device function to move the model's parameters to the right device.

# In[ ]:





# In[ ]:


# Hyperparameters
input_size = img_tensor.shape[1] * img_tensor.shape[2] #728
hidden_size = 128
lr = 0.1
num_epochs = 10

modelNN = MnistModel_NN(input_size, hidden_size, num_classes=10)  
to_device(modelNN, device) # move model parameters to the same device


# In[ ]:


history = fit(modelNN, train_loader, val_loader, num_epochs, lr)


# We can see that the model outperforms simple logistic regression by more than 10%, showing the power of neural networks.

# <a id='Accuracy_loss_plotsNN'></a>
# ## **14. Accuracy and Loss Plots** ##

# In[ ]:


# Plot Accuracy and Loss 
epochs=10

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,epochs+1))
ax1.plot(epoch_list, history['train_acc'], label='Train Accuracy')
ax1.plot(epoch_list, history['val_acc'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, epochs+1, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history['train_loss'], label='Train Loss')
ax2.plot(epoch_list, history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, epochs+1, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")


# <a id='Predict_testNN'></a>
# ## **15. Predicting on Test Set** ##

# In[ ]:


test_loader = DeviceDataLoader(test_loader, device)
preds,labels = test_predict(modelNN, test_loader)


# In[ ]:


# Evaluate Model Performance

get_metrics(true_labels=labels,
            predicted_labels=preds)


# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30,12))
for c,i in enumerate(idxs):
    img_tensor, label = test_dataset[i]
    ax[c//5][c%5].imshow(img_tensor[0,:,:], cmap='gray')
    ax[c//5][c%5].set_title('Label: {}, Prediction: {}'.format(label, preds[i]), fontsize=25)
    ax[c//5][c%5].axis('off')


# Our model now performs better on the test set!
