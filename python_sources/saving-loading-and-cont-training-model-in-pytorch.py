#!/usr/bin/env python
# coding: utf-8

# # Saving and Loading Models in PyTorch
# 
# In this notebook, I'll show you how to save, load and continue to train models with PyTorch.
# 
# [1] and [2] provide a good start for me to expands on this topic.
# 
# Imagine a case when you have been training your model for hours and suddenly the machine crashes or you lose connection to the remote GPU that you have been training your model on. Disaster right? Consider another case that you trained your model for certain epochs which already took a considerable amount of time, but you are not satisfied with the performance and you wish you had trained it for more epochs[2].
# 
# * If you follow along please, make sure checkpoint and best_model directories are created in your work directory:
#     * I am going to save latest checkpoint in checkpoint directory
#     * I am going to save best model/checkpoint in best_model directory
# 
# Reference: 
# - [[1]](https://www.kaggle.com/davidashraf/saving-and-loading-models-in-pytorch) https://www.kaggle.com/davidashraf/saving-and-loading-models-in-pytorch
# - [[2]](https://medium.com/analytics-vidhya/saving-and-loading-your-model-to-resume-training-in-pytorch-cb687352fa61) https://medium.com/analytics-vidhya/saving-and-loading-your-model-to-resume-training-in-pytorch-cb687352fa61
# - [[3]](https://pytorch.org/tutorials/beginner/saving_loading_models.html) https://pytorch.org/tutorials/beginner/saving_loading_models.html

# In[ ]:


get_ipython().run_line_magic('pwd', '')


# In[ ]:


get_ipython().run_line_magic('ls', '')


# In[ ]:


# uncomment if you want to create directory checkpoint, best_model
get_ipython().run_line_magic('mkdir', 'checkpoint best_model')


# In[ ]:


# uncomment if you want to remove folder best_model and checkpoint
# %rm -r best_model/ checkpoint/


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np


# In[ ]:


# check if CUDA is available
use_cuda = torch.cuda.is_available()


# # Define your model

# In[ ]:


# Define your network ( Simple Example )
class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 784
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64,10)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = F.log_softmax(self.fc5(x), dim=1)
        return x


# # Prepare the dataset 

# In[ ]:


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform)

# Download and load the test data
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform)

loaders = {
    'train' : torch.utils.data.DataLoader(trainset,batch_size = 64,shuffle=True),
    'test'  : torch.utils.data.DataLoader(testset,batch_size = 64,shuffle=True),
}


# # Train the network

# In[ ]:


# Create the network, define the criterion and optimizer
model = FashionClassifier()
# move model to GPU if CUDA is available
if use_cuda:
    model = model.cuda()

print(model)


# In[ ]:


#define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[ ]:


import torch
import shutil
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)


# In[ ]:


def train(start_epochs, n_epochs, valid_loss_min_input, loaders, model, optimizer, criterion, use_cuda, checkpoint_path, best_model_path):
    """
    Keyword arguments:
    start_epochs -- the real part (default 0.0)
    n_epochs -- the imaginary part (default 0.0)
    valid_loss_min_input
    loaders
    model
    optimizer
    criterion
    use_cuda
    checkpoint_path
    best_model_path
    
    returns trained model
    """
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input 
    
    for epoch in range(start_epochs, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['test']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
        # calculate average losses
        train_loss = train_loss/len(loaders['train'].dataset)
        valid_loss = valid_loss/len(loaders['test'].dataset)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model


# In[ ]:


trained_model = train(1, 3, np.Inf, loaders, model, optimizer, criterion, use_cuda, "./checkpoint/current_checkpoint.pt", "./best_model/best_model.pt")


# In[ ]:


get_ipython().run_line_magic('ls', './best_model/')


# In[ ]:


get_ipython().run_line_magic('ls', './checkpoint/')


# ---
# ## Saving the model
# The parameters for PyTorch networks are stored in a model's `state_dict`. It includes the parameters matrices for each of the layers. If we just want to save the model, we can just save the `state_dict`

# In[ ]:


#torch.save(checkpoint, 'checkpoint.pth')


# In our case, we want to save a checkpoint that allow us to use these information continue our model traning. Here are the information needed:
# * epoch: a measure of the number of times all of the training vectors are used once to update the weights.
# * valid_loss_min: the minium validation loss, this is needed so that when we continue the training, we can start with this rather than np.Inf value
# * state_dict: model archiecture information, it includes the parameters matrices for each of the layers.
# * optimizer: You need to save optimizer parameters especially when you are using Adam as your optimizer. Adam is an adaptive learning rate method, which means, it computes individual learning rates for different parameters which we would need if we would like to continue your training from where we left off [2].
# 

# In[ ]:


"""
checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
"""


# In[ ]:


# in train method, above we use this to save checkpoint file
# save_ckp(checkpoint, False, checkpoint_path, best_model_path)


# In[ ]:


# in train method, above we use this to save best_model file
# save_ckp(checkpoint, False, checkpoint_path, best_model_path)


# ---
# # Loading the model

# Loading is as simple as Saving.
# * Reconstruct the model
# * Load the state dict to the model
# * Freeze the parameters and enter evaluation mode if you're loading for inference
# 

# In[ ]:


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


# In[ ]:


get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('ls', '')


# In[ ]:


model = FashionClassifier()
# move model to GPU if CUDA is available
if use_cuda:
    model = model.cuda()

print(model)


# In[ ]:


optimizer = optim.Adam(model.parameters(), lr=0.001)
ckp_path = "./checkpoint/current_checkpoint.pt"
model, optimizer, start_epoch, valid_loss_min = load_ckp(ckp_path, model, optimizer)


# In[ ]:


print("model = ", model)
print("optimizer = ", optimizer)
print("start_epoch = ", start_epoch)
print("valid_loss_min = ", valid_loss_min)
print("valid_loss_min = {:.6f}".format(valid_loss_min))


# After we load all the information needed, we can continue training, start_epoch = 4; previously we train the model from 1 to 3

# In[ ]:


trained_model = train(start_epoch, 6, valid_loss_min, loaders, model, optimizer, criterion, use_cuda, "./checkpoint/current_checkpoint.pt", "./best_model/best_model.pt")


# * Noticed, epoch now start from 4 to 6. 
# * The validation loss value from epoch 4 is the validation loss minimum we have in the previous training that we get from the saved checkpoint. 

# # Inference
# Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results [3].

# In[ ]:


trained_model.eval()


# In[ ]:


test_acc = 0.0
for samples, labels in loaders['test']:
    with torch.no_grad():
        samples, labels = samples.cuda(), labels.cuda()
        output = trained_model(samples)
        # calculate accuracy
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(labels)
        test_acc += torch.mean(correct.float())

print('Accuracy of the network on {} test images: {}%'.format(len(testset), round(test_acc.item()*100.0/len(loaders['test']), 2)))

