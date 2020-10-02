#!/usr/bin/env python
# coding: utf-8

# # MNIST with PyTorch CNNs
# This notebook analyzes the MNIST images from the beginners competition using convolutional neural networks (CNNs) implemented in PyTorch.

# In[ ]:


# Load a few helpful modules
import numpy as np
import matplotlib.pyplot as plt
import torch

print(f'Using PyTorch v{torch.__version__}')


# # Import the data
# The first thing we need to do is import the data into the appropriate format

# In[ ]:


import pandas as pd
train = pd.read_csv("../input/digit-recognizer/train.csv")


# In[ ]:


# Construct the transform
import torchvision.transforms as transforms
from   PIL import Image
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

# Get the device we're training on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_digits(df):
    """Loads images as PyTorch tensors"""
    # Load the labels if they exist 
    # (they wont for the testing data)
    labels = []
    start_inx = 0
    if 'label' in df.columns:
        labels = [v for v in df.label.values]
        start_inx = 1
        
    # Load the digit information
    digits = []
    for i in range(df.pixel0.size):
        digit = df.iloc[i].astype(float).values[start_inx:]
        digit = np.reshape(digit, (28,28))
        digit = transform(digit).type('torch.FloatTensor')
        if len(labels) > 0:
            digits.append([digit, labels[i]])
        else:
            digits.append(digit)

    return digits


# In[ ]:


# Load the training data
train_X = get_digits(train)

# Some configuration parameters
num_workers = 0    # number of subprocesses to use for data loading
batch_size  = 64   # how many samples per batch to load
valid_size  = 0.2  # percentage of training set to use as validation

# Obtain training indices that will be used for validation
num_train = len(train_X)
indices   = list(range(num_train))
np.random.shuffle(indices)
split     = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# Define samplers for obtaining training and validation batches
from torch.utils.data.sampler import SubsetRandomSampler
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Construct the data loaders
train_loader = torch.utils.data.DataLoader(train_X, batch_size=batch_size,
                    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_X, batch_size=batch_size, 
                    sampler=valid_sampler, num_workers=num_workers)

# Test the size and shape of the output
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)


# # Visualize digits
# Let's take a look at a random sample of 20 digits to see what they look like

# In[ ]:


# Generate 20 random image indices
rand_indx = np.random.randint(num_train, size=20)

# Construct the subplot space
fig, axes = plt.subplots(4, 5, figsize=(15, 15))

# Plot the images
for i,indx in enumerate(rand_indx):
    # Get the image
    img = train_X[indx][0][0]
    
    # Get the appropriate subplot
    x  = i%5         # Subplot x-coordinate
    y  = int(i/5)    # Subplot y-coordinate
    ax = axes[y][x]
    ax.imshow(img, cmap='gray')

    # Format the plot
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(f"{train_X[indx][1]}")


# That looks pretty good! Now that we have our data and we know it's in an appropriate format, we can move on to building our model.

# # Constructing the model
# Here I will construct the PyTorch CNN model that I will ultimately train on the MNIST image dataset. I'm going to opt for using a Sequential model just for the ease of construction and editing it later on.

# In[ ]:


# Import the necessary modules
import torch.nn as nn

def calc_out(in_layers, stride, padding, kernel_size, pool_stride):
    """
    Helper function for computing the number of outputs from a
    conv layer
    """
    return int((1+(in_layers - kernel_size + (2*padding))/stride)/pool_stride)

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Some helpful values
        inputs      = [1,32,64,64]
        kernel_size = [5,5,3]
        stride      = [1,1,1]
        pool_stride = [2,2,2]

        # Layer lists
        layers = []

        self.out   = 28
        self.depth = inputs[-1]
        for i in range(len(kernel_size)):
            # Get some variables
            padding = int(kernel_size[i]/2)

            # Define the output from this layer
            self.out = calc_out(self.out, stride[i], padding,
                                kernel_size[i], pool_stride[i])

            # convolutional layer 1
            layers.append(nn.Conv2d(inputs[i], inputs[i+1], kernel_size[i], 
                                       stride=stride[i], padding=padding))
            layers.append(nn.ReLU())
            
            # convolutional layer 2
            layers.append(nn.Conv2d(inputs[i+1], inputs[i+1], kernel_size[i], 
                                       stride=stride[i], padding=padding))
            layers.append(nn.ReLU())
            # maxpool layer
            layers.append(nn.MaxPool2d(pool_stride[i],pool_stride[i]))
            layers.append(nn.Dropout(p=0.2))

        self.cnn_layers = nn.Sequential(*layers)
        
        print(self.depth*self.out*self.out)
        
        # Now for our fully connected layers
        layers2 = []
        layers2.append(nn.Dropout(p=0.2))
        layers2.append(nn.Linear(self.depth*self.out*self.out, 512))
        layers2.append(nn.Dropout(p=0.2))
        layers2.append(nn.Linear(512, 256))
        layers2.append(nn.Dropout(p=0.2))
        layers2.append(nn.Linear(256, 256))
        layers2.append(nn.Dropout(p=0.2))
        layers2.append(nn.Linear(256, 10))

        self.fc_layers = nn.Sequential(*layers2)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1, self.depth*self.out*self.out)
        x = self.fc_layers(x)
        return x
    
# create a complete CNN
model = Net()
model


# # Train the model
# Now that we have a working model, we need to train it.

# In[ ]:


import torch.optim as optim

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0005)


# In[ ]:


# number of epochs to train the model
n_epochs = 25 # you may increase this number to train a final model

valid_loss_min = np.Inf # track change in validation loss

# Additional rotation transformation
#rand_rotate = transforms.Compose([
#    transforms.ToPILImage(),
#    transforms.RandomRotation(20),
#    transforms.ToTensor()
#])

# Get the device
print(device)
model.to(device)
tLoss, vLoss = [], []
for epoch in range(n_epochs):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    #########
    # train #
    #########
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        data   = data.to(device)
        target = target.to(device)
        
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
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ############
    # validate #
    ############
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        data   = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    tLoss.append(train_loss)
    vLoss.append(valid_loss)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss


# In[ ]:


# Plot the resulting loss over time
plt.plot(tLoss, label='Training Loss')
plt.plot(vLoss, label='Validation Loss')
plt.legend();


# # Make some predictions
# Now let's load the best fit model...

# In[ ]:


model.load_state_dict(torch.load('model_cifar.pt'));


# ... and see how well it does on our validation data.

# In[ ]:


# track test loss
test_loss     = 0.0
class_correct = [0]*10
class_total   = [0]*10

model.eval()

# For generating confusion matrix
conf_matrix = np.zeros((10,10))

# iterate over test data
for data, target in valid_loader:
    # move tensors to GPU if CUDA is available
    data   = data.to(device)
    target = target.to(device)
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if device == "cpu" else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(target.size(0)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
        
        # Update confusion matrix
        conf_matrix[label][pred.data[i]] += 1

# average test loss
test_loss = test_loss/len(valid_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %3s: %2d%% (%2d/%2d)' % (
            i, 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %3s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))


# Let's generate a confusion matrix to investigate when a number is mis-classified, which number is it most likely classified as.

# In[ ]:


import seaborn as sns
plt.subplots(figsize=(10,9))
ax = sns.heatmap(conf_matrix, annot=True, vmax=20)
ax.set_xlabel('Predicted');
ax.set_ylabel('True');


# There are some non-surprises in there, such as the confusion between `4` and`9` since they are very similar in shape.

# # Make Final Predictions
# Now that we have a trained model with a reasonable amount of accuracy, let's try to make our final predictions for submission to the competition.

# In[ ]:


# Define the test data loader
test        = pd.read_csv("../input/digit-recognizer/test.csv")
test_X      = get_digits(test)
test_loader = torch.utils.data.DataLoader(test_X, batch_size=batch_size, 
                                          num_workers=num_workers)


# In[ ]:


# Create storage objects
ImageId, Label = [],[]

# Loop through the data and get the predictions
for data in test_loader:
    # Move tensors to GPU if CUDA is available
    data = data.to(device)
    # Make the predictions
    output = model(data)
    # Get the most likely predicted digit
    _, pred = torch.max(output, 1)
    
    for i in range(len(pred)):        
        ImageId.append(len(ImageId)+1)
        Label.append(pred[i].cpu().numpy())

sub = pd.DataFrame(data={'ImageId':ImageId, 'Label':Label})
sub.describe


# In[ ]:


# Write to csv file ignoring index column
sub.to_csv("submission.csv", index=False)


# We now have a 'submission.csv' file that we can submit to the competition.

# In[ ]:




