#!/usr/bin/env python
# coding: utf-8

# **<font size="5">DSA 8021 Assignment: Deep Learning for Image Classification</font>**
# 
# **<font size="3">Student number: 40195086</font>**

# This notebook will demonstrate the coding of the three models specified from the assignment, which are Fully Connected Network (FCN), Convolutional Neural Networks (CNN), and Recurrent Neural Network (RNN). As the reader goes through the notebook, brief comments of the codes used and the motives will be given. Further information can be referred from the word document report attached with the assignment. 

# **<font size="5">FCN Model</font>**

# The first of our model here is the FCN model (Fully Connected Network). First of all, unsurprisingly, we import the necessary packages needed for our model, mainly Pytorch and its torchvision, and Matplotlib.pyplot for image plotting. 

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torchvision


# First of all, we define our FCN model (FeedForward in our case). Our model inherits from the subclass of Pytorch's nn.Module, much like any other model. Like theory, we introduce multiple layers within our network, which in our case, the nn.Linear function. Since our image is coloured with a size of 96 by 96, the first layer's input would naturally be 3 * 96 * 96 (3 channels since its RGB, and 96 by 96) followed by the first layer of output we wish to generate, which in our case, 200. The output of the first layer would naturally be the input of our 2nd layer, hence the equal numbers and followed by 100 outputs. The number of layers also play a role in the performance of our model, but since our model performed fairly well after the third layer, the initial 4th model was removed subsequently. 
# 
# The forward method, a.k.a running the forward pass, is called to put the neural network to use in making a prediction. The first argument: flattened, is to flatten to image, to obtain a consistent tensor dimension we wish to use since the dimension of images are defined by batch_size x height x width, which could be [-1, 96 96] or [-1, 1, 96, 96] where the -1 is a placeholder for "unknown" dimension size. 
# Subsequently, we run the forward pass of the flattened image to our first layer of activation (F.ReLU), the Rectified Linear Unit. Repeat the steps depending on the layer introduced previously and then an output can be generated on the final layer of activation. Note log_softmax is also implemented to ensure a much more stable output. 

# In[ ]:


class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.layer1 = nn.Linear(3*96 * 96 , 200)
        self.layer2 = nn.Linear(200, 100)
        self.layer3 = nn.Linear(100, 3) # The 3 here implies that we wisht to have 3 final classifications for our model.
    def forward(self, img):
        #print(img.shape)
        flattened = img.view(-1, 3*96 * 96)
        activation1 = F.relu(self.layer1(flattened))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return  F.log_softmax(output)

model = FeedForward()


# **<font size="5">Data Loading</font>**
# 
# Importing the necessary datasets for our model. Note that the images were transferred to Tensor forms, and Normalize, in order to ensure better processing of our model on the images in later stages. 

# In[ ]:


import torchvision.transforms as transforms

trainTransform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
data_path = 'Data/assignment_train/'
train_dataset = torchvision.datasets.ImageFolder(
    root = data_path,
    transform = trainTransform)

train_iter = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = 64,
    num_workers = 0,
    shuffle = True
)


# Calling a brief overview of our images. The images were changed to grayscale for quicker processing 

# In[ ]:


for i, (images, labels) in enumerate(train_iter):
    if i >= 15:
        break
    plt.subplot(3, 5, i+1)
    imgb = images[0, :,:]
    print(imgb.shape)
    
    plt.imshow(imgb[0,:,:], cmap='gray')
    plt.show()
    print(labels)

        


# Accordingly, we define our training model (train). Our Loss function, the criterion, uses the Cross Entropy Loss, a more general method compared to the Binary Cross Entropy Loss. The Binary Cross-Entropy Loss is much significant when it comes to, unsurprisingly, binary classification. The optimizer includes the learning rate our model operates on. Initially, the SGD was used, but eventually the Adam algorithm was preferred over the former, as it tends to better handle our data. The learning rate is arguably the highlight of our whole training model, as it decided how well our model was performing. Initially, a learning rate of 0.01 was implemented, but the final performance of our model (mainly the Final training accuracy),as you can see below, did not perform well, as the model had a constant 33% training accuracy. Consequently, we then increase(as in decreasing the velocity of our rate) learning rate, achieving our final training accuracy of approximately 72.2%.

# In[ ]:


train_acc_loader = torch.utils.data.DataLoader(train_dataset, batch_size =  64)

def get_accuracy(model, train=False):
    if train:
        data = train_dataset
    correct = 0
    total = 0
    for imgs, labels in train_acc_loader:#torch.utils.data.DataLoader(data, batch_size=64):
        output = model(imgs) # We don't need to run F.softmax
        #print(output)
        #break
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total


# In[ ]:


import torch.optim as optim

#1. Adam, with lr= 0.0001 or 0.001?

def train(model, data, batch_size=64, num_epochs=1):
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 0.01 or 0.001?

    iters, losses, train_acc = [], [], []

    # training
    n = 0 # the number of iterations
    for epoch in range(num_epochs):
        print("EPOCH:",epoch)
        for iteration,(imgs, labels) in enumerate(train_loader):
            model.train()
            out = model(imgs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch

            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size)             # compute *average* loss
            model.eval()
            acc = get_accuracy(model, train=False)
            train_acc.append(acc) # compute training accuracy 
            if iteration%10==0:
                print(acc,loss.item())
            n += 1

    # plotting
    plt.title("Loss performance")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))


# With all of that out of the way, we can finally use our defined model to implement it on to our training dataset. Note that the num_epochs here entirely arbitary, but are highly crucial to the performance of our model. After much trial and error, the minimal number of epochs that are necessary to obtain an optimistic learning curve is at least 70. Although time consuming, but anything below 65 would end up with a very low learning improvement, eventually affecting our training accuracy. From the graphs, we can also see that as the model "learns", our loss performance, a.k.a our training loss had a bit of large spike in the beginning and soon after gradually decrease, finally to a stable low rate. As for our training curve, there were couple of up downs in the middle phase, but overall, it seems to be performing quite ideally. 

# In[ ]:


#print(train_dataset.shape)
#debug = train_dataset[:100,:,:,:]

model = FeedForward()
train(model, train_dataset, num_epochs = 70)


# **<font size="5">Implementing our model to the Test Dataset</font>**

# Now that our FCN model works semmingly well on our training dataset, we then introduce it to our test dataset. Since our test dataset is relatively small, the number of epochs introduced are much less smaller, in our case 10. Nonetheless, the model performed pretty well with a training accuracy of approximately 65%, with a higher accuracy obtainable by introducint more epochs. 

# **Loading our Test Dataset**

# In[ ]:


import torchvision.transforms as transforms

trainTransform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
data_path = 'Data/assignment_test/'
test_dataset = torchvision.datasets.ImageFolder(
    root = data_path,
    transform = trainTransform)

test_iter = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = 64,
    num_workers = 0,
    shuffle = True
)


# In[ ]:


train(model, test_dataset, num_epochs = 50)


# In[ ]:


import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# <font size="5">Confusion matrix for test dataset</font>

# In[ ]:


def get_confusion_matrix(model, data):
    torch.no_grad()
    model.eval()
    preds, actuals = [], []
    
    for imgs, labels in data:
        imgs = imgs.unsqueeze(0)
        out = model(imgs)
        _, predicted = torch.max(out, 1)
        preds.append(predicted)
        actuals.append(labels)
        
    preds = torch.cat(preds).numpy()
    actuals = np.asarray(actuals)
    return confusion_matrix(actuals, preds)


# In[ ]:


get_confusion_matrix(model, test_dataset)


# In[ ]:




