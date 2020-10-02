# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Import Libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Prepare Dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

activation = {
    'relu': nn.ReLU(inplace=True),
    'leakyrelu': nn.LeakyReLU(0.1, inplace=True),
    'selu': nn.SELU(inplace=True),
    'elu': nn.ELU(alpha=1.0, inplace=True),
    'celu': nn.CELU(alpha=1.0, inplace=True),
    'hardsrink': nn.Hardshrink()
}

import torch.nn.functional as F
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist

# Define convolution operator
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)

# Main basic block that will be repeated
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, normalize=None, activation=nn.ReLU(inplace=True), batchnorm=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        # we define batchnorm but filter it out in forward
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Enables us to use different type of activation
        self.activation = activation
        # Batchnorm flag
        self.batchnorm = batchnorm
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.normalize = normalize
        # self.initilize_weight()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # Perform batchnorm if it is enabled
        if self.batchnorm:
          out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        # Perform batchnorm if it is enabled
        if self.batchnorm:
          out = self.bn2(out)
        # Change the shape of the layer to combine the residual
        if self.normalize:
            residual = self.normalize(x)
        out += residual
        out = self.activation(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, activation=nn.ReLU(inplace=True), batchnorm=True, regularization=False):
        super(ResNet, self).__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.regularization = regularization
        # First change from 3 channel input to 16 channel
        self.in_channels = 16
        # This conv layers converts RBG to 16 channel
        self.conv = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        # Create residual basic blocks
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(3072, num_classes)
        # Dropout
        self.drop = nn.Dropout(0.5)


    def make_layer(self, block, out_channels, blocks, stride=1):
        normalize = None
        
        # Define a layer to adjust the shape so residue can be added
        if (stride != 1) or (self.in_channels != out_channels):
            normalize = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        
        # Holds the basic residue block
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, normalize, self.activation, self.batchnorm))
        self.in_channels = out_channels
        
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, 1, None, self.activation, self.batchnorm))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        
        # Perform batchnorm if it is enabled
        if self.batchnorm:
          out = self.bn1(out)

        out = self.layer1(out)
        # Perform regularization
        if self.regularization:
          out = self.drop(out)
        out = self.layer2(out)
        # Perform regularization
        if self.regularization:
          out = self.drop(out)        
        out = self.layer3(out)
        # Perform regularization
        if self.regularization:
          out = self.drop(out)         
        out = self.avg_pool(out)
        # print('apool', out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    
class ResNetHPara:
  def __init__(self, activation=activation['relu'], batchnorm=False, labelSmooth=False, 
               learningRate='constant', regularization=False, init='he', 
               name='baseline', epoch=10, train=None, valid=None, classes=10):
    # batch_size, epoch and iteration
    self.batch_size = 2
    self.num_epochs = epoch
    self.name = name
    self.learningRate = learningRate
    self.labelSmooth = labelSmooth
    self.train_loader = train
    self.test_loader = valid
    # Create ResNet
    net_args = {
        "block": BasicBlock,
        "layers": [1, 1, 2, 1],
        "activation": activation,
        "batchnorm": batchnorm,
        "regularization": regularization,
        "num_classes" : classes
    }
    self.model = ResNet(**net_args)

    if init == 'ortho':
        for name, module in self.model.named_children():
            module.apply(weights_init)

    # Move the model to CUDA
    self.model.to(device)
        
    if not labelSmooth:
        # Cross Entropy Loss 
        self.error = nn.CrossEntropyLoss()
    else:
        # from torch.utils.cross_entropy import CrossEntropyLoss
        # self.error = CrossEntropyLoss(smooth_eps=0.95)
        self.error = LabelSmoothLoss(smoothing=0.1)
        # self.error = nn.BCEWithLogitsLoss()
    
    # Adam Optimizer
    learning_rate = 0.001

    if self.learningRate == 'constant':
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    else:
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.schedular = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.01, max_lr=0.1)       


#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                             download=True, transform=transform)

#     self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
#                                               shuffle=True, num_workers=4)

#     testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                           download=True, transform=transform)

#     self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
#                                             shuffle=False, num_workers=4)


  def train(self):
    self.loss_list = []
    self.iteration_list = []
    self.accuracy_list = []
    count = 0
    for epoch in range(self.num_epochs):
        for i, (images, labels) in enumerate(self.train_loader):
            train = Variable(images).to(device)
            labels = Variable(labels).to(device)           
            # Clear gradients
            self.optimizer.zero_grad()    
            # Forward propagation
            outputs = self.model(train)        
            # Calculate softmax and ross entropy loss
            if not self.labelSmooth:
              loss = self.error(outputs, labels)        
            else:
              # smooth_label = smooth_one_hot(labels, classes=10, smoothing=0.9)
              loss = self.error(outputs, labels)
            # Calculating gradients
            loss.backward()        
            # Update parameters
            if self.learningRate == 'constant':
              self.optimizer.step()      
            else:
              self.optimizer.step()
              self.schedular.step()  
            count += 1      
            if count % 100 == 0:
                # Calculate Accuracy         
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in self.test_loader:               
                    # Inference
                    images = Variable(images).to(device)
                    labels = Variable(labels).to(device)  
                    outputs = self.model(images)   
                    # Get predictions
                    predicted = torch.max(outputs.data, 1)[1]             
                    # Total number of labels
                    total += labels.size(0)              
                    correct += (predicted == labels).sum()         
                accuracy = 100 * correct / float(total)         
                # store loss and iteration
                self.loss_list.append(loss.data)
                self.iteration_list.append(count)
                self.accuracy_list.append(accuracy)
                if count % 100 == 0:
                   # Print Loss
                   print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
        # Save the model in each epoch
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                }, '/tmp/cifar10-'+self.name+'-.pth')


  def count_parameters(self, grad):
      if grad:
          return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
      else:
          return sum(p.numel() for p in self.model.parameters())

      print('Total Parameters: ', count_parameters(self.model, False))
      print('Total Trainable Parameters: ', count_parameters(self.model, True))    

    
def trainonce(para):
  data = []
  for i in range(1):
    model = ResNetHPara(**para)
    model.train()
    data.append(max(model.accuracy_list).data.tolist())

  return data    


# para = {'epoch':1, 'name':'baseline'}
# data = trainonce(para)
# print(data)
