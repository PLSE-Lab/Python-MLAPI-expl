#!/usr/bin/env python
# coding: utf-8

# ### In this notebook, I have used PyTorch to perform Transfer Learning on VGG16 Network and how it can be further used according to our need. CIFAR-10 dataset is used to showcase the resulting network's performance.
# 
# ### Most of the layers of the pretrained model are freezed and only the last two layers are trained for our dataset. It would be much clear in upcoming steps.
# 
# ### PS: Turn on your GPU for quick run!!

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision.datasets as datasets  
import torchvision.transforms as transforms


# ### Setting the hyperparameters

# In[ ]:


num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5


# ### The last two blocks containing the average pooling and classifier units can be altered according to our dataset. Since we have 10 classes in our dataset, we will alter the classifier unit for getting desired output of 10 classes. We will set the avgpooling layer to identity so that it cannot perform any changes on the feature map. 

# In[ ]:


model = torchvision.models.vgg16(pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #To use the GPU if available

model.to(device)


# ### The Identity class returns the output same as the input thus preventing any changes in the input.

# In[ ]:


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# ### We will now freeze the layers above the avgpooling layer so as to perform the backpropagation only on the last two layers while training.

# In[ ]:


for prm in model.parameters():
    prm.requires_grad = False


# ### Doing the required changes on the model

# In[ ]:


model.avgpool = Identity()

model.classifier = nn.Sequential(
    nn.Linear(512, 100), 
    nn.ReLU(), 
    nn.Linear(100, num_classes)
)


model.to(device)


# ### Loading the dataset

# In[ ]:


train_dataset = datasets.CIFAR10(root="/kaggle/working/dataset/", train=True, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[ ]:


for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        
        # If GPU is active, alter the data accordingly
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())
        
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")


# In[ ]:


def check_accuracy(loader, model):
    
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} cases correctly with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


check_accuracy(train_loader, model)


# ## An accuracy of about 63% is obtained when the pretrained layers are freezed and only the last two units are trained for our dataset. 
