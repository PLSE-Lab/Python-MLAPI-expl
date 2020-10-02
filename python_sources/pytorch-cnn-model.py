#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import matplotlib.pyplot as plt


# ### Seting up the dataloader

# Implementing (Inheriting) the dataset class

# In[ ]:


class CactusImageDataset(Dataset):
    "Aerial Cactus Classification Dataset"
    
    def __init__(self, csv_file, root_dir, transform = None):
        """
        Args:
            csv_file (string): Path to csv file with annotations
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.cactus_annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return self.cactus_annotations.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,
                                self.cactus_annotations.iloc[idx, 0])
        image = io.imread(img_name)
        has_cactus = self.cactus_annotations.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
            
        sample = {"image":image, 'label':has_cactus}
        return sample
    
        


# Setting Up the Transforms, Only using transforms available. No custom transforms. 

# In[ ]:


image_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ]),
    'test':transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
}
    


# #### Initilaizing the Dataset

# In[ ]:


dataset = CactusImageDataset('../input/aerial-cactus-identification/train.csv','../input/aerial-cactus-identification/train/train/', image_transforms['train'])


# Here it would be nice to see the distrbution of the input. Let's make a histogram

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(dpi=128, figsize=(3,3))
plt.title('Distribution of Cacti Images')
plt.xticks([0,1])
plt.xlabel('Has Cactus')
plt.ylabel('Number of Images')
dataset.cactus_annotations['has_cactus'].hist(bins=2)


# > **So about 75% of the images contains cacti. A very unbalanced set. But we'll split the dataset into training and dev set which will help us generalize the model somewhat better.**

# #### Splitting into train (80%) and dev set (20%)

# In[ ]:


train_size = int((0.80 * dataset.__len__()))
dev_size = dataset.__len__() - train_size
train_set, dev_set = random_split(dataset, [train_size, dev_size])


# In[ ]:


BATCH_SIZE = 32
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True)


# ### Defining the Model

# > **Here, let me just show how do the dimensions work in the model. This is something that I used to get confused about a lot.** (Still do sometimes)

# * **Input** Dimension: **32x32x3** Here **3** is the number if channels for our image (RGB). Here the number of output channels refers to the number of **filters** which we define to be **3x3**
# 
# * The first layer will make an output of dim $$\lfloor \frac{INPUTDIM + 2*PAD - (KERNEL(FILTER) - 1) - 1}{STRIDE}  + 1 \rfloor$$ The complete one you can see on PyTorch docs. Which in my model gives us : $$\lfloor \frac{32 + 2*1 - (3 - 1) - 1}{1} +1 \rfloor = 32$$  
# And then we pass this to our max pooling layer of kernel size 2 and stride 2 as well which gives us: $$\lfloor \frac{32 + 2*0 - (2 - 1) - 1}{2} +1 \rfloor = 16$$ 
# 
# 
# * The second layer similarly gives us $$\lfloor \frac{16 + 2*1 - (3 - 1) - 1}{1} + 1\rfloor = 16$$ and then the max pooling layer with kernel size 2 and stride 2 gives us: $$\lfloor \frac{16 + 2*0 - (2 - 1) - 1}{2} +1 \rfloor = 8$$
# 
# * The third layer $$\lfloor \frac{8 + 2*1 - (3 - 1) - 1}{1} + 1 \rfloor = 8$$ and then the max pooling layer with kernel size 2 and stride 2 gives us: $$\lfloor \frac{8 + 2*0 - (2 - 1) - 1}{2} +1 \rfloor = 4$$
# 
# * The fourth layer $$\lfloor \frac{4 + 2*1 - (3 - 1) - 1}{1}  + 1\rfloor = 4$$ and then the max pooling layer with kernel size 2 and stride 2 gives us: $$\lfloor \frac{4 + 2*0 - (2 - 1) - 1}{2} +1 \rfloor = 2$$
# 
# At this point, I stopped adding any convolution layers, but we can add if we want to. Now we take the out channels and line them them up as a linear vector of size $2*2*128$ because $2*2$ in the size of image at this point and there are 128 filters as we defined.  After which the linear layer follows. 
# 
# I hope this helps anyone who reads it. Cheers
# 

# In[ ]:


class CactusIdentifier(nn.Module):
    """
        Aerial Cactus Identifier Model
    """
    def __init__(self):
        super(CactusIdentifier, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3, padding = 1)
        self.conv2 = nn.Conv2d(16,32,3, padding = 1)
        self.conv3 = nn.Conv2d(32,64,3, padding = 1)
        self.conv4 = nn.Conv2d(64,128,3, padding = 1)
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(128*2*2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64,2)
        self.act = nn.ReLU()
        self.drop = nn.Dropout()
    
    def forward(self, inp_image):
        out = self.pool(self.act(self.conv1(inp_image)))
        out = self.pool(self.act(self.conv2(out)))
        out = self.pool(self.act(self.conv3(out)))
        out = self.pool(self.act(self.conv4(out)))
        
        out = out.view(-1, 128*2*2)
        out = self.drop(out)
        out = self.act(self.fc1(out))
        out = self.drop(out)
        out = self.act(self.fc2(out))
        out = self.drop(out)
        out = self.fc3(out)
        
        return out


# ### Training Module

# In the training module we implement early stopping in the function train_model to reduce overfitting. We after each epoch check the loss on the dev set, if the loss reduces on an epoch we save the model as a dict. and keep on going for the number of epocs in the same manner.
# Once our training finishes we load the dict and use that model for the results

# In[ ]:


class TrainingModule():
    """
    Training Module to train the model
    """
    
    def __init__(self, model):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.is_cuda = False
        if torch.cuda.is_available():
            self.is_cuda = True
            self.model = model.cuda()
    
    def train_epoch(self, epoch, train_iterator):
        total_loss = 0
        stats_string = "Epoch : {:3d} | Iteration : {:4d} | Loss : {:4.4f}"
        for i, data in enumerate(train_iterator):
            inputs = data['image']
            labels = data['label'].long()
            
            if self.is_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                    
            self.optimizer.zero_grad()
        
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(stats_string.format(epoch+1, i, total_loss/(i+1)))
                total_loss = 0
    
    def train_model(self, train_iterator, dev_iterator, num_epocs = 30):
        self.model.train()
        min_loss = 1000000
        stats_string = "Epoch : {:2d} | Train Loss : {:4.4f} | Dev Loss : {:4.4f}"
        for i in range(num_epocs):
            self.train_epoch(i, train_iterator)
            dev_loss = self.evaluate(dev_iterator)
            train_loss = self.evaluate(train_iterator)
            print(stats_string.format(i+1, train_loss, dev_loss))
            if dev_loss <= min_loss:
                torch.save(self.model.state_dict(), 'best_model')
                min_loss = dev_loss
    
        print("Finished Training")
        
    def evaluate(self, iterator):
        self.model.eval()
        loss_total = 0
        
        for i, data in enumerate(iterator):
            inputs = data['image']
            labels = data['label'].long()
            
            if self.is_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            preds = self.model(inputs)
            loss = self.loss_fn(preds, labels)
            loss_total += loss.item()
        
        return loss_total
            


# ### Initializing the Model

# In[ ]:


model = CactusIdentifier()
print(model)


# ### Training the Model

# In[ ]:


trainer = TrainingModule(model)
trainer.train_model(train_loader, dev_loader, num_epocs = 1)


# ### Setting Up the Test Set

# In[ ]:


test_set = CactusImageDataset("../input/aerial-cactus-identification/sample_submission.csv", "../input/aerial-cactus-identification/test/test", image_transforms['test'])
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


# ### Generating Predictions using the best model

# In[ ]:


model.load_state_dict(torch.load('best_model'))
final_preds = []

for i, data in enumerate(test_loader):
    inputs = data['image']
    labels = data['label'].long()

    if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()

    preds = model(inputs)
    final_preds += preds[:,1].tolist()

test_set.cactus_annotations['has_cactus'] = final_preds
test_set.cactus_annotations.to_csv("samplesubmission.csv", index=False)


# In[ ]:


test_set.cactus_annotations.head()

