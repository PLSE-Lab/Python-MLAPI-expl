#!/usr/bin/env python
# coding: utf-8

# Transfer learning from pretrained model (Resnet50) using Pytorch

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from torchvision import transforms

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


data_dir = '../input'
train_dir = data_dir + '/aptos2019-blindness-detection/train_images/'
test_dir = data_dir + '/aptos2019-blindness-detection/test_images/'


# In[ ]:


labels = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
print(len(labels))
labels.head()


# In[ ]:


# Example of images 
plt.figure(figsize=[15,15])
i = 1
for img_name in labels['id_code'][:10]:
    img = mpimg.imread(train_dir + img_name + '.png')
    plt.subplot(6,5,i)
    plt.imshow(img)
    i += 1
plt.show()


# In[ ]:


class ImageData(Dataset):
    def __init__(self, df, data_dir, transform):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):       
        img_name = self.df.id_code[index] + '.png'
        label = self.df.diagnosis[index]          
        img_path = os.path.join(self.data_dir, img_name)   
            
        image = mpimg.imread(img_path)
        image = (image + 1) * 127.5
        image = image.astype(np.uint8)
        
        image = self.transform(image)
        return image, label


# In[ ]:


data_transf = transforms.Compose([transforms.ToPILImage(mode='RGB'), 
                                  transforms.Resize(265),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor()])
train_data = ImageData(df = labels, data_dir = train_dir, transform = data_transf)
train_loader = DataLoader(dataset = train_data, batch_size=32, drop_last=True)


# In[ ]:


model = models.resnet50()
model.load_state_dict(torch.load("../input/resnet50/resnet50.pth"))


# In[ ]:


# Freeze model weights
for param in model.parameters():
    param.requires_grad = False


# In[ ]:


# Changing number of model's output classes to 5
model.fc = nn.Linear(2048, 5)


# In[ ]:


# Transfer execution to GPU
model = model.to('cuda')


# In[ ]:


optimizer = optim.Adam(model.parameters())
loss_func = nn.CrossEntropyLoss()


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Train model\nloss_log=[]\nfor epoch in range(25):    \n    model.train()        \n    for ii, (data, target) in enumerate(train_loader):        \n        data, target = data.cuda(), target.cuda()              \n        optimizer.zero_grad()\n        output = model(data)                    \n        loss = loss_func(output, target)\n        loss.backward()\n        optimizer.step()          \n        if ii % 1000 == 0:\n            loss_log.append(loss.item())       \n    print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, loss.item()))")


# In[ ]:


submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
test_data = ImageData(df = submit, data_dir = test_dir, transform = data_transf)
test_loader = DataLoader(dataset = test_data, shuffle=False)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Prediction\npredict = []\nmodel.eval()\nfor i, (data, _) in enumerate(test_loader):\n    data = data.cuda()\n    output = model(data)  \n    output = output.cpu().detach().numpy()    \n    predict.append(output[0])')


# In[ ]:


submit['diagnosis'] = np.argmax(predict, axis=1)
submit.head()


# In[ ]:


submit.to_csv('submission.csv', index=False)

