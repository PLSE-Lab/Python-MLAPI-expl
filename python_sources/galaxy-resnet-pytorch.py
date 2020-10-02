#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torchvision
import PIL

import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.__version__, torchvision.__version__


# In[ ]:


# Check GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:


# For data augementation
# https://discuss.pytorch.org/t/anything-like-transformer-for-tensordataset/928/4

# Need to turn on Internet
# https://www.kaggle.com/questions-and-answers/36982#466931

get_ipython().system('pip install git+https://github.com/ncullen93/torchsample')


# In[ ]:


import torchsample
torchsample.__version__


# # Util functions

# In[ ]:


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# In[ ]:


def predict(net, loader, record_y=False):

    y_pred_list = []
    with torch.no_grad():

        total_loss = 0.0
        total_sample = 0
        for i, data in enumerate(loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            current_sample = len(labels)
            total_sample += current_sample
            total_loss += loss.item() * current_sample

            if record_y:
                y_pred_list.append(outputs.cpu().numpy())
                
    avg_loss = total_loss / total_sample
    print(f"Average loss: {avg_loss}")
    
    if record_y:
        y_pred = np.concatenate(y_pred_list)
        return y_pred
    else:
        return avg_loss


# # Prepare training data

# In[ ]:


get_ipython().run_line_magic('time', "ds = xr.open_dataset('../input/galaxy-zoo-cleaned/galaxy_train.nc')")
ds


# In[ ]:


get_ipython().run_cell_magic('time', '', "X = ds['image_train'].transpose('sample', 'channel', 'x' ,'y').data  # pytorch use channel-first, unlike Keras\ny = ds['label_train'].data\n\nX_train, X_valid, y_train, y_valid = train_test_split(\n    X, y, test_size=0.2, random_state=0)")


# In[ ]:


# Without data augmentation
trainset = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train), torch.from_numpy(y_train)
)

# # With Data augmentation 
# # tips from http://benanne.github.io/2014/04/05/galaxy-zoo.html
# transform = torchsample.transforms.Compose([
#     torchsample.transforms.Rotate(90),
#     torchsample.transforms.RandomFlip(),
#     # torchsample.transforms.Translate(0.04)  # translation decreases performance!
# ])

# trainset = torchsample.TensorDataset(
#     torch.from_numpy(X_train), torch.from_numpy(y_train),
#     input_transform = transform
# )

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=2
)

evalset = torch.utils.data.TensorDataset(
    torch.from_numpy(X_valid), torch.from_numpy(y_valid)
)

evalloader = torch.utils.data.DataLoader(
    evalset, batch_size=32, shuffle=False, num_workers=2
)


# In[ ]:


dataiter = iter(trainloader)
data = dataiter.next()
inputs, labels = data[0].to(device), data[1].to(device)
inputs.shape, labels.shape  # batch, channel, x, y


# # Fit model

# In[ ]:


from torchvision.models.resnet import _resnet, BasicBlock

def resnet10(pretrained=False, progress=True, **kwargs):
    # Apapted from https://github.com/pytorch/vision/blob/9cdc8144a1b462fecee4b2efe0967ba172708c4b/torchvision/models/resnet.py#L227
    return _resnet('resnet10', BasicBlock, [1, 1, 1, 1], pretrained, progress,
                   **kwargs)

net = resnet10(pretrained=False, num_classes=37)
# net = torchvision.models.resnet.resnet18(pretrained=False, num_classes=37)


# In[ ]:


get_ipython().run_line_magic('time', 'net.to(device)')
("")


# In[ ]:


net(inputs).shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ncriterion = nn.MSELoss()\noptimizer = optim.Adam(net.parameters())\n\nprint_freq = 400  # print loss per that many steps\n\ntrain_history = []\neval_history = []\nfor epoch in range(20):  # loop over the dataset multiple times\n\n    running_loss = 0.0\n    for i, data in enumerate(trainloader, 0):\n        # get the inputs; data is a list of [inputs, labels]\n        # inputs, labels = data\n        inputs, labels = data[0].to(device), data[1].to(device)\n\n        # zero the parameter gradients\n        optimizer.zero_grad()\n\n        # forward + backward + optimize\n        outputs = net(inputs)\n        loss = criterion(outputs, labels)\n        loss.backward()\n        optimizer.step()\n\n        # print statistics\n        running_loss += loss.item()\n        if i % print_freq == print_freq-1:\n            print('[%d, %5d] loss: %.4f' %\n                  (epoch + 1, i + 1, running_loss / print_freq))\n            running_loss = 0.0\n            \n    print('Training loss:')\n    train_loss = predict(net, trainloader, record_y=False)\n    train_history.append(train_loss)\n    print('Validation loss:')\n    eval_loss = predict(net, evalloader, record_y=False)\n    eval_history.append(eval_loss)\n    print('-- new epoch --')\n\nprint('Finished Training')")


# # Examine learning curve

# In[ ]:


df_history = pd.DataFrame(
    np.stack([train_history, eval_history], axis=1), 
    columns=['train_loss', 'val_loss'],
)
df_history.index.name = 'epoch'
df_history


# In[ ]:


plt.rcParams['font.size'] = 14
df_history.plot(grid=True, marker='o', ylim=[0, None], linewidth=3.0, alpha=0.8)
plt.title('ResNet-10 on Galaxy Zoo (no data augmentation)')
plt.savefig('training_history.png', dpi=144, bbox_inches='tight')


# In[ ]:


# save loss history
df_history.to_csv('train_history.csv')


# In[ ]:


get_ipython().system('head train_history.csv')


# # Evaluation

# In[ ]:


get_ipython().run_cell_magic('time', '', 'y_valid_pred = predict(net, evalloader, record_y=True)')


# In[ ]:


rmse(y_valid, y_valid_pred)


# In[ ]:


r2_score(y_valid, y_valid_pred)


# # Save trained model

# In[ ]:


get_ipython().run_line_magic('time', "torch.save(net.state_dict(), './trained_resnet.pt')")


# In[ ]:


ls -lh trained_resnet.pt


# In[ ]:




