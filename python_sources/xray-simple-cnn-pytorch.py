#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch
import torchvision

torch.__version__, torchvision.__version__


# In[ ]:


# Check GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# # Util function

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


ds = xr.open_dataset('../input/chest-xray-cleaned/chest_xray.nc')
ds


# In[ ]:


ds['image'].isel(sample=slice(0, 12)).plot(col='sample', col_wrap=4, cmap='gray')


# In[ ]:


ds['label'].mean(dim='sample').to_pandas().plot.barh()  # proportion


# In[ ]:


all_labels = ds['feature'].values.astype(str)
all_labels


# In[ ]:


get_ipython().run_cell_magic('time', '', "X_all = ds['image'].data[:, np.newaxis, :, :]  # pytorch use channel-first, unlike Keras\ny_all = ds['label'].data.astype(np.float32)  \n# https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542/4\n# https://discuss.pytorch.org/t/expected-object-of-scalar-type-float-but-got-scalar-type-long-for-argument-2-target/44109\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X_all, y_all, test_size=0.2, random_state=42\n)\n\nX_train.shape, X_test.shape, y_train.shape, y_test.shape")


# In[ ]:


trainset = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train), torch.from_numpy(y_train)
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=2
)

testset = torch.utils.data.TensorDataset(
    torch.from_numpy(X_test), torch.from_numpy(y_test)
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=False, num_workers=2
)


# In[ ]:


dataiter = iter(trainloader)
data = dataiter.next()
inputs, labels = data[0].to(device), data[1].to(device)
inputs.shape, labels.shape  # batch, channel, x, y


# In[ ]:


inputs.dtype, labels.dtype


# # Fit model

# In[ ]:


class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        
        channel_1 = 16
        channel_2 = 32
        channel_3 = 64
        
        self.conv1 = nn.Conv2d(1, channel_1, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(channel_1, channel_2, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(channel_2, channel_3, 3, padding=1)
        
        last_x = 32  # spatial size = /128 / 2 / 2
        self.last_size = channel_3 * last_x * last_x
        self.fc1 = nn.Linear(self.last_size, 14)  # need to flatten to filter * x * y

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.pool1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.last_size)  # flatten
        x = self.fc1(x)
        x = torch.sigmoid(x)
        # https://discuss.pytorch.org/t/usage-of-cross-entropy-loss/14841/5
        return x

net = Net()


# In[ ]:


get_ipython().run_line_magic('time', 'net.to(device)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# criterion = nn.MSELoss()\ncriterion = nn.BCELoss()\n\noptimizer = optim.Adam(net.parameters())\n\nprint_freq = 400  # print loss per that many steps\n\ntrain_history = []\neval_history = []\nfor epoch in range(5):  # loop over the dataset multiple times\n\n    running_loss = 0.0\n    for i, data in enumerate(trainloader, 0):\n        # get the inputs; data is a list of [inputs, labels]\n        # inputs, labels = data\n        inputs, labels = data[0].to(device), data[1].to(device)\n\n        # zero the parameter gradients\n        optimizer.zero_grad()\n\n        # forward + backward + optimize\n        outputs = net(inputs)\n        loss = criterion(outputs, labels)\n        loss.backward()\n        optimizer.step()\n\n        # print statistics\n        running_loss += loss.item()\n        if i % print_freq == print_freq-1:\n            print('[%d, %5d] loss: %.4f' %\n                  (epoch + 1, i + 1, running_loss / print_freq))\n            running_loss = 0.0\n            \n    print('Training loss:')\n    train_loss = predict(net, trainloader, record_y=False)\n    train_history.append(train_loss)\n    \n    # we shouldn't actually monitor test loss; just for convenience\n    print('Validation loss:')\n    eval_loss = predict(net, testloader, record_y=False)\n    eval_history.append(eval_loss)\n    print('-- new epoch --')\n\nprint('Finished Training')")


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
plt.title('Simple CNN on Chest-xray')


# # Evaluation

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# do not shuffle, so that the output order matches true label\ny_train_pred = predict(\n    net, \n    torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=2),\n    record_y=True\n)\n\ny_test_pred = predict(net, testloader, record_y=True)')


# In[ ]:


def plot_ruc(y_true, y_pred):
    fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    
    for (idx, c_label) in enumerate(all_labels):
        fpr, tpr, thresholds = roc_curve(y_true[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))

    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')


# In[ ]:


plot_ruc(y_test, y_test_pred)
plt.title('ROC test set')


# In[ ]:


plot_ruc(y_train, y_train_pred)
plt.title('ROC train set')


# In[ ]:




