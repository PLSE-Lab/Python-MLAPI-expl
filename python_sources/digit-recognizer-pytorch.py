#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

import torch as th
import torch.nn.functional as F

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings('ignore')

device = th.device('cpu')


# ### Importing data

# In[ ]:


df_train = pd.read_csv('../input/digit-recognizer/train.csv')
df_test = pd.read_csv('../input/digit-recognizer/test.csv')

df_train.head()


# In[ ]:


print('Train shape:', df_train.shape)
print('Test shape:', df_test.shape)


# In[ ]:


df_train['label'].value_counts()


# ### Splitting data into features and labels
# x = features
# 
# y = labels

# In[ ]:


x = df_train[df_train.columns[1:]].values
y = df_train['label'].values

x_to_submit = df_test[df_train.columns[1:]].values


# ### Reshaping features to transform it to a 28x28 image

# In[ ]:


x = np.reshape(x, (x.shape[0], 28, 28))
x_to_submit = np.reshape(x_to_submit, (x_to_submit.shape[0], 28, 28))


# In[ ]:


for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(x[i])
    plt.title(y[i])


# ### Splitting data for training, testing and validation

# In[ ]:


x_train, x_test_val, y_train, y_test_val = train_test_split(x, y, test_size = .15, 
                                                            stratify=y, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_test_val, y_test_val, test_size = .25, 
                                                stratify=y_test_val, random_state=42)

print('X train shape:', x_train.shape)
print('Y train shape:', y_train.shape)
print()
print('X test shape:', x_test.shape)
print('Y test shape:', y_test.shape)
print()
print('X validation shape:', x_val.shape)
print('Y validation shape:', y_val.shape)


# ### Transforming data to tensor
# 
# Now we have (batch, height, width) features.
# 
# First we will need to add the channel dimension with numpy's expand_dims, now having features with shape of (batch, height, width, channel).
# 
# Tensors needs to have shape of (batch, channel, height, width), so we will transpose our data.

# In[ ]:


def img2tensor(arr, device):
    arr_expand = np.expand_dims(arr, 3)
    arr_transpose = np.transpose(arr_expand, (0, 3, 1, 2))
    tensor = th.from_numpy(arr_transpose).to(device)
    
    return tensor.float()

x_train = img2tensor(x_train, device)
x_test = img2tensor(x_test, device)
x_val = img2tensor(x_val, device)

x_to_submit = img2tensor(x_to_submit, device)

y_train = th.from_numpy(y_train).to(device)
y_test = th.from_numpy(y_test).to(device)
y_val = th.from_numpy(y_val).to(device)


# ### Creating a DataLoader

# In[ ]:


dataset = th.utils.data.TensorDataset(x_train, y_train)

data_loader = th.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)


# ### Creating our Convolutional neural network

# In[ ]:


class CNN(th.nn.Module):
    
    def __init__(self, hidden_size, droupout_p, output_size):
        super(CNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.droupout_p = droupout_p
        self.output_size = output_size
        
        self.norm = th.nn.BatchNorm2d(1)
        
        self.conv1 = th.nn.Conv2d(1, 16, (3, 3))
        self.conv2 = th.nn.Conv2d(16, 32, (4, 4))
        
        self.dropout = th.nn.Dropout(self.droupout_p)
        
        self.fc1 = th.nn.Linear(32*5*5, self.hidden_size)
        self.fc2 = th.nn.Linear(self.hidden_size, self.output_size)
        
        
    def forward(self, x):
        out = self.norm(x)
        
        out = F.relu(self.conv1(out))
        out = F.max_pool2d(out, (2, 2))
        
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, (2, 2))
        
        out = out.view(-1, 32*5*5)
        
        out = F.tanh(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
    
    def predict(self, x):
        out = self.forward(x)
        _, y_pred = th.max(out, 1)
    
        return y_pred


# Decay the learning rate by 10 every 30 epochs

# In[ ]:


def decrease_lr(opt, epoch, init_lr):
    lr =  init_lr * (0.1 ** (epoch // 30))
    for param_group in opt.param_groups:
        param_group['lr'] = lr


# In[ ]:


lr = 0.001
epochs = 90

net = CNN(hidden_size=32, droupout_p=.3, output_size=10).to(device)

opt = th.optim.Adam(net.parameters(), lr)
criterion = th.nn.CrossEntropyLoss()


# ### Training our net

# In[ ]:


losses = []
test_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    print('Epoch:', epoch+1)
    
    decrease_lr(opt, epoch, lr)
    
    net.train()
    batch_losses = []
    for feats, labels in tqdm(data_loader):      
        output = net(feats)
        
        loss = criterion(output, labels)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        batch_losses.append(loss.item())
    
    net.eval()
    
    loss_mean = np.array(batch_losses).mean()
    test_acc = accuracy_score(y_test, net.predict(x_test).data.numpy())
    val_acc = accuracy_score(y_val, net.predict(x_val).data.numpy())
    
    losses.append(loss_mean)
    test_accuracies.append(test_acc)
    val_accuracies.append(val_acc)
    
    print('Loss:', loss_mean)
    print('Test acc:', test_acc)
    print('Validation acc:', val_acc)


# In[ ]:


plt.plot(losses, label='loss')
plt.legend()


# In[ ]:


plt.plot(test_accuracies, label='test')
plt.plot(val_accuracies, label='val')
plt.legend()


# In[ ]:


th.save(net.state_dict(), 'model-mnist.pth.tar')


# In[ ]:


def plot_confusion_matrix(cm, title):
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')


# In[ ]:


net.eval()
y_test_pred =  net.predict(x_test).data.numpy()
cm_test = confusion_matrix(y_test, y_test_pred)

print(cm_test)
plot_confusion_matrix(cm_test, 'Test')


# In[ ]:


net.eval()
y_val_pred =  net.predict(x_val).data.numpy()
cm_val = confusion_matrix(y_val, y_val_pred)

print(cm_val)
plot_confusion_matrix(cm_val, 'Validation')


# In[ ]:


net.eval()
y_to_submit = net.predict(x_to_submit)


# In[ ]:


df_submit = pd.DataFrame()
df_submit['ImageId'] = range(1, y_to_submit.size()[0]+1)
df_submit['Label'] = y_to_submit


# In[ ]:


df_submit.to_csv('submission.csv', index=False)

