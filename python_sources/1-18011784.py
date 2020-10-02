#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip uninstall -y kaggle')
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install kaggle==1.5.6')

get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle')
get_ipython().system('chmod=600 ~/.kaggle/kaggle.json')

get_ipython().system('kaggle competitions download -c 2020-ai-exam-fashionmnist-1')
get_ipython().system('unzip 2020-ai-exam-fashionmnist-1.zip')


# In[ ]:


import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)


# In[ ]:


pd_train = pd.read_csv("mnist_train_label.csv")
pd_train.info()


# In[ ]:


x_train = torch.FloatTensor(np.array(pd_train.iloc[:,1:])).to(device)
y_train = torch.LongTensor(np.array(pd_train.iloc[:,0])).to(device)

print(x_train.shape)
print(y_train.shape)


# In[ ]:


traindata = torch.utils.data.TensorDataset(x_train,y_train)

train_loader = torch.utils.data.DataLoader(dataset = traindata,
                                           batch_size = 100,
                                           shuffle = True,
                                           drop_last = True)


# In[ ]:


class NN(nn.Module):
  def __init__(self):
    super(NN, self).__init__()

    self.linear = nn.Linear(784, 10, bias = True)
    nn.init.kaiming_uniform_(self.linear.weight)
    nn.init.constant_(self.linear.bias, 0.01)

  def forward(self, x):
    out = self.linear(x)

    return out


# In[ ]:


model = NN().to('cuda')
loss = nn.CrossEntropyLoss().to('cuda')
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov= True)


# In[ ]:


Epochs = 15
for epoch in range(Epochs+1):
  avg_cost = 0
  model.train()

  for x,y in train_loader:
    optimizer.zero_grad()
    H = model(x)
    cost = loss(H,y)
    cost.backward()
    optimizer.step()
    avg_cost += cost

  print('Epoch : {:4d}, avg_cost : {:.6f}'.format(epoch,
                                                  avg_cost/len(train_loader)))


# In[ ]:


pd_test = pd.read_csv('mnist_test.csv')
pd_test


# In[ ]:


y_train = torch.FloatTensor(np.array(pd_test.iloc[:,:])).to(device)

print(y_train.shape)


# In[ ]:



with torch.no_grad():
  model.eval()
  H = model(y_train)
  predict = torch.argmax(H,dim=1)

ID = np.array([i for i in range(len(y_train))]).reshape(-1,1)
cate = predict.to('cpu').detach().numpy().reshape(-1,1)
result = np.hstack((ID,cate))
result


# In[ ]:


df = pd.DataFrame(result, columns=['Id', 'Category'])
df.to_csv("submit_example.csv",index=False,header=True)


# In[ ]:


get_ipython().system('kaggle competitions submit -c 2020-ai-exam-fashionmnist-1 -f submit_example.csv -m "Message"')


# In[ ]:




