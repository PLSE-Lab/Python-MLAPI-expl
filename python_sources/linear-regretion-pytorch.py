#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pylab import *


# In[ ]:


from sklearn.datasets import load_boston


# In[ ]:


boston_dataset = load_boston()


# In[ ]:


import pandas as pd


# In[ ]:


boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
boston.head()


# In[ ]:


scatter(boston['RM'], boston['MEDV'], alpha=0.3)
show()


# In[ ]:


X = array(boston['RM'])
Y = array(boston['MEDV'])


# In[ ]:


import torch


# In[ ]:


print(X.shape, Y.shape)


# In[ ]:


XT = torch.Tensor(X.reshape(506, 1))
YT = torch.Tensor(Y.reshape(506, 1))


# In[ ]:


print(XT.shape, YT.shape)


# In[ ]:


class LinearRegretion(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.F = torch.nn.Linear(input_dim, 1)
        self.loss = None
        self.accuracy = None
        
    def forward(self, x):
        return self.F(x)
    
    def fit(self, x, y, epochs=1, lr=0.01):
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        
        self.train()
        
        for i in range(0, epochs):
            y_ = self.forward(x)
            loss = loss_fn(y_, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.loss = loss.detach().numpy()
        self.accuracy = self.loss / x.shape[0]


# In[ ]:


h = LinearRegretion(1)


# In[ ]:


hy = array(h(XT).detach())
scatter(X, Y, alpha=0.3)
plot(X, hy, c="brown")
show()


# In[ ]:


print(h.F.weight)


# In[ ]:


print(h.F.bias)


# In[ ]:


h.fit(XT, YT, epochs=50000, lr=0.003)

hy = array(h(XT).detach())
scatter(X, Y, alpha=0.3)
plot(X, hy, c="brown")
show()


# In[ ]:


for param in h.parameters():
    print(param)


# In[ ]:


print(h.F.weight)


# In[ ]:


print(h.F.bias)


# In[ ]:


print("Loss: ", h.loss)
print("Accuracy: ", h.accuracy)


# In[ ]:


print((h(torch.Tensor([6]))*1000).detach().numpy())


# In[ ]:


print(h)


# In[ ]:


print(h.state_dict())


# In[ ]:


torch.save(h.state_dict(), "linear_state_dict.pt")

