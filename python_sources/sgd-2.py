#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.basics import *


# In[ ]:


n = 100


# Create a tensor for a1x1^2 + a2x2 + a3x3 where x3=1

# In[ ]:


x = torch.ones(n,3) 
x[:,1].uniform_(-3.,3)
x[:,0] = x[:,1]**2
x[:5]


# In[ ]:


a = tensor(2.,1, -2); a #tensor of a1 a2 and a3


# In[ ]:


y = x@a + torch.rand(n) 
y[:5]


# In[ ]:


plt.scatter(x[:,1], y);


# In[ ]:


def mse(y_hat, y): return ((y_hat-y)**2).mean()


# In[ ]:


a = tensor(-3.,1,10)


# In[ ]:


y_hat = x@a
mse(y_hat, y)


# In[ ]:


plt.scatter(x[:,1],y)
plt.scatter(x[:,1],y_hat);


# In[ ]:


a = nn.Parameter(a); a


# What gradient calculates is the direction where we need to go to find the correct parameters. But once we move into that direction, it doesnt mean that the direction remains the same. That is why we multiply the direction with learning rate and calculate new gradient, i.e. new direction. 

# In[ ]:


def update():
    y_hat = x@a
    loss = mse(y, y_hat)
    loss.backward()
    if t % 50 == 0: 
        print(loss)
    with torch.no_grad():
        a.sub_(lr * a.grad)
        a.grad.zero_()


# In[ ]:


lr = 5e-2
for t in range(150): update()


# In[ ]:


plt.scatter(x[:,1],y)
plt.scatter(x[:,1],x@a);


# In[ ]:


from matplotlib import animation, rc
rc('animation', html='jshtml')


# In[ ]:


a = nn.Parameter(tensor(-3.,1,10))

fig = plt.figure()

def animate(i):
    update() 
    plt.clf()
    plt.scatter(x[:,1], y, c='orange')
    return plt.scatter(x[:,1], x@a)

animation.FuncAnimation(fig, animate, np.arange(0, 150), interval=10)


# In[ ]:




