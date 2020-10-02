#!/usr/bin/env python
# coding: utf-8

# If you think it's useful, please give me a upvote, thanks.
# 
# You can see more in [here](https://www.kaggle.com/c/google-quest-challenge/discussion/121582)

# # Let's begin to learn NLP step by step.

# ## It's easy to make out why its name is Recurrent Neural Network.
# ![0](https://upload-images.jianshu.io/upload_images/17947883-9d3116d2b8fb06fd.png?imageMogr2/auto-orient/strip|imageView2/2/w/529/format/webp)

# ## Now, spread it.
# ![1](https://upload-images.jianshu.io/upload_images/17947883-d221f4e6ebbdce16.png?imageMogr2/auto-orient/strip|imageView2/2/w/1083/format/webp)

# ## Formula :
# ![image.png](attachment:image.png)
# 
# You can see, the computation is acutally very easy.

# ## Now, we will build our own RNN model with pytorch.
# 
# the code is from [here ]( https://blog.csdn.net/out_of_memory_error/article/details/81456501)

# In[ ]:


import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt


# In[ ]:


# firstly, define your RNN model.
class RNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.rnn = nn.RNN(input_size=input_size,
                         hidden_size=32,
                         num_layers=1,
                         batch_first=True)
        
        # you can change here according to your needs, for this example, we need to output a regression value.
        self.out = nn.Linear(32, 1)
    
    # two input, and two output, you can see it in above picture.
    def forward(self, x_t, h_t_1):
        h_t, a_t = self.rnn(x_t, h_t_1)
        
        outputs = []
        
        # h_t.size(1) = time_step, we can regard each input as a vector and its elements' number equal time_step
        for i in range(h_t.size(1)):
            outputs.append(self.out(h_t[:, i, :]))
            
        return torch.stack(outputs, dim=1), a_t


# In[ ]:


# data is like this.
steps = np.linspace(0, np.pi*2, 100, dtype=np.float)
x_np = np.sin(steps)
y_np = np.cos(steps)

plt.plot(steps, x_np, 'b-', label='input(sin)')
plt.plot(steps, y_np, 'r-', label='target(cos)')
plt.legend(loc='best')


# In[ ]:


# train
time_step = 10
input_size = 1
learning_rate = 0.02


# In[ ]:


model = RNN(input_size)
print(model)


# In[ ]:


loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[ ]:


a_state = None


# In[ ]:


# imaging it's a sentence with 300 words, each step you input a word, and get a prediction.
# What do you think this model can do?
for step in range(300):
    start, end = step * np.pi, (step + 1) * np.pi
    
    steps = np.linspace(start, end, time_step, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
    
    # Recurrent is in here.
    prediction, a_state = model(x, a_state)
    a_state = a_state.data
    
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[ ]:


# plot final step
plt.plot(steps, y_np.flatten(), 'r-')
plt.plot(steps, prediction.data.numpy().flatten(), 'b-')


# Good result.
