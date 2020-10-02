#!/usr/bin/env python
# coding: utf-8

# # Animating embedding space
# 
# This notebook creates a toy neural net that tries to learn how to add and subtract numbers. Input is strings of the form `12+34` and the target output is a float of the result, i.e. `46.0`. Each character is represented by an embedding with three dimensions. This makes it easy to represent the embeddings as a 3D scater chart without any dimensionality reduction tricks. By saving the embedding values as we train, we can animate the chart and visualise the learning process.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import torch
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import plotly.offline as py
import plotly.graph_objs as go

# put plotly in Jupyter mode
py.init_notebook_mode(connected=True)


# ## Model and data
# 
# Let's define our model and a function to generate random sums.

# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.numInputs = 7     # max length of input string
        self.numOutputs = 1    # single number out
        self.numVocab = 13     # number of unique characters
        self.numDimensions = 3 # each embedding has three numbers
        self.numHidden = 10    # hidden fully connected layer
                
        self.embedding = nn.Embedding(self.numVocab, self.numDimensions)
        self.lin1 = nn.Linear(self.numInputs * self.numDimensions, self.numHidden)
        self.lin2 = nn.Linear(self.numHidden, self.numOutputs)

    def forward(self, input):
        x = self.embedding(input).view(-1, self.numInputs * self.numDimensions)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return torch.squeeze(x)

def generateData(num):
    train = []
    for i in range(0, num):
        a = random.randint(0, 100)
        b = random.randint(-100-a, 100-a)
        train.append((f'{a}{b:+}'.ljust(7), a+b))
    return train


# Now we create our data and network. No need to create a validation set since we're only interested in the embeddings.

# In[ ]:


vocab = list('0123456789-+ ')
char2index = {char: i for i, char in enumerate(vocab)}

train = generateData(100000)
trainloader = torchdata.DataLoader(train, batch_size=100)
net = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=5e-4, momentum=0.9)
embeddingFrames = []


# ## Train
# 
# And train the network. This is really fast so no need for GPU. Run this cell again and again to keep training if you like.

# In[ ]:


# loop over the dataset multiple times
for epoch in range(1): 
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        # save emdebbings for creating the animation later
        embeddingFrames.append(net.embedding(torch.tensor(range(0, len(vocab)))))
        
        # prepare batch
        sums, actuals = data
        input = torch.tensor([[char2index[c] for c in sum] for sum in sums], dtype=torch.long)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(input)
        actualsTensor = actuals.type(torch.FloatTensor)
        loss = criterion(outputs, actualsTensor)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200}')
            running_loss = 0.0

print('Finished Training')


# ## Analyse the embeddings
# 
# All that's left is to plot the embeddings. Once the animation completes you can rotate the chart to look around.

# In[ ]:


frames = []
for embeddings in embeddingFrames[0:300:5]:
    x, y, z = torch.transpose(embeddings, 0, 1).tolist()
    data = [go.Scatter3d(x=x,y=y,z=z, mode='text', text=vocab)]
    frame = dict(data=data)
    frames.append(frame)
    
fig = dict(data=frames[0]['data'], frames=frames)
py.iplot(fig)


# ## Observations
# - Generally you'll find that the digits `1`-`9` line up nicely and are equally spaced. Note that the line they form can be at any angle, which means that no single embedding parameter represents that dimension. 
# - `0` is usually a little out of alignment, which makes sense since it is a bit different to the other digits (it never appears at the start of a number, for example). 
# - Unsurprisingly, `+` and `-` end up on diametrically opposed sides of the space.
