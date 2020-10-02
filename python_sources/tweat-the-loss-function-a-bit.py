#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import numpy as np


# # Start from CrossEntropyLoss

# In [abhishek](https://www.kaggle.com/abhishek)'s great baseline notebook, the loss function used is the `CrossEntropyLoss`.  
# 
# However, one drawback of `CrossEntropyLoss` if that it doesn't care "the position of the error".   
# 
# We can see from a simple example.   
# 
# Let's say we got a sentence of length 5 (index starts from 0), and the correct answer is position 3.   
# 
# The first prediction has its highest probability at position 4, while the second prediction put it at position 0.

# In[ ]:


inputs = torch.Tensor([[0.1, 0.1, 0.1, 0.1, 0.8]]).float() # pred as 4
targets = torch.Tensor([3]).long()

loss_func = torch.nn.CrossEntropyLoss()
print(loss_func(inputs, targets))


# In[ ]:


inputs = torch.Tensor([[0.8, 0.1, 0.1, 0.1, 0.1]]).float() # pred as 0
targets = torch.Tensor([3]).long()

loss_func = torch.nn.CrossEntropyLoss()
print(loss_func(inputs, targets))


# As you can see, the loss is the same, but 4 is much closer to 3 than 0. Then how can we optimise the loss function a bit?

# # Penalty based on the position gap

# Now let's add more penalty when our "argmax prediction" is far away from our target.   
# 
# This function also allows you to add different penalty to "argmax prediction" before/after the target.

# In[ ]:


def pos_weight(pred_tensor, pos_tensor, neg_weight=1, pos_weight=1):
    # neg_weight for when pred position < target position
    # pos_weight for when pred position > target position
    gap = torch.argmax(pred_tensor, dim=1) - pos_tensor
    gap = gap.type(torch.float32)
    return torch.where(gap < 0, -neg_weight * gap, pos_weight * gap)


# In[ ]:


inputs = torch.Tensor([[0.1, 0.1, 0.1, 0.1, 0.8]]).float() # pred as 4
targets = torch.Tensor([3]).long()

pos_weight(inputs, targets, 1, 1)


# In[ ]:


inputs = torch.Tensor([[0.8, 0.1, 0.1, 0.1, 0.1]]).float() # pred as 0
targets = torch.Tensor([3]).long()

pos_weight(inputs, targets, 1, 1)


# The larger the gap is, the more penalty it'll get.

# # Combine them

# Now it's time to combine them. A simply tweak on the original `loss_fn` is all we need.  
# 
# You can either multiple `loss_fct` with `pos_weight`, or squeeze `pos_weight` first and then add them together.

# In[ ]:


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss(reduce='none') # do reduction later
    
    start_loss = loss_fct(start_logits, start_positions) * pos_weight(start_logits, start_positions, 1, 1)
    end_loss = loss_fct(end_logits, end_positions) * pos_weight(end_logits, end_positions, 1, 1)
    
    start_loss = torch.mean(start_loss)
    end_loss = torch.mean(end_loss)
    
    total_loss = (start_loss + end_loss)
    return total_loss


# In[ ]:


# argmax pred for the start is 3, target is 1
# argmax pred for the end is 3, target is 3
start = torch.Tensor([[0.1, 0.1, 0.1, 0.8, 0.1]]).float()
start_target = torch.Tensor([1]).long()

end = torch.Tensor([[0.1, 0.1, 0.1, 0.8, 0.1]]).float()
end_target = torch.Tensor([3]).long()


# In[ ]:


loss_fn(start, end, start_target, end_target)


# In[ ]:


# argmax pred for the start is 2, target is 1
# argmax pred for the end is 3, target is 3
start = torch.Tensor([[0.1, 0.1, 0.8, 0.1, 0.1]]).float()
start_target = torch.Tensor([1]).long()

end = torch.Tensor([[0.1, 0.1, 0.1, 0.8, 0.1]]).float()
end_target = torch.Tensor([3]).long()


# In[ ]:


loss_fn(start, end, start_target, end_target)


# As you can see, the loss for the second one is smaller, which also aligns with the jaccard.
