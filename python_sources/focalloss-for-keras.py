#!/usr/bin/env python
# coding: utf-8

# This [great kernel](https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-448-public-lb#) uses FocalLoss written for PyTorch and has a success with it. In case you'd like to use same loss function in Keras, I've rewritten it here.
# 
# There is [another good implementation of FocalLoss here](https://github.com/mkocabas/focal-loss-keras), but it differs for the one used by lafoss.
# This kernel is aimed for people who would like to replicate his results step by step in Keras.

# In[ ]:


from keras import backend as K
import tensorflow as tf

def KerasFocalLoss(target, input):
    
    gamma = 2.
    input = tf.cast(input, tf.float32)
    
    max_val = K.clip(-input, 0, 1)
    loss = input - input * target + max_val + K.log(K.exp(-max_val) + K.exp(-input - max_val))
    invprobs = tf.log_sigmoid(-input * (target * 2.0 - 1.0))
    loss = K.exp(invprobs * gamma) * loss
    
    return K.mean(K.sum(loss, axis=1))


# # Sanity check
# 
# Check that my Keras implementation returns same values.

# In[ ]:


import numpy as np
from fastai.conv_learner import *
from fastai.dataset import *


# credits: https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-448-public-lb#
# credits originally: https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):

        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()


# In[ ]:


# define some results
Y_true = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]])
Y_pred = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1]], dtype=np.float32)


# In[ ]:


fc = FocalLoss()

print(fc.forward(torch.from_numpy(Y_pred), torch.from_numpy(Y_true.astype(np.float32))))
print(K.eval(KerasFocalLoss(Y_true, Y_pred)))


# 
