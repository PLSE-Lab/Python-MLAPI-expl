#!/usr/bin/env python
# coding: utf-8

# I was curious about how much an adversarial image can be, with the limits of 4-16 infinity norm.
# So I generated this NB to share my results.

# In[ ]:


import numpy as np # linear algebra
import cv2
import matplotlib.pyplot as plt


# I notice that the infinity norm is related to the range of a uniform distribution...

# In[ ]:


np.linalg.norm(np.random.uniform(0, 16, size=1000), ord=np.inf)


# So I applied uniform noise to a target image in order to visualize its effects. Here are the results:

# In[ ]:


im_norm = lambda i: (i.astype(np.float32)-i.min()) / (i.max()-i.min())

im = cv2.imread('../input/nips-2017-adversarial-learning-development-set/images/be1c43056c5d0085.png')

im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

im_adversarial = lambda pert: im + np.random.uniform(0, pert, size=im.shape)

fig, ax = plt.subplots(4, 2, figsize=(14, 28))

ax = ax.ravel()

perturbations = [4, 8, 16, 32, 64, 128, 256]

ax[0].imshow(im_norm(im), interpolation='nearest')
ax[0].set_title('Original')

for i, perturbation in enumerate(perturbations, 1):
    ax[i].imshow(im_norm(im_adversarial(perturbation)), interpolation='nearest')
    ax[i].set_title('Perturbation @{}'.format(perturbation))


# Final note: I'm not 100% sure this implementation is correct regarding the inf-norm... please comment if I made some mistake.
