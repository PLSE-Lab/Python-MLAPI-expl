#!/usr/bin/env python
# coding: utf-8

# We will predict number of fingers (not left/right hand)
# 
# The first mask I tried got 97% accuracy, then I lost that so I had to spend a while finding a mask which achieved the same test accuracy.

# In[ ]:


import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from pathlib import Path
import skimage.measure
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
Path.ls = lambda p: list(p.iterdir())


# In[ ]:


train = Path("../input/fingers/train")
test = Path("../input/fingers/test")


# # Create fist mask

# In[ ]:


train.ls()


# In[ ]:


fist = np.array(Image.open(train/"d4b08243-4cd3-493a-8616-e83c5ea23b7a_0R.png"))


# In[ ]:


fist_mask = fist > 80


# In[ ]:


plt.imshow(fist_mask)


# Fill in holes

# In[ ]:


for i in range(5):
    new_fist_mask = fist_mask.copy()
    for x in range(128):
        for y in range(128):
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    x_ = x + dx
                    y_ = y + dy
                    if 0 <= x_ < 128 and 0 <= y_ < 128:
                        if fist_mask[y_][x_]:
                            new_fist_mask[y][x] = True
    fist_mask = new_fist_mask


# In[ ]:


plt.imshow(fist_mask)


# In[ ]:


fist_mask[80:, ] = True


# In[ ]:


plt.imshow(fist_mask)


# # Use mask to leave only fingers

# In[ ]:


im = np.array(Image.open(train.ls()[140]))

fingers = im * (1-fist_mask) > 85
fingers = skimage.measure.label(fingers)


# In[ ]:


plt.imshow(fingers)


# In[ ]:


print(fingers.max(), "fingers")


# # Make test predictions

# In[ ]:


test_y = []
test_pred = []
    
for fn in tqdm(test.ls()):
    im = Image.open(fn)
    test_y.append(int(fn.name[-6:-5]))
    fingers = im * (1-fist_mask) > 85
    fingers = skimage.measure.label(fingers)
    pred = fingers.max()
    test_pred.append(pred)


# In[ ]:


accuracy_score(test_y, test_pred)

