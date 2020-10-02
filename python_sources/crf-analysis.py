#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import jaccard

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, unary_from_softmax, create_pairwise_gaussian


# Find the folder name

# In[ ]:


ls ../input -a


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


H = W = 96


# I define a simple IOU function to compare the results of the basic model with the different CRF models.
# 
# TODO: The jaccard function currently gives a RuntimeError when the mask is empty. Solution: preset the 'random' image...

# In[ ]:


def iou(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    return(1 - jaccard(y_true, y_pred))


# # Read in the data, masks, and predicted masks

# `x` is a numpy array of shape (batch_size, H, W).
# 
# It is the images for a specific batch.
# 
# 
# 
# `y` is a numpy array of shape (batch_size, H, W, 1).
# 
# It is the ground truth masks for a specific batch.
# 
# 
# 
# `preds_valid` is a numpy array of shape (batch_size, H, W, 1).
# 
# It is the result of the U-net model for a specific batch, given as probabilities.

# In[ ]:


images = np.load('../input/validation-set/x.npy')
true_masks = np.load('../input/validation-set/y.npy')[..., 0]
mask_probabilities = np.load('../input/validation-set/preds_valid.npy')[..., 0]


# A random image is selected.

# In[ ]:


ix = np.random.randint(images.shape[0])
ix = 87
img = images[ix, ..., 0]
mask = true_masks[ix]
mask_proba = mask_probabilities[ix]
mask_probas = np.rollaxis(np.stack([1 - mask_proba, mask_proba], axis = 2), 2, 0)


# The threshold is determined after training by iterating over multiple thresholds and calculating the mean precision over the given IOU thresholds for all validation images. This calculation is done in the training kernel.

# In[ ]:


threshold = 0.69
mask_pred = np.int32(mask_proba > threshold)


# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 2, sharey=True,sharex=True)
# ax1.set_aspect('equal')
ax1[0].imshow(img, cmap='seismic'); ax1[0].axis('off'); ax1[0].set_title('Input Image')
ax1[1].imshow(mask); ax1[1].axis('off'); ax1[1].set_title('Ground Truth')
ax2[0].imshow(mask_proba); ax2[0].axis('off'); ax2[0].set_title('Mask Probabilities')
ax2[1].imshow(mask_pred); ax2[1].axis('off'); ax2[1].set_title('Mask Prediction')
# plt.subplots_adjust(wspace=0.8)
plt.show()


# In[ ]:


initial_iou = iou(mask, mask_pred)


# # Construct the CRF

# During training, I downsampled the images slightly to fit in the U-net. I believe the number of labels should be two: 0 - background, 1 - salt. Also both class probabilies need to be used (i.e. mask_probabilities and 1 - mask_probabilities.

# In[ ]:


d_l = dcrf.DenseCRF2D(H, W, 2)
d_p = dcrf.DenseCRF2D(H, W, 2)


# I built the unary both from the mask probabilities and the mask predictions.

# In[ ]:


U_from_labels = unary_from_labels(mask_pred, 2, gt_prob=0.7, zero_unsure=False)
U_from_proba = unary_from_softmax(mask_probas)


# # Run inference with only the unary

# This is more of a baseline test. This method does not account for a pixel's neighbors.

# In[ ]:


d_l.setUnaryEnergy(U_from_labels)
d_p.setUnaryEnergy(U_from_proba)


# In[ ]:


Q_l = d_l.inference(10)
Q_p = d_p.inference(10)


# In[ ]:


map_l = np.argmax(Q_l, axis=0).reshape((H, W))
map_p = np.argmax(Q_p, axis=0).reshape((H, W))


# Inspect the unary and MAP from labels:

# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(U_from_labels.reshape((2, H,W))[0]); ax1.axis('off'); ax1.set_title('Unary from labels')
ax2.imshow(map_l); ax2.axis('off'); ax2.set_title('MAP from labels');


# Change in IOU

# In[ ]:


iou(mask, map_l) - initial_iou


# Inspect the unary and MAP from probabilities:

# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(U_from_proba.reshape((2, H,W))[0]); ax1.axis('off'); ax1.set_title('Unary from proba')
ax2.imshow(map_p); ax2.axis('off'); ax2.set_title('MAP from proba');


# Change in IOU

# In[ ]:


iou(mask, map_p) - initial_iou


# # Pairwise terms

# The whole point of DenseCRFs is to use some form of content to smooth out predictions. This is done via "pairwise" terms, which encode relationships between elements.

# ### Add bilateral term
# 
# The bilateral term makes the assumption that pixels with either a similar color or a similar location are likely to belong to the same class.
# 
# The sdims parameter defines the strenght of the location bilateral and the schan parameter defines the strength of the image content bilateral.

# In[ ]:


pairwise_bilateral = create_pairwise_bilateral(sdims=(5, 5), schan=(0.01,), img=np.expand_dims(img, -1), chdim=2)


# ### Run inference with the pairwise bilateral term

# In[ ]:


d_l = dcrf.DenseCRF2D(H, W, 2)
d_p = dcrf.DenseCRF2D(H, W, 2)


# In[ ]:


d_l.setUnaryEnergy(U_from_labels)
d_l.addPairwiseEnergy(pairwise_bilateral, compat=10)

d_p.setUnaryEnergy(U_from_proba)
d_p.addPairwiseEnergy(pairwise_bilateral, compat=10)


# Using the code from thepydensecrf repo, we can do the inference in steps to track intermediate solutions as well as the KL-divergence which indicates how well we have converged.

# In[ ]:


def run_inference(d):
    Q, tmp1, tmp2 = d.startInference()
    for _ in range(2):
        d.stepInference(Q, tmp1, tmp2)
    kl1 = d.klDivergence(Q) / (H*W)
    map_soln1 = np.argmax(Q, axis=0).reshape((H,W))

    for _ in range(8):
        d.stepInference(Q, tmp1, tmp2)
    kl2 = d.klDivergence(Q) / (H*W)
    map_soln2 = np.argmax(Q, axis=0).reshape((H,W))

    for _ in range(16):
        d.stepInference(Q, tmp1, tmp2)
    kl3 = d.klDivergence(Q) / (H*W)
    map_soln3 = np.argmax(Q, axis=0).reshape((H,W))
    return(map_soln1, kl1, map_soln2, kl2, map_soln3, kl3)


# ### Unary from labels

# In[ ]:


map_soln1, kl1, map_soln2, kl2, map_soln3, kl3 = run_inference(d_l)

img_en = pairwise_bilateral.reshape((-1, H, W))  # Reshape just for plotting
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(map_soln1);
plt.title('MAP Solution with DenseCRF\n(2 steps, KL={:.2f})'.format(kl1)); plt.axis('off');
plt.subplot(1,3,2); plt.imshow(map_soln2);
plt.title('MAP Solution with DenseCRF\n(8 steps, KL={:.2f})'.format(kl2)); plt.axis('off');
plt.subplot(1,3,3); plt.imshow(map_soln3);
plt.title('MAP Solution with DenseCRF\n(16 steps, KL={:.2f})'.format(kl3)); plt.axis('off');


# Change in IOU

# In[ ]:


iou(mask, map_soln3) - initial_iou


# ### Unary from proba

# In[ ]:


map_soln1, kl1, map_soln2, kl2, map_soln3, kl3 = run_inference(d_p)

img_en = pairwise_bilateral.reshape((-1, H, W))  # Reshape just for plotting
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(map_soln1);
plt.title('MAP Solution with DenseCRF\n(2 steps, KL={:.2f})'.format(kl1)); plt.axis('off');
plt.subplot(1,3,2); plt.imshow(map_soln2);
plt.title('MAP Solution with DenseCRF\n(8 steps, KL={:.2f})'.format(kl2)); plt.axis('off');
plt.subplot(1,3,3); plt.imshow(map_soln3);
plt.title('MAP Solution with DenseCRF\n(16 steps, KL={:.2f})'.format(kl3)); plt.axis('off');


# Change in IOU

# In[ ]:


iou(mask, map_soln3) - initial_iou


# # Summary

# The pairwise bilateral term usually degrades the final binary mask.
# 
# Creating the unary from the probabilities usually gives a better final binary mask than creating the unary from the predicted mask.

# # Futher work

# The summary is based on individually analyzing images. A more automated approach that considers all images should be used.
# 
# The convergence of the basic CRF model should be explored similarly to what is done with the pairwise bilateral model.
# 
# The hyperparameters of the pairwise bilateral term may not be optimal.
# 
# A pairwise gaussian term should be explored along with its hyperparameters.
# 
# Finally, the gt_prop (which used when creating the unary from labels) parameter may need to be tuned. It is interpreted as our confidence in the predicted mask, so it should be different for each image. I don't think it would be worth the effort to use a bayesian U-net model just for this parameter, especially when the unary from softmax performs much better already.
