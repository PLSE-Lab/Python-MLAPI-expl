#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from nilearn.image import resample_img
import pylab as plt
import nibabel as nb
import numpy as np


# In[ ]:


orig_nii = nb.load("../input/ixi-example/ixi002-guys-0828-t1.nii/IXI002-Guys-0828-T1.nii")


# In[ ]:


np.round(orig_nii.affine)


# In[ ]:


orig_nii.shape


# In[ ]:


orig_nii.header.get_zooms()


# In[ ]:


plt.imshow(orig_nii.dataobj[:,:,80])


# In[ ]:


orig_rotated_nii = nb.as_closest_canonical(orig_nii)


# In[ ]:


np.round(orig_rotated_nii.affine)


# In[ ]:


orig_rotated_nii.shape


# In[ ]:


plt.imshow(orig_rotated_nii.dataobj[:,:,80])


# # Downsampling

# In[ ]:


downsampled_nii = resample_img(orig_rotated_nii, target_affine=np.eye(3)*2., interpolation='nearest')


# In[ ]:


downsampled_nii.affine


# In[ ]:


downsampled_nii.shape


# In[ ]:


plt.imshow(downsampled_nii.dataobj[:,:,50])


# # Upsampling

# In[ ]:


upsampled_nii = resample_img(orig_rotated_nii, target_affine=np.eye(3)*0.5, interpolation='nearest')


# In[ ]:


upsampled_nii.affine


# In[ ]:


upsampled_nii.shape


# In[ ]:


plt.imshow(upsampled_nii.dataobj[:,:,200])


# # Resampling with cropping/padding

# In[ ]:


target_shape = np.array((240,40,100))
new_resolution = [2,]*3
new_affine = np.zeros((4,4))
new_affine[:3,:3] = np.diag(new_resolution)
# putting point 0,0,0 in the middle of the new volume - this could be refined in the future
new_affine[:3,3] = target_shape*new_resolution/2.*-1
new_affine[3,3] = 1.
downsampled_and_cropped_nii = resample_img(orig_rotated_nii, target_affine=new_affine, target_shape=target_shape, interpolation='nearest')


# In[ ]:


downsampled_and_cropped_nii.affine


# In[ ]:


downsampled_and_cropped_nii.shape


# In[ ]:


plt.imshow(downsampled_and_cropped_nii.dataobj[:,:,70])


# In[ ]:




