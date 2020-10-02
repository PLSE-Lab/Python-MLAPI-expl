#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # Import libraries
# we use skimage (scikit-image) to calculate properties, matplotlib to do the figures,  pandas to save output

# In[ ]:


from skimage.io import imread, imsave
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # a nice progress bar
import pandas as pd


# # Load in the image
# The image is a 3D tiff stack which is an output of tomographic reconstruction. The volume is already binarized (from gray scale image we decide which voxels will belong to the water and which to the air phase based on their gray values in the war reconstruction)
# The sample here is a liquid foam - very similar to a shampoo or washing-up liquid foam and was produced by C. Raufaste, B. Dollet and S. Santucci during a synchrotron experiment. The scan was acquired at the TOMCAT beamline at the Paul Scherrer Institut in Switzerland. 

# In[ ]:


stack_image = imread('../input/rec_8bit_ph03_cropC_kmeans_scale510.tif')
stack_image= stack_image[:,:,:]
print(stack_image.shape, stack_image.dtype)


# # Show projections through the binary volume 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))
for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):
    cax.imshow(np.sum(stack_image,i).squeeze(), interpolation='none', cmap = 'bone_r')
    cax.set_title('%s Projection' % clabel)
    cax.set_xlabel(clabel[0])
    cax.set_ylabel(clabel[1])


# In[ ]:


plt.imshow(stack_image[100],cmap='bone') # showing slice No.100


# # Create Bubble Image
# The bubble image is the reverse of the plateau border image (where there is water there can't be air) with a convex hull used to find the mask of the image
# $$ \text{Bubbles} = \text{ConvexHull}(\text{PlateauBorder}) - \text{PlateauBorder} $$
# This step is needed to select the cylinder which contains the foam and avoid the voxels that are outside the sample.

# In[ ]:


from skimage.morphology import binary_opening, convex_hull_image as chull
bubble_image = np.stack([chull(csl>0) & (csl==0) for csl in stack_image])
plt.imshow(bubble_image[5]>0, cmap = 'bone')


# # Invert image

# In[ ]:


bubble_inver=np.invert(bubble_image)


# In[ ]:


plt.imshow(bubble_inver[100], cmap='bone')
water = 0
air = 0
for layer in bubble_inver:
    for i in layer:
        for j in i:
            if j == 0:
                water += 1
            else:
                air += 1
print("The liquid fraction is", water/air)
            


# # Create a distance map
# Next we will give each black voxel (air) a value that corresponds to its distance to the closest white (water) voxel in the image.

# In[ ]:


from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt as distmap
bubble_dist = distmap(bubble_inver)


# In[ ]:


plt.imshow(bubble_dist[100,:,:], interpolation = 'none', cmap ='jet')


# # Calculate local maxima of the distance map
# The local maxima are those black voxels which have the maximum distance from the nearest white voxles. You can imagine this as being the centers of the air bubbles in the image. 

# In[ ]:


from skimage.feature import peak_local_max
bubble_candidates = peak_local_max(bubble_dist, min_distance=12)
print('Found',len(bubble_candidates), 'bubbles')


# These local maxima are the seeds of the future bubbles. We save them to a csv file which can be opened with any standard tool that can handle values in tables. In this table there is a list of the maxima found and their spatial coordinates. 

# In[ ]:


df = pd.DataFrame(data=bubble_candidates, columns=['x','y','z'])
df.to_csv('bubble.candidates.csv')


# # Watershed segmentation
# After finding the seeds we need to grow the bubbles. This is done by starting a 'flod' from the seeds and painting the voxels by a value that corresponds to the nearest seed number. This flod is done until neighbouring regions meet. Then we have all the bubbles in the system labeled. 
# We often have quite important uncertainties in this processa and should be replaced by something more robust in the future.

# In[ ]:


from skimage.morphology import watershed
bubble_seeds = peak_local_max(bubble_dist, min_distance=12, indices=False)
plt.imshow(np.sum(bubble_seeds,0).squeeze(), interpolation = 'none', cmap='bone_r')


# In[ ]:


markers = ndi.label(bubble_seeds)[0]
cropped_markers = markers[:,:,:]
cropped_bubble_dist=bubble_dist[:,:,:]
cropped_bubble_inver=bubble_inver[:,:,:]
labeled_bubbles= watershed(-cropped_bubble_dist, cropped_markers, mask=cropped_bubble_inver)
print(len(labeled_bubbles))


# In[ ]:


plt.imshow(labeled_bubbles[50,:,:], cmap=plt.cm.Spectral, interpolation='nearest')


# # Feature properties
# We find feature properties using the scikit-image library (for documentation and examples see: scikit-image.org)

# In[ ]:


from skimage.measure import regionprops
props=regionprops(labeled_bubbles)
props[20].filled_area
print(len(props))
filled = 0
endindex = len(props)
for i in range(0,endindex):
    filled += props[i].filled_area
print(filled)


# # Saving results
# just as we did for the bubble seeds we can save any other properties in csv files

# In[ ]:


bubble_volume=[prop.filled_area for prop in props]
bubble_volume_mean=np.mean(bubble_volume)
dfV = pd.DataFrame(data=bubble_volume, columns=['volume [pix^3]'])
dfV.to_csv('bubbleVolumes.csv')
Vm = {'mean volume': [1,bubble_volume_mean]}
dfVm=pd.DataFrame(data=Vm)


# # Saving processed 3D images
# For reference we can also save the labeled 3D volume. This can be opened in ImageJ for example 

# In[ ]:


from tifffile import imsave
imsave('labeled_bubbles.tif', labeled_bubbles)


# # Bubble shape and Area
# The shape of the bubbles is characterized by using the extent function giving the ratio of the volume of the bubble to that of the smallest bounding box. In the case of a sphere, the value would become pi/6 = 0.523.
# The bubble area is approximated by extracting the dimensions of the bounding box using bbox and then proceding to use an appriximate formula for the area of an ellipsoid calculated from the three half axes.
# 

# In[ ]:


bub_eccentricity = [prop.extent for prop in props]
dfV = pd.DataFrame(data=bub_eccentricity, columns=['eccentricity'])
dfV.to_csv('bubbleEccentricities.csv')

bub_bbox = [prop.bbox for prop in props]
approx_area = [4 * np.pi * np.power((np.power(t[3] * t[3] / 4,1.6075) + np.power(t[3] * t[5] / 4,1.6075) + np.power(t[4] * t[5] / 4,1.6075)) / 3,1/1.6075) for t in bub_bbox]
dfV = pd.DataFrame(data=approx_area, columns=['area [pix^2]'])
dfV.to_csv('bubbleArea.csv')


# 

# In[ ]:




