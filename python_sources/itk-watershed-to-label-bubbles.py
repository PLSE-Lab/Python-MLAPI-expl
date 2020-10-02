#!/usr/bin/env python
# coding: utf-8

# # Example using ITK Watershed to Segment Bubbles
# Here we use the distance maps and watershed functions in ITK to segment bubbles in a 3D image. The advantage of using ITK over skimage / scipy is the tools are better built for 3D data, and can be easily combined into pipelines for efficiently running a number of steps sequentially. Additionally the ITK tools support a wide range of data-types some of which are better suited to complicated datasets like Diffusion Tensor Imaging and Diffraction experiments. 

# In[2]:


from skimage.io import imread, imsave
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.util.montage import montage2d # for showing slices in a montage
import itk


# In[3]:


stack_image = imread('../input/plateau_border.tif')
print(stack_image.shape, stack_image.dtype)


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))
for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):
    cax.imshow(np.sum(stack_image,i).squeeze(), interpolation='none', cmap = 'bone_r')
    cax.set_title('%s Projection' % clabel)
    cax.set_xlabel(clabel[0])
    cax.set_ylabel(clabel[1])


# # Create Bubble Image
# The bubble image is the reverse of the plateau border image (where there is water there can't be air) with a convex hull used to find the mask of the image
# $$ \text{Bubbles} = \text{ConvexHull}(\text{PlateauBorder}) - \text{PlateauBorder} $$

# In[5]:


from skimage.morphology import binary_opening, convex_hull_image as chull
bubble_image = np.stack([chull(csl>0) & (csl==0) for csl in stack_image])
plt.imshow(bubble_image[5], cmap = 'bone')


# # Watershed on Bubbles
# Here we use the ITK watershed operation (requires going to settings for the kernel and adding a itk to the list of *pip install* components. The code is adapted from this [example](https://itk.org/ITKExamples/src/Segmentation/Watersheds/SegmentWithWatershedImageFilter/Documentation.html) and the distance map example

# In[6]:


def apply_watershed(in_vol, 
                    threshold = 0.01, 
                    level = 0.5):
    """
    threshold:  is used to set the absolute minimum height value used during processing. 
        Raising this threshold percentage effectively decreases the number of local minima in the input, 
        resulting in an initial segmentation with fewer regions. 
        The assumption is that the shallow regions that thresholding removes are of of less interest.
    level: parameter controls the depth of metaphorical flooding of the image. 
        That is, it sets the maximum saliency value of interest in the result. 
        Raising and lowering the Level influences the number of segments 
        in the basic segmentation that are merged to produce the final output. 
        A level of 1.0 is analogous to flooding the image up to a 
        depth that is 100 percent of the maximum value in the image. 
        A level of 0.0 produces the basic segmentation, which will typically be very oversegmented. 
        Level values of interest are typically low (i.e. less than about 0.40 or 40% ), 
        since higher values quickly start to undersegment the image.
    """
    #(A rule of thumb is to set the Threshold to be about 1 / 100 of the Level.)
    Dimension = len(np.shape(in_vol))
    # convert to itk array and normalize
    itk_vol_img = itk.GetImageFromArray((in_vol*255.0).clip(0,255).astype(np.uint8))
    InputImageType = itk.Image[itk.ctype('unsigned char'), Dimension]
    OutputImageType = itk.Image[itk.ctype('float'), Dimension]
    dmapOp = itk.SignedMaurerDistanceMapImageFilter[InputImageType, OutputImageType].New(Input = itk_vol_img)
    dmapOp.SetInsideIsPositive(False)
    watershedOp = itk.WatershedImageFilter.New(Input=dmapOp.GetOutput())
    watershedOp.SetThreshold(threshold)
    watershedOp.SetLevel(level)
    watershedOp.Update()
    return itk.GetArrayFromImage(dmapOp), itk.GetArrayFromImage(watershedOp)


# In[7]:


get_ipython().run_cell_magic('time', '', 'dmap_vol, ws_vol = apply_watershed(bubble_image)')


# ## Cut-through
# We show the values at a slice in the middle as a way to get a feeling for what the watershed and distance map did

# In[8]:


mid_slice = ws_vol.shape[0]//2
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 7))
ax1.imshow(bubble_image[mid_slice], cmap ='bone')
ax1.set_title('Bubble Image')
m_val = np.abs(dmap_vol[mid_slice]).std()
ax2.imshow(dmap_vol[mid_slice], cmap = 'RdBu', vmin = -m_val, vmax = m_val)
ax2.set_title('Distance Image\nMin:%2.2f, Max:%2.2f, Mean: %2.2f' % (dmap_vol[mid_slice].min(),
                                                                    dmap_vol[mid_slice].max(),
                                                                    dmap_vol[mid_slice].mean()))
ax3.imshow(ws_vol[mid_slice], cmap = 'nipy_spectral')
ax3.set_title('Watershed\nLabels Found:{}'.format(len(np.unique(ws_vol[ws_vol>0]))));


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))
for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):
    cax.imshow(np.max(ws_vol,i).squeeze(), interpolation='none', cmap = 'nipy_spectral')
    cax.set_title('%s Projection' % clabel)
    cax.set_xlabel(clabel[0])
    cax.set_ylabel(clabel[1])


# # Remove Small and Relabel
# We remove the small bubbles and relabel in order. Additionally we perform a closing operation to clean up the images a bit and remove fluff from the bubbles

# In[11]:


from skimage.morphology import binary_opening, ball
bubble_label_image = np.zeros(ws_vol.shape).astype(np.uint16)
new_idx = 1
bubble_ids = [(idx, np.sum(ws_vol[ws_vol==idx]>0)) for idx in np.unique(ws_vol[ws_vol>0])]


# In[12]:


from tqdm import tqdm
for old_idx, vol in tqdm(sorted(bubble_ids, key = lambda x: x[1])):
    if vol>40000 and vol<400000:
        old_img = ws_vol==old_idx
        # bubbles are round
        cleaned_img = binary_opening(old_img, ball(3))
        bubble_label_image[old_img] = new_idx
        new_idx += 1
print(new_idx, 'total bubbles kept of ', len(bubble_ids))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))
for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):
    cax.imshow(np.max(bubble_label_image,i), 
               interpolation='none', 
               cmap = plt.cm.jet, vmin = 0, vmax = new_idx)
    cax.set_title('%s Projection' % clabel)
    cax.set_xlabel(clabel[0])
    cax.set_ylabel(clabel[1])


# # Show 3D Rendering

# In[ ]:


from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from tqdm import tqdm
def show_3d_mesh(image, thresholds):
    p = image[::-1].swapaxes(1,2)
    cmap = plt.cm.get_cmap('nipy_spectral_r')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i, c_threshold in tqdm(list(enumerate(thresholds))):
        verts, faces, _, _ = measure.marching_cubes(p==c_threshold, 0)
        mesh = Poly3DCollection(verts[faces], alpha=0.25, edgecolor='none', linewidth = 0.1)
        mesh.set_facecolor(cmap(i / len(thresholds))[:3])
        mesh.set_edgecolor([1, 0, 0])
        ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
    ax.view_init(45, 45)
    return fig


# In[ ]:


_ = show_3d_mesh(bubble_label_image, range(1,np.max(bubble_label_image), 10))


# # Calculate Bubble Centers

# In[ ]:


def meshgrid3d_like(in_img):
    return np.meshgrid(range(in_img.shape[1]),range(in_img.shape[0]), range(in_img.shape[2]))
zz, xx, yy = meshgrid3d_like(bubble_label_image)


# In[ ]:


out_results = []
for c_label in np.unique(bubble_label_image): # one bubble at a time
    if c_label>0: # ignore background
        cur_roi = bubble_label_image==c_label
        out_results += [{'x': xx[cur_roi].mean(), 'y': yy[cur_roi].mean(), 'z': zz[cur_roi].mean(), 
                         'volume': np.sum(cur_roi)}]


# In[ ]:


import pandas as pd
out_table = pd.DataFrame(out_results)
out_table.to_csv('bubble_volume.csv')
out_table.sample(5)


# In[ ]:


out_table['volume'].plot.density()


# In[ ]:


out_table.plot.hexbin('x', 'y', gridsize = (5,5))


# # Compare with the Training Values

# In[ ]:


train_values = pd.read_csv('../input/bubble_volume.csv')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
_, n_bins, _ = ax1.hist(np.log10(train_values['volume']), bins = 20, label = 'Training Volumes')
ax1.hist(np.log10(out_table['volume']), n_bins, alpha = 0.5, label = 'Watershed Volumes')
ax1.legend()
ax1.set_title('Volume Comparison\n(Log10)')
ax2.plot(out_table['x'], out_table['y'], 'r.',
        train_values['x'], train_values['y'], 'b.')
ax2.legend(['Watershed Bubbles', 
            'Training Bubbles'])


# In[ ]:





# In[ ]:




