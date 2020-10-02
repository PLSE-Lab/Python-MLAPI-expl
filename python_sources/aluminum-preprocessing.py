#!/usr/bin/env python
# coding: utf-8

# # Setup
# Load the packages and make plots look nice-ish

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2 # opencv 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage.color import rgb2gray
from skimage.util import montage as montage2d
from tqdm import tqdm_notebook
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})


# # Read and Process Data

# In[ ]:


def video_to_frames(in_path):
    """Read video and output frames and time-stamps as a generator."""
    c_cap = cv2.VideoCapture(in_path)
    if (not c_cap.isOpened()): 
        raise ValueError("Error opening video stream or file")
    status = True
    while status:
        ret, frame = c_cap.read()
        if not ret: break
        time_stamp = c_cap.get(cv2.CAP_PROP_POS_MSEC)
        yield time_stamp, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    c_cap.release()


# In[ ]:


video_path = '../input/slice821_tser.avi'
video_src = video_to_frames(video_path)
video_src


# In[ ]:


ts_0, frame_0 = next(video_src)
plt.imshow(frame_0)
print(ts_0)


# ## Import all Frames

# In[ ]:


time_axis = []
frame_axis = []
for ts, frame in tqdm_notebook(video_to_frames(video_path)):
    time_axis += [ts]
    frame_axis += [rgb2gray(frame)]
    


# ## Combine all Frames

# In[ ]:


time_vec = np.stack(time_axis, 0)
frame_vec = np.stack(frame_axis, 0)
del time_axis
del frame_axis
print(time_vec.shape, frame_vec.shape)


# ## Visualize All Frames

# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
ax1.imshow(montage2d(frame_vec[::2, ::4, ::4]), cmap='gray')


# # Image Enhancement

# In[ ]:


from scipy.ndimage import median_filter, zoom
# filter and downsample
filt_frame = zoom(median_filter(frame_vec, [2, 3, 3]), [1, 0.5, 0.5])


# In[ ]:


# keep memory usage low
import gc
del frame_vec # remove the unfiltered data
gc.collect()


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(15, 15))
ax1.imshow(montage2d(filt_frame[::2, ::4, ::4]), cmap='gray')


# # Segmentation

# In[ ]:


from skimage.filters import try_all_threshold
try_all_threshold(filt_frame[10]) # first 
try_all_threshold(filt_frame[filt_frame.shape[0]//2]) # middle
try_all_threshold(filt_frame[-1]) # last


# ## Mean
# It looks like the mean value looks best, we can try applying it slice by slice or by using a constant for the whole image (green vs blue)

# In[ ]:


from skimage.filters import threshold_mean
fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
bins = np.linspace(0, 1, 255)
ax1.hist(filt_frame.ravel(), bins)
ax1.axvline(threshold_mean(filt_frame), label='Whole Image Threshold', c='b', lw=2)
for i, c_slice in enumerate(filt_frame):
    ax1.axvline(threshold_mean(c_slice), label='Slice Threshold', c='g', alpha=0.25)
    if i==0:
        ax1.legend()
ax1.set_yscale('log')
ax1.set_ylabel('Pixel Count')
ax1.set_xlabel('Intensity')


# ## Applying Threshold
# We can try applying the threshold to the whole stack or frame by frame and compare the results. If they are similar we prefer the same threshold for the whole image since it is more consistent between slices

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
# whole image
ax1.imshow(montage2d(filt_frame[::12, ::4, ::4]>threshold_mean(filt_frame)), cmap='gray')
ax1.set_title('Whole Image Threshold')
ax2.imshow(montage2d(np.stack([c_img>threshold_mean(c_img) for c_img in filt_frame[::12]],0)[:, ::4, ::4]), cmap='gray')
ax2.set_title('Slice-based Threshold')


# In[ ]:


seg_frames = filt_frame>threshold_mean(filt_frame)


# ## Segmenting Bubbles
# We have now segmented the aluminum, but we are interested in the bubbles. We can use convex hull in order to compute the boundary of aluminum 

# In[ ]:


from skimage.morphology import convex_hull_image
hull_frames = np.stack([convex_hull_image(c_img) for c_img in seg_frames], 0)
bubble_frames = (hull_frames^seg_frames) & hull_frames
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
# whole image
ax1.imshow(montage2d(hull_frames[::12, ::4, ::4]), cmap='gray')
ax1.set_title('Convex Hull')
ax2.imshow(montage2d(bubble_frames[::12, ::4, ::4]), cmap='gray')
ax2.set_title('Segmented Bubbles')


# ## Cleaning up Hull Images
# We can see that the hulls are a bit inaccurate, so we can first remove small objects (erosion) in the aluminum image and then erode the convex hull to avoid messy boundaries

# In[ ]:


from skimage.morphology import erosion, ball
hull_frames = erosion( # erode the hull
    np.stack([convex_hull_image(c_img) 
              for c_img in 
              erosion(seg_frames, ball(3)) # erode the segmentation
             ], 0), 
    ball(3))

bubble_frames = (hull_frames^seg_frames) & hull_frames

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
# whole image
ax1.imshow(montage2d(hull_frames[::12, ::4, ::4]), cmap='gray')
ax1.set_title('Convex Hull')
ax2.imshow(montage2d(bubble_frames[::12, ::4, ::4]), cmap='gray')
ax2.set_title('Segmented Bubbles')


# In[ ]:


bubble_frames.shape


# In[ ]:


from skimage.io import imsave
imsave('bubbles.tif', bubble_frames.astype(np.uint8)) # convert to 8-bit format to save


# # Labeling Bubbles

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from skimage.measure import label\nbubble_labels = label(bubble_frames)')


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
ax1.hist(bubble_labels[bubble_labels>0], np.arange(1+np.max(bubble_labels.ravel())))
ax1.set_yscale('log')
ax1.set_title('Connected Size Distribution')
ax1.set_xlabel('Component Number')
ax1.set_ylabel('Pixel Count')


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
bin_counts, _ = np.histogram(bubble_labels[bubble_labels>0], np.arange(1+np.max(bubble_labels.ravel())))
ax1.hist(bin_counts, np.logspace(0, 7, 30))
ax1.set_xscale('log')
ax1.set_title('Object Size')
ax1.set_xlabel('Number of Components')
ax1.set_xlabel('Pixel Count')


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))
# whole image
ax1.imshow(montage2d(bubble_frames[::12, ::4, ::4]), cmap='gray')
ax1.set_title('Convex Hull')
ax2.imshow(montage2d(bubble_labels[::12, ::4, ::4]), cmap=plt.cm.nipy_spectral)
ax2.set_title('Segmented Bubbles')


# ## Labeling and Shape Analysis
# Run the labeling individually on each slice and then track the bubbles

# In[ ]:


from warnings import warn
import warnings
def scalar_attributes_list(im_props):
    """
    Makes list of all scalar, non-dunder, non-hidden
    attributes of skimage.measure.regionprops object
    """

    attributes_list = []

    for i, test_attribute in enumerate(dir(im_props[0])):

        # Attribute should not start with _ and cannot return an array
        # does not yet return tuples
        try:
            if test_attribute[:1] != '_' and not                     isinstance(getattr(im_props[0], test_attribute),
                               np.ndarray):
                attributes_list += [test_attribute]
        except Exception as e:
            warn("Not implemented: {} - {}".format(test_attribute, e),
                 RuntimeWarning)

    return attributes_list

def regionprops_to_df(im_props):
    """
    Read content of all attributes for every item in a list
    output by skimage.measure.regionprops
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        attributes_list = scalar_attributes_list(im_props)

    # Initialise list of lists for parsed data
    parsed_data = []

    # Put data from im_props into list of lists
    for i, _ in enumerate(im_props):
        parsed_data += [[]]

        for j in range(len(attributes_list)):
            parsed_data[i] += [getattr(im_props[i], attributes_list[j])]

    # Return as a Pandas DataFrame
    return pd.DataFrame(parsed_data, columns=attributes_list)


# In[ ]:


from skimage.measure import label, regionprops
region_frames = [regionprops(label(c_bubbles)) for c_bubbles in bubble_frames]


# In[ ]:


from IPython.display import clear_output
all_regions_df = pd.concat([regionprops_to_df(c_frame).assign(time_ms=c_ts) 
                            for c_ts, c_frame in zip(time_vec, region_frames) 
                            if len(c_frame)>0], sort=False).reset_index(drop=True)
clear_output() # makes a huge mess
all_regions_df.sample(3)


# In[ ]:


all_regions_df['area'].hist(bins=np.linspace(0, 1000, 50))


# In[ ]:


all_regions_df['x'] = all_regions_df['centroid'].map(lambda x: x[0])
all_regions_df['y'] = all_regions_df['centroid'].map(lambda x: x[1])
all_regions_df.to_csv('tracked_bubbles.csv', index=False)
all_regions_df.to_json('tracked_bubbles.json')


# # Animations

# In[ ]:


all_regions_df.query('area>50').plot.scatter('x', 'y')


# In[ ]:


import plotly_express as px
scatter_plot = px.scatter(x='x', y='y', 
                          animation_frame='time_ms', 
                          color='area', 
                          data_frame=all_regions_df.query('area>50'))
scatter_plot


# In[ ]:


import plotly
plotly.offline.plot(scatter_plot, filename='animation.html')


# In[ ]:


from matplotlib.animation import FuncAnimation
big_points_df = all_regions_df.query('area>100')
fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
ax1.plot(big_points_df['x'].values, big_points_df['y'].values, '.', alpha=0.1, label='All Points')

def draw_frame(in_df):
    [c_line.remove() 
       for c_ax in [ax1]
       for c_line in c_ax.get_lines() 
       if c_line.get_label().startswith('_')];
    ax1.plot(in_df['x'].values, in_df['y'].values, 'bs', alpha=0.5)
out_anim = FuncAnimation(fig, draw_frame, [c_df for _, c_df in big_points_df.groupby('time_ms')])


# In[ ]:


if False:
    from IPython.display import HTML
    HTML(out_anim.to_jshtml())
else:
    out_anim.save('alu_frames.gif', bitrate=8000, fps=8)

