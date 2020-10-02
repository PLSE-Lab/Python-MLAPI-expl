#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:





# ## convert dicom imgaes to jpeg

# In[ ]:





# In[ ]:


import os
import pydicom
import glob
from PIL import Image

inputdir = '../input/sample images/'
outdir = './'

test_list = [os.path.basename(x) for x in glob.glob(inputdir + './*.dcm')]
for f in test_list:  
    ds = pydicom.read_file( inputdir + f) # read dicom image
    img = ds.pixel_array # get image array
    img_mem = Image.fromarray(img) # Creates an image memory from an object exporting the array interface
    
#   There is an exception in Kaggle kernel about "encoder jpeg2k not available", please test following code on your local workstation
#   img_mem.save(outdir + f.replace('.dcm','.jp2'))


# In[ ]:


outdir


# In[ ]:





# In[ ]:


import os
import pydicom
import glob
import imageio

inputdir = '../input/'
outdir = './shukla1'
os.mkdir(outdir)
test_list = [os.path.basename(x) for x in glob.glob(inputdir + './*.dcm')]
for f in test_list:  
    ds = pydicom.read_file(inputdir + f) # read dicom image
    img = ds.pixel_array # get image array
    
#   There is an exception in Kaggle kernel about "encoder jpeg2k not available", please test following code on your local workstation
#   imageio.imwrite(outdir + f.replace('.dcm','.jp2'), img)


# In[ ]:


outdir


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf


# In[ ]:


pip install -q tensorflow-io


# In[ ]:


import tensorflow_io as tfio

image_bytes = tf.io.read_file('/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00136637202224951350618/253.dcm')

image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)

skipped = tfio.image.decode_dicom_image(image_bytes, on_error='skip', dtype=tf.uint8)

lossy_image = tfio.image.decode_dicom_image(image_bytes, scale='auto', on_error='lossy', dtype=tf.uint8)


fig, axes = plt.subplots(1,2, figsize=(10,10))
axes[0].imshow(np.squeeze(image.numpy()), cmap='gray')
axes[0].set_title('image')
axes[1].imshow(np.squeeze(lossy_image.numpy()), cmap='gray')
axes[1].set_title('lossy image');


# In[ ]:


lossy_image


# In[ ]:


#%reload_ext signature
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
init_notebook_mode(connected=True) 


# In[ ]:


data_path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00136637202224951350618/"
#os.mkdir(output_path)
output_path = working_path = "/image/"
#os.mkdir(output_path)
g = glob(data_path + '/*.dcm')

# Print out the first 5 file names to verify we're in the right folder.
print ("Total of %d DICOM images.\nFirst 5 filenames:" % len(g))
print('\n'.join(g[:5]))


# In[ ]:


def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

id=0
patient = load_scan(data_path)
imgs = get_pixels_hu(patient)


# In[ ]:


os.mkdir('/image1/')


# In[ ]:


np.save(output_path + "fullimages_%d.npy" % (id), imgs)


# In[ ]:


file_used=output_path+"fullimages_%d.npy" % id
imgs_to_process = np.load(file_used).astype(np.float64) 

plt.hist(imgs_to_process.flatten(), bins=50, color='c')
plt.xlabel("images data ")
plt.ylabel("Frequency")
plt.show()


# In[ ]:


id = 0
imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))

def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

sample_stack(imgs_to_process)


# In[ ]:


print("Slice Thickness: %f" % patient[0].SliceThickness)
print("Pixel Spacing (row, col): (%f, %f) " % (patient[0].PixelSpacing[0], patient[0].PixelSpacing[1]))


# In[ ]:


patient[0].SliceThickness


# This means we have 2.5 mm slices, and each voxel represents 0.7 mm.
# 
# Because a CT slice is typically reconstructed at 512 x 512 voxels, each slice represents approximately 370 mm of data in length and width.
# 
# Using the metadata from the DICOM we can figure out the size of each voxel as the slice thickness. In order to display the CT in 3D isometric form (which we will do below), and also to compare between different scans, it would be useful to ensure that each slice is resampled in 1x1x1 mm pixels and slices.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#spacing = np.array([slices[0].SliceThickness, slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]], dtype=np.float32)
#np.array([patient[0].SliceThickness , patient[0].PixelSpacing[0])


# In[ ]:





# In[ ]:


#return np.array(image, dtype=np.int16), np.array([slices[0].SliceThickness, slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]], dtype=np.float32)


# # 3D Plotting

# In[ ]:


#img = imgs_after_resamp[260]
#make_lungmask(img, display=True)


# FOR MORE DETAIL PLEASE MOVE TO SECOND NOTEBOOK
# https://www.kaggle.com/shubham9455999082/read-dicom-images-plotting-and-analysis?scriptVersionId=38404981

# In[ ]:




