#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import dicom
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# Input data files are available in the "../input/" directory.

PATH_BASE = '../input'
print('{} {}\n'.format('Files:', os.listdir(PATH_BASE)))

EXT_SAMPLE_IMAGES = 'sample_images'
folders = os.listdir(os.path.join(PATH_BASE, EXT_SAMPLE_IMAGES))
print('# Folders in sample_images: {}'.format(len(folders)))    


# In[ ]:


patients = []
counter = 0
for i, folder in enumerate(folders):
    files = os.listdir(os.path.join(PATH_BASE, EXT_SAMPLE_IMAGES, folder))
    print('Folder {}: {} has {} files'.format(i+1, folder, len(files)))
    images = []
    for j, file in enumerate(files):
        images.append(dicom.read_file(os.path.join(PATH_BASE, EXT_SAMPLE_IMAGES, folder, file)))
    counter += len(images)
    images.sort(key = lambda z: int(z.InstanceNumber))
    patients.append(images)


# In[ ]:


n = len(patients[0])
print(n)


# ### Exploring the lung images

# In[ ]:


import matplotlib.animation as animation
get_ipython().run_line_magic('matplotlib', 'nbagg')

fig = plt.figure()
im = plt.imshow(np.zeros([512, 512]), cmap=plt.cm.bone , animated=True)

def update_fig(x):
    for i, patient in enumerate(patients):
        for j, image_slice in enumerate(patient):
            image = image_slice.pixel_array
            image[image == -2000] = 0
            im.set_array(image)
            return image

ani = animation.FuncAnimation(fig, update_fig, frames=range(counter), interval=50, blit=True)
plt.show()


# In[ ]:




