#!/usr/bin/env python
# coding: utf-8

# ## Motivation

# If someone is going to use pre-trained computer vision models, it might be useful first to scale the data in the range from 0 to 255. First of all, one needs to window the CT slices correctly. This windowing approach is based on the description written [here](http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html). More about CT and Windowing is [here](https://www.youtube.com/watch?v=VnpqylFYtqI). I hope that radiologists in Kaggle can verify this.

# In[ ]:


from os.path import join
from glob import glob
import pandas as pd
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Windowing function for Default LINEAR Function

# In[ ]:


def get_first_val(val):
    """Returns first tag value."""
    return val[0] if type(val) == pydicom.multival.MultiValue else val


def window_ct(dcm, c, w, ymin, ymax):
    """Windows a CT slice.
    http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html
    Note: Windows Default LINEAR Function, there are other functions such as LINEAR_EXACT and sigmoid

    Args:
        dcm (pydicom.dataset.FileDataset): CT slice.
        c: Window Center parameter.
        w: Window Width parameter.
        ymin: Minimum output value.
        ymax: Maximum output value.

    Returns:
        Windowed slice.
    """
    # convert to HU
    b = dcm.RescaleIntercept
    m = dcm.RescaleSlope
    x = m * dcm.pixel_array + b

    # windowing C.11.2.1.2.1 Default LINEAR Function
    y = np.zeros_like(x)
    y[x <= (c - 0.5 - (w - 1) / 2)] = ymin
    y[x > (c - 0.5 + (w - 1) / 2)] = ymax
    y[(x > (c - 0.5 - (w - 1) / 2)) & (x <= (c - 0.5 + (w - 1) / 2))] =         ((x[(x > (c - 0.5 - (w - 1) / 2)) & (x <= (c - 0.5 + (w - 1) / 2))] - (c - 0.5)) / (w - 1) + 0.5) * (
                ymax - ymin) + ymin

    return y


# ## Load data

# In[ ]:


DATA_DIR = "../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/"
TRAIN_CSV = "../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv"

train_df = pd.read_csv(TRAIN_CSV)
train_df['image'] = train_df['ID'].str.slice(stop=12)
train_df['diagnosis'] = train_df['ID'].str.slice(start=13)
train_df.head()


# ## Sample slice for each of the labels

# In[ ]:


titles = []
imgs = []
labels = train_df["diagnosis"].unique()

# Grayscale
ymin = 0
ymax = 255
random_seed = 2019
for label in labels:
    img_id = train_df[(train_df["diagnosis"] == label) & 
                      (train_df["Label"] == 1)].sample(1, random_state=random_seed).image.values[0]
    dcm = pydicom.dcmread(join(DATA_DIR, img_id + ".dcm"))
    
    # default image without windowing
    imgs.append(dcm.pixel_array)
    titles.append("{}, Label: {}".format(img_id, label))
    
    # default parameters from .dcm file
    c = int(get_first_val(dcm.WindowCenter))
    w = int(get_first_val(dcm.WindowWidth))
    imgs.append(window_ct(dcm, c, w, ymin, ymax))
    titles.append("Windowing (.dcm): c={}, w={}".format(c, w))
    
    # brain specific parameters
    c = 40
    w = 80
    imgs.append(window_ct(dcm, c, w, ymin, ymax))
    titles.append("Windowing (brain): c={}, w={}".format(c, w))


# ## Show slices with different windowing parameters

# In[ ]:


fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(15, 30))
for i, ax in enumerate(axs.flat):
    ax.imshow(imgs[i], cmap="bone")
    ax.set_title(titles[i])
    ax.set_yticks([])
    ax.set_xticks([])


# I would appreciate any feedback. Thanks.

# In[ ]:




