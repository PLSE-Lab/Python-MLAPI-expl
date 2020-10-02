#!/usr/bin/env python
# coding: utf-8

# # Special Thanks To @dlibenzi (github) for all his help;

# In[ ]:


get_ipython().system('echo $TPU_NAME')


# In[ ]:


get_ipython().system('env')


# In[ ]:


import os; os.environ["XRT_TPU_CONFIG"] = "tpu_worker;0;10.0.0.2:8470"


# In[ ]:


import collections
from datetime import datetime, timedelta
import os
import tensorflow as tf
import numpy as np
import requests, threading

_VersionConfig = collections.namedtuple('_VersionConfig', 'wheels,server')
VERSION = "torch_xla==nightly"
CONFIG = {
    'torch_xla==nightly': _VersionConfig('nightly', 'XRT-dev{}'.format(
        (datetime.today() - timedelta(1)).strftime('%Y%m%d'))),
}[VERSION]

DIST_BUCKET = 'gs://tpu-pytorch/wheels'
TORCH_WHEEL = 'torch-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)
TORCH_XLA_WHEEL = 'torch_xla-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)
TORCHVISION_WHEEL = 'torchvision-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)


# In[ ]:


CONFIG.wheels


# In[ ]:


get_ipython().system('export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH')
get_ipython().system('apt-get install libomp5 -y')
get_ipython().system('apt-get install libopenblas-dev -y')

# Install COLAB TPU compat PyTorch/TPU wheels and dependencies
get_ipython().system('pip uninstall -y torch torchvision')
get_ipython().system('gsutil cp "$DIST_BUCKET/$TORCH_WHEEL" .')
get_ipython().system('gsutil cp "$DIST_BUCKET/$TORCH_XLA_WHEEL" .')
get_ipython().system('gsutil cp "$DIST_BUCKET/$TORCHVISION_WHEEL" .')
get_ipython().system('pip install "$TORCH_WHEEL"')
get_ipython().system('pip install "$TORCH_XLA_WHEEL"')
get_ipython().system('pip install "$TORCHVISION_WHEEL"')


# In[ ]:


import torch_xla.core.xla_model as xm
import torch_xla.distributed.data_parallel as dp # http://pytorch.org/xla/index.html#running-on-multiple-xla-devices-with-multithreading
import torch_xla.distributed.xla_multiprocessing as xmp # http://pytorch.org/xla/index.html#running-on-multiple-xla-devices-with-multiprocessing
import torch_xla.distributed.parallel_loader as pl


# In[ ]:


from kaggle_datasets import KaggleDatasets

# Data access
GCS_DS_PATH = KaggleDatasets().get_gcs_path()

# Configuration
IMAGE_SIZE = [512, 512]
EPOCHS = 20
BATCH_SIZE = 16 * 1

GCS_PATH_SELECT = { # available image sizes
    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',
    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',
    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',
    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'
}
GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]

TRAINING_FILENAMES   = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
TEST_FILENAMES       = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')


# In[ ]:


TRAINING_FILENAMES


# In the 100 flowers dataset, the format of each TFRecord of labeled data is:
#     - "image": list of bytestrings containing 1 bytestring (the JPEG-ecoded image bytes)
#     - "label": list of int64 containing 1 int64

# In[ ]:


# REFERENCE https://gist.githubusercontent.com/dlibenzi/c9868a1090f6f8ef9d79d2cfcbadd8ab/raw/947fbec325cbdeda91bd53acb5e126caa4115348/more_tf_stuff.py
# Thanks A Lot For Your Help!!!

from PIL import Image
import numpy as np
import hashlib
import os
import sys
import torch
import torch_xla.utils.tf_record_reader as tfrr

a = """
image/class/label       tensor([82])
image/class/synset      n01796340
image/channels  tensor([3])
image/object/bbox/label tensor([], dtype=torch.int64)
image/width     tensor([900])
image/format    JPEG
image/height    tensor([600])
image/class/text        ptarmigan
image/object/bbox/ymin  tensor([])
image/encoded   tensor([ -1, -40,  -1,  ..., -30,  -1, -39], dtype=torch.int8)
image/object/bbox/ymax  tensor([])
image/object/bbox/xmin  tensor([])
image/filename  n01796340_812.JPEG
image/object/bbox/xmax  tensor([])
image/colorspace        RGB
"""

def decode(ex):

    w = 512 # ex['image/width'].item()
    h = 512 # ex['image/height'].item()
    imgb = ex['image'].numpy().tobytes()
    
    # m = hashlib.md5()
    # m.update(imgb)
    # print('HASH = {}'.format(m.hexdigest()))
    
    image = Image.frombytes("RGB", (w, h), imgb,
                            "JPEG".lower(),
                            'RGB', None
                           )
    npa = np.asarray(image)
    return torch.from_numpy(npa), image


def readem(path, img_path=None):
    count = 0
    transforms = {}  
    r = tfrr.TfRecordReader(path, compression='', transforms=transforms)
    while True:
        ex = r.read_example()
        if not ex: break
        # print('\n')
        # for lbl, data in ex.items():
            # print('{}\t{}'.format(lbl, data))
        img_tensor, image = decode(ex)
        if img_path:
            image.save(os.path.join(img_path, str(count) + '.jpg'))
        count += 1
    print('\n\nDecoded {} samples'.format(count))


# In[ ]:


get_ipython().system('ls && pwd')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nimport os;\nfor idx, file in enumerate(TRAINING_FILENAMES):\n    img_path = f"/kaggle/working/flower_images_{idx}"\n    os.makedirs(img_path, exist_ok=True)\n    print(file)\n    readem(path = file, img_path = img_path)')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_figures(figures, nrows = 1, ncols=1):
    """
    Plot a dictionary of figures.
    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,20))
    for ind,title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.jet())
        # axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()


# In[ ]:


# generation of a dictionary of (title, images)
w, h = 10, 10
number_of_im = w*h

figures = {'im'+str(i): Image.open(f"./flower_images_0/{i}.jpg") for i in range(number_of_im)}

# plot of the images in a figure, with 5 rows and 4 columns
plot_figures(figures, w, h)
plt.show()


# In[ ]:


# generation of a dictionary of (title, images)
w, h = 10, 10
number_of_im = w*h

figures = {'im'+str(i): Image.open(f"./flower_images_1/{i}.jpg") for i in range(number_of_im)}

plot_figures(figures, w, h)
plt.show()


# In[ ]:


# generation of a dictionary of (title, images)
w, h = 10, 10
number_of_im = w*h

figures = {'im'+str(i): Image.open(f"./flower_images_2/{i}.jpg") for i in range(number_of_im)}

plot_figures(figures, w, h)
plt.show()


# In[ ]:


# generation of a dictionary of (title, images)
w, h = 10, 10
number_of_im = w*h

figures = {'im'+str(i): Image.open(f"./flower_images_3/{i}.jpg") for i in range(number_of_im)}

plot_figures(figures, w, h)
plt.show()


# In[ ]:


# generation of a dictionary of (title, images)
w, h = 10, 10
number_of_im = w*h

figures = {'im'+str(i): Image.open(f"./flower_images_10/{i}.jpg") for i in range(number_of_im)}

plot_figures(figures, w, h)
plt.show()

