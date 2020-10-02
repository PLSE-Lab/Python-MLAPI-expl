#!/usr/bin/env python
# coding: utf-8

# Loading all needed modules

# In[ ]:


get_ipython().system('apt install libturbojpeg0')
get_ipython().system('apt -y install libvips libvips-dev')
get_ipython().system('pip install jpeg4py')
get_ipython().system('pip install pyvips')


# In[ ]:


import glob
import time
import os
import jpeg4py as jpeg
from PIL import Image
import struct
import imghdr
import skimage.io
import imageio
import cv2
import shutil
import numpy as np
from io import BytesIO


# Get image links and copy images in memory (/dev/shm/)

# In[ ]:


INPUT_PATH_PNG = "../input/aptos2019-blindness-detection/train_images/"
files_png_init = sorted(glob.glob(INPUT_PATH_PNG + '*.png'))
files_png_init = files_png_init[:300]
print('PNG Files: {}'.format(len(files_png_init)))

os.mkdir('/dev/shm/1/')
files_png = []
for f in files_png_init:
    new_path = '/dev/shm/1/' + os.path.basename(f)
    shutil.copy(f, new_path)
    files_png.append(new_path)

INPUT_PATH_JPG_SMALL = "../input/open-images-2019-object-detection/test/"
files_jpg_small_init = sorted(glob.glob(INPUT_PATH_JPG_SMALL + '*.jpg'))
files_jpg_small_init = files_jpg_small_init[:3000]
print('JPG small files: {}'.format(len(files_jpg_small_init)))

os.mkdir('/dev/shm/2/')
files_jpg_small = []
for f in files_jpg_small_init:
    new_path = '/dev/shm/2/' + os.path.basename(f)
    shutil.copy(f, new_path)
    files_jpg_small.append(new_path)

INPUT_PATH_JPG_BIG = "../input/sp-society-camera-model-identification/train/"
files_jpg_big_init = sorted(glob.glob(INPUT_PATH_JPG_BIG + '*/*.jpg'))
files_jpg_big_init = files_jpg_big_init[:300]
print('JPG big files: {}'.format(len(files_jpg_big_init)))

os.mkdir('/dev/shm/3/')
files_jpg_big = []
for f in files_jpg_big_init:
    new_path = '/dev/shm/3/' + os.path.basename(f)
    shutil.copy(f, new_path)
    files_jpg_big.append(new_path)


# Test jpeg4py (libjpeg-turbo) speed

# In[ ]:


start_time = time.time()
d = []
for f in files_jpg_small:
    a = jpeg.JPEG(f).decode()
    d.append(a)
print('Time to read {} JPEGs small for libjpeg-turbo (jpeg4py): {:.2f} sec'.format(len(files_jpg_small), time.time() - start_time))

start_time = time.time()
d = []
for f in files_jpg_big:
    a = jpeg.JPEG(f).decode()
    d.append(a)
print('Time to read {} JPEGs big for libjpeg-turbo (jpeg4py): {:.2f} sec'.format(len(files_jpg_big), time.time() - start_time))


# Test cv2 with conversion from BGR to RGB format

# In[ ]:


start_time = time.time()
d = []
for f in files_jpg_small:
    b = cv2.imread(f)
    # b = np.transpose(b, (1, 0, 2))
    # b = np.flip(b, axis=0)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    d.append(b)
print('Time to read {} JPEGs small for cv2 with BGR->RGB conversion: {:.2f} sec'.format(len(files_jpg_small), time.time() - start_time))

start_time = time.time()
d = []
for f in files_jpg_big:
    b = cv2.imread(f)
    # b = np.transpose(b, (1, 0, 2))
    # b = np.flip(b, axis=0)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    d.append(b)
print('Time to read {} JPEGs big for cv2 with BGR->RGB conversion: {:.2f} sec'.format(len(files_jpg_big), time.time() - start_time))

start_time = time.time()
d = []
for f in files_png:
    b = cv2.imread(f)
    # b = np.transpose(b, (1, 0, 2))
    # b = np.flip(b, axis=0)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    d.append(b)
print('Time to read {} PNGs for cv2 with BGR->RGB conversion: {:.2f} sec'.format(len(files_png), time.time() - start_time))


# Test cv2 without conversion

# In[ ]:


start_time = time.time()
d = []
for f in files_jpg_small:
    b = cv2.imread(f)
    d.append(b)
print('Time to read {} JPEGs small for cv2 no conversion: {:.2f} sec'.format(len(files_jpg_small), time.time() - start_time))

start_time = time.time()
d = []
for f in files_jpg_big:
    b = cv2.imread(f)
    d.append(b)
print('Time to read {} JPEGs big for cv2 no conversion: {:.2f} sec'.format(len(files_jpg_big), time.time() - start_time))

start_time = time.time()
d = []
for f in files_png:
    b = cv2.imread(f)
    d.append(b)
print('Time to read {} PNGs for cv2 no conversion: {:.2f} sec'.format(len(files_png), time.time() - start_time))


# Test PIL Image

# In[ ]:


start_time = time.time()
d = []
for f in files_jpg_small:
    c = Image.open(f)
    c = np.array(c)
    d.append(c)
print('Time to read {} JPEGs small for PIL: {:.2f} sec'.format(len(files_jpg_small), time.time() - start_time))

start_time = time.time()
d = []
for f in files_jpg_big:
    c = Image.open(f)
    c = np.array(c)
    d.append(c)
print('Time to read {} JPEGs big for PIL: {:.2f} sec'.format(len(files_jpg_big), time.time() - start_time))

start_time = time.time()
d = []
for f in files_png:
    c = Image.open(f)
    c = np.array(c)
    d.append(c)
print('Time to read {} PNGs for PIL: {:.2f} sec'.format(len(files_png), time.time() - start_time))


# Test SKImage

# In[ ]:


start_time = time.time()
d = []
plugin = 'matplotlib'
for f in files_jpg_small:
    c = skimage.io.imread(f, plugin=plugin)
    c = np.array(c)
    d.append(c)
print('Time to read {} JPEGs small for skimage.io Plugin: {}: {:.2f} sec'.format(len(files_jpg_small), plugin, time.time() - start_time))

start_time = time.time()
d = []
plugin = 'matplotlib'
for f in files_jpg_big:
    c = skimage.io.imread(f, plugin=plugin)
    c = np.array(c)
    d.append(c)
print('Time to read {} JPEGs big for skimage.io Plugin: {}: {:.2f} sec'.format(len(files_jpg_big), plugin, time.time() - start_time))

start_time = time.time()
d = []
plugin = 'matplotlib'
for f in files_png:
    c = skimage.io.imread(f, plugin=plugin)
    c = np.array(c)
    d.append(c)
print('Time to read {} PNGs for skimage.io Plugin: {}: {:.2f} sec'.format(len(files_png), plugin, time.time() - start_time))


# Test Imageio

# In[ ]:


start_time = time.time()
d = []
for f in files_jpg_small:
    c = imageio.imread(f)
    d.append(c)
print('Time to read {} JPEGs small for Imageio (no rotate): {:.2f} sec'.format(len(files_jpg_small), time.time() - start_time))

start_time = time.time()
d = []
for f in files_jpg_big:
    c = imageio.imread(f)
    d.append(c)
print('Time to read {} JPEGs big for Imageio (no rotate): {:.2f} sec'.format(len(files_jpg_big), time.time() - start_time))

start_time = time.time()
d = []
for f in files_png:
    c = imageio.imread(f)
    d.append(c)
print('Time to read {} PNGs for Imageio (no rotate): {:.2f} sec'.format(len(files_png), time.time() - start_time))


# Test PyVips

# In[ ]:


import pyvips

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

start_time = time.time()
d = []
for f in files_png:
    c = pyvips.Image.new_from_file(f, access='sequential')
    c = np.ndarray(buffer=c.write_to_memory(),
                   dtype=format_to_dtype[c.format],
                   shape=[c.height, c.width, c.bands])
    d.append(c)
print('Time to read {} PNGs for PyVips: {:.2f} sec'.format(len(files_png), time.time() - start_time))

