#!/usr/bin/env python
# coding: utf-8

# In this notebook I will create images from signal data.  
# An output of the notebook may be used to create a model based on CNN.
# 
# The approach is taken from the following gist:
#     - https://gist.github.com/oguiza/26020067f499d48dc52e5bcb8f5f1c57
#  
# More info on  Gramian Angular Fields, Markov Transition Fields and  Recurrence Plots may be found here:
#     - https://medium.com/analytics-vidhya/encoding-time-series-as-images-b043becbdbf3 (GAF)
#     - http://coral-lab.umbc.edu/wp-content/uploads/2015/05/10179-43348-1-SM1.pdf (MTF)
#     - https://en.wikipedia.org/wiki/Recurrence_plot (RP)

# In[ ]:


get_ipython().system(' pip install pyts')


# In[ ]:


import csv
import os
import re
import numpy as np
import pickle
from pyts.approximation import PAA
from pyts.image import GADF, MTF, RecurrencePlots
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook
from time import time
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')


# # Signal -> images

# In[ ]:


input_csv_path = '../input/train.csv'
sample_size = 150_000
n_rows = 629145480
n_samples = n_rows // sample_size
IMG_SIZE = 512  # bigger images won't get inside of 5.2GB of Disk allocated by Kaggle kernel


# In[ ]:


def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
        
def _make_log10(num):
    if num == 0:
        return 0.
    new_num = np.log10(abs(num)) + 0.05
    if num < 0:
        new_num = -new_num
    return new_num

make_log10 = np.vectorize(_make_log10)

def create_image(array, name=None, folder=None, func=None):
    if func:
        array = func(array)
    uvts = PAA(output_size=IMG_SIZE).fit_transform([array])
    encoder1 = RecurrencePlots()
    encoder2 = MTF(IMG_SIZE, n_bins=IMG_SIZE//20, quantiles='gaussian')
    encoder3 = GADF(IMG_SIZE)
        
    r = np.squeeze(encoder1.fit_transform(uvts)) 
    g = np.squeeze(encoder2.fit_transform(uvts))
    b = np.squeeze(encoder3.fit_transform([array]))
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    shape = r.shape
    r = scaler.fit_transform(r.reshape(-1, 1)).reshape(shape)
    g = scaler.fit_transform(g.reshape(-1, 1)).reshape(shape)
    b = scaler.fit_transform(b.reshape(-1, 1)).reshape(shape)
    rgbArray = np.zeros((IMG_SIZE, IMG_SIZE, 3), 'uint8')
    rgbArray[..., 0] = r * 256
    rgbArray[..., 1] = g * 256
    rgbArray[..., 2] = b * 256
    
    if not (name and folder):
        plt.imshow(rgbArray)
    else:
        filename = name + ".png"
        plt.imsave(os.path.join(folder, filename),
                   rgbArray)

def create_dataset(path, n_samples, sample_size, folder):
    create_folder(folder)
    with open(path, "r") as read_f:
        reader = csv.reader(read_f)
        counter = 0
        pbar = tqdm_notebook(total=n_samples)
        row = np.zeros(sample_size)
        y = np.zeros(n_samples)
        next(reader)
        for val, ttf in reader:
            n = counter // sample_size
            m = counter % sample_size
            row[m] = int(val)
            if m == sample_size - 1:
                create_image(row.copy(), str(n), folder,
                             make_log10)
                y[n] = float(ttf)
                row = np.zeros(sample_size)
                pbar.update(1)
            counter += 1
        return y

def print_rand_images(folder, y=None):
    fig=plt.figure(figsize=(10, 12))
    columns, rows = 3, 3
    ax = []
    for i in range(columns*rows):
        img_name = np.random.choice(os.listdir(folder))
        path = os.path.join(folder, img_name)
        img = mpimg.imread(path)
        ax.append(fig.add_subplot(rows, columns, i+1))
        if y is not None:
            img_id = int(os.path.splitext(img_name)[0])
            img_name = f"ttf: {y[img_id]:.5f}"
        ax[-1].set_title(img_name)
        plt.imshow(img)
    plt.show()


# # Create images for training set

# In[ ]:


start = time()
y = create_dataset(input_csv_path, n_samples, sample_size, "train")
print("Completed in {} seconds".format(int(time()-start)))


#  ### Example of the resulting images:

# In[ ]:


print_rand_images("train", y)


# ### Saving results

# In[ ]:


with open("y_train.pkl", "wb") as f:
    f.write(pickle.dumps(y))


# In[ ]:


get_ipython().system(' tar -zcvf train.tar.gz train')


# In[ ]:


get_ipython().system(' rm -rf train')


# # Create images for test set

# In[ ]:


test_folder = "test"
create_folder(test_folder)
test_input_dir = "../input/test/"
test_files = [os.path.join(test_input_dir, x) for x in os.listdir(test_input_dir)]


# In[ ]:


start = time()
for file in tqdm_notebook(test_files):
    array = np.loadtxt(file, skiprows=1)
    name = os.path.splitext(os.path.basename(file))[0]
    create_image(array, name, test_folder, make_log10)
print("Completed in {} seconds".format(int(time()-start)))


#  ### Example of the resulting images:

# In[ ]:


print_rand_images(test_folder)


# ### Saving results

# In[ ]:


get_ipython().system(' tar -zcvf test.tar.gz test')


# In[ ]:


get_ipython().system(' rm -rf test')

