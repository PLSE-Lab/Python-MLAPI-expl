#!/usr/bin/env python
# coding: utf-8

# # 2. Understanding and plotting rle bounding boxes
# ### Airbus Ship Detection Challenge - A quick overview for computer vision noobs
# 
# &nbsp;
# 
# 
# Hi, and welcome! This is the second kernel of the series `Airbus Ship Detection Challenge - A quick overview for computer vision noobs.` In this short kernel we will explain the run-length encoded bounding boxes, translate the rle code into a list of pixels with pure python and plot that list of pixels as a mask on top of the pictures with matplotlib.
# 
# 
# The full series consist of the following notebooks:
# 1. [Loading and visualizing the images](https://www.kaggle.com/julian3833/1-loading-and-visualizing-the-images)
# 2. *[Understanding and plotting rle bounding boxes](https://www.kaggle.com/julian3833/2-understanding-and-plotting-rle-bounding-boxes)*
# 3. [Basic exploratory analysis](https://www.kaggle.com/julian3833/3-basic-exploratory-analysis)
# 4. [Exploring public models](https://www.kaggle.com/julian3833/4-exploring-models-shared-by-the-community)
# 5. [1.0 submission: submitting the test file](https://www.kaggle.com/julian3833/5-1-0-submission-submitting-the-test-file)
# 
# This is an ongoing project, so expect more notebooks to be added to the series soon. Actually, we are currently working on the following ones:
# * Understanding and exploiting the data leak
# * A quick overview of image segmentation domain
# * Jumping into Pytorch
# * Understanding U-net
# * Proposing a simple improvement to U-net model

# <a id='understand'></a>
# # 1. Understanding run-length encoding
# 
# &nbsp;
# 
# There is no clear information about this encoding on the Challenge's Data tab - may be it's too obvious? : in any case, it is definitely new for us. There are some comments about the encoding  on the [Evaluation](https://www.kaggle.com/c/airbus-ship-detection#evaluation) tab and - yes - there is an entry on [wikipedia](https://es.wikipedia.org/wiki/Run-length_encoding) explaining the  `run-length encoding` idea.  RLE, for short, is a simple morse-like representation of shapes in 2d images. In this case, what's encoded are some rectangular shapes - the bounding boxes - where the ships are located in the respective images.
# 
# The encoded string looks like this: `start, length, start, length, ...` , where each pair of (`start`, `length`) draws a line of `length` pixeles starting from position `start.`  The `start` position, in turn, is not a  `(x, y)` coordinate but an index of the 1-d array resulting of flattening the 2-d image into a rows-after-row 1-d sequence of pixels.  Knowing the shape of the images we can just unfold this 1-d representating into a 2-dimensions mask using  `//` and `%`.  
# 
# Let's start by checking a csv for a rle code example. It's stored in the column `EncodedPixels`:

# In[ ]:


import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../input/train_ship_segmentations_v2.csv", index_col=0).dropna()
display(df.head())
df['EncodedPixels']['000155de5.jpg']


# Ok: let's parse it with pure python. In the next cell, we just map this string into a list of (`start`, `length`) pairs:

# In[ ]:


# turn rle example into a list of ints
rle = [int(i) for i in df['EncodedPixels']['000155de5.jpg'].split()]
# turn list of ints into a list of (`start`, `length`) `pairs`
pairs = list(zip(rle[0:-1:2], rle[1::2])) 
pairs[:3]


# On the other hand, we can trivially encode and decode from a `start` scalar position like `264661` into a 2-d coordinate in our 768$\times$768 pictures using `%`, `//` and `*`:

# In[ ]:


start = pairs[0][0]
print(f"Original start position: {start}")

coordinate = (start % 768, start // 768)
print(f"Maps to this coordinate: {coordinate}")

back = 768 * coordinate[1] + coordinate[0]
print(f"And back: {back}")


# With this in mind, we can map the list of (`start`, `length`) pairs into a list of `pixels` very easily in one line.
# There are some python gotchas so let's comment a little what does this line do:
# 1. Map each pair (`start`, `length`) into a list of `positions` [`start`, `start + 1`, `...` `start + length`] using <span style='color:green'>range</span>
# 2. Flatten those lists using a `nested for` (note: python's [nested for](https://stackoverflow.com/questions/17657720/python-list-comprehension-double-for/17657966) looks weird)
# 3. Map the list of  `positions` into a list of (`x`, `y`) `coordinates` using `%` and `//` as explained above.

# In[ ]:


pixels = [(pixel_position % 768, pixel_position // 768) 
                            for start, length in pairs 
                            for pixel_position in range(start, start + length)]
pixels[:3]


# Finally, the following function puts it all together, translating from the RLE string into a list of pixels in a (768, 768) image:

# In[ ]:


def rle_to_pixels(rle_code):
    '''
    Transforms a RLE code string into a list of pixels of a (768, 768) canvas
    '''
    rle_code = [int(i) for i in rle_code.split()]
    pixels = [(pixel_position % 768, pixel_position // 768) 
                 for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2])) 
                 for pixel_position in range(start, start + length)]
    return pixels

# First three pixels of this particular bounding box:
rle_to_pixels(df['EncodedPixels']['000155de5.jpg'])[0:3]


# # 2. Plotting the bounding boxes as a mask

# The first thing we can do with this list of pixels is to plot them on a monochrome (768, 768) map.  To do that, we need to create a `0` and `1` mask matrix and plot it using [imshow](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html)  as we covered in the [previous kernel](https://www.kaggle.com/julian3833/1-loading-and-visualizing-images) of the series:

# In[ ]:


# Create a matrix of shape (768, 768) full of zeros
canvas = np.zeros((768, 768))

# numpy arrays can't be indexed by a list of pairs [(x1, y1), (x2, y2)]
# but it can be indexed with a tuple with ([x1, x2,..., xn], [y1, y2... yn])
# tuple(zip(*)) does exactly this map.... 
# ref: https://stackoverflow.com/questions/28491230/indexing-a-numpy-array-with-a-list-of-tuples
canvas[tuple(zip(*pixels))] = 1

plt.imshow(canvas);


# Running the cell below, you can get some random samples of the bounding boxes (Press Ctrl+Enter to run and stay in the cell):

# In[ ]:


canvas = np.zeros((768, 768))
pixels = rle_to_pixels(np.random.choice(df['EncodedPixels']))
canvas[tuple(zip(*pixels))] = 1
plt.imshow(canvas);


# # 3. Masking the images with the bounding boxes

# To finish, we will plot these masks over the corresponding images.  We can apply a mask to an image by just overriding some colors for the relevant pixels (the ones obtained with `rle_to_pixels()`). In this case we saturate completely the red and green coordinates and leave the blue one as it was.
# 
# This cell uses some simple functions (open, array, imshow) covered on the [previous kernel](https://www.kaggle.com/julian3833/1-loading-and-visualizing-the-images) of the series.

# In[ ]:


# An image may have more than one row in the df, 
# Meaning that the image has more than one ship present
# Here we merge those n-ships into the a continuos rle-code for the image....
df = df.groupby("ImageId")[['EncodedPixels']].agg(lambda rle_codes: ' '.join(rle_codes)).reset_index()

load_img = lambda filename: np.array(PIL.Image.open(f"../input/train_v2/{filename}"))

def apply_mask(image, mask):
    for x, y in mask:
        image[x, y, [0, 1]] = 255
    return image

img = load_img(df.loc[0, 'ImageId'])
mask_pixels = rle_to_pixels(df.loc[0, 'EncodedPixels'])
img = apply_mask(img, mask_pixels)
plt.imshow(img);


# To summarize, we present some random examples of masked pictures on a grid  (we covered subplot on [this](https://www.kaggle.com/julian3833/1-loading-and-visualizing-the-images) kernel):

# In[ ]:


w = 6
h = 6

_, axes_list = plt.subplots(h, w, figsize=(2*w, 2*h))

for axes in axes_list:
    for ax in axes:
        ax.axis('off')
        row_index = np.random.randint(len(df)) # take a random row from the df
        ax.imshow(apply_mask(load_img(df.loc[row_index, 'ImageId']), rle_to_pixels(df.loc[row_index, 'EncodedPixels'])))
        ax.set_title(df.loc[row_index, 'ImageId'])


# ### References
# * [Airbus ship data vizualization](https://www.kaggle.com/meaninglesslives/airbus-ship-detection-data-visualization) - a nice data visualization and exploratory data analysis kernel. We didn't copy code from there, but it helped us to quick start the project.
# 
# ### What's next?
# You can check the [next kernel](https://www.kaggle.com/julian3833/3-basic-exploratory-analysis) of the series, where we explore the data and present the class imbalance problem of the dataset.
