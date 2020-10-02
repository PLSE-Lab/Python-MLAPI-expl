#!/usr/bin/env python
# coding: utf-8

# # Introduction
# A desirable image augmentation is the horizontal flipping. This means
# - flipping the image horizontally
# - changing the labels for each car as...
#   - y=-y
#   - yaw=-yaw (the actual yaw as defined e.g. [here](https://www.google.com/imgres?imgurl=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F283951857%2Ffigure%2Ffig2%2FAS%3A319897696522260%401453280962401%2FRoll-pitch-yaw-angles-of-cars-and-other-land-based-vehicles-10.png&imgrefurl=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FRoll-pitch-yaw-angles-of-cars-and-other-land-based-vehicles-10_fig2_283951857&docid=sZPJ9uehdFLqlM&tbnid=D-qwg3dzlKNduM%3A&vet=10ahUKEwjot8PcyYXnAhVVPcAKHZxcCpgQMwhbKAYwBg..i&w=600&h=401&bih=846&biw=1707&q=yaw%20car%20&ved=0ahUKEwjot8PcyYXnAhVVPcAKHZxcCpgQMwhbKAYwBg&iact=mrc&uact=8))
#   - roll=-roll
# 
# ## Problem
# However, as [some people have noticed](https://www.kaggle.com/c/pku-autonomous-driving/discussion/125591), there is an issue: If you project the xyz coordinates to 2D, the resulting uv-coordinates will not match the cars in the image anymore. The reason is, that the principal point of the camera does not lie exactly at the image center, but slightly off.
# 
# ## Solution
# Instead of a simple horizontal flipping, we need to flip the image exactly at the camera principal point (see [explanation of principal point](http://ksimek.github.io/2013/08/13/intrinsic/)). Therefore, I provide the function below. It flips the image with a precision of 0.5 pixels without using interpolation.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import cv2

def flip_hor_at_u(img, cx, flag_plot=False):
    # determine line for flipping rounded to 0.5 pixel
    cx_rounded = np.round(cx * 2).astype(np.int)
    u_flip = cx_rounded / 2

    # determine new width
    height, width, nchannels = img.shape
    if cx_rounded % 2 == 1:
        # if flipping line lies between two pixels...
        width_left = np.ceil(u_flip)
        width_right = np.floor(width - u_flip)
        width_new = 2 * max(width_left, width_right)
        pad_left = width_new / 2 - width_left
        pad_right = width_new / 2 - width_right
    else:
        # if flipping line lies at a pixel...
        width_left = np.round(u_flip)
        width_right = np.round(width - u_flip - 1)
        width_new = 2 * max(width_left, width_right) + 1
        pad_left = (width_new - 1) / 2 - width_left
        pad_right = (width_new - 1) / 2 - width_right

    # create new image and flip horizontally
    bg = img.mean(1, keepdims=True).astype(img.dtype)
    bg_left = np.repeat(bg, pad_left, axis=1)
    bg_right = np.repeat(bg, pad_right, axis=1)
    img_padded = np.hstack((bg_left, img, bg_right))
    img_padded_flipped = img_padded[:, ::-1, :]

    # crop back to org size s.t. cx=const
    dim_right = width_new-pad_right
    img_cropped = img_padded_flipped[:,
                  pad_left.astype(np.int):dim_right.astype(np.int)
                  :]
    width_cropped = img_cropped.shape[1]
    assert width_cropped== width, "width changed during flipping ?!"

    # plot images
    if flag_plot:
        fig_width, fig_height = max(4,width/100), max(6, 2*height/100)
        fig,ax = plt.subplots(2,1, sharey=True, sharex=True, figsize=(fig_width,fig_height))
        ax[0].imshow(img)
        ax[1].imshow(img_cropped)
        for axi in ax:
            axi.vlines(u_flip, 0, height-1)
        plt.show()

    return img_cropped


# # Examples
# ## Small dummy ones, to see effect

# In[ ]:


# flip image at u=2
img = np.random.rand(4, 6, 3) * 255
img = img.astype(np.uint8)
img_flipped = flip_hor_at_u(img, cx=2, flag_plot=True)


# In[ ]:


# flip image at u=1.5
img = np.random.rand(4, 6, 3) * 255
img = img.astype(np.uint8)
img_flipped = flip_hor_at_u(img, cx=1.5, flag_plot=True)


# ## Actual images used in competition

# In[ ]:


path_img = '/kaggle/input/pku-autonomous-driving/train_images/ID_00ac30455.jpg'
cam_K = np.array([[2304.5479, 0, 1686.2379],
                  [0, 2305.8757, 1354.9849],
                  [0, 0, 1]], dtype=np.float32)
cx = cam_K[0, 2]
img = cv2.imread(path_img)[:,:,::-1] #BGR to RGB
img_flipped = flip_hor_at_u(img, cx, flag_plot=True)

