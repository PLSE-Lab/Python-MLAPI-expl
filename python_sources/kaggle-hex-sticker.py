#!/usr/bin/env python
# coding: utf-8

# This kernel uses [Fridex's hexsticker package](https://github.com/fridex/hexsticker) to create a hex sticker from a .png of the Kaggle logo. 
# 
# You can use this code to create your own sticker by:
# 
# 1. Forking this notebook
# 2. Uploading a dataset with your own .png files
# 3. Start a new kernel on your dataset & replace the "../input/k-logo-white-square.png" with the path to the picture you want to lose
# 
# You'll probably want to adjust the padding size (depending on the size ratio of your photo), the border size and the border color (which is Kaggle blue here). 
# 
# Happy stickering! :)

# In[ ]:


# use the ! magic to run the command line hexsticker command
get_ipython().system("hexsticker ../input/k-logo-white-square.png -o kaggle-sticker-logo.png --padding-size 1500 --border-size 1500 --border-color '#20beff'")

# display our image (it's saved in the current working directory)
from IPython.display import Image
Image(filename='kaggle-sticker-logo.png') 

