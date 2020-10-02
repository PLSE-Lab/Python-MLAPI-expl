#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


from IPython.display import Image
import os
get_ipython().system('ls ../input/')


# To add images from your pc click the + Add Data Button on the right hand side of your Kernel

# In[ ]:


Image("../input/AddingData1.JPG")


# Click the Upload button on the pop-up box

# In[ ]:


Image("../input/AddingData2.JPG")


# Select Files to upload will allow you to navigate to the files on your PC you wish to upload.

# In[ ]:


Image("../input/AddingData3.JPG")


# You will need to give the directory a name, but be warned even though it shows as case sensitive in 
# the draft environment on the right the path will only take lower case.

# In[ ]:


Image("../input/AddingData4.JPG")


# In the example below I brought in a python file and called the directory "myapitoken." You can use this same process though to bring in image files and use the  method below to display the image.
# 
# from IPython.display import Image
# Image("../input/yourimage.JPG")

# In[ ]:


Image("../input/AddingData5.JPG")


# I have noticed that aside from the case sensitivity of the directory it also can be a little hinky in other ways. For this notebook I brought in the images to a directory in input which should be ../input/picsfordemo/here is where they should be. However, after I got an error I checked !ls ../input/ and that is where the images were. Good luck and happy learning all. 
