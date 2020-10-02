#!/usr/bin/env python
# coding: utf-8

# About a year ago I applied for a deep learning position in computer vision and I was assigned a task. The task is to detect house roofs on satellite images. Back then I was a newbie in computer vision and I had absolutely no idea that it was an object detection problem. 
# 
# 
# When I received the task, I was given sets of images. One set is the label and the other one is the original images. They look like this.
# 
# 

# In[ ]:


from IPython.display import Image
Image("../input/satellite/images/121.png")


# In[ ]:


Image("../input/satellite/labels/121.png")


# The original image is the satellite image and the label is an image with all the roofs are white while the rest regions are all black.
# 
# Back then I had no idea that these white regions are called masks. They in fact belong to the image segmentation problem. That is, not only the roofs need to be detected but also the entire roof regions need to be segmented from the rest.

# In[ ]:



