#!/usr/bin/env python
# coding: utf-8

# In[ ]:


image_fn = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/ISIC_0077735.jpg"
small_image_fn = "small.jpg"


# In[ ]:


get_ipython().run_cell_magic('time', '', 'import io\nf = open(image_fn, "rb")\ndata = f.read()\nprint(len(data))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'import PIL.Image as Image\nimage = Image.open(io.BytesIO(data))\ndown = image.resize((224, 224))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'down.save(small_image_fn)')


# In[ ]:


from IPython.display import Image
Image(image_fn)


# In[ ]:


Image(small_image_fn)

