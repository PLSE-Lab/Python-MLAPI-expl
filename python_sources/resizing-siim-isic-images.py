#!/usr/bin/env python
# coding: utf-8

# ## Resizing JPG images from SIIM-ISIC Melanoma Classification into sizes of:
# * 224 x 224
# * 300 x 300
# * 480 x 480
# * 640 x 640

# In[ ]:


from PIL import Image 
import os

from matplotlib import pyplot as plt

import multiprocessing 
import time 


# In[ ]:


PATH = '/kaggle/input/siim-isic-melanoma-classification/jpeg'


# In[ ]:


out_path = 'siim-isic-melanima-classification-resize-images'
os.mkdir(out_path)


# In[ ]:


fol = 'train'
out_dir = os.path.join(out_path, '224x224')
# os.mkdir(out_dir)
# os.mkdir(out_dir+'/train')
# os.mkdir(out_dir+'/test')
images_name = os.listdir(os.path.join(PATH, fol))

for img_name in images_name:
    img = Image.open(os.path.join(PATH, fol, img_name)).convert('RGB')
    plt.imshow(img)
    plt.show()
    
    img1 = img.resize((224, 224))
    plt.imshow(img1)
    plt.show()
    
    img2 = img.resize((640, 640))
    plt.imshow(img2)
    plt.show()
        
    break


# In[ ]:


# out_dir = os.path.join(out_path, '224x224')
# os.mkdir(out_dir)
# os.mkdir(out_dir+'/train')
# os.mkdir(out_dir+'/test')

out_dir = os.path.join(out_path, '300x300')
os.mkdir(out_dir)
os.mkdir(out_dir+'/train')
os.mkdir(out_dir+'/test')

# out_dir = os.path.join(out_path, '480x480')
# os.mkdir(out_dir)
# os.mkdir(out_dir+'/train')
# os.mkdir(out_dir+'/test')

out_dir = os.path.join(out_path, '640x640')
os.mkdir(out_dir)
os.mkdir(out_dir+'/train')
os.mkdir(out_dir+'/test')


# In[ ]:


def resize_img(img_name):
    img = Image.open(os.path.join(PATH, img_name)).convert('RGB')
#     img1 = img.resize((224, 224))
    img2 = img.resize((300, 300))
#     img3 = img.resize((480, 480))
    img4 = img.resize((640, 640))
#     img1.save(os.path.join(out_path, '224x224', img_name))
    img2.save(os.path.join(out_path, '300x300', img_name))
#     img3.save(os.path.join(out_path, '480x480', img_name))
    img4.save(os.path.join(out_path, '640x640', img_name))


# In[ ]:


pool = multiprocessing.Pool() 
pool = multiprocessing.Pool(processes=4) 


# In[ ]:


imgs_paths = []
for fol in ['train', 'test']:
    images_name = os.listdir(os.path.join(PATH, fol))
    for img_name in images_name:
        imgs_paths.append(fol + '/' + img_name)
        
pool.map(resize_img, imgs_paths) 
print("Done")    


# In[ ]:


import shutil
shutil.make_archive('siim-isic-melanima', 'zip', out_path)
shutil.rmtree(out_path)


# In[ ]:


out_path


# In[ ]:




