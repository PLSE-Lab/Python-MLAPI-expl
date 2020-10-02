#!/usr/bin/env python
# coding: utf-8

# # [Monk Object Detection](https://github.com/Tessellate-Imaging/Monk_Object_Detection)

# *A one-stop repository for low-code easily-installable object detection pipelines.*

# # Monk Format
# 
# # Dataset Directory Structure
#   
#   
#   root_dir
#   
#       |
#       | 
#       |         
#       |----train_img_dir
#       |       |
#       |       |---------img1.jpg
#       |       |---------img2.jpg
#       |                |---------..........(and so on) 
#       |
#       |----train_mask_dir
#       |       |
#       |       |---------img1.jpg
#       |       |---------img2.jpg
#       |                |---------..........(and so on)
#       |
#       |----val_img_dir (optional)
#       |       |
#       |       |---------img1.jpg
#       |       |---------img2.jpg
#       |                |---------..........(and so on)
#       |
#       |----val_mask_dir (optional)
#       |       |
#       |       |---------img1.jpg
#       |       |---------img2.jpg
#       |                |---------..........(and so on)

# # Converting the dataset into MONK format

# Convert .tif image into .jpeg or .png format and also structure the dataset as required

# In[ ]:


get_ipython().system(' mkdir trainJPEG testJPEG')


# In[ ]:


get_ipython().system(' mkdir trainJPEG/trainimg trainJPEG/trainmask testJPEG/testimg testJPEG/testmask')


# In[ ]:


get_ipython().system(' pwd')


# In[ ]:


import os, sys
from PIL import Image


# In[ ]:


for infile in os.listdir("/kaggle/input/ultrasound-nerve-segmentation/train/"):
    #print("file : " + infile)
    
    if infile[-3:] == "tif":
        
        if infile[-8:-4] == "mask":
            file = "/kaggle/input/ultrasound-nerve-segmentation/train/" + infile
            outfile = "/kaggle/working/trainJPEG/trainmask/"+ infile[:-9] + ".jpeg"
            im = Image.open(file)
            out = im.convert("RGB")
            out.save(outfile, "JPEG", quality=100)
        else:
            fileImg = "/kaggle/input/ultrasound-nerve-segmentation/train/" + infile
            outfileImg = "/kaggle/working/trainJPEG/trainimg/"+ infile[:-3] + "jpeg"
            imImg = Image.open(fileImg)
            outImg = imImg.convert("RGB")
            outImg.save(outfileImg, "JPEG", quality=100)


# In[ ]:


for infile in os.listdir("/kaggle/input/ultrasound-nerve-segmentation/test/"):
    #print("file : " + infile)
    
    if infile[-3:] == "tif":
        
        if infile[-8:-4] == "mask":
            file = "/kaggle/input/ultrasound-nerve-segmentation/test/" + infile
            outfile = "/kaggle/working/testJPEG/testmask/"+ infile[:-9] + ".jpeg"
            im = Image.open(file)
            out = im.convert("RGB")
            out.save(outfile, "JPEG", quality=100)
        else:
            fileImg = "/kaggle/input/ultrasound-nerve-segmentation/test/" + infile
            outfileImg = "/kaggle/working/testJPEG/testimg/"+ infile[:-3] + "jpeg"
            imImg = Image.open(fileImg)
            outImg = imImg.convert("RGB")
            outImg.save(outfileImg, "JPEG", quality=100)  


# In[ ]:


import cv2
import matplotlib.pyplot as plt
f, axarr = plt.subplots(2,4)

img1 = cv2.imread('/kaggle/working/trainJPEG/trainimg/10_103.jpeg')
img2 = cv2.imread('/kaggle/working/trainJPEG/trainmask/10_103.jpeg')
img3 = cv2.imread('/kaggle/working/trainJPEG/trainimg/10_104.jpeg')
img4 = cv2.imread('/kaggle/working/trainJPEG/trainmask/10_104.jpeg')

img5 = cv2.imread('/kaggle/working/trainJPEG/trainimg/10_109.jpeg')
img6 = cv2.imread('/kaggle/working/trainJPEG/trainmask/10_109.jpeg')
img7 = cv2.imread('/kaggle/working/trainJPEG/trainimg/10_112.jpeg')
img8 = cv2.imread('/kaggle/working/trainJPEG/trainmask/10_112.jpeg')

axarr[0,0].imshow(img1)
axarr[0,1].imshow(img2)
axarr[1,0].imshow(img3)
axarr[1,1].imshow(img4)

axarr[0,2].imshow(img5)
axarr[0,3].imshow(img6)
axarr[1,2].imshow(img7)
axarr[1,3].imshow(img8)


# # Installation

# Run these commands
# 
# * git clone https://github.com/Tessellate-Imaging/Monk_Object_Detection.git
# 
# * cd Monk_Object_Detection/9_segmentation_models/installation
# 
# Select the right requirements file and run
# 
# cat requirements_cuda9.0.txt | xargs -n 1 -L 1 pip install

# In[ ]:


get_ipython().system(' git clone https://github.com/Tessellate-Imaging/Monk_Object_Detection.git')


# * For colab use the command below
# 
# ! cd Monk_Object_Detection/9_segmentation_models/installation && cat requirements_colab.txt | xargs -n 1 -L 1 pip install
# 
# 
# * For Local systems and cloud select the right CUDA version
# 
# ! cd Monk_Object_Detection/9_segmentation_models/installation && cat requirements_cuda10.0.txt | xargs -n 1 -L 1 pip install

# In[ ]:


get_ipython().system(' cd Monk_Object_Detection/9_segmentation_models/installation && cat requirements_colab.txt | xargs -n 1 -L 1 pip install')


# # Training your own segmenter

# In[ ]:


DATA_DIR = '/kaggle/working/'


# In[ ]:


import os
import sys
sys.path.append("Monk_Object_Detection/9_segmentation_models/lib/");


# In[ ]:


from train_segmentation import Segmenter


# In[ ]:


gtf = Segmenter();


# In[ ]:


img_dir = "/kaggle/working/trainJPEG/trainimg/";
mask_dir = "/kaggle/working/trainJPEG/trainmask/";


# In[ ]:


classes_dict = {
    'background': 0, 
    'nerves': 1,
};
classes_to_train = ['background', 'nerves'];


# # Load Dataset

# In[ ]:


gtf.Train_Dataset(img_dir, mask_dir, classes_dict, classes_to_train)


# In[ ]:


img_dir = "/kaggle/working/testJPEG/testimg/";
mask_dir = "/kaggle/working/testJPEG/testmask/";


# In[ ]:


gtf.Val_Dataset(img_dir, mask_dir)


# In[ ]:


gtf.List_Backbones();


# In[ ]:


gtf.Data_Params(batch_size=2, backbone="efficientnetb3", image_shape=[580, 420])


# # Load Model

# In[ ]:


gtf.List_Models();


# In[ ]:


gtf.Model_Params(model="Linknet")


# # Train Params

# In[ ]:


gtf.Train_Params(lr=0.001)


# # Setup

# In[ ]:


gtf.Setup();


# # Train

# In[ ]:


gtf.Train(num_epochs=5);


# In[ ]:


gtf.Visualize_Training_History();


# # Inference

# In[ ]:


from infer_segmentation import Infer


# In[ ]:


gtf = Infer();


# In[ ]:


classes_dict = {
    'background': 0, 
    'nerves': 1,
};
classes_to_train = ['nerves'];


# In[ ]:


gtf.Data_Params(classes_dict, classes_to_train, image_shape=[580, 420])


# In[ ]:


gtf.Model_Params(model="Linknet", backbone="efficientnetb3", path_to_model='best_model.h5')


# In[ ]:


gtf.Setup();


# In[ ]:


gtf.Predict("/kaggle/working/trainJPEG/trainimg/10_103.jpeg", vis=True);


# In[ ]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/kaggle/working/trainJPEG/trainmask/10_103.jpeg", 0)
cv2.imwrite("tmp.jpg", img)

from IPython.display import Image
Image(filename="tmp.jpg")


# # [Tessellate Imaging](https://www.tessellateimaging.com/)
# 
# # Check out 
# 
# # [Monk AI](https://github.com/Tessellate-Imaging/monk_v1)
# 
# *Monk is a low code Deep Learning tool and a unified wrapper for Computer Vision.*
# 
# **Monk features**
# 
#     - low-code
#     - unified wrapper over major deep learning framework - keras, pytorch, gluoncv
#     - syntax invariant wrapper
# 
# **Enables developers**
# 
#     - to create, manage and version control deep learning experiments
#     - to compare experiments across training metrics
#     - to quickly find best hyper-parameters
#     
# 
# # [Monk GUI](https://github.com/Tessellate-Imaging/Monk_Gui)
# 
# *A Graphical user Interface for deep learning and computer vision over Monk Libraries*
# 
# # [Pytorch Tutorial](https://github.com/Tessellate-Imaging/Pytorch_Tutorial)
# 
# *A set of jupyter notebooks on pytorch functions with examples*

# # To contribute to Monk AI or Monk Object Detection repository raise an issue in the git-repo or DM us on linkedin
# 
# * Abhishek - https://www.linkedin.com/in/abhishek-kumar-annamraju/
# * Akash - https://www.linkedin.com/in/akashdeepsingh01/
