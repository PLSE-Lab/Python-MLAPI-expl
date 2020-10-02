#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from fastai.vision import *
from fastai import *
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from tqdm.notebook import tqdm
import gc
from pylab import imread,subplot,imshow,show
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


path = "/kaggle/input/kermany2018/OCT2017 /"


# In[ ]:


data


# In[ ]:


size = 224
bs=6


# In[ ]:


size = 224
bs = 64
data = ImageDataBunch.from_folder(path, 
                                  ds_tfms=get_transforms(max_rotate=0.1,max_lighting=0.15),
                                  valid_pct=0.2, 
                                  size=size, 
                                  bs=bs)


# In[ ]:


data.show_batch(rows=4)


# In[ ]:


fb = FBeta()
fb.average='macro'


# In[ ]:


learner = cnn_learner(data, models.resnet18, metrics=[accuracy])


# In[ ]:


learner.fit_one_cycle(1,1e-3)


# In[ ]:


learner.save('model_retina2')


# In[ ]:


learner.model_dir='/kaggle/working/'


# In[ ]:


learner.export('/kaggle/working/export2.pkl')


# In[ ]:


l=load_learner('/kaggle/working/')


# In[ ]:


learner=load_learner('/kaggle/input/modelx/')


# In[ ]:


learner


# In[ ]:


learner.load('/kaggle/input/modelx/model_retina')


# In[ ]:


interp= ClassificationInterpretation.from_learner(learner)


# In[ ]:


preds,y,losses = learner.get_preds(with_loss=True)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


import cv2
import numpy as np
img = cv2.imread('/kaggle/input/kermany2018/OCT2017 /val/DME/DME-9721607-2.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = np.zeros_like(img)
img2[:,:,0] = gray
img2[:,:,1] = gray
img2[:,:,2] = gray


# TESTING

# In[ ]:


path2='/kaggle/input/kermany2018/OCT2017 /test/'


# In[ ]:


img=pil2tensor(img2,dtype=np.float32)


# In[ ]:


cl=['CNV','DME','DRUSEN','NORMAL']


# In[ ]:


for i in range(len(cl)):
    for file in os.listdir(path2+str(cl[i])):
        path3='/kaggle/input/kermany2018/OCT2017 /test/'+str(cl[i])+"/"+str(file)
        
        img=cv2.imread(str(path3))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros_like(img)
        img2[:,:,0] = gray
        img2[:,:,1] = gray
        img2[:,:,2] = gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros_like(img)
        img2[:,:,0] = gray
        img2[:,:,1] = gray
        img2[:,:,2] = gray
        print(cl[i])
        img=pil2tensor(img2,dtype=np.float32)
        print(learner.predict(Image(img)))
        

        
    
    


# In[ ]:


pre=[]
for i in preds:
    a=max(i)
    pre.append(a)
pre = np.array(pre, dtype=np.float32)


# In[ ]:


path2='/kaggle/input/kermany2018/OCT2017 /val/'
img=cv2.imread('/kaggle/input/kermany2018/OCT2017 /val/CNV/CNV-6294785-1.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = np.zeros_like(img)
img2[:,:,0] = gray
img2[:,:,1] = gray
img2[:,:,2] = gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = np.zeros_like(img)
img2[:,:,0] = gray
img2[:,:,1] = gray
img2[:,:,2] = gray

img=pil2tensor(img2,dtype=np.float32)
        


# In[ ]:


print(learner.predict(Image(img)))


# In[ ]:




