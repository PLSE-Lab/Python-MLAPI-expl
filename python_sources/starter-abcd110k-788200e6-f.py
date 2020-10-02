#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Image Segmentation using bbd100k dataset with fastai library, obtaining 85.83% accuracy, 128x128 size.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *

import matplotlib.pyplot as plt


# In[ ]:


path = Path('../input/bdd100k_seg/bdd100k/seg/')
path.ls()


# In[ ]:


img = open_image(path/'images'/'val'/'a00d3a96-00000000.jpg')
color_label = open_mask(path/'color_labels'/'val'/'a00d3a96-00000000_train_color.png')
label = open_image(path/'labels'/'val'/'a00d3a96-00000000_train_id.png')


# In[ ]:


img.show()
color_label.show()


# In[ ]:


color_label.data


# # Convert mask values between 0~N-1

# In[ ]:


import PIL.Image as PilImage

def getClassValues(label_names):

    containedValues = set([])

    for i in range(len(label_names)):
        tmp = open_mask(label_names[i])
        tmp = tmp.data.numpy().flatten()
        tmp = set(tmp)
        containedValues = containedValues.union(tmp)
    
    return list(containedValues)

def replaceMaskValuesFromZeroToN(mask, 
                                 containedValues):

    numberOfClasses = len(containedValues)
    newMask = np.zeros(mask.shape)

    for i in range(numberOfClasses):
        newMask[mask == containedValues[i]] = i
    
    return newMask

def convertMaskToPilAndSave(mask, 
                            saveTo):

    imageSize = mask.squeeze().shape

    im = PilImage.new('L',(imageSize[1],imageSize[0]))
    im.putdata(mask.astype('uint8').ravel())
    im.save(saveTo)

def convertMasksToGrayscaleZeroToN(pathToLabels,
                                   saveToPath):

    label_names = get_image_files(pathToLabels)
    containedValues = getClassValues(label_names)

    for currentFile in label_names:
        currentMask = open_mask(currentFile).data.numpy()
        convertedMask = replaceMaskValuesFromZeroToN(currentMask, containedValues)
        convertMaskToPilAndSave(convertedMask, saveToPath/f'{currentFile.name}')
    
    print('Conversion finished!')


# In[ ]:


get_ipython().system('mkdir converted_masks/')


# In[ ]:


get_ipython().system('mkdir converted_masks/train')


# In[ ]:


get_ipython().system('mkdir converted_masks/val')


# In[ ]:


pathToLabels = path/'color_labels/train'
saveToPath = Path('/kaggle/working/converted_masks/train')
convertMasksToGrayscaleZeroToN(pathToLabels, saveToPath)

pathToLabels = path/'color_labels/val'
saveToPath = Path('/kaggle/working/converted_masks/val')
convertMasksToGrayscaleZeroToN(pathToLabels, saveToPath)


# In[ ]:


mask_path = Path('/kaggle/working/converted_masks')


# In[ ]:


mask2=open_mask(saveToPath/'a00d3a96-00000000_train_color.png')
mask2.show()
mask2.data.shape


# # Train

# In[ ]:


def get_y_func(x):
    y = mask_path/f'{x.parts[-2]}/{x.stem}_train_color.png'
#     y = saveToPath/f'{x.stem}_train_color.png'
    return y


# In[ ]:


classes = ['banner',
'billboard',
'lane divider',
'parking sign',
'pole',
'polegroup',
'street light',
'traffic cone',
'traffic device',
'traffic light',
'traffic sign',
'sign frame',
'person',
'rider',
'bicycle',
'bus',
'car',
'caravan',
'motorcycle']
# 'trailer',
# 'train',
# 'truck',
# 'void']


# In[ ]:


src = (SegmentationItemList.from_folder(path/'images')
                            .split_by_folder('train', 'val')
                            .label_from_func(get_y_func, classes=classes))


# In[ ]:


src


# In[ ]:


data = (src.transform(tfms=get_transforms(), size=(512,512), tfm_y=True)
           .databunch(bs=8).normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=2)


# In[ ]:


def acc_bdd(input, target):
    target = target.squeeze(1)
#     mask = target != void_code
    return (input.argmax(dim=1)==target).float().mean()


# In[ ]:


learn=unet_learner(data, models.resnet50,metrics=acc_bdd,path='/kaggle/working')


# In[ ]:


learn.lr_find(num_it=200)


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr=2e-5


# In[ ]:


learn.fit_one_cycle(8,max_lr=lr)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find(num_it=400)


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(60,max_lr=slice(5e-7,lr/5))
learn.recorder.plot_losses()

