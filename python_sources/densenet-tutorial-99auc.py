#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
'''
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''

# Any results you write to the current directory are saved as output.


# In[ ]:


import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision import Image as Im


# In[ ]:


#path = Path('../input/dataset/data/data')
#gray dataset
path =Path('../input/graydataset/')
test_images = ('../input/test-eye/images_test/images_test/testresim')


# In[ ]:


train = path/'train'
#test =  path/'test'
path.ls()


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train="train", valid_pct=0.2,
ds_tfms=get_transforms(do_flip=False,flip_vert=False,max_rotate=0, max_zoom=0.5,
                       max_lighting=0, max_warp=0), size=[150,400],bs=64, num_workers=4).normalize(imagenet_stats)


# In[ ]:


data.classes,len(data.train_ds),len(data.valid_ds)


# In[ ]:


data.show_batch(rows=2, figsize=(7, 8))


# In[ ]:


learn = cnn_learner(data, models.densenet169, metrics=accuracy,model_dir='/kaggle/working/models')


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(2,4.37E-03)


# In[ ]:


learn.export('/kaggle/working/strage1.pkl')
learn.save('strage1')


# In[ ]:


learn.unfreeze() # must be done before calling lr_find
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(50, max_lr=1e-2,callbacks=[SaveModelCallback(learn, every='improvement', monitor='accuracy')])


# In[ ]:


learn.unfreeze()
learn.export('/kaggle/working/strage2.pkl')
learn.save('strage2')


# In[ ]:


learn.unfreeze() # must be done before calling lr_find
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(8,8))


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


interp.plot_top_losses(4, figsize=(7,7))


# In[ ]:


#testing
classes = ['Esotropia','Exotropia','Normal']

test_list = glob.glob('../input/test-eye/images_test/images_test/testresim/*.jpg')

'''tfms = get_transforms(max_rotate=25)
def get_ex(i): 
    return open_image('C:/Users/Doruk/Desktop/Ophthalmology/')

def plots_f(rows, cols, width, height, **kwargs):
    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(
        rows,cols,figsize=(width,height))[1].flatten())]''' 


learn = learn.load('strage2')

d = {'pred': [], 'actual': []}
df = pd.DataFrame(data=d)


# In[ ]:


for i in (test_list):

    img = open_image(i)
    #img =open_image(i).apply_tfms([crop_pad()], size=256, resize_method=ResizeMethod.PAD, padding_mode='zeros')
    ac_class = i.split('/')[6]
   
    if ac_class.__contains__('Exotropia')==True:
        temp = 'Exotropia'
    elif ac_class.__contains__('Esotropia')==True:
        temp = 'Esotropia'
    elif ac_class.__contains__('Normal')==True:
        temp = 'Normal'

    pred_class,pred_idx,outputs = learn.predict(img)
    df = df.append({'pred': str(pred_class), 'actual' : temp}, ignore_index=True)
    prediction_list = list(map(lambda x: int(float(x*100)), outputs))
    predictionary = dict(zip(classes, prediction_list))
    print(temp,pred_class,sorted(predictionary.items(),key=operator.itemgetter(1), reverse=True))


# In[ ]:


df['result'] = np.where(df['pred'] == df['actual'], 'true', 'false')
true = len(df.loc[(df['result'] =='true')])
pred_ac =int(100*true/len(df))
print("true_count:{}".format(true))
print("score:{}".format(pred_ac))

