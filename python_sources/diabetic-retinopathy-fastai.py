#!/usr/bin/env python
# coding: utf-8

# ## 1) Introduction
# * This kernel trains a ResNet34 model on the colored images of Diabetic Retinopathy dataset. All the intermediate models, and the final model is being exported. 
# * You can either download the final model or use it in your own kernels as well.

# In[ ]:


from fastai.vision import *
import os


# In[ ]:


os.makedirs('/root/.cache/torch/checkpoints')


# In[ ]:


get_ipython().system('cp ../input/resnet34fastai/resnet34.pth /root/.cache/torch/checkpoints/resnet34-333f7ec4.pth')


# In[ ]:


model_path = 'models'
plot_path = 'plots'

if not os.path.exists(model_path):
    os.makedirs(model_path)
    os.makedirs(os.path.join(model_path, plot_path))


# In[ ]:


'''
Severity Levels

0 - 'No_DR',
1 - 'Mild',
2 - 'Moderate',
3 - 'Severe',
4 - 'Proliferate_DR'
'''

classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']


# In[ ]:


path = Path('../input/diabetic-retinopathy-224x224-2019-data/colored_images')
path.ls()


# In[ ]:


# remove the images that we cannot open
# for c in classes:
#     print(c)
#     verify_images(path/c, delete=True, max_size=500)


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2, 
                                  ds_tfms=get_transforms(), size=224, 
                                  num_workers=4, bs=16).normalize(imagenet_stats)


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3, figsize=(10, 7))


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir='/kaggle/working/models')
# learner = load('../input/resnet34-fastai/resnet34.pth')
# learn = cnn_learner(data, learner, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(20)    


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


get_ipython().system('pwd')
print(os.listdir('../../'))


# In[ ]:


learn.save('colored_stage1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3, max_lr=slice(1e-6, 1e-5))


# In[ ]:


learn.save('colored_stage2')


# In[ ]:


learn.load('colored_stage2')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


learn.export('/kaggle/working/models/colored_export.pkl')


# In[ ]:





# In[ ]:




