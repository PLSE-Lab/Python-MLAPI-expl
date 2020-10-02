#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# * In this kernel we use a pretrained ResNet34 model to train the retina images.
# * This kernel is only for getting started as I have only trained the model for one epoch.
# * **The main purpose of this kernel is to:**
#     * **Show how to use fastai library correctly to train models in Kaggle kernels, save models, plots and also create an export pickle file. Many starters find it cofusing to use Fastai library in Kaggle kernels and how to save the models and download them correctly. I hope this kernel helps.**
#     * **To create a baseline for getting started, further this can be extended.**
#     * **How to PyTorch pretrained models without turning the internet on.**
# 
#     
# <h3 style="color:green">If you find this kernel helpful,then please upvote.</h3>

# In[ ]:


from fastai.vision import *
import os


# In[ ]:


# this is where we will copy our pretrained models
os.makedirs('/root/.cache/torch/checkpoints')


# In[ ]:


get_ipython().system('cp ../input/resnet34/resnet34.pth /root/.cache/torch/checkpoints/resnet34-333f7ec4.pth')


# In[ ]:


# to save the models
model_path = 'models'
# to save the plots
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


path = Path('../input/diabetic-retinopathy-2015-data-colored-resized/colored_images/colored_images/')
path.ls()


# In[ ]:


'''
Remove the images that we cannot open. 
Execute this only once per kernel run.
'''
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2, 
                                  ds_tfms=get_transforms(), size=224, 
                                  num_workers=4, bs=32).normalize(imagenet_stats)


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=3, figsize=(10, 7))


# In[ ]:


learn = cnn_learner(data, models.resnet34, 
                    metrics=error_rate, 
                    model_dir='/kaggle/working/models')


# In[ ]:


learn.fit_one_cycle(1)    


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('colored_stage1')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(1, max_lr=slice(1e-5, 1e-4))


# In[ ]:


learn.save('colored_stage2')


# In[ ]:


learn.load('colored_stage2')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()
plt.savefig('models/plots/interp.png')
plt.show()


# In[ ]:




