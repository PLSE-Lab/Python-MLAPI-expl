#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('curl -s https://course.fast.ai/setup/colab | bash')
from fastai.vision import *
from fastai.tabular import *
from pathlib import Path


# ## View data

# In[ ]:


cp -R /kaggle/input/platesv2/plates/plates /kaggle/input/plate


# In[ ]:


ls /kaggle/input/plate


# In[ ]:


path = Path('/kaggle/input/plate')


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path/"train", train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), bs=16, size=224, num_workers=4).normalize(imagenet_stats)


# In[ ]:


data.classes


# In[ ]:


data.show_batch(rows=5, figsize=(7,8))


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# ## Train model

# In[ ]:


learn = cnn_learner(data, models.resnet18, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(100)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(200, max_lr=slice(3e-6,2e-6))


# In[ ]:


learn.save("/kaggle/working/stage")


# ## Interpretation

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# # Result

# ## One Picture

# In[ ]:


img = open_image(path/'test/0000.jpg')
img


# In[ ]:


pred_class,pred_idx,outputs = learn.predict(img)
pred_class


# ## CSV result

# In[ ]:


from os import listdir
sample_list = [f[:4] for f in listdir(path/"test")]
sample_list.sort()


# In[ ]:


pred_list_cor = []
for f in sample_list :
    file = f+".jpg"
    p,_,_ = learn.predict(open_image(path/"test"/file))
    pred_list_cor.append(p.obj)


# In[ ]:


final_df = pd.DataFrame({'id':sample_list,'label':pred_list_cor})
final_df.to_csv('/kaggle/working/plate_submission.csv', header=True, index=False)


# In[ ]:


final_df.head()

