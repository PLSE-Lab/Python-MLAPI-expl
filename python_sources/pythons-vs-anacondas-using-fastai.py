#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


path = Path('../input/anacondas_pythons')
path.ls()


# In[ ]:


predict_python = get_image_files(path/"valid/python")
predict_anaconda = get_image_files(path/"valid/anaconda")


# In[ ]:


data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224, bs = 8)
data.normalize(imagenet_stats)


# In[ ]:


DatasetType.Train


# In[ ]:


data.show_batch(rows=3)


# In[ ]:


print(data.classes)


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/models")


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


learn.save("stage-1")


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.most_confused()


# In[ ]:


learn.unfreeze()
learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


def do_prediction(files, expected):
    p = []
    for f in files:
        p_img = open_image(f)
        pred_class,pred_idx,outputs = learn.predict(p_img)
        if str(pred_class) != expected:
            p.append(p_img)
    return p


# In[ ]:


wrong_anaconda_pred = do_prediction(predict_anaconda, "anaconda")
wrong_python_pred = do_prediction(predict_python, "python")


# # Anacondas errorneously predicted as Pythons

# In[ ]:


for f in wrong_anaconda_pred:
    show_image(f)


# # Pythons errorneously predicted as Anacondas

# In[ ]:


for f in wrong_python_pred:
    show_image(f)

