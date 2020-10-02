#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import fastai
from fastai.callbacks import *
from fastai.imports import *
from fastai.vision import *
from fastai.metrics import *
from fastai.gen_doc.nbdoc import *


# In[ ]:


model = models.resnet50
WORK_DIR = os.getcwd()
IMAGE_DIR = Path('../input/')
image_size=299
batch_size=16


# In[ ]:


fnames = get_image_files(IMAGE_DIR / 'train' / 'train_kaggle')
fnames[:5]


# In[ ]:


data = ImageDataBunch.from_name_re(path=IMAGE_DIR,
                                   fnames=fnames, 
                                   pat=r'/([^/]+)_\d+\.jpg$', 
                                   ds_tfms=get_transforms(), 
                                   test='test/test_kaggle',
                                   size=image_size, 
                                   bs=batch_size,
                                   num_workers=0).normalize(imagenet_stats)
data


# In[ ]:


data.show_batch(rows=3)


# In[ ]:


learn = cnn_learner(data, model, metrics=accuracy, model_dir=WORK_DIR)


# In[ ]:


learn.fit_one_cycle(8, callbacks=[EarlyStoppingCallback(learn, patience=2)])


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.unfreeze()
learn.lr_find(1e-9, 1)
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-5))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


preds, _ = learn.TTA(ds_type=DatasetType.Test)


# In[ ]:


preds


# In[ ]:


print(data.classes)
dict_label_order = {label:order for order,label in enumerate(data.classes)}
print(dict_label_order)


# In[ ]:


n_dogs = dict_label_order['frankfurter'], dict_label_order['chili-dog'], dict_label_order['hotdog']
n_dogs


# In[ ]:


prob_dogs = preds[:,n_dogs].numpy().sum(axis=1)
prob_dogs


# In[ ]:


plt.hist(prob_dogs)


# In[ ]:


ids = [file.name for file in data.test_ds.x.items]
ids[:10]


# In[ ]:


import pandas as pd
submission = pd.DataFrame({'image_id': ids, 'label': [1 if x >= 0.5 else 0 for x in prob_dogs]})
submission = submission.sort_values(by=['image_id'])


# In[ ]:


submission.head()


# In[ ]:


submission.label.describe()


# In[ ]:


submission.to_csv('submission.csv', index=False)

