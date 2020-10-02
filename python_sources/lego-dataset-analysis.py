#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[ ]:


bs = 64


# In[ ]:


path = Path('/kaggle/input/lego-dataset')
path.ls()


# In[ ]:


path_anno = path/'train'
fn_paths = get_image_files(path_anno)


# In[ ]:


df = pd.read_csv(path/'Train.csv')
df.head()


# In[ ]:


def get_labels(file_path):
        for row in df.itertuples():
            if '/'+row.name in str(file_path):            
                return row.category
    


# In[ ]:


labels = list(map(get_labels, fn_paths))


# In[ ]:


tfms = get_transforms()
data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=200, bs=bs, valid_pct=0.25
                                  ).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=20, figsize=(20,20))


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[ ]:


get_ipython().system('mkdir -p /root/.cache/torch/checkpoints/')
get_ipython().system('cp /kaggle/input/fast-ai-models/resnet50-19c8e357.pth /root/.cache/torch/checkpoints/resnet50-19c8e357.pth')


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir=Path('/kaggle/input/fast-ai-models'))


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


learn.model_dir = '/kaggle/output/fast-ai-models/'


# In[ ]:


learn.save('/kaggle/output/fast-ai-models/stage-1-50')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(20, figsize=(20,20))


# In[ ]:


interp.plot_confusion_matrix(figsize=(20,20), dpi=100)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(2)


# In[ ]:


learn.load('/kaggle/output/fast-ai-models/stage-1-50');


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(10, max_lr=slice(1e-6,1e-4))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(50, figsize=(20,20))


# In[ ]:


interp.plot_confusion_matrix(figsize=(20,20), dpi=100)


# In[ ]:


interp.most_confused(min_val=1)


# In[ ]:


learn.save('/kaggle/output/fast-ai-models/stage-2-50')


# In[ ]:


path = learn.path


# In[ ]:


learn.export('/kaggle/output/fast-ai-models/lego.pkl')


# In[ ]:


defaults.device = torch.device('cpu')


# In[ ]:


lego_learn = load_learner('/kaggle/output/fast-ai-models', 'lego.pkl')


# In[ ]:


pred_path = path/'test'
pred_fn_paths = get_image_files(pred_path)


# In[ ]:


for pred_fn_path in pred_fn_paths:
    img = open_image(pred_fn_path)
    pred_class,pred_idx,outputs = lego_learn.predict(img)
    print(pred_fn_path, pred_class)


# In[ ]:


img = open_image('/kaggle/input/lego-dataset/test/5491.png')
img


# In[ ]:


pred_class,pred_idx,outputs = lego_learn.predict(img)
print(str(pred_class))


# In[ ]:




