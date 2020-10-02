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


path = Path('/kaggle/input/emergency-vehicles-identification/Emergency_Vehicles')
path.ls()


# In[ ]:


path_anno = path/'train'
fn_paths = get_image_files(path_anno)


# In[ ]:


len(fn_paths)


# In[ ]:


train_df = pd.read_csv(path/'train.csv')
train_df.head()


# In[ ]:


def get_labels(file_path):
        for row in train_df.itertuples():
            if '/'+row.image_names in str(file_path):           
                return row.emergency_or_not


# In[ ]:


labels = list(map(get_labels, fn_paths))


# In[ ]:


len(labels)


# In[ ]:


tfms = get_transforms()
data = ImageDataBunch.from_lists(path, fn_paths, labels=labels, ds_tfms=tfms, size=224, bs=bs, valid_pct=0.25).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=20, figsize=(20,20))


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[ ]:


get_ipython().system('mkdir -p /root/.cache/torch/checkpoints/')
get_ipython().system('cp /kaggle/input/resnet152/resnet152.pth /root/.cache/torch/checkpoints/resnet152.pth')


# In[ ]:


learn = cnn_learner(data, models.resnet152, metrics=accuracy, model_dir=Path('/kaggle/input/resnet152'))


# In[ ]:


learn.model


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


learn.model_dir = '/kaggle/output/resnet152/'


# In[ ]:


learn.save('/kaggle/output/resnet152/stage-1-152')


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


learn.fit_one_cycle(4)


# In[ ]:


learn.load('/kaggle/output/resnet152/stage-1-152');


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-5,1e-4))


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


learn.save('/kaggle/output/resnet152/stage-2-152')


# In[ ]:


path = learn.path


# In[ ]:


learn.export('/kaggle/output/resnet152/emergency_vehicles.pkl')


# In[ ]:


defaults.device = torch.device('cpu')


# In[ ]:


lego_learn = load_learner('/kaggle/output/resnet152', 'emergency_vehicles.pkl')


# In[ ]:


pred_path = path/'test'
pred_fn_paths = get_image_files(pred_path)


# In[ ]:


for pred_fn_path in pred_fn_paths:
    img = open_image(pred_fn_path)
    pred_class,pred_idx,outputs = learn.predict(img)
    print(pred_fn_path, pred_class)


# In[ ]:


img = open_image('/kaggle/input/emergency-vehicles-identification/Emergency_Vehicles/test/841.jpg')
img


# In[ ]:


pred_class,pred_idx,outputs = lego_learn.predict(img)
print(str(pred_class))


# In[ ]:


img = open_image('/kaggle/input/emergency-vehicles-identification/Emergency_Vehicles/test/1287.jpg')
img


# In[ ]:


pred_class,pred_idx,outputs = lego_learn.predict(img)
print(str(pred_class))


# In[ ]:




