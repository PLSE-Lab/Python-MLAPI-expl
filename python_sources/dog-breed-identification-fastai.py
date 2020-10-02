#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *


# In[ ]:


path = Path('/kaggle/input/dog-breed-identification')


# In[ ]:


path.ls()


# In[ ]:


df = pd.read_csv(path/'labels.csv')
df.head()


# In[ ]:


np.random.seed(42) # set random seed so we always get the same validation set
src = (ImageList.from_csv(path, 'labels.csv', folder='train', suffix='.jpg')
                .split_by_rand_pct(0.2)
                .label_from_df(cols='breed')
                .add_test_folder(test_folder = 'test'))


# In[ ]:


# Data augmentation
tfms = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4, p_affine = 1., p_lighting=1.)


# In[ ]:


# Starting with smaller size 224x224 before using the full size 352x352
bs, size = 64, 224


# In[ ]:


data = src.transform(tfms, size=size).databunch(bs=bs).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(12, 9))


# In[ ]:


# Visualizing transformations
def _plot(i,j,ax):
    x,y = data.train_ds[4]
    x.show(ax, y=y)
    
plot_multi(_plot, 3, 3, figsize=(8,8))


# In[ ]:


arch = models.resnet50
metrics = [error_rate, accuracy]


# In[ ]:


learn = cnn_learner(data, arch, metrics=metrics)


# In[ ]:


learn.model_dir = '/kaggle/working'


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


min_grad_lr = learn.recorder.min_grad_lr
print(min_grad_lr)


# In[ ]:


learn.fit_one_cycle(3, slice(min_grad_lr))


# In[ ]:


learn.save('stage-1-224-rn50')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


min_grad_lr = learn.recorder.min_grad_lr
print(min_grad_lr)


# In[ ]:


learn.fit_one_cycle(3, slice(min_grad_lr))


# In[ ]:


learn.save('stage-2-224-rn50')


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


# Interpret the result
interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(4, figsize=(12, 9))


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


# Make predictions of the test folder 
predictions, targets = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


# Make predictions on the first 9 images
classes = predictions.argmax(1)
class_dict = dict(enumerate(learn.data.classes))
labels = [class_dict[i] for i in list(classes[:9].tolist())]
test_images = [i.name for i in learn.data.test_ds.items][:9]


# In[ ]:


plt.figure(figsize=(12,9))

for i, fn in enumerate(test_images):
    img = plt.imread(path/'test'/fn, 0)
    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title(labels[i])
    plt.axis("off")


# In[ ]:


from sklearn.metrics import log_loss
preds, y = learn.TTA()
print(accuracy(preds,y))
print(log_loss(y, preds))


# In[ ]:


# Save the predictions into "submission.csv" file

preds_test, y_test = learn.TTA(ds_type=DatasetType.Test)

df = pd.DataFrame(array(preds_test))
df.columns = data.classes

# Extract the id name from the file name
df.insert(0, "id", [str(pth).split('/')[5][:-4] for pth in data.test_ds.items])

df.to_csv("submission.csv", index=False)


# In[ ]:


# Double check the submission
df.head()

