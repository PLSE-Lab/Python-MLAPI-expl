#!/usr/bin/env python
# coding: utf-8

# [This](https://medium.com/@oneironaut.oml/solving-captchas-with-deeplearning-part-2-single-character-classification-ac0b2d102c96) blog post explains what's going in in this kernel.

# In[ ]:


from fastai.vision import *
import os

path = Path("../input/samples/samples")
print(os.listdir(path)[:10])


# In[ ]:


def plot_lr(learn):
    lr_find(learn)
    learn.recorder.plot()


# In[ ]:


def char_from_path(path, position):
    return path.name[position]


# In[ ]:


data = (ImageList.from_folder(path) #Where to find the data? -> in path and its subfolders
        .split_by_rand_pct(0.2)              #How to split in train/valid? -> use the folders
        .label_from_func(partial(char_from_path, position=0))            #How to label? -> depending on the folder of the filenames
        .transform(get_transforms(do_flip=False))       #Data augmentation? -> use tfms with a size of 64
        .databunch())                   #Finally? -> use the defaults for conversion to ImageDataBunch


# In[ ]:


data.show_batch(3, figsize=(10,10))


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir='/tmp', ps=0.)


# In[ ]:


plot_lr(learn)


# In[ ]:


lr = 5e-2
learn.fit_one_cycle(5, lr)


# In[ ]:


learn.save('pretrained')


# In[ ]:


learn.load('pretrained')
learn.unfreeze()


# In[ ]:


plot_lr(learn)


# In[ ]:


learn.fit_one_cycle(15, slice(5e-4, lr/5))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix(figsize=(7,7))


# In[ ]:


interp.plot_top_losses(4, heatmap_thresh=14, largest=False)


# In[ ]:


def data_from_position(position):
    data = (ImageList.from_folder(path) #Where to find the data? -> in path and its subfolders
        .split_by_rand_pct(0.2)              #How to split in train/valid? -> use the folders
        .label_from_func(partial(char_from_path, position=position))            #How to label? -> depending on the folder of the filenames
        .transform(get_transforms(do_flip=False))       #Data augmentation? -> use tfms with a size of 64
        .databunch())                   #Finally? -> use the defaults for conversion to ImageDataBunch
    return data


# In[ ]:


learners = []
for i in range(5):
    data = data_from_position(i)
    
    learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir='/tmp', ps=0.)
    
    lr = 5e-2
    learn.fit_one_cycle(5, lr)
    
    learn.unfreeze()
    learn.fit_one_cycle(15, slice(5e-4, lr/5))
    
    learners.append(learn)


# In[ ]:


figures = []
for learner in learners:
    figures.append(learner.interpret().plot_top_losses(4, heatmap_thresh=14, figsize=(8,8), largest=False, return_fig=True))


# In[ ]:


for e,f in enumerate(figures):
    f.suptitle('')
    for a in f.axes: a.set_title(f'Position: {e+1}')
    f.savefig(f'{e}_heatmap.png', bbox_inches='tight')


# In[ ]:


def predict_captcha(img, learners):
    return ''.join([str(learner.predict(img)[0]) for learner in learners])


# In[ ]:


fig, ax = plt.subplots(ncols=5, figsize=(20,10))
for a, (img, lbl) in zip(ax.flatten(), learners[0].data.valid_ds):
    show_image(img, a)
    a.set_title(f'predicted: {predict_captcha(img, learners)}')
plt.show()


# In[ ]:


img_paths = learners[0].data.valid_ds.items
count = 0
correct = 0

for img_path in img_paths:
    lbl = img_path.name[:-4]
    img = open_image(img_path)
    predicted = predict_captcha(img, learners)
    if lbl==predicted: correct +=1
    count += 1
correct/count

