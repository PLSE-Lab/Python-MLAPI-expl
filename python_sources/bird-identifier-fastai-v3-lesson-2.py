#!/usr/bin/env python
# coding: utf-8

# Note: 
# 
# **This is a mirror of the FastAI Lesson 2 Nb. Big thanks to FastAI team for the notebook, that has been used as skeleton for this notebook.
# For complete info on the course, visit course.fast.ai**
# 
# [Original FastAI Lesson 2 Notebook](https://www.kaggle.com/init27/fastai-v3-lesson-2)
# 
# [Lesson Video Link](https://course.fast.ai/videos/?lesson=2)
# 
# [Lesson resources and updates](https://forums.fast.ai/t/lesson-2-official-resources-and-updates/28630)
# 
# [Lesson chat](https://forums.fast.ai/t/lesson-2-chat/28722)
# 
# [Further discussion thread](https://forums.fast.ai/t/lesson-2-further-discussion/28706)

# # Initial notes

# This notebook is based on [FastAI Lesson 2 Notebook](https://www.kaggle.com/init27/fastai-v3-lesson-2). As the training pipeline was painless in that notebook, I decided to test if recognizing two similar bird species in the same way will be painless as well. 
# 
# In this notebook we'll train the model to recognize birds from *regulus* familly:
# 
# **Goldcrest** (*Regulus regulus*)             |  **Firecrest** (*Regulus ignicapilla*)
# :-------------------------:|:-------------------------:
# ![Goldcrest](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Regulus_regulus_-Marwell_Wildlife%2C_Hampshire%2C_England-8.jpg/320px-Regulus_regulus_-Marwell_Wildlife%2C_Hampshire%2C_England-8.jpg)  |  ![Firecrest](https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Firecrest_-_Appenines_-_Italy_S4E5222_%2817014042119%29.jpg/320px-Firecrest_-_Appenines_-_Italy_S4E5222_%2817014042119%29.jpg)

# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.widgets import *
from fastai.callbacks.hooks import *

import time


# In[ ]:


## Pytorch seed for reproducibility
def random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


# # Creating dataset
# 
# *by: Francisco Ingham and Jeremy Howard. Inspired by [Adrian Rosebrock](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/)*
# 
# The images has been collected from Google Images: 
# 1. **Search** for the suitable images as long as ~300 images at the top will be suitable
# 2. **Download** the list of the images **urls** 
# 3. **Upload** the list to kaggle server: `("../input/bird-recognition-regulus-urls")`
# 
# The details of this process are described in [FastAi Lesson 2 Notebook](https://www.kaggle.com/init27/fastai-v3-lesson-2). 
# 
# **The lists are already uploaded for this notebook.**

# # Load the data

# In[ ]:


classes = ['r_regulus','r_ignicapilla']
path = Path('data/regulus')


# Copy the data from `../input/bird-recognition-regulus-urls` data to `data` folder, as `input` directory is read-only.
# 
# Use `max_pics=200` images, as the further images on the list are less accurate Google hits.

# In[ ]:


def copy_download_files_from_urls(folder, file, path=path, max_pics=200):
    dest = path/folder
    dest.mkdir(parents=True, exist_ok=True)

    get_ipython().system('cp ../input/* {path}/')
    try:
        download_images(os.path.join("..", "input","bird-recognition-regulus-urls", file), dest, max_pics=max_pics)
    except:
        try:
            # try once again
            download_images(os.path.join("..", "input","bird-recognition-regulus-urls", file), dest, max_pics=max_pics)
        except:
            pass


# In[ ]:


copy_download_files_from_urls("r_regulus", "r_regulus.csv", max_pics=200)


# In[ ]:


copy_download_files_from_urls("r_ignicapilla", "r_ignicapilla.csv", max_pics=200)


# Remove images that can't be opened:

# In[ ]:


for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)


# ## View data

# As Pytorch always set initial weights randomly, `random_seed` is used **for reproducibility** before createing data, creating model and calling `fit`. 
# 
# *Thanks to rpcoelho for solving the problem on [FastAI forum](https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/13?).*

# In[ ]:


# you can manipulate with the seed
random_seed(333)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(max_warp=0), size=81, num_workers=0).normalize(imagenet_stats)


# In[ ]:



random_seed(323)
data.show_batch(rows=9, ds_type=DatasetType.Valid, figsize=(12,12))


# You can manipulate with the seed to get different batch of images. I've choosed `333` as it results with good validation set. If you choose different seed you will see invalid images (eggs, stenograms, drawings, other species). We'll train 
# 
# **Keep in mind!**
# 
# Even though we use seed for reproducubility, **you may got different accuracy** if at least one of the images is not downloaded (timeout, doesn't exist anymore). It seems that even the commit has different accuracy due to that.

# # Initial model
# Before training the right model, we'll train simple model to detect images with the smallest loss. The smallest loss indicates that the images have improved the model just a little. 
# 
# It's probable that among those images are these, that we want do delete - eggs, stenograms, drawings etc.

# In[ ]:


def seed_and_fit_cycle(learner,*args, seed=123, **kwargs):
    random_seed(seed)
    learner.fit_one_cycle(*args, **kwargs)


# In[ ]:


random_seed(111)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
seed_and_fit_cycle(learn, 3, seed=42)


# ## Delete unwanted images
# 
# Some of our top losses aren't due to bad performance by our model. There are images in our data set that shouldn't be.
# 
# Using the `ImageCleaner` widget from `fastai.widgets` we can prune our top losses, removing photos that don't belong.
# 
# Notice that the widget will not delete images directly from disk but it will create a new csv file `cleaned.csv` from where you can create a new ImageDataBunch with the corrected labels to continue training your model.
# 
# **I deletete images of juveniles, drawings and any images without the two bird species.**

# In[ ]:


#ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=150)


# In[ ]:


#ImageCleaner(ds, idxs, path)


# # The right model
# Information about unwanted images has been saved to `cleaned(1).csv`. Recreate the `data` from the file and train the proper model.

# In[ ]:


size=225
np.random.seed(42)
data = ImageDataBunch.from_csv(path, valid_pct=0.3, csv_labels="../../../input/regulus-cleaned/cleaned(1).csv", ds_tfms=get_transforms(max_warp=0), size=225, num_workers=4).normalize(imagenet_stats)


# In[ ]:


random_seed(111)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
seed_and_fit_cycle(learn, 4)


# In[ ]:


learn.unfreeze()


# In[ ]:


random_seed(111)
learn.lr_find()
learn.recorder.plot()


# Following the FastAI lesson tips, I choose the learning rate with steepest interval for further training (for this learning rate the model learns the fastest).

# In[ ]:


seed_and_fit_cycle(learn, 6, max_lr=slice(2e-5,1e-3))


# ## Confusion Matrix

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# ## Missclassified files
# The code from FastAI Lesson 3 displays heatmap to show the parts of images recognized as bird species. 
# 
# We use it to make sure that **the model doesn't sufer for overfitting** (i.e. it recognizes birds, but not the other parts of the image).

# In[ ]:


wrong_labeled_valid_idx = []
for idx in range(len(data.valid_ds)):
    img,y = data.valid_ds[idx]
    pred_class,pred_idx,outputs = learn.predict(img)
    if (str(y) != str(pred_class)):
        wrong_labeled_valid_idx.append(idx)


# In[ ]:


def heatmap(idx):
    x,y = data.valid_ds[idx]
    m = learn.model.eval();

    def hooked_backward(cat=y):
        with hook_output(m[0]) as hook_a: 
            with hook_output(m[0], grad=True) as hook_g:
                preds = m(xb)
                preds[0,int(cat)].backward()
        return hook_a,hook_g

    xb,_ = data.one_item(x)
    xb_im = Image(data.denorm(xb)[0])
    xb = xb.cuda()

    def show_heatmap(hm):
        _,ax = plt.subplots()
        xb_im.show(ax)
        ax.set_title(label="True_label: " + str(y))
        return ax.imshow(hm, alpha=0.6, extent=(0,size,size,0),
                  interpolation='bilinear', cmap='magma')

    hook_a,hook_g = hooked_backward()
    acts  = hook_a.stored[0].cpu()
    avg_acts = acts.mean(0)
    return show_heatmap(avg_acts)


# **Species misclasified in my model:**

# In[ ]:


for idx in wrong_labeled_valid_idx:
    heatmap(idx)


# ## End
