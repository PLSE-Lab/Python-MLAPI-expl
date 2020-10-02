#!/usr/bin/env python
# coding: utf-8

# ## Multi-label classification

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().system('pip install fastai==0.7.0 --no-deps')
get_ipython().system('pip install torch==0.4.1 torchvision==0.2.1')

from fastai.conv_learner import *


# In[ ]:


print(os.listdir("../input/"))
PATH = '../input/planet-understanding-the-amazon-from-space/'


# In[ ]:


TMP_PATH = "/tmp/tmp"
MODEL_PATH = "/tmp/model"


# In[ ]:


ls {PATH}


# In[ ]:


df = pd.read_csv(PATH+'sample_submission_v2.csv')
df.head(5)


# ## Multi-label versus single-label classification

# In[ ]:


from fastai.plots import *


# In[ ]:


def get_1st(path): return glob(f'{path}/*.*')[0]


# In[ ]:


dc_path = '../input/newdogscats/dogscats/dogscats/train/'
list_paths = [get_1st(f"{dc_path}cats"), get_1st(f"{dc_path}dogs")]
plots_from_files(list_paths, titles=["cat", "dog"], maintitle="Single-label classification")


# In single-label classification each sample belongs to one class. In the previous example, each image is either a *dog* or a *cat*.

# In[ ]:


list_paths = [f"{PATH}train-jpg/train_0.jpg", f"{PATH}train-jpg/train_1.jpg"]
titles=["haze primary", "agriculture clear primary water"]
plots_from_files(list_paths, titles=titles, maintitle="Multi-label classification")


# In multi-label classification each sample can belong to one or more classes. In the previous example, the first images belongs to two classes: *haze* and *primary*. The second image belongs to four classes: *agriculture*, *clear*, *primary* and  *water*.

# ## Multi-label models for Planet dataset

# In[ ]:


# planet.py

from fastai.imports import *
from fastai.transforms import *
from fastai.dataset import *
from sklearn.metrics import fbeta_score
import warnings

def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max([fbeta_score(targs, (preds>th), 2, average='samples')
                    for th in np.arange(start,end,step)])

metrics=[f2]
f_model = resnet34


# In[ ]:


label_csv = f'{PATH}train_v2.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)


# We use a different set of data augmentations for this dataset - we also allow vertical flips, since we don't expect vertical orientation of satellite images to change our classifications.

# In[ ]:


def get_data(sz):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    return ImageClassifierData.from_csv(PATH, 'train-jpg', label_csv, tfms=tfms,
                    suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg-v2')


# In[ ]:


data = get_data(256)


# In[ ]:


x,y = next(iter(data.val_dl))


# In[ ]:


y


# In[ ]:


list(zip(data.classes, y[0]))


# In[ ]:


plt.imshow(data.val_ds.denorm(to_np(x))[0]*1.4);


# In[ ]:


sz=64


# In[ ]:


data = get_data(sz)


# In[ ]:


data = data.resize(int(sz*1.3), TMP_PATH)


# In[ ]:


learn = ConvLearner.pretrained(f_model, data, metrics=metrics)


# In[ ]:


lrf=learn.lr_find()
learn.sched.plot()


# In[ ]:


lr = 0.2


# In[ ]:


learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# In[ ]:


lrs = np.array([lr/9,lr/3,lr])


# In[ ]:


learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)


# In[ ]:


learn.save(f'{sz}')


# In[ ]:


learn.sched.plot_loss()


# In[ ]:


sz=128


# In[ ]:


learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# In[ ]:


learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')


# In[ ]:


sz=256


# In[ ]:


learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


# In[ ]:


learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')


# In[ ]:


multi_preds, y = learn.TTA()
preds = np.mean(multi_preds, 0)


# In[ ]:


f2(preds,y)


# In[ ]:


multi_preds, y = learn.TTA(is_test=True)


# In[ ]:


preds = np.mean(multi_preds, 0)
preds.shape


# In[ ]:


test_fnames = [os.path.basename(f).split(".")[0] for f in data.test_ds.fnames]
classes = np.array(data.classes, dtype=str)
res = [" ".join(classes[np.where(pp > 0.2)]) for pp in preds] 
test_df = pd.DataFrame(res, index=test_fnames, columns=['tags'])
test_df.head(5)
test_df.to_csv('submission.csv', index_label='image_name')
df = pd.read_csv('submission.csv')
df.head(20)


# ### End
