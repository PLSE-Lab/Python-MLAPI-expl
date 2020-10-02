#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.conv_learner import *


# In[ ]:


torch.cuda.is_available()


# In[ ]:


torch.backends.cudnn.enabled


# In[ ]:


INPUT_PATH = '../input/airbus-ship-detection/'


# In[ ]:


TRAIN = os.path.join(INPUT_PATH, 'train')
TEST = os.path.join(INPUT_PATH, 'test')


# In[ ]:


get_ipython().system('mkdir tmp')
get_ipython().system('mkdir model')
TMP = '/kaggle/working/tmp'
MODEL = '/kaggle/working/model'
get_ipython().system('ls')


# In[ ]:


masks = pd.read_csv(os.path.join(INPUT_PATH, 'train_ship_segmentations.csv'))
masks.head()


# In[ ]:


def is_boat(s):
  s = str(s)
  if len(s)>0 and ('nan' not in str.lower(s)):
    return 1
  else: return 0


# In[ ]:


masks['EncodedPixels']=masks['EncodedPixels'].apply(is_boat)


# In[ ]:


masks.drop_duplicates(inplace=True)
masks.head()


# In[ ]:


masks.hist()


# In[ ]:


masks.to_csv('boat_count.csv',index=False)


# In[ ]:


def get_data(sz):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    return ImageClassifierData.from_csv(INPUT_PATH, 'train', label_csv, tfms=tfms,
                    suffix='', val_idxs=val_idxs, test_name='test')
      
bs=64; 
f_model = resnet34


# In[ ]:


n = len(list(open('boat_count.csv')))-1
print(n)
label_csv = 'boat_count.csv'
val_idxs = get_cv_idxs(n)


# In[ ]:


sz = 64
data = get_data(sz)


# In[ ]:


get_ipython().system('mkdir  /kaggle/working/tmp/')


# In[ ]:


get_ipython().system('ls ../input/airbus-ship-detection/train/../input/airbus-ship-detection/')


# In[ ]:


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[ ]:


data = data.resize(int(sz*1.3), TMP)


# In[ ]:


x,y = next(iter(data.val_dl))
plt.imshow(data.val_ds.denorm(to_np(x))[0]*1.4);


# In[ ]:


# !cp ../input/resnet34/resnet34.pth /tmp/.torch/models/resnet34-333f7ec4.pth


# In[ ]:


learner = ConvLearner.pretrained(f_model, data, tmp_name=TMP,models_name=MODEL)


# In[ ]:


lrf=learner.lr_find()
learner.sched.plot()


# In[ ]:


lr = (1E-2)/2


# In[ ]:


learner.fit(lr, 3, cycle_len=1, cycle_mult=2)
learner.save(f'{sz}')


# In[ ]:


lrs = np.array([lr/9,lr/3,lr])


# In[ ]:


learner.unfreeze()
learner.fit(lrs, 3, cycle_len=1, cycle_mult=2)


# In[ ]:


learner.save(f'{sz}-lrs ')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




