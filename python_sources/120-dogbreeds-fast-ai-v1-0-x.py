#!/usr/bin/env python
# coding: utf-8

# # **1.Peapare Enviroment**

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


print(torch.cuda.is_available(), torch.backends.cudnn.enabled)


# # **2.Peapare Data**

# In[ ]:


path_model='/kaggle/working/'
path_input="/kaggle/input/"
label_df = pd.read_csv(f"{path_input}labels.csv")
label_df.head()


# In[ ]:


label_df.shape


# In[ ]:


label_df.pivot_table(index='breed',aggfunc=len).sort_values('id',ascending=False)


# In[ ]:


data = ImageDataBunch.from_csv(
                      path_input,
                      folder='train',
                      valid_pct=0.2,
                      ds_tfms=get_transforms(flip_vert=True,max_rotate=20., max_zoom=1.1),
                      size=224,
                      test='/kaggle/input/test/test',
                      suffix='.jpg',
                      bs=64,
                      num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(8,8))


# In[ ]:


[print(len(e)) for e in [data.train_ds, data.valid_ds, data.test_ds]]


# In[ ]:


files = os.listdir(f'{path_input}train/')[:5]
img = plt.imread(f'{path_input}train/{files[0]}')
plt.imshow(img)


# # **3.Create & Train Network**

# In[ ]:


#learner = Learner(data, models.resnet50, metrics=[accuracy], )
learner = create_cnn(data,models.resnet50,metrics=[accuracy],model_dir=f'{path_model}')


# In[ ]:


learner.fit_one_cycle(3)


# ---

# # **4.Predict**
# 

# In[ ]:


np.set_printoptions(precision=6, suppress=True)
test_result = learner.get_preds(ds_type=DatasetType.Test)


# In[ ]:


for i in range(0, 12):
    print(np.array(test_result[0][1][i*10:i*10+10]))


# # **5.Submission**
# following the [format of result](https://www.kaggle.com/c/dog-breed-identification#evaluation):
# 
# > &nbsp;&nbsp;id,affenpinscher,afghan_hound,..,yorkshire_terrier<br/>
# &nbsp;&nbsp;000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083<br/>
# &nbsp;&nbsp;etc.
# 

# In[ ]:


pd.options.display.float_format = '{:.6f}'.format
df = pd.DataFrame(np.array(test_result[0]))
df.columns = data.classes
df.head()


# In[ ]:


df.shape


# In[ ]:


# insert clean ids - without folder prefix and .jpg suffix - of images as first column
df.insert(0, "id", [e.name[:-4] for e in data.test_ds.x.items])


# In[ ]:


df.head()


# In[ ]:


df.to_csv(f"dog-breed-identification-submission.csv", index=False)

