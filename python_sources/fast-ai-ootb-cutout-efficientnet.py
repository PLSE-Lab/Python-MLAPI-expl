#!/usr/bin/env python
# coding: utf-8

# Reference
# - https://www.kaggle.com/sujoykg/xception-keras

# Try
# 
# - Use mixup
# - fp_16
# - Oversampling
# - cutout
# - efficientnet b3

# In[1]:


from fastai.vision import *
from fastai.metrics import *
PATH = Path('../input')


# In[2]:


ann_file = '../input/train2019.json'
with open(ann_file) as data_file:
        train_anns = json.load(data_file)

train_anns_df = pd.DataFrame(train_anns['annotations'])[['image_id','category_id']]
train_img_df = pd.DataFrame(train_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
df_train_file_cat = pd.merge(train_img_df, train_anns_df, on='image_id')
df_train_file_cat['category_id']=df_train_file_cat['category_id'].astype(str)
df_train_file_cat = df_train_file_cat.drop(['image_id'],axis=1)
df_train_file_cat.head()


# In[3]:


get_ipython().run_cell_magic('time', '', "# Try Oversampling\n\nres = None\nsample_to = df_train_file_cat.category_id.value_counts().max() # which is 500\n\nfor grp in df_train_file_cat.groupby('category_id'):\n    n = grp[1].shape[0]\n    additional_rows = grp[1].sample(0 if sample_to < n  else sample_to - n, replace=True)\n    rows = pd.concat((grp[1], additional_rows))\n    \n    if res is None: res = rows\n    else: res = pd.concat((res, rows))")


# In[4]:


res.category_id.value_counts()[:10]


# In[5]:


test_ann_file = '../input/test2019.json'
with open(test_ann_file) as data_file:
        test_anns = json.load(data_file)
test_img_df = pd.DataFrame(test_anns['images'])[['file_name','id']].rename(columns={'id':'image_id'})
test_img_df.head()


# In[22]:


src = (
ImageList.from_df(df=res,path=PATH/"train_val2019")
    .use_partial_data(0.3)
    .split_by_rand_pct(0.1)
    .label_from_df()
    .add_test(ImageList.from_df(df=test_img_df,path=PATH/"test2019"))
)


# In[23]:


data = (
    src
    .transform(get_transforms(),size=128)
    .databunch(bs=64*2)
    .normalize(imagenet_stats)
)


# In[24]:


get_ipython().system('pip install efficientnet_pytorch')


# In[25]:


from efficientnet_pytorch import EfficientNet


# In[26]:


model_name = 'efficientnet-b3'
def getModel(pret):
    model = EfficientNet.from_pretrained(model_name)
#     model._bn1 = nn.Identity()
    model._fc = nn.Linear(1536,data.c)
    return model


# In[27]:


# learn = cnn_learner(data,models.densenet201,metrics=[error_rate],model_dir='/kaggle/working',pretrained=True,loss_func=LabelSmoothingCrossEntropy()).mixup()


# In[28]:


learn = Learner(data,getModel(False),metrics=[error_rate],model_dir='/kaggle/working',loss_func=LabelSmoothingCrossEntropy()).mixup().to_fp16()


# In[29]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3,1e-3)


# In[ ]:


SZ=224
cutout_frac = 0.25
p_cutout = 0.75
cutout_sz = round(SZ*cutout_frac)
cutout_tfm = cutout(n_holes=(1,1), length=(cutout_sz, cutout_sz), p=p_cutout)


# In[ ]:


learn.data = (
    src
    .transform(get_transforms(xtra_tfms=[cutout_tfm]),size=SZ)
    .databunch(bs=64)
    .normalize(imagenet_stats)
)


# In[ ]:


learn.fit_one_cycle(7,1e-3)


# In[ ]:


learn.save('cutout-efficient')


# In[ ]:


# learn.unfreeze()
# learn.fit_one_cycle(8,slice(1e-6,1e-4))


# In[ ]:


preds,y = learn.get_preds(DatasetType.Test)


# In[ ]:


results = torch.topk(preds,5)


# In[ ]:


out = []
for i in results[1].numpy():
    temp = ""
    for j in i:
        temp += (" "+str(data.classes[j])) 
    out.append(temp)
# print(out)


# In[ ]:


sam_sub_df = pd.read_csv('../input/kaggle_sample_submission.csv')
# sam_sub_df.head()
sam_sub_df["predicted"] = out
sam_sub_df.head()


# In[ ]:


sam_sub_df.to_csv("submission.csv",index=False)

