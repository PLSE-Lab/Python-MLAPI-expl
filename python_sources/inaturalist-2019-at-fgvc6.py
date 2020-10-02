#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


from fastai.vision import *
from fastai.metrics import *
PATH = Path("../input")


# In[ ]:


ann_file = "../input/train2019.json"

with open(ann_file) as data_file:
    train_anns = json.load(data_file)
    
train_anns_df = pd.DataFrame(train_anns["annotations"])[["id", "category_id"]]
train_img_df = pd.DataFrame(train_anns["images"])[["id", "file_name"]]


# In[ ]:


df_train = pd.merge(train_img_df, train_anns_df, on = "id")
df_train.drop(["id"], axis = 1, inplace = True)


# In[ ]:


#sample_to = df_train.category_id.value_counts().max()
#res = None

#for grp in df_train.groupby("category_id"):
#    n = grp[1].shape[0]
#    additional_rows = grp[1].sample(0 if sample_to < n else sample_to - n, replace=True)
#    rows = pd.concat((grp[1], additional_rows))
#    res = pd.concat((res, rows))


# In[ ]:


test_ann_file = PATH/"test2019.json"

with open(test_ann_file) as data_file:
    test_anns = json.load(data_file)

test_img_df = pd.DataFrame(test_anns["images"])[["file_name", "id"]]
#test_img_df.head()


# In[ ]:


df_train_sub = df_train[:10000]
#print(df_train_sub)
#print(df_train_sub.shape)
#res_sub = res[:10000]
#res_sub.head()
test_img_df_sub = test_img_df[:1000]


# In[ ]:


src = (ImageList.from_df(df=df_train, path=PATH/"train_val2019", cols = 0)
    #.use_partial_data(0.2)
    .split_by_rand_pct(0.1)
    .label_from_df("category_id")
    .add_test(ImageList.from_df(df=test_img_df, path=PATH/"test2019", cols = 0))
    )


# In[ ]:


data = (src
       .transform(get_transforms(), size = 128)
       .databunch(bs=32)
       .normalize(imagenet_stats))


# In[ ]:


data.classes
data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


#learn = cnn_learner(data, models.resnet34, metrics = accuracy, model_dir="/tmp/model/")
#learn.save("StaticWeights_resnet34_v1")
#learn.lr_find()
#learn.recorder.plot()
#learn.unfreeze()
#learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-1))
#learn.save("FittedWeights_resnet34_v1")
#interp = ClassificationInterpretation.from_learner(learn)
#losses,idxs = interp.top_losses()
#interp.plot_top_losses(9, figsize=(15,11))
#interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
#interp.most_confused(min_val=2)


# In[ ]:


import re
for grp in df_train.groupby("category_id"):
    n = grp[1].iloc[0,0]
    n = re.search("(?<=/)(.*)(?=/.*/)",n).group(0)
    cat = grp[1].iloc[0,1]
    print("Type {} has label {}".format(n, cat))


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics = accuracy, model_dir="/tmp/model/")
#learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-1))
#learn.save("StaticWeights_resnet50_v1")


# In[ ]:


learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(10, max_lr=slice(1e-6,3e-3))
learn.save("FittedWeights_resnet50_v1")


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


preds, y = learn.get_preds(DatasetType.Test)


# In[ ]:


#pred_t, _ = learn.TTA(ds_type=DatasetType.Test)
#pred_t_max = np.argmax(pred_t, 1); pred_t_max[0:5]


# In[ ]:


#result = torch.topk(pred_t, 5)
results = torch.topk(preds, 5)


# In[ ]:


predictions = []
for i in results[1].numpy():
    temp = ""
    for j in i:
        temp += (" "+str(data.classes[j]))
    predictions.append(temp)


# In[ ]:


submission_df = pd.read_csv(PATH/"kaggle_sample_submission.csv")
#submission_df_sub = submission_df[:1000]
#submission_df_sub["predicted"] = predictions
submission_df["predicted"] = predictions


# In[ ]:


#submission_df_sub.to_csv("submission_sub.csv", index = False)
submission_df.to_csv("submission.csv", index = False)

