#!/usr/bin/env python
# coding: utf-8

# # fastai v2 example
# 
# How far does one lecture of fastai takes you? Or two? This notebook will be improved with some new stuff as lectures go on.

# In[ ]:


get_ipython().system('pip install fastai2>=0.0.11 graphviz ipywidgets matplotlib nbdev>=0.2.12 pandas scikit_learn azure-cognitiveservices-search-imagesearch sentencepiece')


# In[ ]:


get_ipython().system('pip install --upgrade fastprogress')


# In[ ]:


from fastai2.vision.all import *
from sklearn.metrics import roc_auc_score


# In[ ]:


data_path = Path("../input/plant-pathology-2020-fgvc7/")


# In[ ]:


df = pd.read_csv(data_path/"train.csv")


# In[ ]:


df.head()


# A funny thing is that it is single-label classification task, not a multi-label, which can be checked like this:

# In[ ]:


df.iloc[:, 1:].sum(axis=1).value_counts()


# In[ ]:


imglabels = list(df.columns[1:])


# In[ ]:


df["labels"] = df.apply(lambda x: imglabels[x.values[1:].argmax()], axis=1)


# In[ ]:


df.head()


# In[ ]:


dls = ImageDataLoaders.from_df(df,
                               path=data_path, 
                               suff=".jpg", 
                               folder="images",
                               label_col="labels",
                               item_tfms=RandomResizedCrop(512, min_scale=0.5), # note that we use a bigger image size
                               batch_tfms=aug_transforms(),
                               valid_pct=0.05,
                               bs=16,
                               val_bs=16
                               )


# In[ ]:


dls.show_batch()


# In[ ]:


def mean_roc_auc(preds, targets, num_cols=4):
    """The competition metric
    
    Quoting: 'Submissions are evaluated on mean column-wise ROC AUC. 
    In other words, the score is the average of the individual AUCs 
    of each predicted column. '
    
    Unfortunately, we cannot use in validation, as it can happen that
    all files in a batch has the same label, and ROC is undefined
    """
    aucs = []
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    for i in range(num_cols):
        # grab a column from the networks output
        cpreds = preds[:, i]
        # see which objects have the i-th label
        ctargets = [x == i for x in targets]
        aucs.append(roc_auc_score(ctargets, cpreds))
    return sum(aucs) / num_cols


# In[ ]:


learn = cnn_learner(dls, resnet50, metrics=[accuracy], model_dir="/kaggle/working")


# We now know a bit more about setting a correct learning rate, so let's do it by finding a good LR with the learning rate finder technique.

# In[ ]:


learn.lr_find()


# In[ ]:


learn.fit(4, lr=1e-3)


# Great. Now let's unfreeze the lower layers and look at the suggested LR again.

# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.save("model")


# In[ ]:


learn.fit_one_cycle(16, lr_max=slice(1e-6,1e-5), cbs=[SaveModelCallback(every='epoch', monitor="accuracy")])


# In[ ]:


learn.load("model")


# Aaand prediction time!

# In[ ]:


test_image_ids = [img.split(".")[0] for img in os.listdir(data_path/"images") if img.startswith("Test")]
test_images = [data_path/"images"/f"{img}.jpg" for img in test_image_ids]
preds = learn.get_preds(dl=dls.test_dl(test_images, shuffle=False, drop_last=False))


# In[ ]:


# ensure that the order of columns in preds matches the imglabels
preds = preds[0].cpu().numpy()
vocab = list(dls[0].dataset.vocab)
column_permutation = [vocab.index(l) for l in imglabels]
preds = preds[:, column_permutation]

submission = pd.DataFrame()
submission["image_id"] = test_image_ids
for i in range(len(imglabels)):
    submission[imglabels[i]] = preds[:, i]
submission.to_csv("submission.csv", index=False)


# In[ ]:


submission.head(10)

