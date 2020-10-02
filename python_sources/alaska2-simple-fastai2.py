#!/usr/bin/env python
# coding: utf-8

# This notebook demonstrates how fastai2 makes it easier to do practical DL for domain experts. Imagine you are a detective.

# 22
# 
# Fixed metrics
# 
# 
# 17
# 
# Added metrics

# In[ ]:


get_ipython().system('pip install -q git+https://github.com/fastai/fastai2')
get_ipython().system('pip install -q git+https://github.com/fastai/fastcore')


# In[ ]:


from fastai2.vision.all import *


# In[ ]:


path = Path("/kaggle/input/alaska2-image-steganalysis")


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(15)


# # Behold, the DataBlock

# In[ ]:


# def label_func(f): return False if f.parent.name == "Cover" else True
def label_func(f): return f.parent.name


# In[ ]:


read_files = partial(get_image_files, folders=["JUNIWARD", "JMiPOD", "Cover", "UERD"])


# In[ ]:


files = read_files(path)


# In[ ]:


valid_idx = np.concatenate([np.random.permutation(75_000)[:15_000],
                            np.random.permutation(range(135_000, 150_000))[:15_000],
                            np.random.permutation(range(210_000, 225_000))[:15_000],
                            np.random.permutation(range(285_000, 300_000))[:15_000]])


# In[ ]:


def get_data(bs=8):
    return DataBlock(blocks=(ImageBlock, CategoryBlock),
                     get_items=lambda x:files,
                     get_y=label_func,
                     splitter=IndexSplitter(valid_idx),
                     item_tfms=None,
                     #only flips
                     batch_tfms=aug_transforms(flip_vert=True, max_rotate=0, min_zoom=1,
                                               max_zoom=1, max_lighting=0, max_warp=0),
                      ).dataloaders(path, bs=bs)


# In[ ]:


# dls = get_data()


# In[ ]:


# dls.show_batch()


# In[ ]:


# dls.vocab


# In[ ]:


# len(dls.train_ds), len(dls.valid_ds)


# In[ ]:


# del dls


# # Model

# In[ ]:


from sklearn import metrics
        
def alaska_weighted_auc(y_true, y_valid):
    """
    https://www.kaggle.com/anokas/weighted-auc-metric-updated
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)
    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (y_max > tpr)
        if mask.sum() == 0:
            continue

        x_padding = np.linspace(fpr[mask][-1], 1, 100)
        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric / normalization

def weighted_roc_auc(preds, targs):
    return alaska_weighted_auc(targs, 1 - preds.clamp(0,1).numpy()[:, 0])


# In[ ]:


def get_learner(bs, model):
    dls = get_data(bs)
    display(dls.vocab)
    return cnn_learner(dls, model,
                       metrics=[error_rate, AccumMetric(weighted_roc_auc, flatten=False)],
                       ).to_fp16()


# In[ ]:


# learn = get_learner(bs=60, model=resnet50)
learn = get_learner(bs=160, model=resnet34)


# In[ ]:


# learn.lr_find()


# In[ ]:


# learn.recorder.plot_lr_find(skip_end=10)


# In[ ]:


lr = 1e-3


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(5, lr)


# # Interpretation

# In[ ]:


# interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


# interp.plot_top_losses(9, figsize=(15, 10))


# In[ ]:


# interp.plot_confusion_matrix()


# # Submission

# In[ ]:


tst_dl = learn.dls.test_dl(get_image_files(path/"Test"))


# In[ ]:


preds, _ = learn.get_preds(dl=tst_dl)


# In[ ]:


subm = pd.read_csv(path/"sample_submission.csv")


# In[ ]:


subm.head()


# In[ ]:


# subm.iloc[:, 1:] = preds[:, 1]
subm.iloc[:, 1:] = 1- preds.numpy()[:, 0]


# In[ ]:


subm.to_csv("submission.csv", index=False)


# In[ ]:


pd.read_csv("submission.csv")


# In[ ]:




