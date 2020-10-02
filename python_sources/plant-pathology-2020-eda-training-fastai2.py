#!/usr/bin/env python
# coding: utf-8

# # Plant Pathology 2020 with fastai2
# 
# The goal of this notebook is to showcase some of the features in the soon to be released [fastai2](https://github.com/fastai/fastai2) library and to provide a starter example for classifying categories of foliar diseases in apple trees.
# 
# ## Changelog
# 
# ### v10 (2020-05-1)
# 
# * Bug fixes.
# * Add test data to EDA.
# 
# ### v8,v9 (2020-03-15)
# 
# * Train 5 folds.
# * Clean up some broken parts.
# 
# ### v7 (2020-03-13)
# 
# * Resize to 1024
# * Less oversampling.
# 
# ### v6 (2020-03-13)
# 
# * Label smoothing.
# * Fix validation bug.
# 
# ### v5 (2020-03-13)
# 
# * Oversampling minority class.
# 
# ### v4 (2020-03-12)
# 
# * Add competition metric to learner output.
# * Add confusion matrix.
# 
# ### v3. 2020-03-12
# 
# * Bigger image size.
# 
# ### v2. 2020-03-12
# 
# * First working version

# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastcore > /dev/null')
get_ipython().system('pip install git+https://github.com/fastai/fastai2 > /dev/null')
get_ipython().system('pip install iterative-stratification > /dev/null')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import pandas as pd
import sys
from collections import Counter
from pathlib import Path

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm.notebook import tqdm
from torchvision.models import densenet121
from sklearn.metrics import roc_auc_score
from torch.utils.data.sampler import WeightedRandomSampler

from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ## Params

# In[ ]:


DATA_PATH = Path('../input/plant-pathology-2020-fgvc7')
IMG_PATH = DATA_PATH / 'images'
LABEL_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']

IMG_SIZE = 512
SEED = 420
N_FOLDS = 5
BS = 16
N_FOLDS = 5

ARCH = densenet121


# In[ ]:


seed_everything(SEED)


# ## EDA

# In[ ]:


train_df = pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')
test_df = pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')


# In[ ]:


train_df.head()


# ### Dataset size

# In[ ]:


(len(train_df), len(test_df))


# ### Label distribution

# In[ ]:


_, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True, figsize=(10, 3))
for ax, column in zip(axes, LABEL_COLS):
    train_df[column].value_counts().plot.bar(title=column, ax=ax)
plt.show()


# In[ ]:


plt.title('Label dist')
train_df[LABEL_COLS].idxmax(axis=1).value_counts().plot.bar()


# Let's see how many times the labels appear together.

# In[ ]:


train_df.iloc[:,1:-1].sum(axis=1).value_counts()


# In[ ]:


train_df[['healthy', 'multiple_diseases', 'rust', 'scab']].sum(axis=1).unique()


# Looks like never. So this appears to be multiclass but not multilabel classification.

# ### Img size distribution

# In[ ]:


def get_size(df):
    hs, ws = [], []
    for _, row in tqdm(df.iterrows(), total=len(train_df)):
        img = Image.open(IMG_PATH/(row.image_id+'.jpg'))
        h, w = img.size
        hs.append(h)
        ws.append(w)
        
    return hs, ws


# In[ ]:


train_hs, train_ws = get_size(train_df)
test_hs, test_ws = get_size(test_df)


# In[ ]:


for set_label, set_size in ('train', [train_hs, train_ws]), ('test', [test_hs, test_ws]):
    print(f'{set_label} height val counts: {Counter(set_size[0])}')
    print(f'{set_label} width val counts: {Counter(set_size[1])}')

    _, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True, figsize=(10, 3))
    for ax, column, vals in zip(axes, ['heights', 'widths'], set_size):
        ax.hist(vals, bins=100)
        ax.set_title(f'{set_label} {column} hist')

plt.show()


# All images are either: 2048x1365 or 1365x2048.

# ### Colour distribution

# In[ ]:


def plot_colour_hist(df, title):
    red_values = []; green_values = []; blue_values = []; all_channels = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img = np.array(Image.open(IMG_PATH/(row.image_id+'.jpg')))
        red_values.append(np.mean(img[:, :, 0]))
        green_values.append(np.mean(img[:, :, 1]))
        blue_values.append(np.mean(img[:, :, 2]))
        all_channels.append(np.mean(img))
        
    _, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True, figsize=(16, 3), sharey=True)
    for ax, column, vals, c in zip(
        axes,
        ['red', 'green', 'blue', 'all colours'],
        [red_values, green_values, blue_values, all_channels],
        'rgbk'
    ):
        ax.hist(vals, bins=100, color=c)
        ax.set_title(f'{column} hist')

    plt.suptitle(title)
    plt.show()


# In[ ]:


plot_colour_hist(train_df, title='Train colour dist')


# In[ ]:


plot_colour_hist(test_df, title='Test colour dist')


# ## Create folds

# I'll use iterative stratification to create balanced folds.

# In[ ]:


train_df['fold'] = -1

strat_kfold = MultilabelStratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
for i, (_, test_index) in enumerate(strat_kfold.split(train_df.image_id.values, train_df.iloc[:,1:].values)):
    train_df.iloc[test_index, -1] = i
    
train_df['fold'] = train_df['fold'].astype('int')


# In[ ]:


_, axes = plt.subplots(ncols=5, nrows=1, constrained_layout=True, figsize=(16, 3), sharey=True)
for ax, fold in zip(axes, range(5)):
    train_df.query(f'fold == {fold}')[LABEL_COLS].idxmax(axis=1).value_counts().plot.bar(ax=ax)
    ax.set_title(f'Fold {fold} label dist') 

plt.show()


# In[ ]:


train_df.to_csv('train_with_strat_folds.csv', index=False)


# ## Data (inc oversampling)

# In[ ]:


train_df['label'] = train_df[LABEL_COLS].idxmax(axis=1)


# Right now, I'm x2 the number of multiple diseases labels as that appears to be the majorly unrepresented class.

# In[ ]:


def get_data(fold):
    train_df_no_val = train_df.query(f'fold != {fold}')
    train_df_just_val = train_df.query(f'fold == {fold}')

    train_df_bal = pd.concat(
        [train_df_no_val.query('label != "multiple_diseases"'), train_df_just_val] +
        [train_df_no_val.query('label == "multiple_diseases"')] * 2
    ).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=LABEL_COLS)),
        getters=[
            ColReader('image_id', pref=IMG_PATH, suff='.jpg'),
            ColReader('label')
        ],
        splitter=IndexSplitter(train_df_bal.loc[train_df_bal.fold==fold].index),
        item_tfms=Resize(IMG_SIZE),
        batch_tfms=aug_transforms(size=IMG_SIZE, max_rotate=30., min_scale=0.75, flip_vert=True, do_flip=True)
    )
    return datablock.dataloaders(source=train_df_bal, bs=BS)


# In[ ]:


dls = get_data(fold=0)


# In[ ]:


dls.show_batch()


# ## Training

# In[ ]:


def comp_metric(preds, targs, labels=range(len(LABEL_COLS))):
    # One-hot encode targets
    targs = np.eye(4)[targs]
    return np.mean([roc_auc_score(targs[:,i], preds[:,i]) for i in labels])

def healthy_roc_auc(*args):
    return comp_metric(*args, labels=[0])

def multiple_diseases_roc_auc(*args):
    return comp_metric(*args, labels=[1])

def rust_roc_auc(*args):
    return comp_metric(*args, labels=[2])

def scab_roc_auc(*args):
    return comp_metric(*args, labels=[3])


# In[ ]:


def get_learner(fold_num, lr=1e-3):
    opt_func = partial(Adam, lr=lr, wd=0.01, eps=1e-8)

    data = get_data(fold_num)

    learn = cnn_learner(
        data, ARCH, opt_func=opt_func,
        loss_func=LabelSmoothingCrossEntropy(),
        metrics=[
            AccumMetric(healthy_roc_auc, flatten=False),
            AccumMetric(multiple_diseases_roc_auc, flatten=False),
            AccumMetric(rust_roc_auc, flatten=False),
            AccumMetric(scab_roc_auc, flatten=False),
            AccumMetric(comp_metric, flatten=False)]
        ).to_fp16()

    return learn


# In[ ]:


get_learner(fold_num=0).lr_find()


# In[ ]:


def print_metrics(val_preds, val_labels):
    comp_metric_fold = comp_metric(val_preds, val_labels)
    print(f'Comp metric: {comp_metric_fold}')
    
    healthy_roc_auc_metric = healthy_roc_auc(val_preds, val_labels)
    print(f'Healthy metric: {healthy_roc_auc_metric}')
    
    multiple_diseases_roc_auc_metric = multiple_diseases_roc_auc(val_preds, val_labels)
    print(f'Multi disease: {multiple_diseases_roc_auc_metric}')
    
    rust_roc_auc_metric = rust_roc_auc(val_preds, val_labels)
    print(f'Rust metric: {rust_roc_auc_metric}')
    
    scab_roc_auc_metric = scab_roc_auc(val_preds, val_labels)
    print(f'Scab metric: {scab_roc_auc_metric}')


# In[ ]:


all_val_preds = []
all_val_labels = []
all_test_preds = []

for i in range(N_FOLDS):
    print(f'Fold {i} results')

    learn = get_learner(fold_num=i)

    learn.fit_one_cycle(4)
    learn.unfreeze()

    learn.fit_one_cycle(8, slice(1e-5, 1e-4))
    
    learn.recorder.plot_loss()
    
    learn.save(f'model_fold_{i}')
    
    val_preds, val_labels = learn.get_preds()
    
    print_metrics(val_preds, val_labels)
    
    all_val_preds.append(val_preds)
    all_val_labels.append(val_labels)
    
    test_dl = dls.test_dl(test_df)
    test_preds, _ = learn.get_preds(dl=test_dl)
    all_test_preds.append(test_preds)
    
plt.show()


# In[ ]:


print_metrics(np.concatenate(all_val_preds), np.concatenate(all_val_labels))


# ## Interpret

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(9, figsize=(15, 10))


# In[ ]:


interp.plot_confusion_matrix(normalize=True, figsize=(6, 6))


# ## Test predictions

# In[ ]:


test_df_output = pd.concat([test_df, pd.DataFrame(np.mean(np.stack(all_test_preds), axis=0), columns=LABEL_COLS)], axis=1)


# In[ ]:


test_df_output.head()


# In[ ]:


test_df_output.to_csv('submission.csv', index=False)


# In[ ]:


get_ipython().system('head -n 5 submission.csv')

