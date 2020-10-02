#!/usr/bin/env python
# coding: utf-8

# Unfreeze, progressive resizing, cross_validation, devise

# In[ ]:


import fasttext as ft
from fastai2.vision.all import * 
from sklearn.metrics import roc_auc_score
from timm import create_model
import pandas as pd
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[ ]:


BS=32
IMG_SIZE = 224
N_FOLDS = 5
SEED = 2
seed_everything(SEED)


# In[ ]:


IMG_PATH = Path('/home/bf/Documents/Plant/images')


# In[ ]:


train = pd.read_csv('train.csv')


# In[ ]:


def get_tag(row):
    if row.healthy:
        return "healthy"
    if row.multiple_diseases:
        return "multiple_diseases"
    if row.rust:
        return "rust"
    if row.scab:
        return "scab"
train['label'] = [get_tag(train.iloc[idx]) for idx in train.index]
train.drop(columns=['healthy', 'multiple_diseases', 'rust', 'scab'], inplace=True)
LABEL_COLS = ['healthy', 'multiple_diseases', 'rust', 'scab']


# In[ ]:


train['fold'] = -1

strat_kfold = MultilabelStratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
for i, (_, test_index) in enumerate(strat_kfold.split(train.image_id.values, train.iloc[:,1:].values)):
    train.iloc[test_index, -1] = i
    
train['fold'] = train['fold'].astype('int')


# In[ ]:


PATH = Path('devise')


# In[ ]:


PATH.mkdir(exist_ok=True)


# In[ ]:


ft_vecs = ft.load_model(str((PATH/'wiki.en.bin')))


# In[ ]:


lc_vec_d = {w.lower(): ft_vecs.get_word_vector(w) for w in LABEL_COLS}


# In[ ]:


train.head()


# In[ ]:


vec = np.empty([train.shape[0], 300])
for i in range(train.shape[0]):
    vec[i] = (np.array(lc_vec_d[train.iloc[i, 1]]))


# In[ ]:


train = pd.concat([train, pd.DataFrame(vec)], 1)


# In[ ]:


train.head()


# In[ ]:


LAB_COL = train.columns.tolist()[3:]


# In[ ]:


item_tfms = RandomResizedCrop(IMG_SIZE, min_scale=0.75, ratio=(1.,1.))
batch_tfms=[*aug_transforms(size=IMG_SIZE, max_rotate=30., min_scale=0.75, flip_vert=True, do_flip=True)]


# In[ ]:


def get_data(fold):
    train_no_val = train.query(f'fold != {fold}')
    train_just_val = train.query(f'fold == {fold}')

    train_bal = pd.concat(
        [train_no_val.query('label != "multiple_diseases"'), train_just_val] +
        [train_no_val.query('label == "multiple_diseases"')] * 2
    ).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    datablock = DataBlock(
        blocks=(ImageBlock, RegressionBlock(c_out=300)),
        getters=[
            ColReader('image_id', pref=IMG_PATH, suff='.jpg'),
            ColReader(LAB_COL)
        ],
        splitter=IndexSplitter(train_bal.loc[train_bal.fold==fold].index),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )
    return datablock.dataloaders(source=train_bal, bs=BS)


# In[ ]:


def cos_loss(inp,targ): return 1 - F.cosine_similarity(inp,targ).mean()


# In[ ]:


dls = get_data(fold=0)


# In[ ]:


net = create_model('efficientnet_b3a', pretrained=True)


# In[ ]:


def create_timm_body(arch:str, pretrained=True, cut=None):
    model = create_model(arch, pretrained=pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NamedError("Cut must be either integer or function")


# In[ ]:


body = create_timm_body('efficientnet_b3a', pretrained=True)
nf = num_features_model(nn.Sequential(*body.children()))* (2)
head = create_head(nf, dls.c)
model = nn.Sequential(body, head)


# In[ ]:


dls.c


# In[ ]:


mixup = MixUp()


# In[ ]:


learn = Learner(dls, model, 
                cbs=mixup,
                #opt_func = partial(ranger, eps=1e-7),
                opt_func = partial(Lamb),
                loss_func= cos_loss).to_fp16()


# In[ ]:


#learn.summary()


# In[ ]:


learn.lr_find()


# In[ ]:


#learn.fine_tune(10, 1e-3)
learn.fit_one_cycle(10, 3e-3)


# In[ ]:


learn.unfreeze()
learn.lr_find()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-6, 1e-4))


# In[ ]:


learn.save('devise')


# In[ ]:


import nmslib

def create_index(a):
    index = nmslib.init(space='angulardist')
    index.addDataPointBatch(a)
    index.createIndex()
    return index

def get_knns(index, vecs):
     return zip(*index.knnQueryBatch(vecs, k=10, num_threads=4))

def get_knn(index, vec): return index.knnQuery(vec, k=10)


# In[ ]:


k2i = {'healthy':1, 'multiple_diseases':2, 
      'rust':3, 'scab':4}


# In[ ]:


syn_wv = [(k2i[k], v) for k,v in lc_vec_d.items()]
syns, wvs = list(zip(*syn_wv))


# In[ ]:


nn_wvs = create_index(wvs)


# In[ ]:


test_df = pd.read_csv('test.csv')
test_dl = dls.test_dl(test_df)


# In[ ]:


test_preds = learn.get_preds(dl=test_dl)


# In[ ]:


idxs,dists = get_knns(nn_wvs, test_preds[0])


# In[ ]:


labels = [idx[0] for idx in idxs]


# In[ ]:


subm = pd.read_csv('sample_submission.csv')


# In[ ]:


subm.healthy = 0
subm.multiple_diseases = 0
subm.rust = 0
subm.scab = 0


# In[ ]:


for i in range(subm.shape[0]):
    subm.iloc[i, labels[i]+1] = 1


# In[ ]:


subm


# In[ ]:


subm.to_csv('Devise.csv', index=False)


# In[ ]:





# In[ ]:




