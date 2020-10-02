#!/usr/bin/env python
# coding: utf-8

# Let's install the things we will need :)
# 
# The below will pull the latest version of the [fastai v2 library](https://github.com/fastai/fastai_dev). The library is under active development and there can be breaking changes. The notebook runs at the moment but should you encounter any issues in the future, it is best to verify whether the API changed or maybe rerun the notebook within a day or two.
# 
# If you find a genuine bug while getting acquainted with the library, it is best to [report it on github](https://github.com/fastai/fastai_dev/issues)!

# In[ ]:


pip install git+https://github.com/fastai/fastai_dev


# In[ ]:


dir_csv = '../input/rsna-intracranial-hemorrhage-detection'
dir_train_img = '../input/rsna-train-stage-1-images-png-224x/stage_1_train_png_224x'
dir_test_img = '../input/rsna-test-stage-1-images-png-224x/stage_1_test_png_224x'


# In[ ]:


from fastai2.torch_basics import *
from fastai2.test import *
from fastai2.layers import *
from fastai2.data.all import *
from fastai2.optimizer import *
from fastai2.learner import *
from fastai2.metrics import *
from fastai2.vision.all import *
from fastai2.vision.learner import *
from fastai2.vision.models import *
from fastai2.callback.all import *


# In[ ]:


items = get_image_files(dir_train_img)
items = [i for i in items if '(copy)' not in i.name]


# Processing the `stage_1_train.csv` into something more parsable.

# In[ ]:


get_ipython().system('mkdir -p data')

df_train = pd.read_csv(f'{dir_csv}/stage_1_train.csv')
df_train['fn'] = df_train.ID.apply(lambda x: '_'.join(x.split('_')[:2]) + '.png')
df_train.columns = ['ID', 'probability', 'fn']
df_train['label'] = df_train.ID.apply(lambda x: x.split('_')[-1])
df_train.drop_duplicates('ID', inplace=True)
pivot = df_train.pivot(index='fn', columns='label', values='probability')
pivot.reset_index(inplace=True)
pivot.to_csv('data/train_pivot.csv', index=False)

from collections import defaultdict

d = defaultdict(list)
for fn in df_train.fn.unique(): d[fn]

for tup in df_train.itertuples():
    if tup.probability: d[tup.fn].append(tup.label)
        
ks, vs = [], []

for k, v in d.items():
    ks.append(k), vs.append(' '.join(v))
    
pd.DataFrame(data={'fn': ks, 'labels': vs}).to_csv('data/train_labels_as_strings.csv', index=False)


# In[ ]:


class Labeller():
    '''path to label, eg. path -> ['subdural', 'any']'''
    def __init__(self):
        self.df = pd.read_csv('data/train_labels_as_strings.csv')
        self.df.set_index('fn', inplace=True)
    def __call__(self, path):
        fn = path.name
        labels_txt = self.df.loc[fn].labels
        if isinstance(labels_txt, float) or labels_txt == ' ': return []
        return labels_txt.split(' ')


# In[ ]:


labeler = Labeller()


# In[ ]:


classes = L(pd.read_csv('data/train_pivot.csv').columns.tolist()[1:])
classes


# In[ ]:


mcat = MultiCategorize(vocab=classes)
mcat.o2i = classes.val2idx()


# In[ ]:


tfms = [PILImage.create, [Labeller(), mcat, OneHotEncode(mcat.vocab)]]

ds_img_tfms = [ToTensor()]
dsrc = DataSource(items, tfms, splits=RandomSplitter()(items))


# In[ ]:


dsrc[0]


# In[ ]:





# In[ ]:


test_paths = get_image_files(dir_test_img)
test_tfms = [PILImage.create, [lambda x: np.array([0,0,0,0,0,0])]]
dsrc_test = DataSource([test_paths[0]] + test_paths, test_tfms, splits=[[0], L(range(len(test_paths))).map(lambda x: x + 1)])


# In[ ]:


dsrc_test[0]


# In[ ]:


# %%time

# means, stds = [], []

# for batch in dbch.train_dl:
#     reshaped = batch[0].permute(1,0,2,3).reshape((3, -1))
#     means.append(reshaped.mean(1)), stds.append(reshaped.std(1))

# torch.stack(means).mean(0)

# torch.stack(stds).mean(0)


# I have not tried whether normalizing with imagenet stats (stats of the dataset the models were pretrained on) or normalizing with stats specific to this dataset work better.
# 
# My approach to this competition is to cut corners where I feel little value can be had (intuition is often misleading though!). I do want to focus on the aspects of the competition that are interesting and fun to me, and where I feel I can learn the most, even at the cost of a lower overall score.

# In[ ]:


means = [0.1627, 0.1348, 0.1373]
st_devs = [0.2961, 0.2605, 0.1889]

dataset_stats = (means, st_devs)
dataset_stats = broadcast_vec(1, 4, *dataset_stats)


# In[ ]:


ds_img_tfms = [ToTensor()]
dl_tfms = [Cuda(), ByteToFloatTensor(), Normalize(*dataset_stats)]

dbch = dsrc.databunch(after_item=ds_img_tfms, after_batch=dl_tfms, bs=128, num_workers=4)
dbch_test = dsrc_test.databunch(after_item=ds_img_tfms, after_batch=dl_tfms, bs=128, num_workers=4)


# In[ ]:


dbch.show_batch(max_n=9)


# In[ ]:


model = create_cnn_model(resnet18, 6, -2)


# I define how I want to split the model for applying differential learing rates

# In[ ]:


model_segments = model[0][:6], model[0][6:], model[1]


# In[ ]:


def trainable_params_mod(model): return L(trainable_params(segment) for segment in model_segments)


# In[ ]:


opt_func = partial(Adam, wd=0.01, eps=1e-3)


# In[ ]:


learn = Learner(
    dbch,
    model,
    loss_func=BCEWithLogitsLossFlat(),
    metrics=[accuracy_multi],
    opt_func=opt_func,
    splitter=trainable_params_mod
)


# In[ ]:


learn.freeze_to(-1)


# In[ ]:


learn.lr_find(start_lr=1e-6, end_lr=1)


# In[ ]:


learn.fit_one_cycle(1, 2e-2)


# In[ ]:


learn.save('phase-1')


# In[ ]:





# In[ ]:


learn.load('phase-1');


# In[ ]:


learn.freeze_to(-2)


# In[ ]:


learn.lr_find(start_lr=1e-8, end_lr=1e-1)


# In[ ]:


learn.fit_one_cycle(1, [1e-3, 1e-4, 1e-5])


# In[ ]:


learn.recorder.plot_loss()


# In[ ]:


learn.save('phase-2')


# In[ ]:





# In[ ]:


learn.load('phase-2');


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(1, np.array([1e-3, 1e-4, 1e-5]))


# In[ ]:


learn.save('phase-3')


# In[ ]:


learn.load('phase-3')


# In[ ]:


learn.metrics = [PrecisionMulti(), RecallMulti()]


# In[ ]:


learn.validate()


# One on hand, it is amazing that the switching of the databunches is possible. This is what flexibility and modularity looks like :) On the other hand, this is a hack.
# 
# This is only necessary right now because either this functionality in fastai v2 has not been added yet or I do not know how to go about this in a more proper way. I am sure the need for doing this will go away soon!

# In[ ]:


learn.dbunch = dbch_test


# In[ ]:


preds, targs = learn.get_preds()


# In[ ]:


ids = []
labels = []

for path, pred in zip(test_paths, preds):
    for i, label in enumerate(classes):
        ids.append(f"{path.name.split('.')[0]}_{label}")
        predicted_probability = '{0:1.10f}'.format(pred[i].item())
        labels.append(predicted_probability)


# In[ ]:


pd.DataFrame({'ID': ids, 'Label': labels}).to_csv(f'submission.csv', index=False)

