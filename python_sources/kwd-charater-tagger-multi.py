#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %reload_ext autoreload
# %autoreload 2
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from fastai import *

from fastai.vision import *
from tqdm import tqdm_notebook as tqdm


# # Get Data

# In[ ]:


get_ipython().system('wget "https://drive.google.com/uc?export=download&id=1BTQQQcb0a_MWmTRdKt0EGIKk2UfpnZJK" -O KWD_characters.txt')


# In[ ]:


with open('KWD_characters.txt', 'r') as f:
    s = f.read()
#     print(s)
    data_urls = eval(s)


# In[ ]:


urls = [ it[0] for it in data_urls]
with open('strips.txt', 'w') as f:
    f.write("\n".join(urls))
    
fnames = [url[url.rfind("/")+1:] for url in urls]
#add unique prefix
fnames = [f'{i:08d}_{f}' for i,f in zip(range(len(fnames)), fnames)]

with open('strips_names.txt', 'w') as f:
    f.write("\n".join(fnames))
    
labels = [[c.replace(" ", "_") for c in it[1]] for it in data_urls]


# In[ ]:


get_ipython().system('cat strips.txt | head -n5')
get_ipython().system('cat strips_names.txt | head -n5')


# In[ ]:


# TODO: Clear it.
# FROM:
# https://forums.fast.ai/t/using-download-images-retaining-source-file-name/39463/2
def download_image(url,dest, timeout=4):
    try: r = download_url(url, dest, overwrite=True, show_progress=False, timeout=timeout)
    except Exception as e: print(f"Error {url} {e}")

def _download_image_inner(dest, info, i, timeout=4):
    url = info[0]
    name = info[1]
#     suffix = re.findall(r'\.\w+?(?=(?:\?|$))', url)
#     suffix = suffix[0] if len(suffix)>0  else '.jpg'
    download_image(url, dest/name, timeout=timeout)

def download_images_with_names(urls:Collection[str], dest:PathOrStr, names:PathOrStr=None,  max_pics:int=1000, max_workers:int=8, timeout=4):
    "Download images listed in text file `urls` to path `dest`, at most `max_pics`"
    urls = open(urls).read().strip().split("\n")[:max_pics]
    if names:
      names = open(names).read().strip().split("\n")[:max_pics]
    else:
      names = [f"{index:08d}" for index in range(0,len(urls))]
    info_list = list(zip(urls, names))
    dest = Path(dest)
    dest.mkdir(exist_ok=True)
    parallel(partial(_download_image_inner, dest, timeout=timeout), info_list, max_workers=max_workers)


# In[ ]:


path = Path('/tmp/strips/')
get_ipython().system('rm -rf {path}')
get_ipython().system('mkdir {path}')
download_images_with_names('strips.txt',path ,'strips_names.txt')
os.listdir(path)[:5]


# In[ ]:


import pprint
from collections import Counter
cnt = Counter([c for group in labels for c in group])
# pprint.pprint(cnt)
good_chars = [k for k,v in cnt.items() if v > 5]
good_chars.remove('Patron')
# chars = labels[0]
filtered_labels = [[c for c in chars if c in good_chars] for chars in labels]
cnt.most_common()


# In[ ]:


# labeled_data = [(f,l)  for (f,l) in zip(fnames, labels) if l]
use_data = [bool(l) for l in labels]
labeled_data = [(f," ".join(l)) for (f,l,u) in zip(fnames, filtered_labels,use_data) if u] 
print('Total images:', len(fnames))
print('Labeled images:', len(labeled_data))


# # Multi class

# In[ ]:


df = pd.DataFrame(labeled_data)
df[:10]


# In[ ]:


#strips are 1000x350
print(f'{fnames[0]} : {open_image(path/fnames[0]).size}')
np.random.seed = 42
data = ImageDataBunch.from_df(path,
                              df, 
                              label_delim=' ', 
                              ds_tfms=get_transforms(), 
                              size=(175,500), 
                              valid_pct=0.2).normalize(imagenet_stats)
# data = ImageDataBunch.from_df('strips', df, label_delim=' ',size=(256,512), valid_pct=0.2)


# In[ ]:


data.show_batch(rows=3, figsize=(15,15))


# In[ ]:


acc_05 = partial(accuracy_thresh, thresh=0.5)
f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, models.resnet50, metrics=[acc_05, f_score])


# In[ ]:


learn.lr_find(); learn.recorder.plot()


# In[ ]:


lr = 2e-2


# In[ ]:


get_ipython().run_cell_magic('time', '', "learn.fit_one_cycle(5, slice(lr))\nlearn.save('stage-1')")


# In[ ]:


learn.load('stage-1')
learn.fit_one_cycle(10, slice(lr))
learn.save('stage-1a')


# In[ ]:


# learn.load('stage-1')
# learn.fit_one_cycle(5, slice(lr))
# learn.save('stage-1b')


# In[ ]:


learn.unfreeze()
learn.lr_find(); learn.recorder.plot()


# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', "# learn.load('stage-1')\nlearn.unfreeze()\nlearn.fit_one_cycle(10, max_lr=slice(1e-4,lr/5))")


# In[ ]:


learn.save('stage-2')


# In[ ]:


learn.lr_find(); learn.recorder.plot()


# In[ ]:


learn.load('stage-2')
learn.fit_one_cycle(10, max_lr=slice(1e-4,lr/5))
learn.save('stage-3')


# In[ ]:


learn.lr_find(); learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5,lr/10))


# In[ ]:


# learn.fit_one_cycle(10, slice(1e-5,1e-4))
learn.save('stage-4')


# In[ ]:


learn.lr_find(); learn.recorder.plot()


# In[ ]:


np.random.seed = 42
data2 = ImageDataBunch.from_df(path, 
                               df, 
                               label_delim=' ', 
                               ds_tfms=get_transforms(), 
                               size=(350,1000), 
                               valid_pct=0.2,
                               bs=16).normalize(imagenet_stats)

learn.data = data2


# In[ ]:


learn.freeze()


# In[ ]:


learn.lr_find(); learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-3))


# In[ ]:


learn.unfreeze()
learn.lr_find(); learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-5,(1e-3)/5))


# In[ ]:


learn.fit_one_cycle(10, slice(1e-5,(1e-3)/5))


# In[185]:


model_name = 'KWD-rn50-500-1000.pkl' 
learn.export(model_name)
get_ipython().system('cp /tmp/strips/{model_name} .')
get_ipython().system('ls')


# In[183]:





# In[ ]:


def sort_predictions(preds, classes, thresh=1e-2):
    predictions = list(zip(preds, classes))
    predictions.sort(reverse=True)
    predictions = "\n".join([f'{p:.3f} : {c}' for p,c in predictions if p>thresh])
    return predictions


# In[ ]:


img = open_image(path/fnames[0])
img.show(figsize=(10,20))
cat, indice, preds = learn.predict(img)
print(sort_predictions(preds, learn.data.classes))


# In[ ]:


download_image('http://www.konradokonski.com/KWD/wp-content/uploads/2019/06/KWD_573.png','/tmp/KWD_573.png')
img = open_image('/tmp/KWD_573.png')
img.show(figsize=(10,20))
cat, indice, preds = learn.predict(img)
print(sort_predictions(preds, learn.data.classes))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_multi_top_losses(1)


# In[ ]:


# import random
# random.seed=666
unlabeled_files = fnames[-100:]
for file in random.sample(unlabeled_files, 20):
    img = open_image(path/file)
    cat, indice, preds = learn.predict(img)
    img.show(title=f'{sort_predictions(preds, learn.data.classes)}\n\nPredicted: {cat}', 
             figsize=(15,30))


# In[ ]:




