#!/usr/bin/env python
# coding: utf-8

# # Working with 3D data - fastai2
# In this notebook I show an example of how to work with 3D data using fastai2. I use MNIST and create sequences from 0 to 9 in order to make the example more intuitive and easier to debug.
# 
# A detailed explanation of the code is available at: https://towardsdatascience.com/working-with-3d-data-fastai2-5e2baa09037e
# 

# In[ ]:


# Instaling fastai2
get_ipython().system('pip install git+https://github.com/fastai/fastai2 ')
get_ipython().system('pip install git+https://github.com/fastai/fastcore')


# In[ ]:


from fastai2.vision.all import *


# In[ ]:


get_ipython().system('wget {URLs.MNIST}')
get_ipython().system('tar -xf mnist_png.tgz')


# In[ ]:


path = Path('mnist_png')
path.ls()


# In[ ]:


files = [get_image_files(path/f'training/{i}')[:100] for i in range(10)]
files = np.concatenate(files)
sequence_order = [int(f.parent.stem) for f in files]
sequence_id = [f's{i:02d}' for i in range(100)]*10

df = pd.DataFrame({'file': files, 'sequence_id': sequence_id, 'sequence_order': sequence_order,
                   'label': [int(s[1:]) for s in sequence_id]})


# In[ ]:


df.head()


# In[ ]:


df.loc[df.sequence_id=='s00']


# In[ ]:


def int2float(o:TensorImage):
    return o.float().div_(255.)

class ImageSequence(Tuple):
    @classmethod
    def create(cls, fns): return cls(tuple(PILImage.create(f) for f in fns))

def ImageSequenceBlock(): 
    return TransformBlock(type_tfms=ImageSequence.create, batch_tfms=int2float)

class SequenceGetItems():
    def __init__(self, filename_col, sequence_id_col, label_col):
        self.fn = filename_col
        self.seq = sequence_id_col
        self.label = label_col
        
    def __call__(self, df):
        data = []
        for fn in progress_bar(df[self.seq].unique()):
            similar = df[self.seq] == fn
            similar = df.loc[similar]
            fns = similar[self.fn].tolist()
            lbl = similar[self.label].values[0]
            data.append([*fns, lbl])
        return data

def create_batch(data):
    xs, ys = [], []
    for d in data:
        xs.append(d[0])
        ys.append(d[1])
    xs = torch.cat([TensorImage(torch.cat([im[None] for im in x], dim=0))[None] for x in xs], dim=0)
    ys = torch.cat([y[None] for y in ys], dim=0)
    return TensorImage(xs), TensorCategory(ys)

def show_sequence_batch(max_n=4):
    xb, yb = dls.one_batch()
    fig, axes = plt.subplots(ncols=10, nrows=max_n, figsize=(12,6), dpi=120)
    for i in range(max_n):
        xs, ys = xb[i], yb[i]
        for j, x in enumerate(xs):
            axes[i,j].imshow(x.permute(1,2,0).cpu().numpy())
            axes[i,j].set_title(ys.item())
            axes[i,j].axis('off')


# In[ ]:


class SequenceTfms(Transform):
    def __init__(self, tfms): 
        self.tfms = tfms
        
    def encodes(self, x:TensorImage): 
        bs, seq_len, ch, rs, cs = x.shape
        x = x.view(bs, seq_len*ch, rs, cs)
        x = compose_tfms(x, self.tfms)
        x = x.view(bs, seq_len, ch, rs, cs) 
        return x
    
class BatchTfms(Transform):
    def __init__(self, tfms): 
        self.tfms = tfms
        
    def encodes(self, x:TensorImage): 
        bs, seq_len, ch, rs, cs = x.shape
        x = x.view(bs*seq_len, ch, rs, cs)
        x = compose_tfms(x, self.tfms)
        x = x.view(bs, seq_len, ch, rs, cs) 
        return x


# In[ ]:


dblock = DataBlock(
    blocks    = (ImageSequenceBlock, CategoryBlock),
    get_items = SequenceGetItems('file', 'sequence_id', 'label'), 
    get_x     = lambda t : t[:-1],
    get_y     = lambda t : t[-1],
    splitter  = RandomSplitter(valid_pct=0.2, seed=2020))

dls = dblock.dataloaders(df, bs=8, create_batch=create_batch)
show_sequence_batch()


# In[ ]:


affine_tfms, light_tfms = aug_transforms(flip_vert=True)
brightness = lambda x : x.brightness(p=0.75, max_lighting=0.9)
contrast   = lambda x : x.contrast(p=0.75, max_lighting=0.9)

dblock = DataBlock(
    blocks     = (ImageSequenceBlock, CategoryBlock),
    get_items  = SequenceGetItems('file', 'sequence_id', 'label'), 
    get_x      = lambda t : t[:-1],
    get_y      = lambda t : t[-1],
    splitter   = RandomSplitter(valid_pct=0.2, seed=2020),
    item_tfms  = Resize(128),
    batch_tfms = [SequenceTfms([affine_tfms]), BatchTfms([brightness, contrast])])

dls = dblock.dataloaders(df, bs=8, create_batch=create_batch)
show_sequence_batch()


# In[ ]:


xb, yb = dls.one_batch()
xb.shape, yb.shape

