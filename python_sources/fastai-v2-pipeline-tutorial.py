#!/usr/bin/env python
# coding: utf-8

# # Tutorial: fastai v2 low level data APIs
# 
# > Using `DataSource`, `Pipeline`, `TfmdList` and `Transform`
# 
# Note that this tutorial is for advanced features of fastai v2 for working with data formats that are not supported directly by fastai applications. However most common applications are supported by the Data Blocks API, which is more suitable for beginners and for rapid development.
# 
# **NB: fastai v2 is not officially released at this point. This is using the pre-release**. We have to install directly from GitHub since it isn't yet released on PyPi or any conda channel.

# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastai_dev ')


# Once released, the module will be `fastai`, but for now it's called `fastai2`.

# In[ ]:


from fastai2.torch_basics import *
from fastai2.test import *
from fastai2.data.all import *
from fastai2.vision.core import *
from fastai2.notebook.showdoc import *


# ## Loading the Pets dataset using only `Transform`
# 
# Let's see how to use `fastai.data` to process the Pets dataset. We use *source* to refer to the underlying source of our data (e.g. a directory on disk, a database connection, a network connection, etc).

# In[ ]:


source = Path('../input/the-oxfordiiit-pet-dataset/images')


# First we grab the items and split indices:

# In[ ]:


items = get_image_files(source)
split_idx = RandomSplitter()(items)


# We'll use this function to create consistently sized tensors from image files:

# In[ ]:


def resized_image(fn:Path, sz=128):
    x = PILImage.create(fn).convert('RGB').resize((sz,sz))
    return tensor(array(x)).permute(2,0,1).float()/255.


# Before we can create a `Transform`, we need a type that knows how to show itself. Here we define a `TitledImage`:

# In[ ]:


class TitledImage(Tuple):
    def show(self, ctx=None, **kwargs): show_titled_image(self, ctx=ctx, **kwargs)


# In[ ]:


img = resized_image(items[0])


# In[ ]:


TitledImage(img,'test').show()


# We can now create a `Transform` that converts from an index to our `x` and `y` for modeling.

# In[ ]:


class PetTfm(Transform):
    def __init__(self, items, train_idx):
        self.items,self.train_idx = items,train_idx
        self.labeller = RegexLabeller(pat = r'/([^/]+)_\d+.jpg$')
        vals = map(self.labeller, items[train_idx])
        self.vocab,self.o2i = uniqueify(vals, sort=True, bidir=True)

    def encodes(self, i):
        o = self.items[i]
        return resized_image(o), self.o2i[self.labeller(o)]
    
    def decodes(self, x): return TitledImage(x[0],self.vocab[x[1]])


# Note that `decodes` is intended to operate on the return value of `encodes`. In the case above, `decodes` takes in a tuple consisting of a Tensor representing the image and an integer being the class idx and returns an instance of `TitledImage`.
# 
# It's important to give the type that can show itself to fully decoded elements because when in a `Pipeline`, we stop decoding as soon as we can find a `show` method.

# In[ ]:


pets = PetTfm(items, split_idx[0])


# In[ ]:


x,y = pets(0)
x.shape,y


# In[ ]:


dec = pets.decode((x,y))
dec.show()


# ## Using `Pipeline` to create Siamese model dataset
# 
# A *Siamese* model needs a dataset with two images as input, and a boolean output that is `True` if the two images are the "same" (e.g. are pictures of the same breed of pet). Custom structures like this are often easiest to create using your own `Pipeline`, which makes no assumptions about the structure of your input or output data.
# 
# We'll be creating a dataset that returns two images and a boolean. So let's first define a showable type for a tuple with those things:

# In[ ]:


class SiameseImage(Tuple):
    def show(self, ctx=None, **kwargs): 
        img1,img2,same_breed = self
        return show_image(torch.cat([img1,img2], dim=2), title=same_breed, ctx=ctx)


# In[ ]:


SiameseImage(img,img,True).show();


# We create a `SiamesePair` transform that creates the tuple we'll need for a `SiameseImage`.

# In[ ]:


class SiamesePair(Transform):
    def __init__(self,items,labels):
        self.items,self.labels,self.assoc = items,labels,self
        sortlbl = sorted(enumerate(labels), key=itemgetter(1))
        # dict of (each unique label) -- (list of indices with that label)
        self.clsmap = {k:L(v).itemgot(0) for k,v in itertools.groupby(sortlbl, key=itemgetter(1))}
        self.idxs = range_of(self.items)
        
    def encodes(self,i):
        "x: tuple of `i`th image and a random image from same or different class; y: True if same class"
        othercls = self.clsmap[self.labels[i]] if random.random()>0.5 else self.idxs
        otherit = random.choice(othercls)
        return SiameseImage(self.items[i], self.items[otherit], self.labels[otherit]==self.labels[i])


# Then we want to open and resize the image filenames but not the boolean. This kind of filtering is done with type annotations. We indicated with the return type annotation of `SiamesePair.encodes` the types of our objects and we can then create a new transform that opens and resizes `Path` objects but leaves other types untouched. 

# Then the `Pipeline` will compose our two transforms, and create the `SiameseImage`.

# In[ ]:


OpenAndResize = TupleTransform(resized_image)
labeller = RegexLabeller(pat = r'/([^/]+)_\d+.jpg$')
sp = SiamesePair(items, items.map(labeller))
pipe = Pipeline([sp, OpenAndResize], as_item=True)
x,y,z = t = pipe(0)
x.shape,y.shape,z


# In[ ]:


for _ in range(3): pipe.show(pipe(0))


# ## Using `DataSource`
# 
# `DataSource` applies a list of list of transforms (or list of `Pipeline`s) lazily to items of a collection, creating one output per list of transforms/`Pipeline`. This makes it easier for us to separate out steps of a process, so that we can re-use them and modify the process more easily. For instance, we could add data augmentation, data normalization, etc. Here we separate out the steps of the basic pets process.

# In[ ]:


class ImageResizer(Transform):
    order=10
    "Resize image to `size` using `resample"
    def __init__(self, size, resample=Image.BILINEAR):
        if not is_listy(size): size=(size,size)
        self.size,self.resample = (size[1],size[0]),resample

    def encodes(self, o:PILImage): return o.resize(size=self.size, resample=self.resample)
    def encodes(self, o:PILMask):  return o.resize(size=self.size, resample=Image.NEAREST)


# In[ ]:


tfms = [[PILImage.create, ImageResizer(128), ToTensor(), ByteToFloatTensor()],
        [labeller, Categorize()]]
dsrc = DataSource(items, tfms)


# In[ ]:


t = dsrc[0]
type(t[0]),type(t[1])


# In[ ]:


x,y = dsrc.decode(t)
x.shape,y


# In[ ]:


dsrc.show(t);


# The types being properly propagated and dispatched, we can do the same thing with `ImageResizer`, `ImageToByteTensor`, `ByteToFloatTensor` being passed as tranforms over the tuple. This is done in `TfmdDL` by passing them to `after_item`. They won't do anything to the category but will only be applied to the inputs.

# In[ ]:


tfms = [[PILImage.create], [labeller, Categorize()]]
dsrc = DataSource(items, tfms)
tdl = TfmdDL(dsrc, bs=1, after_item=[ImageResizer(128), ToTensor(), ByteToFloatTensor()])


# In[ ]:


t = tdl.one_batch()
x,y = tdl.decode_batch(t)[0]
x.shape,y


# In[ ]:


dsrc.show((x,y));


# ## Adding splits

# When you create a `Datasource`, you can pass along `splits` indices that represent the split between train and validation set (there can be multiple validation sets) on top of the items and tfms.

# In[ ]:


pets = DataSource(items, tfms, splits=split_idx)


# To access an element we need to specify the subset (either with `train`/`valid` or with `subset(i)`):

# In[ ]:


x,y = pets.subset(1)[0]
x.shape,y


# ...or equivalently:

# In[ ]:


x2,y2 = pets.valid[0]
test_eq(x.shape,x2.shape)
test_eq(y,y2)


# We can decode an element for display purposes, either passing a tuple to `decode` or by passing an index to `decode_at`.

# In[ ]:


xy = pets.valid.decode((x,y))
xy[1]


# In[ ]:


xy2 = decode_at(pets.valid, 0)
test_eq(type(xy2[1]), Category)
test_eq(xy2, xy)


# We're now ready to show our items.

# In[ ]:


pets.show((x,y));


# We can use this to feed a `DataLoader` and view a batch. It's faster to convert to float on GPU, so we'll do it as a DataLoader transform, after CUDA in `after_batch`.

# In[ ]:


ds_img_tfms = [ImageResizer(128), ToTensor()]
dl_tfms = [Cuda(), ByteToFloatTensor()]

trn_dl = TfmdDL(pets.train, bs=9, after_item=ds_img_tfms, after_batch=dl_tfms)
b = trn_dl.one_batch()

test_eq(len(b[0]), 9)
test_eq(b[0][0].shape, (3,128,128))
test_eq(b[0].type(), 'torch.cuda.FloatTensor' if default_device().type=='cuda' else 'torch.FloatTensor')


# In[ ]:


bd = trn_dl.decode_batch(b)

test_eq(len(bd), 9)
test_eq(bd[0][0].shape, (3,128,128))


# In[ ]:


_,axs = plt.subplots(3,3, figsize=(9,9))
trn_dl.show_batch(ctxs=axs.flatten())


# In[ ]:


dbch = pets.databunch(bs=9, after_item=ds_img_tfms, after_batch=dl_tfms)
dbch.train_dl.show_batch()


# ## Segmentation

# By using the same transforms in `after_item` and `after_batch` but a different kind of targets (here segmentation masks), the targets are automatically processed as they should with the type-dispatch system.

# In[ ]:


cv_source = Path('../input/camvid/camvid/CamVid')
cv_items = get_image_files(cv_source/'train')
cv_splitter = RandomSplitter(seed=42)
cv_split = cv_splitter(cv_items)
cv_label = lambda o: cv_source/'train_labels'/f'{o.stem}_L{o.suffix}'


# In[ ]:


tfms = [[PILImage.create], [cv_label, PILMask.create]]
camvid = DataSource(cv_items, tfms, splits=cv_split)
trn_dl = TfmdDL(camvid.train,  bs=4, after_item=ds_img_tfms, after_batch=dl_tfms)


# In[ ]:


_,axs = plt.subplots(2,2, figsize=(6,6))
trn_dl.show_batch(ctxs=axs.flatten())


# In[ ]:




