#!/usr/bin/env python
# coding: utf-8

# # `fastai2` Starter Kernel
# 
# This kernel will walk you through how to set up the `DataBlock` for this competition!

# First let's grab `fastai2` (make sure your internet is turned on!)

# In[ ]:


get_ipython().system('pip install fastai2 --quiet')


# And now let's grab the `vision` library and set up our `Path`

# In[ ]:


from fastai2.vision.all import *

path = Path('../input/global-wheat-detection')


# In[ ]:


path.ls()


# Our labels are inside of the `train.csv`, let's take alook

# In[ ]:


df = pd.read_csv(path/'train.csv')


# In[ ]:


df.head()


# In[ ]:


df['bbox'].isna().sum()


# So we can see that we get an `image_id` and one label per row. We'll need to remember that in a little!

# In[ ]:


imgs = get_image_files(path/'train')


# In[ ]:


len(imgs) == df['image_id'].nunique()


# In[ ]:


len(imgs) - df['image_id'].nunique()


# So we have 49 images that *aren't* labelled. Let's build on this now by simply making a custom set of `Paths` that contain our working images, and build a `get_items` for it

# In[ ]:


im_df = df['image_id'].unique()


# In[ ]:


im_df = [fn + '.jpg' for fn in im_df]


# In[ ]:


im_df[:5]


# In[ ]:


fns = [Path(str(path/'train') + f'/{fn}') for fn in im_df]


# In[ ]:


fns[0]


# In[ ]:


fns[0].name[:-4]


# We'll make a `get_items` that simply returns our *good* images

# In[ ]:


def get_items(noop): return fns


# ## DataFrame Format:
# 
# `image_id` is the same, `bbox` contains bounding boxes, all labels are `wheat`
# 
# Keeping everything in the `DataFrame` is super inefficient, so we'll move everything to a NumPy array to load it faster.

# In[ ]:


df['label'] = 'wheat'


# In[ ]:


df_np = df.to_numpy()


# `get_y` needs to return the coordinates then the label. Let's look at an example quickly

# In[ ]:


coco_source = untar_data(URLs.COCO_TINY)
images, lbl_bbox = get_annotations(coco_source/'train.json')
img2bbox = dict(zip(images, lbl_bbox))


# In[ ]:


fn = images[0]; fn


# In[ ]:


img2bbox[fn][0][0]


# Now that we know the format, let's work with our `DataFrame` to return something `fastai2` wants. First let's convert our bounding boxes into something we can use. NumPy has a nice `np.fromstring`, but it wants just numbers whereas our `DataFrame` will give us: `"[0,0,0,0]"`, which we don't want! So we'll replace both of the brackets first. This is just temporary for now though, as we want to make everything run on NumPy for efficiency

# In[ ]:


def get_tmp_bbox(fn):
    "Grab bounding boxes from `DataFrame`"
    rows = np.where((df_np[:, 0] == fn.name[:-4]))
    bboxs = df_np[rows][:,3]
    bboxs = [b.replace('[', '').replace(']', '') for b in bboxs]
    return np.array([np.fromstring(b, sep=',') for b in bboxs])


# We don't require as much for the labels, as all of them are simply a string of "wheat"

# In[ ]:


def get_tmp_lbl(fn):
    "Grab label from `DataFrame`"
    rows = np.where((df_np[:, 0] == fn.name[:-4]))
    return df_np[rows][:,5]


# In[ ]:


fnames = df['image_id'].unique(); fnames[:3]


# Now let's start building our ground truth data. We'll want an initial array to add to

# In[ ]:


bboxs = get_tmp_bbox(fns[0])
lbls = get_tmp_lbl(fns[0])
arr = np.array([fns[0].name[:-4], bboxs, lbls])


# In[ ]:


arr


# And now we can add the rest of the data

# In[ ]:


for fname in fns[1:]:
    bbox = get_tmp_bbox(fname)
    lbl = get_tmp_lbl(fname)
    arr2 = np.array([fname.name[:-4], bbox, lbl])
    arr = np.vstack((arr, arr2))


# Now we have our actual data array, we need to make some adjustments. Currently our coordinates are x,y,w,h and we want x1,y1,x2,y2. So let's look at converting those!

# In[ ]:


arr[:,1][0][0][0] + arr[:,1][0][0][2]


# To convert it, we need to add our width and height to the respective x and y. We can do this like so:

# In[ ]:


arr[:,1][0][1]


# In[ ]:


for i, im in enumerate(arr[:,1]):
    for j, box in enumerate(im):
        arr[:,1][i][j][2] = box[0]+box[2]
        arr[:,1][i][j][3] = box[1]+box[3]


# In[ ]:


arr[0][1][0]


# That looks much better!

# In[ ]:


np.save('data.npy', arr)


# Let's make our true `get_bbox` and `get_lbl`. We'll want to first search our NumPy array for a matching filename, then grab the second or third index for the bounding box or the label respectively 

# In[ ]:


def get_bbox(fn):
    "Gets bounding box from `fn`"
    idx = np.where((arr[:,0] == fn.name[:-4]))
    return arr[idx][0][1]


# In[ ]:


def get_lbl(fn):
    "Get's label from `fn`"
    idx = np.where((arr[:,0] == fn.name[:-4]))
    return arr[idx][0][2]


# For a true test of speed, to get the first value with pandas it takes ~6.4-6.6 milliseconds for each function. Let's see how ours does:

# In[ ]:


get_ipython().run_cell_magic('timeit', '', '_ = get_bbox(fns[0])')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', '_ = get_lbl(fns[0])')


# *Much* more efficent to use `NumPy` here.
# 
# # DataLoaders

# For our `DataLoaders`, we're going to want to use a `ImageBlock` for our input, and the `BBoxBlock` and `BBoxLblBlock` for our outputs, our custom `get_items`, along with some `get_y`'s. I chose some very simple transforms for us to use here. Finally we need to specify the number of inputs to simply be 1, telling `fastai` we have two outputs

# In[ ]:


wheat = DataBlock(blocks=(ImageBlock, BBoxBlock, BBoxLblBlock),
                 get_items=get_items,
                 splitter=RandomSplitter(),
                 get_y=[get_bbox, get_lbl],
                 item_tfms=Resize(256, method=ResizeMethod.Pad),
                 n_inp=1)


# And now we can build our `DataLoaders` and you're done!

# In[ ]:


dls = wheat.dataloaders(path,bs=32)


# In[ ]:


dls.show_batch(max_n=1, figsize=(12,12))


# In[ ]:


batch = dls.one_batch()


# In[ ]:


batch[1].shape


# For a time comparison, the Pandas method took ~17 seconds to build the dataloaders and 1.3 seconds per batch. Using NumPy we reduce this to 90ms and 838ms per batch (with most of that time taken up by shuffling the data)
