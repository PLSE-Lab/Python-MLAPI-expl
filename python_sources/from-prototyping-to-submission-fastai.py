#!/usr/bin/env python
# coding: utf-8

# In the notebook "[Cleaning the data for rapid prototyping](https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai)" I showed how to create a small, fast, ready-to-use dataset for prototyping our models. The dataset created in that notebook, along with the metadata files it uses, are now [available here](https://www.kaggle.com/jhoward/rsna-hemorrhage-jpg).
# 
# So let's use them to create a model! In this notebook we'll see the whole journey from pre-training using progressive resizing on our prototyping sample, through to fine-tuning on the full dataset, and then submitting to the competition.
# 
# I'm intentionally not doing any tricky modeling in this notebook, because I want to show the power of simple techniques and simples architectures. You should take this as a starting point and experiment! e.g. try data augmentation methods, architectures, preprocessing approaches, using the DICOM metadata, and so forth...
# 
# We'll be using the fastai.medical.imaging library here - for more information about this see the notebook [Some DICOM gotchas to be aware of](https://www.kaggle.com/jhoward/some-dicom-gotchas-to-be-aware-of-fastai). We'll also use the same basic setup that's in the notebook.
# 
# *Update: I'm out of GPU hours and Kaggle isn't freezing when running the current version of the notebook. To see a complete run, see [this version](https://www.kaggle.com/jhoward/from-prototyping-to-submission-fastai?scriptVersionId=22577538). I've commented out the GPU calls in this run so I can run it end to end.*

# In[ ]:


get_ipython().system('pip install torch torchvision feather-format kornia pyarrow --upgrade   > /dev/null')
get_ipython().system('pip install git+https://github.com/fastai/fastai_dev                    > /dev/null')


# In[ ]:


from fastai2.basics           import *
from fastai2.vision.all       import *
from fastai2.medical.imaging  import *
from fastai2.callback.tracker import *
from fastai2.callback.all     import *

np.set_printoptions(linewidth=120)
matplotlib.rcParams['image.cmap'] = 'bone'


# First we read in the metadata files (linked in the introduction).

# In[ ]:


path = Path('../input/rsna-intracranial-hemorrhage-detection/')
path_trn = path/'stage_1_train_images'
path_tst = path/'stage_1_test_images'

path_inp = Path('../input')
path_xtra = path_inp/'rsna-hemorrhage-jpg'
path_meta = path_xtra/'meta'/'meta'
path_jpg = path_xtra/'train_jpg'/'train_jpg'


# In[ ]:


df_comb = pd.read_feather(path_meta/'comb.fth').set_index('SOPInstanceUID')
df_tst  = pd.read_feather(path_meta/'df_tst.fth').set_index('SOPInstanceUID')
df_samp = pd.read_feather(path_meta/'wgt_sample.fth').set_index('SOPInstanceUID')
bins = (path_meta/'bins.pkl').load()


# ## Train vs valid

# To get better validation measures, we should split on patients, not just on studies, since that's how the test set is created.
# 
# Here's a list of random patients:

# In[ ]:


set_seed(42)
patients = df_comb.PatientID.unique()
pat_mask = np.random.random(len(patients))<0.8
pat_trn = patients[pat_mask]


# We can use that to take just the patients in a dataframe that match that mask:

# In[ ]:


def split_data(df):
    idx = L.range(df)
    mask = df.PatientID.isin(pat_trn)
    return idx[mask],idx[~mask]

splits = split_data(df_samp)


# Let's double-check that for a patient in the training set that their images are all in the first split.

# In[ ]:


df_trn = df_samp.iloc[splits[0]]
p1 = L.range(df_samp)[df_samp.PatientID==df_trn.PatientID[0]]
assert len(p1) == len(set(p1) & set(splits[0]))


# ## Prepare sample DataBunch

# We will grab our sample filenames for the initial pretraining.

# In[ ]:


def filename(o): return os.path.splitext(os.path.basename(o))[0]

fns = L(list(df_samp.fname)).map(filename)
fn = fns[0]
fn


# We need to create a `DataBunch` that contains our sample data, so we need a function to convert a filename (pointing at a DICOM file) into a path to our sample JPEG files:

# In[ ]:


def fn2image(fn): return PILCTScan.create((path_jpg/fn).with_suffix('.jpg'))
fn2image(fn).show();


# We also need to be able to grab the labels from this, which we can do by simply indexing into our sample `DataFrame`.

# In[ ]:


htypes = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']
def fn2label(fn): return df_comb.loc[fn][htypes].values.astype(np.float32)
fn2label(fn)


# If you have a larger GPU or more workers, change batchsize and number-of-workers here:

# In[ ]:


bs,nw = 128,4


# We're going to use fastai's new [Transform Pipeline API](http://dev.fast.ai/pets.tutorial.html) to create the DataBunch, since this is extremely flexible, which is great for intermediate and advanced Kagglers. (Beginners will probably want to stick with the Data Blocks API). We create two transform pipelines, one to open the image file, and one to look up the label and create a tensor of categories.

# In[ ]:


tfms = [[fn2image], [fn2label,EncodedMultiCategorize(htypes)]]
dsrc = DataSource(fns, tfms, splits=splits)
nrm = Normalize(tensor([0.6]),tensor([0.25]))
aug = aug_transforms(p_lighting=0.)
batch_tfms = [IntToFloatTensor(), nrm, Cuda(), *aug]


# To support progressive resizing (one of the most useful tricks in the deep learning practitioner's toolbox!) we create a function that returns a dataset resized to a requested size:

# In[ ]:


def get_data(bs, sz):
    return dsrc.databunch(bs=bs, num_workers=nw, after_item=[ToTensor],
                          after_batch=batch_tfms+[AffineCoordTfm(size=sz)])


# Let's try it out!

# In[ ]:


dbch = get_data(128, 96)
xb,yb = to_cpu(dbch.one_batch())
dbch.show_batch(max_n=4, figsize=(9,6))
xb.mean(),xb.std(),xb.shape,len(dbch.train_dl)


# Let's track the accuracy of the *any* label as our main metric, since it's easy to interpret.

# In[ ]:


def accuracy_any(inp, targ, thresh=0.5, sigmoid=True):
    inp,targ = flatten_check(inp[:,0],targ[:,0])
    if sigmoid: inp = inp.sigmoid()
    return ((inp>thresh)==targ.bool()).float().mean()


# The loss function in this competition is weighted, so let's train using that loss function too.

# In[ ]:


def get_loss(scale=1.0):
    loss_weights = tensor(2.0, 1, 1, 1, 1, 1).cuda()*scale
    return BaseLoss(nn.BCEWithLogitsLoss, pos_weight=loss_weights, floatify=True, flatten=False, 
        is_2d=False, activation=torch.sigmoid)


# We'll scale the loss initially to account for our sampling (since the original data had 14% rows with a positive label, and we resampled it to 50/50).

# In[ ]:


loss_func = get_loss(0.14*2)
opt_func = partial(Adam, wd=0.01, eps=1e-3)
metrics=[accuracy_multi,accuracy_any]


# Now we're ready to create our learner. We can use mixed precision (fp16) by simply adding a call to `to_fp16()`!

# In[ ]:


def get_learner():
    dbch = get_data(128,128)
    learn = cnn_learner(dbch, xresnet50, loss_func=loss_func, opt_func=opt_func, metrics=metrics)
    return learn.to_fp16()


# In[ ]:


learn = get_learner()


# Leslie Smith's famous LR finder will give us a reasonable learning rate suggestion.

# In[ ]:


# lrf = learn.lr_find()


# ## Pretrain on sample

# Here's our main routine for changing the size of the images in our DataBunch, doing one fine-tuning of the final layers, and then training the whole model for a few epochs.

# In[ ]:


def do_fit(bs,sz,epochs,lr, freeze=True):
    learn.dbunch = get_data(bs, sz)
    if freeze:
        if learn.opt is not None: learn.opt.clear_state()
        learn.freeze()
        learn.fit_one_cycle(1, slice(lr))
    learn.unfreeze()
    learn.fit_one_cycle(epochs, slice(lr))


# Now we can pre-train at different sizes.

# In[ ]:


# do_fit(128, 96, 4, 1e-2)


# In[ ]:


# do_fit(128, 160, 3, 1e-3)


# ## Scale up to full dataset

# Now let's fine tune this model on the full dataset. We'll need all the filenames now, not just the sample.

# In[ ]:


fns = L(list(df_comb.fname)).map(filename)
splits = split_data(df_comb)


# These functions are copied nearly verbatim from our [earlier cleanup notebook](https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai), so have a look there for details.

# In[ ]:


def fix_pxrepr(dcm):
    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept<-100: return
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000


# In[ ]:


def dcm_tfm(fn): 
    fn = (path_trn/fn).with_suffix('.dcm')
    try:
        x = fn.dcmread()
        fix_pxrepr(x)
    except Exception as e:
        print(fn,e)
        raise SkipItemException
    if x.Rows != 512 or x.Columns != 512: x.zoom_to((512,512))
    px = x.scaled_px
    return TensorImage(px.to_3chan(dicom_windows.brain,dicom_windows.subdural, bins=bins))


# In[ ]:


dcm = dcm_tfm(fns[0])
show_images(dcm)
dcm.shape


# We have some slight changes to our data source

# In[ ]:


tfms = [[dcm_tfm], [fn2label,EncodedMultiCategorize(htypes)]]
dsrc = DataSource(fns, tfms, splits=splits)
batch_tfms = [nrm, Cuda(), *aug]


# In[ ]:


def get_data(bs, sz):
    return dsrc.databunch(bs=bs, num_workers=nw, after_batch=batch_tfms+[AffineCoordTfm(size=sz)])


# Now we can test it out:

# In[ ]:


dbch = get_data(64,256)
x,y = to_cpu(dbch.one_batch())
dbch.show_batch(max_n=4)
x.shape


# We need to remove the sample scaling from our loss function, since we're using the full dataset.

# In[ ]:


learn.loss_func = get_loss(1.0)


# We can now fine-tune the final layers.

# In[ ]:


def fit_tune(bs, sz, epochs, lr):
    dbch = get_data(bs, sz)
    learn.dbunch = dbch
    learn.opt.clear_state()
    learn.unfreeze()
    learn.fit_one_cycle(epochs, slice(lr))


# In[ ]:


# fit_tune(64, 256, 2, 1e-3)


# ## Prepare for submission

# Now we're ready to submit. We can use the handy `test_dl` function to get an inference `DataLoader` ready, then we can check it looks OK.

# In[ ]:


test_fns = [(path_tst/f'{filename(o)}.dcm').absolute() for o in df_tst.fname.values]


# In[ ]:


tst = test_dl(dbch, test_fns)
x = tst.one_batch()[0]
x.min(),x.max()


# We pass that to `get_preds` to get our predictions, and then clamp them just in case we have some extreme values.

# In[ ]:


preds,targs = learn.get_preds(dl=tst)
preds_clipped = preds.clamp(.0001, .999)


# I'm too lazy to write a function that creates a submission file, so this code is stolen from Radek, with minor changes.

# In[ ]:


ids = []
labels = []

for idx,pred in zip(df_tst.index, preds_clipped):
    for i,label in enumerate(htypes):
        ids.append(f"{idx}_{label}")
        predicted_probability = '{0:1.10f}'.format(pred[i].item())
        labels.append(predicted_probability)


# In[ ]:


# df_csv = pd.DataFrame({'ID': ids, 'Label': labels})
# df_csv.to_csv(f'submission.csv', index=False)
# df_csv.head()


# Run the code below if you want a link to download the submission file.

# In[ ]:


from IPython.display import FileLink, FileLinks
# FileLink('submission.csv')

