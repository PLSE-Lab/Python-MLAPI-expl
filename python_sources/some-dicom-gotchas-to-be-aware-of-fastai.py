#!/usr/bin/env python
# coding: utf-8

# Let's explore the data, taking advantage of the features of the `fastai.medical.imaging` library, available in the prerelease of [fastai v2](https://dev.fast.ai). If you're interested in learning more about fastai v2, check out the [dedicated forum](https://forums.fast.ai/c/fastai-users/fastai-v2). For a deep dive, have a look at the 10 [fastai v2 code walkthru videos](https://forums.fast.ai/t/fastai-v2-daily-code-walk-thrus/53839). The overall approach in the library is described in detail (and many parts implememted from scratch) in the course [Deep Learning from the Foundations](https://course.fast.ai/part2.html).
# 
# Since this is a prerelease, it's not installed on Kaggle yet, so we'll install it and import the necessary modules.

# In[ ]:


get_ipython().system('pip install torch torchvision feather-format kornia pyarrow --upgrade   > /dev/null')
get_ipython().system('pip install git+https://github.com/fastai/fastai_dev             > /dev/null')

from fastai2.basics           import *
from fastai2.medical.imaging  import *

np.set_printoptions(linewidth=120)


# In[ ]:


path_inp = Path('../input')
path = path_inp/'rsna-intracranial-hemorrhage-detection'
path_trn = path/'stage_1_train_images'
path_tst = path/'stage_1_test_images'


# # Reading the data

# It's much faster and easier to analyze DICOM metadata when it's in a DataFrame. Converting a bunch of DICOM files into a metadata DataFrame is as simple as calling `pd.DataFrame.from_dicoms`. However it takes quite a while (particularly on Kaggle, due to the lack of RAM), so we'll import the DataFrame we've already created and saved in the [Creating a metadata DataFrame](https://www.kaggle.com/jhoward/creating-a-metadata-dataframe/) kernel. See that kernel for more details on how this is made.

# In[ ]:


path_df = path_inp/'creating-a-metadata-dataframe'

df_lbls = pd.read_feather(path_df/'labels.fth')
df_tst = pd.read_feather(path_df/'df_tst.fth')
df_trn = pd.read_feather(path_df/'df_trn.fth')


# Let's merge the labels and DICOM data together into a single DataFrame. It's always good practice after a merge to assert that you haven't had any failed matches in your join.

# In[ ]:


comb = df_trn.join(df_lbls.set_index('ID'), 'SOPInstanceUID')
assert not len(comb[comb['any'].isna()])


# This next output is worth studying - it's all the information we have (other than the actual pixels, of course!) about each of our images. As well as the DICOM metadata elements (which you can learn about with a Google search, since DICOM is a widely used standard), you'll see we also have `img_min`, `img_max`, `img_mean`, and `img_std` (which are the basic statistics of that images pixels), along with the labels for that image.

# In[ ]:


comb.head().T


# # Looking at metadata - BitsStored and PixelRepresentation

# Two interesting fields are `BitsStored` and `PixelRepresentation`. These tell you whether the data is 12 bit or 16 bit, and whether it's stored as signed on unsiged data. Let's look at some image, metadata, and label statistics grouping on these fields. We'll use Pandas' powerful `pivot_table` function (which isn't as powerful as MS Excel's eponymous tool, but is pretty great nonetheless!)

# In[ ]:


repr_flds = ['BitsStored','PixelRepresentation']
comb.pivot_table(values=['img_mean','img_max','img_min','PatientID','any'], index=repr_flds,
                   aggfunc={'img_mean':'mean','img_max':'max','img_min':'min','PatientID':'count','any':'mean'})


# The different frequencies of labels shown in the `any` column is interesting, suggesting that these groups may be from different institutions. Assuming that the stage 2 data will be from the same distribution (which the competition organizers have said is the case) then it should be safe to take advantage of this information. In general, using institution and machine metadata in this way can actually be helpful in practice in production models (as long as it's fine-tuned appropriately when used at new institutions).
# 
# As expected, the unsigned DICOMs have a minimum pixel value of zero. The range of the 16 bit data is of some concern, however, since in theory hounsfield units (which CT scans use) are not meant to be so extreme.
# 
# In theory, there are some DICOM elements that tell us how to scale and look at our data. Perhaps these will fix up the different representations for us. Let's look at a summary of them.

# In[ ]:


comb.pivot_table(values=['WindowCenter','WindowWidth', 'RescaleIntercept', 'RescaleSlope'], index=repr_flds,
                   aggfunc={'mean','max','min','std','median'})


# Well... that doesn't look good! Although when `PixelRepresentation` is zero, there's *normally* a `RescaleIntercept` of `-1024` to give us signed data (as we would expect, since hounsfield units can be negative), the max `RescaleIntercept` for that row is `1.0`. Curiously, `RescaleIntercept` is normally also `-1024` when `PixelRepresentation` is one, even although it's signed data, which means that shouldn't be necessary.
# 
# Also, `RescaleSlope` is always `1.0`, so it's not going to fix the very extreme pixel values in the 16 bit data. Let's see how often that happens. To make our analyses easier, we'll create DataFrames for each subset we're interested in.

# In[ ]:


df1 = comb.query('(BitsStored==12) & (PixelRepresentation==0)')
df2 = comb.query('(BitsStored==12) & (PixelRepresentation==1)')
df3 = comb.query('BitsStored==16')
dfs = [df1,df2,df3]


# Now we can see how often images occur with extreme values. We'll use a little function that summarizes the distribution of values.

# In[ ]:


def distrib_summ(t):
    plt.hist(t,40)
    return array([t.min(),*np.percentile(t,[0.1,1,5,50,95,99,99.9]),t.max()], dtype=np.int)


# In[ ]:


distrib_summ(df3.img_max.values)


# In[ ]:


distrib_summ(df3.img_min.values)


# We can see that extreme values occur in only a very small number of images. We may want to clip these images in the dataset. We'll need to be careful, because these extreme values appear in the test set too:

# In[ ]:


distrib_summ(df_tst.img_max.values)


# # Looking at image data

# We'll open a few images so we can look at their pixel data.

# In[ ]:


dcms = path_trn.ls(10).map(dcmread)
dcm = dcms[0]
dcm


# Let's see what format the pixel data is in.

# In[ ]:


dcms_px = dcms.attrgot('pixel_array')


# In[ ]:


list(zip(dcms_px.attrgot('dtype'),
         dcms.attrgot('PixelRepresentation'),
         dcms.attrgot('BitsStored')))


# We see here that we are automatically given unsigned or signed int16 arrays, based on the `PixelRepresentation` element.
# 
# **Be Careful!** This means that if you use `RescaleIntercept` to rescale the uint16 arrays, you'll end up wrapping around to large positive numbers, instead of creating negative numbers!
# 
# Instead of doing this manually, use fastai's `scaled_px` attribute, which turns everything into a float and applies rescaling automatically. This is a good example of the kind of nasty "gotchas" that can occur when working with medical images. It's best, where possible, to automatically handle these inside libraries, since otherwise the bugs that can creep in can be hard to spot.

# In[ ]:


dcm.scaled_px.type()


# Another benefit of using `scaled_px` is that it returns a PyTorch tensor, which allows you to do image transformations in full floating point precision, accelerated on the GPU. fastai v2 has a wide range of GPU-accelerated transformations, which we will look at in a future notebook.
# 
# fastai will attempt to automatically normalize the image for basic viewing, without having to worry about windowing.

# In[ ]:


dcm.show(figsize=(6,6))


# A range of windows are also provided, based on recommendations from [Radiopaedia](https://radiopaedia.org/articles/windowing-ct) (or you can set up your own), or you can also look at the raw data.

# In[ ]:


scales = False, True, dicom_windows.brain, dicom_windows.subdural
titles = 'raw','normalized','brain windowed','subdural windowed'
for s,a,t in zip(scales, subplots(2,2,imsize=5)[1].flat, titles):
    dcm.show(scale=s, ax=a, title=t)


# I've analyzed the issue of windowing and leveraging floating point data in much more detail in this notebook: [DON'T see like a radiologist](https://www.kaggle.com/jhoward/don-t-see-like-a-radiologist-fastai). It builds on top of the work done in this current notebook.
# 
# I'll announce new kernels on my [Twitter account](https://twitter.com/jeremyphoward), and don't forget to drop by the  [fastai v2 forum](https://forums.fast.ai/c/fastai-users/fastai-v2) for more resources, Q&A, and discussions.
