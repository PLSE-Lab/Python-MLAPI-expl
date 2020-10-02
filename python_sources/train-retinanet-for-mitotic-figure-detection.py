#!/usr/bin/env python
# coding: utf-8

# # Train RetinaNet model on CCMCT data set
# 
# Train a RetinaNet model for mitotic figure detection on the MITOS_WSI_CCMCT data set. Note that this kaggle-version of the data set was converted to the DICOM format, and for the sake of dataset size only the lowest (i.e., highest resolution) layer of data was exported.
# 
# As of writing, the MITOS_WSI_CCMCT data set is the only set providing mitotic figure annotations and mitotic figure look-alikes for the complete microscopy slide image (in total, 44k mitotic figures on 32 WSI).
# 
# This is an excerpt of the data set. For the complete set and more information about it, please see our publication in Scientific Data:
# - Bertram, C.A., Aubreville, M., Marzahl, C. et al. A large-scale dataset for mitotic figure assessment on whole slide images of canine cutaneous mast cell tumor. Sci Data 6, 274 (2019) doi:10.1038/s41597-019-0290-4
# 
# Credits to some of the library code:
# - https://github.com/ChristianMarzahl/ObjectDetection
# - https://github.com/rafaelpadilla/Object-Detection-Metrics
# 
# Additionally, some of my code:
# - https://github.com/maubreville/MITOS_WSI_CCMCT
# - https://github.com/maubreville/SlideRunner
# 
# 

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
from fastai import *
from fastai.vision import *
from fastai.callbacks import *

"""
         Own libraries
"""
from sliderunnerdatabase import Database
from objectdetectiontools import get_slides, PascalVOCMetric, create_anchors,ObjectItemListSlide, SlideObjectCategoryList, bb_pad_collate_min, show_anchors_on_images, slide_object_result 
from retinanet import RetinaNet,RetinaNetFocalLoss


# In[ ]:


path = Path('/kaggle/input/mitosis-wsi-ccmct-training-set/')

database = Database()
database.open(str(path/'MITOS_WSI_CCMCT_ODAEL_train_dcm.sqlite'))


getslides = """SELECT filename FROM Slides"""
all_slides = database.execute(getslides).fetchall()


# In[ ]:


lbl_bbox, train_slides,val_slides,files = get_slides(slidelist_test=[1,2,3,4,5,6,7,8,9,10,11,12,13,14], size=512, positive_class=2, negative_class=7, database=database,basepath=str(path))
            


# In[ ]:


bs = 16
train_images = 5000
val_images = 5000
size=512

img2bbox = dict(zip(files, np.array(lbl_bbox)))
get_y_func = lambda o:img2bbox[o]

tfms = get_transforms(do_flip=True,
                      flip_vert=True,
                      max_rotate=90,
                      max_lighting=0.0,
                      max_zoom=1.,
                      max_warp=0.0,
                      p_affine=0.5,
                      p_lighting=0.0,
                      #xtra_tfms=xtra_tfms,
                     )
train_files = list(np.random.choice([files[x] for x in train_slides], train_images))
valid_files = list(np.random.choice([files[x] for x in val_slides], val_images))


train =  ObjectItemListSlide(train_files, path=path)
valid = ObjectItemListSlide(valid_files, path=path)
valid = ObjectItemListSlide(valid_files, path=path)
item_list = ItemLists(path, train, valid)
lls = item_list.label_from_func(get_y_func, label_cls=SlideObjectCategoryList) #
lls = lls.transform(tfms, tfm_y=True, size=size)
data = lls.databunch(bs=bs, collate_fn=bb_pad_collate_min, num_workers=4).normalize()


# Let's have a look at the data - from the data set, only mitotic figure cells are selected. Additionally, the training provides images with a high probability of mitotic-figure-lookalikes (see the paper).

# In[ ]:


data.show_batch(rows=2, ds_type=DatasetType.Train, figsize=(15,15))


# Create anchors for object detection and show them on the image.

# In[ ]:


anchors = create_anchors(sizes=[(32,32)], ratios=[1], scales=[0.6, 0.7,0.8])
not_found = show_anchors_on_images(data, anchors)


# Looks cool. 
# 
# Now create the network. Note that for this step, internet access is required, since fast.ai wants to download the pre-trained weights for the ResNet18 stem.

# In[ ]:


crit = RetinaNetFocalLoss(anchors)
encoder = create_body(models.resnet18, True, -2)
model = RetinaNet(encoder, n_classes=data.train_ds.c, n_anchors=3, sizes=[32], chs=128, final_bias=-4., n_conv=3)


voc = PascalVOCMetric(anchors, size, [str(i-1) for i in data.train_ds.y.classes[1:]])
learn = Learner(data, model, loss_func=crit, callback_fns=[ShowGraph], #BBMetrics, ShowGraph
                metrics=[voc]
               )


# Let's now find the optimal learning rate.

# In[ ]:


learn.split([model.encoder[6], model.c5top5])
learn.freeze_to(-2)
learn.model_dir='/kaggle/working/'

learn.lr_find()
learn.recorder.plot()


# 1e-4 seems to be a good value. Let's fit the (frozen, except the heads) network for a cycle of 1 epoch (5000 randomly cropped images)

# In[ ]:


learn.fit_one_cycle(1, 1e-4)


# Let's have a quick peek at the preliminary state ...

# In[ ]:


slide_object_result(learn, anchors, detect_thresh=0.3, nms_thresh=0.2, image_count=10)


# Not great, but a good start. Let's now unfreeze the network (i.e., allow all layers to train)

# In[ ]:


learn.unfreeze()


# And now let's fit the network using the cyclic learning rate scheme for 10 "epochs"

# In[ ]:


learn.fit_one_cycle(10,1e-4)


# Let's see how the network performs now.

# In[ ]:


slide_object_result(learn, anchors, detect_thresh=0.3, nms_thresh=0.2, image_count=10)


# Our model was trained vor several more epochs, but GPU time on kaggle is rare so we stop here with our proof of principle.

# In[ ]:


learn.export('/kaggle/working/RetinaNetMitoticFigures')

