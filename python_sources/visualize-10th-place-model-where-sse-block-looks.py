#!/usr/bin/env python
# coding: utf-8

# # Visualization of squeezed features: Where sSE Block looks?

#  I suppose that **sSE Block** plays an important role as well as RandomErasing in [my solution](https://www.kaggle.com/c/bengaliai-cv19/discussion/136815).  
# This notebook shows you where on images **sSE Block** pays its attention, by visualizing squeezed features for some images in train dataset.
#  
# Used model is cycle2 (70 epoch) model, which achived the best single private score(0.9499) in my submissions.
# 
# <br>
# 
# Thank you Peter(@pestipeti)! I can visualize each component by your code and dataset in [Bengali - Quick EDA](https://www.kaggle.com/pestipeti/bengali-quick-eda). 

# ## preparation

# ### import

# In[ ]:


import sys
import os
import time
import gc
import random
from pathlib import Path
from collections import Counter
from itertools import chain
from PIL import Image, ImageDraw, ImageFont

from typing import Union, Tuple, Dict, Optional

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

import chainer

from matplotlib import pyplot as plt
import seaborn as sns

from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.max_rows = 2000
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 500


# In[ ]:


# # set Path
ROOT = Path(".").absolute().parents[0]
INPUT_ROOT = ROOT / "input"
RAW_DATA = INPUT_ROOT / "bengaliai-cv19"

SRC_PATH = INPUT_ROOT / "bengali-src-final-05"
PROC_DATA = INPUT_ROOT / "bengali-processed-data"
TRIANED_MODELS_PATH = INPUT_ROOT / "bengali-final-serx50sse-128x224-3x35ep-all-fold"


# In[ ]:


cd ../input/bengali-src-final-05


# In[ ]:


# # import from my src
# sys.path.append(SRC_PATH.as_posix())

from competition_utils import utils as my_utils

from nn_for_image_data import backborn_chains as my_backborns
from nn_for_image_data import global_pooling_chains as my_global_poolings
from nn_for_image_data import classifer_chains as my_classifers
from training_utils import nn_training as my_nn_tr

import config as my_config


# ### make dataset

# In[ ]:


# # read train label info
train_df = pd.read_csv(PROC_DATA / "train_add-4fold-index.csv").drop(["character_id", "fold"], axis=1)
clm = pd.read_csv(RAW_DATA / "class_map.csv")


# I select some of graphemes to visualize, giving priority to `consonant_diacritic`.

# In[ ]:


train_df = train_df.drop_duplicates(
    subset=["vowel_diacritic", "consonant_diacritic"]
).sort_values(by=["consonant_diacritic", "vowel_diacritic"]).reset_index(drop=True)


# add str of each component

# In[ ]:


# # add `grapheme_root_str`
train_df = train_df.merge(
    clm.query("component_type == 'grapheme_root'")[["component", "label"]].rename(
        columns={"component": 'grapheme_root_str', "label": "grapheme_root"}),
    on="grapheme_root", how="left")
# # add `vowel_diacritic_str`
train_df = train_df.merge(
    clm.query("component_type == 'vowel_diacritic'")[["component", "label"]].rename(
        columns={"component": 'vowel_diacritic_str', "label": "vowel_diacritic"}),
    on="vowel_diacritic", how="left")
# # add `consonant_diacritic`
train_df = train_df.merge(
    clm.query("component_type == 'consonant_diacritic'")[["component", "label"]].rename(
        columns={"component": 'consonant_diacritic_str', "label": "consonant_diacritic"}),
    on="consonant_diacritic", how="left")


# In[ ]:


train_df


# #### make labels arr

# In[ ]:


train_labels_arr = train_df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values.astype("i")


# #### make chainer dataset for extract squeezed features

# In[ ]:


## prepare a dataset for visualize to feed into the trained model.
ext_dataset = chainer.datasets.LabeledImageDataset(
    pairs=list(
        zip((train_df["image_id"] + ".png").tolist(),  train_labels_arr)),
    root=(PROC_DATA / "train").as_posix())

# # set transforms
# # # Note: `CustomTranspose` just changes position of channel. This is because I use chainer datasets class and albumentations.
# # # Albumentations requires [H, W, C] format images while chainer datasets class output [C, H, W] format images.
ext_dataset = chainer.datasets.TransformDataset(
    ext_dataset,
    my_nn_tr.ImageTransformer([
        # # change format from [C, H, W] to [H, W, C]
        ["CustomTranspose", {"always_apply": True, "axis": [1, 2, 0]}],
        # # Pad
        ["PadIfNeeded", {
            "always_apply": True, "min_height": 140, "min_width": 245, "border_mode": 0, "value": 253}],
        # # Resize
        ["Resize", {"always_apply": True, "height": 128, "width": 224}],
        # # Normalize
        ["Normalize", {
            "always_apply": True, "mean": [0.946967259411315,], "std": [0.06580952928802959,]}],
        # # change format from [H, W, C] to [C, H, W]
        ["CustomTranspose", {"always_apply": True, "axis": [2, 0, 1]}],
    ])
)


# ### init and load model
# #### make new class which inherits ImageClassificationModels
# Add new method for extract features squeezed  by sSE Block output.

# In[ ]:


class CustomICM(my_nn_tr.ImageClassificationModel):
    """Custom Class"""
    
    def __init__(
        self, extractor: chainer.Chain,
        global_pooling: Optional[chainer.Chain], classifier: chainer.Chain
    ) -> None:
        """Initialization."""
        super(CustomICM, self).__init__(extractor, global_pooling, classifier)
        
    def extract_squeezed_features(self, x: chainer.Variable) -> Tuple[chainer.Variable]:
        """New method for extraction,"""
        # # x: (bs, 1, H, W) => feature_map: (bs, 2048, H // 32, W // 32)
        # # # [in this case] x: (bs, 1,128, 224) => feature_map: (bs, 2048, 4, 7)
        feature_map = self.extractor(x)

        # # feature_map: (bs, 2048, H // 32, W // 32) => sqfeat_XXX: (bs, 1, H // 32, W // 32) (the same shape)
        # # # for root
        sqfeat_root = chainer.functions.sigmoid(
            self.global_pooling.pool1.sse.channel_squeeze(feature_map))
        # # # for vowel
        sqfeat_vowel = chainer.functions.sigmoid(
            self.global_pooling.pool2.sse.channel_squeeze(feature_map))
        # # # for consonant
        sqfeat_consonant = chainer.functions.sigmoid(
                self.global_pooling.pool3.sse.channel_squeeze(feature_map))
        
        return (sqfeat_root , sqfeat_vowel, sqfeat_consonant)


# #### init and load model

# In[ ]:


# # init model
model = CustomICM(
    # # backborn
    extractor=getattr(my_backborns, "ImageFeatureExtractor")(
        backborn_model="SEResNeXt50Conv1MeanTo1ch", pretrained_model_path=None, extract_layers=["res5"]),
    # # global pooling
    global_pooling=getattr(my_global_poolings, "TripleHeadPoolingLayer")(
        pooling_layer="sSEAvgPool", pooling_kwargs={}),
    # # head classifier
    classifier=getattr(my_classifers, "TripleInOutClassificationLayer")(
        n_classes=[168, 11, 7], classification_layer="LADL")
)

# # load the model of cycle2(70 epoch), which is my best single model at private LB (0.9499). 
chainer.serializers.load_npz(
    TRIANED_MODELS_PATH / "model_snapshot_70.npz", model)


# ## extract

# for extract squeezed features, rewrite inference loop function.

# In[ ]:


def loop_for_extraction_of_squeezed_features(
    model: chainer.Chain, test_iter: chainer.iterators.MultiprocessIterator, gpu_device: int=-1
# ) -> Tuple[np.ndarray]:
) -> Tuple[Tuple[np.ndarray], np.ndarray]:
    """Inference roop for extraction of squeezed_features for each component"""
#     test_pred_list = []
    sqfeat_root_list = []
    sqfeat_vowel_list = []
    sqfeat_consonant_list = []

    test_label_list = []
    
    iter_num = 0
    epoch_test_start = time.time()

    while True:
        test_batch = test_iter.next()
        iter_num += 1
        print("\rtmp_iteration: {:0>5}".format(iter_num), end="")
        in_arrays = chainer.dataset.concat_examples(test_batch, gpu_device)

        # Forward the test data
        with chainer.no_backprop_mode() and chainer.using_config("train", False):
#             prediction_test = model.inference(*in_arrays[:-1])
#             test_pred_list.append(prediction_test)
            (test_sqfeat_root, test_sqfeat_vowel, test_sqfeat_consonant) = model.extract_squeezed_features(*in_arrays[:-1])
            sqfeat_root_list.append(test_sqfeat_root)
            sqfeat_vowel_list.append(test_sqfeat_vowel)
            sqfeat_consonant_list.append(test_sqfeat_consonant)

            test_label_list.append(in_arrays[-1])
#             prediction_test.unchain_backward()
            test_sqfeat_root.unchain_backward()
            test_sqfeat_vowel.unchain_backward()
            test_sqfeat_consonant.unchain_backward()

        if test_iter.is_new_epoch:
            print(" => test end: {:.2f} sec".format(time.time() - epoch_test_start))
            test_iter.reset()
            break

#     test_pred_all = chainer.cuda.to_cpu(functions.concat(test_pred_list, axis=0).data)
    sqfeat_root_all = chainer.cuda.to_cpu(chainer.functions.concat(sqfeat_root_list, axis=0).data)
    sqfeat_vowel_all = chainer.cuda.to_cpu(chainer.functions.concat(sqfeat_vowel_list, axis=0).data)
    sqfeat_consonant_all = chainer.cuda.to_cpu(chainer.functions.concat(sqfeat_consonant_list, axis=0).data)

    test_label_all = chainer.cuda.to_cpu(chainer.functions.concat(test_label_list, axis=0).data)
#     del test_pred_list
    del sqfeat_root_list
    del sqfeat_vowel_list
    del sqfeat_consonant_list
    del test_label_list
#     return test_pred_all, test_label_all
    return (sqfeat_root_all, sqfeat_vowel_all, sqfeat_consonant_all), test_label_all


# extact

# In[ ]:


# # create iterator
ext_iter = chainer.iterators.MultiprocessIterator(
    ext_dataset, batch_size=64, repeat=False, shuffle=False, n_processes=2)

# # extract
# model.to_gpu(0)
(
    sqfeat_root_arr, sqfeat_vowel_arr, sqfeat_consonant_arr
), label_arr = loop_for_extraction_of_squeezed_features(model, ext_iter, gpu_device=-1)
# model.to_cpu()

# # check shapes
print((sqfeat_root_arr.shape, sqfeat_vowel_arr.shape, sqfeat_consonant_arr.shape), label_arr.shape)


# ## Visualization

# prepare a dataset for visualization.

# In[ ]:


grapheme_arr = train_df[
    ["image_id", "grapheme", "grapheme_root_str", "vowel_diacritic_str", "consonant_diacritic_str"]].values


# In[ ]:


viz_dataset = chainer.datasets.LabeledImageDataset(
    pairs=list(
        zip((train_df["image_id"] + ".png").tolist(),  train_labels_arr)),
    root=(PROC_DATA / "train").as_posix())

# # Now I want to get original images, not apply `Normalize`.
viz_dataset = chainer.datasets.TransformDataset(
    viz_dataset,
    my_nn_tr.ImageTransformer([
        # # change format from [C, H, W] to [H, W, C]
        ["CustomTranspose", {"always_apply": True, "axis": [1, 2, 0]}],
        # # Pad
        ["PadIfNeeded", {
            "always_apply": True, "min_height": 140, "min_width": 245, "border_mode": 0, "value": 253}],
        # # Resize
        ["Resize", {"always_apply": True, "height": 128, "width": 224}],
        # # Normalize
#         ["Normalize", {
#             "always_apply": True, "mean": [0.946967259411315,], "std": [0.06580952928802959,]}],
        # # change format from [H, W, C] to [C, H, W]
        ["CustomTranspose", {"always_apply": True, "axis": [2, 0, 1]}],
    ])
)


# define visualization function

# In[ ]:


def image_from_char(char, width=224, height=128):
    """
    Make image from char.
    reference: https://www.kaggle.com/pestipeti/bengali-quick-eda
    """
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)
    myfont = ImageFont.truetype('/kaggle/input/bengaliai/hind_siliguri_normal_500.ttf', 120)
    w, h = draw.textsize(char, font=myfont)
    draw.text(((width - w) / 2,(height - h) / 2), char, font=myfont)

    return image

def get_sqfeat_image(idx, sqfeat_arr, img_size, cmap, size=(224, 128)):
    """Create sqeezed feature image."""
    # # min_max normalize for clear visualization
    arr = (sqfeat_arr[idx][0] - sqfeat_arr[idx][0].min()) / (sqfeat_arr[idx][0].max() - sqfeat_arr[idx][0].min())
    # # convert to RGB color image by color map
    img =  Image.fromarray((cmap(arr)*255).astype("uint8")).resize(size).convert("RGB")
    return img


# In[ ]:


def visualize_squeezed_features(
    img_idxs: np.ndarray, img_dataset: chainer.datasets.LabeledImageDataset,
    root_arr: np.ndarray, vowel_arr: np.ndarray, consonant_arr: np.ndarray,
    label_arr: np.ndarray, grapheme_arr:np.ndarray
):
    """Vidualize squeezed features by blending with original image."""
    cmap_autumn = plt.get_cmap("autumn_r")
    num_img = len(img_idxs)
    fig = plt.figure(figsize=(16, 2 * 3 * num_img))
    
    for i, idx in enumerate(img_idxs):
        image_id, grapheme_str, root_str, vowel_str, consonant_str = grapheme_arr[idx]
        root_label, vowel_label, consonant_label = label_arr[idx]
        img_org = Image.fromarray(img_dataset[idx][0][0].astype("uint8")).convert("RGB")
        img_org_font = image_from_char(grapheme_str)
        img_root = get_sqfeat_image(idx, root_arr, img_org.size, cmap_autumn)
        img_root_font = image_from_char(root_str)
        img_vowel = get_sqfeat_image(idx, vowel_arr, img_org.size, cmap_autumn)
        img_vowel_font = image_from_char(vowel_str)
        img_consonant = get_sqfeat_image(idx, consonant_arr, img_org.size, cmap_autumn)
        img_consonant_font = image_from_char(consonant_str)
    
        ax1_1 = fig.add_subplot(num_img * 2, 4, 4 * 2 * i + 1)
        ax2_1 = fig.add_subplot(num_img * 2, 4, 4 * (2 * i + 1) + 1)
        ax1_2 = fig.add_subplot(num_img * 2, 4, 4 * 2 * i + 2)
        ax2_2 = fig.add_subplot(num_img * 2, 4, 4 * (2 * i + 1) + 2)
        ax1_3 = fig.add_subplot(num_img * 2, 4, 4 * 2 * i + 3)
        ax2_3 = fig.add_subplot(num_img * 2, 4, 4 * (2 * i + 1) + 3)
        ax1_4 = fig.add_subplot(num_img * 2, 4, 4 * 2 * i + 4)
        ax2_4 = fig.add_subplot(num_img * 2, 4, 4 * (2 * i + 1) + 4)

        ax1_1.imshow(img_org)
        ax1_1.xticks = None
        ax2_1.imshow(img_org_font)
        ax1_2.imshow(Image.blend(img_org, img_root, 0.7))
        ax2_2.imshow(img_root_font)
        ax1_3.imshow(Image.blend(img_org, img_vowel, 0.7))
        ax2_3.imshow(img_vowel_font)
        ax1_4.imshow(Image.blend(img_org, img_consonant, 0.7))
        ax2_4.imshow(img_consonant_font)
        
        ax1_1.set_title("[grapheme] {}".format(image_id), fontsize=14)
        ax1_1.tick_params(labelbottom=False, labelleft=False)
        ax2_1.tick_params(labelbottom=False, labelleft=False)
        ax1_2.set_title("[grapheme_root] id: {}".format(root_label), fontsize=14)
        ax1_2.tick_params(labelbottom=False, labelleft=False)
        ax2_2.tick_params(labelbottom=False, labelleft=False)
        ax1_3.set_title("[vowel_diacritic]  id: {}".format(vowel_label), fontsize=14)
        ax1_3.tick_params(labelbottom=False, labelleft=False)
        ax2_3.tick_params(labelbottom=False, labelleft=False)
        ax1_4.set_title("[consonant_diacritic] id: {}".format(consonant_label), fontsize=14)
        ax1_4.tick_params(labelbottom=False, labelleft=False)
        ax2_4.tick_params(labelbottom=False, labelleft=False)


# ### for consonant_diacrtic == 0 (nothing)

# In[ ]:


train_df.query("consonant_diacritic == 0")


# In[ ]:


visualize_squeezed_features(
    train_df.query("consonant_diacritic == 0").index.values, viz_dataset,
    sqfeat_root_arr, sqfeat_vowel_arr, sqfeat_consonant_arr, label_arr, grapheme_arr)


# ### for consonant_diacrtic == 1

# In[ ]:


train_df.query("consonant_diacritic == 1")


# In[ ]:


visualize_squeezed_features(
    train_df.query("consonant_diacritic == 1").index.values, viz_dataset,
    sqfeat_root_arr, sqfeat_vowel_arr, sqfeat_consonant_arr, label_arr, grapheme_arr)


# ### for consonant_diacrtic == 2

# In[ ]:


train_df.query("consonant_diacritic == 2")


# In[ ]:


visualize_squeezed_features(
    train_df.query("consonant_diacritic == 2").index.values, viz_dataset,
    sqfeat_root_arr, sqfeat_vowel_arr, sqfeat_consonant_arr, label_arr, grapheme_arr)


# ### for consonant_diacrtic == 3

# In[ ]:


train_df.query("consonant_diacritic == 3")


# In[ ]:


visualize_squeezed_features(
    train_df.query("consonant_diacritic == 3").index.values, viz_dataset,
    sqfeat_root_arr, sqfeat_vowel_arr, sqfeat_consonant_arr, label_arr, grapheme_arr)


# ### for consonant_diacrtic == 4

# In[ ]:


train_df.query("consonant_diacritic == 4")


# In[ ]:


visualize_squeezed_features(
    train_df.query("consonant_diacritic == 4").index.values, viz_dataset,
    sqfeat_root_arr, sqfeat_vowel_arr, sqfeat_consonant_arr, label_arr, grapheme_arr)


# ### for consonant_diacrtic == 5

# In[ ]:


train_df.query("consonant_diacritic == 5")


# In[ ]:


visualize_squeezed_features(
    train_df.query("consonant_diacritic == 5").index.values, viz_dataset,
    sqfeat_root_arr, sqfeat_vowel_arr, sqfeat_consonant_arr, label_arr, grapheme_arr)


# ### for consonant_diacrtic == 6

# In[ ]:


train_df.query("consonant_diacritic == 6")


# In[ ]:


visualize_squeezed_features(
    train_df.query("consonant_diacritic == 6").index.values, viz_dataset,
    sqfeat_root_arr, sqfeat_vowel_arr, sqfeat_consonant_arr, label_arr, grapheme_arr)


# ## Discussion
# 
# For me, difference between squeezed features of each component looks not so large.
# 
# But this result gives me some findings.
# 
# ### In most cases, All the sSE Blocks look the grapheme, **_not the white space_**.
# I think this behavior contributes to model performance. Simple GAP doesn't do this.
# 
# ### sSE Blocks of each component look different areas.
# 
# As you see above, this difference seems to be just a little. But there is **certainly** a difference.  
# Roughly, sSE Block of `grapheme_root` looks relatively wide area, and one of `consonant_diacritic` relatively narrow area.
# 
# <br>
# 
# We know **_the devils are in the details_**, I think these small differences contribute to my solo gold place solution.
# 
# Thank you for reading! I'm glad to share this somewhat interesting result.
