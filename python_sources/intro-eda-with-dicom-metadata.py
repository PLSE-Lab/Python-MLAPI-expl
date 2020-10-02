#!/usr/bin/env python
# coding: utf-8

# ## Sections
# - [Parsing Tabular Data](#Parsing-Tabular-Data)
# - [Parsing Metadata from DICOM Object](#Parsing-Metadata-from-DICOM-Object)
# - [Cleaning DICOM Metadata and Merging to Tabular Data](#Cleaning-DICOM-Metadata-and-Merging-to-Tabular-Data)
# - [EDA on Metadata and Initial Tabular Data](#EDA-on-Metadata-and-Initial-Tabular-Data)
#     - [ViewPosition](#ViewPosition)
#     - [PatientAge](#PatientAge)
#     - [PatientSex](#PatientSex)
#     - [PixelSpacing](#PixelSpacing)
#     - [Initial Tabular Data](#Initial-Tabular-Data)
#     - [Bounding Box](#Bounding-Box)
#         - [Bounding Box Data Manipulation](#Bounding-Box-Data-Manipulation)
#         - [Bounding Box Plots](#Bounding-Box-Plots)
#     - [Image Intensity Distribution](#Image-Intensity-Distribution)
# - [Summary (TL;DR)](#Summary-%28TL%3BDR%29)
# - [Simple LGBM Binary Classifier](#Simple-LGBM-Binary-Classifier)

# ## Summary (TL;DR)
# - ViewPosition, PatientAge, PatientSex, PixelSpacing likely the only fields in metadata with possible value [[relevant section](#Initial-takeaways)]
# - ViewPosition seems to be a vital feature [[relevant section](#ViewPosition)]
# - Bounding boxes are more prevalent on the right lung [[relevant section](#Bounding-Box-Plots)]

# In[ ]:


from functools import partial
from collections import defaultdict
import pydicom
import os
import glob
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

np.warnings.filterwarnings('ignore')


# ## Parsing Tabular Data

# In[ ]:


labels = pd.read_csv('../input/stage_1_train_labels.csv')
details = pd.read_csv('../input/stage_1_detailed_class_info.csv')
# duplicates in details just have the same class so can be safely dropped
details = details.drop_duplicates('patientId').reset_index(drop=True)
labels_w_class = labels.merge(details, how='inner', on='patientId')


# ## Parsing Metadata from DICOM Object

# In[ ]:


# get lists of all train/test dicom filepaths
train_dcm_fps = glob.glob('../input/stage_1_train_images/*.dcm')
test_dcm_fps = glob.glob('../input/stage_1_test_images/*.dcm')

# read each file into a list (using stop_before_pixels to avoid reading the image for speed and memory savings)
train_dcms = [pydicom.read_file(x, stop_before_pixels=True) for x in train_dcm_fps]
test_dcms = [pydicom.read_file(x, stop_before_pixels=True) for x in test_dcm_fps]


# In[ ]:


def parse_dcm_metadata(dcm):
    unpacked_data = {}
    group_elem_to_keywords = {}
    # iterating here to force conversion from lazy RawDataElement to DataElement
    for d in dcm:
        pass
    # keys are pydicom.tag.BaseTag, values are pydicom.dataelem.DataElement
    for tag, elem in dcm.items():
        tag_group = tag.group
        tag_elem = tag.elem
        keyword = elem.keyword
        group_elem_to_keywords[(tag_group, tag_elem)] = keyword
        value = elem.value
        unpacked_data[keyword] = value
    return unpacked_data, group_elem_to_keywords

train_meta_dicts, tag_to_keyword_train = zip(*[parse_dcm_metadata(x) for x in train_dcms])
test_meta_dicts, tag_to_keyword_test = zip(*[parse_dcm_metadata(x) for x in test_dcms])


# ### Using the easily interpretable keyword instead of the DICOM tag (group, element) for column names. However, the DICOM tag might be useful for searching online for more info / detailed explanation (e.g. DICOMLookup).

# In[ ]:


# join all the dicts
unified_tag_to_key_train = {k:v for dict_ in tag_to_keyword_train for k,v in dict_.items()}
unified_tag_to_key_test = {k:v for dict_ in tag_to_keyword_test for k,v in dict_.items()}

# quick check to make sure there are no different keys between test/train
assert len(set(unified_tag_to_key_test.keys()).symmetric_difference(set(unified_tag_to_key_train.keys()))) == 0

tag_to_key = {**unified_tag_to_key_test, **unified_tag_to_key_train}
tag_to_key


# In[ ]:


# using from_records here since some values in the dicts will be iterables and some are constants
train_df = pd.DataFrame.from_records(data=train_meta_dicts)
test_df = pd.DataFrame.from_records(data=test_meta_dicts)
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'
df = pd.concat([train_df, test_df])


# In[ ]:


df.head(1)


# ## Cleaning DICOM Metadata and Merging to Tabular Data

# In[ ]:


# separating PixelSpacing list to single values
df['PixelSpacing_x'] = df['PixelSpacing'].apply(lambda x: x[0])
df['PixelSpacing_y'] = df['PixelSpacing'].apply(lambda x: x[1])
df = df.drop(['PixelSpacing'], axis='columns')

# x and y are always the same
assert sum(df['PixelSpacing_x'] != df['PixelSpacing_y']) == 0


# In[ ]:


# ReferringPhysicianName appears to just be empty strings
assert sum(df['ReferringPhysicianName'] != '') == 0

# SeriesDescription appears to be 'view: {}'.format(ViewPosition)
set(df['SeriesDescription'].unique())

# so these two columns don't have any useful info and can be safely dropped


# ### Initial takeaways
# - Many of the fields are identical throughout all the samples (probably little to gain from looking into these)
# - Looks like PatientAge, PatientSex, PixelSpacing, and ViewPosition are the only metadata items with possible value
# - Wide range of ages
# - Small number of different Pixel Spacings (maybe relevant in terms of resolution and specific to certain imaging machines or setups?)
# - 2 different view positions

# In[ ]:


nunique_all = df.aggregate('nunique')
nunique_all


# In[ ]:


# drop constant cols and other two from above
df = df.drop(nunique_all[nunique_all == 1].index.tolist() + ['ReferringPhysicianName', 'SeriesDescription'], axis='columns')

# now that we have a clean metadata dataframe we can merge back to our initial tabular data with target and class info
df = df.merge(labels_w_class, how='left', left_on='PatientID', right_on='patientId')

df['PatientAge'] = df['PatientAge'].astype(int)


# In[ ]:


# df now has multiple rows for some patients (those with multiple bounding boxes in label_w_class)
# so creating one with no duplicates for patients
df_deduped = df.drop_duplicates('PatientID', keep='first')


# ## EDA on Metadata and Initial Tabular Data

# ## ViewPosition
# 
# ### View Position is likely important since it describes the positioning of the patient when the radiograph is taken. PA = posterior-anterior (ray enters back first, so back would be at the 'top' of the image) and AP = anterior-posterior (ray enters chest first). My understanding is that PA would be the desired position for detecting lung opacity, however it frequently can't be done in that way as the patient is to ill to stand (which is required for PA). I think there are several important elements here: the pose of the patient is different (standing vs laying), the image has a different 'view' (back-first vs chest-first) and as such different parts of the body may be occluded or obscured to an extent, image quality itself is probably better in PA.
# 
# Note: I am not a medical professional so this may be an incorrect or incomplete understanding.
# 
# ### Takeaways from below plots:
# - Test/Train seems relatively well balanced AP vs PA
# - AP vs PA have a big shift in the 'Normal' and 'Lung Opacity' classes
# - Target 1/0 is well split for AP, but for PA there are many more 0s
# - Gender seems relatively well balanced AP vs PA
# - Age distributions seem similar AP vs PA
# - Different pixel spacing AP vs PA (probably due to different location of patient relative to imaging machine)

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 7))
p = sns.countplot(x='ViewPosition', hue='dataset', data=df, ax=axes[0])
_ = p.set_title('With multiple')
p = sns.countplot(x='ViewPosition', hue='dataset', data=df_deduped, ax=axes[1])
_ = p.set_title('Single row per patient')


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 7))
p = sns.countplot(x='ViewPosition', hue='class', data=df[df['dataset']=='train'], ax=axes[0])
_ = p.set_title('With multiple')
p = sns.countplot(x='ViewPosition', hue='class', data=df_deduped[df_deduped['dataset']=='train'], ax=axes[1])
_ = p.set_title('Single row per patient')


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 7))
p = sns.countplot(x='ViewPosition', hue='Target', data=df[df['dataset']=='train'], ax=axes[0])
_ = p.set_title('With multiple')
p = sns.countplot(x='ViewPosition', hue='Target', data=df_deduped[df_deduped['dataset']=='train'], ax=axes[1])
_ = p.set_title('Single row per patient')


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 7))
p = sns.countplot(x='ViewPosition', hue='PatientSex', data=df, ax=axes[0])
_ = p.set_title('With multiple')
p = sns.countplot(x='ViewPosition', hue='PatientSex', data=df_deduped, ax=axes[1])
_ = p.set_title('Single row per patient')


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 7))

p = sns.distplot(df[df['ViewPosition']=='AP']['PatientAge'], hist=True, kde=False, color='red', label='AP', ax=axes[0])
p = sns.distplot(df[df['ViewPosition']=='PA']['PatientAge'], hist=True, kde=False, color='gray', label='PA', ax=axes[0])
_ = p.set_ylabel('Count')
_ = p.legend()
_ = p.set_title('With multiple')

p = sns.distplot(df_deduped[df_deduped['ViewPosition']=='AP']['PatientAge'], hist=True, kde=False, color='red', label='AP', ax=axes[1])
p = sns.distplot(df_deduped[df_deduped['ViewPosition']=='PA']['PatientAge'], hist=True, kde=False, color='gray', label='PA', ax=axes[1])
_ = p.set_ylabel('Count')
_ = p.legend()
_ = p.set_title('Single row per patient')


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 7))

p = sns.countplot(df['PixelSpacing_x'], hue='ViewPosition', data=df, ax=axes[0])
_ = p.set_xticklabels([x for x in p.get_xticklabels()],rotation=90)
_ = p.set_title('With multiple')

p = sns.countplot(df_deduped['PixelSpacing_x'], hue='ViewPosition', data=df_deduped, ax=axes[1])
_ = p.set_xticklabels([x for x in p.get_xticklabels()],rotation=90)
_ = p.set_title('Single row per patient')


# ### Viewing the actual images for AP vs PA. The PA images seem much clearer to my untrained eye.

# In[ ]:


def read_img(patient_id):
    train_fp = '../input/stage_1_train_images/{}.dcm'.format(patient_id)
    if os.path.exists(train_fp):
        dcm = pydicom.read_file(train_fp)
    else:
        test_fp = '../input/stage_1_test_images/{}.dcm'.format(patient_id)
        dcm = pydicom.read_file(test_fp)
    return dcm


# In[ ]:


def plot_grid(df, pid_sample_list, nrows=3, ncols=3, draw_bbox=True, ax_off=True):
    fig = plt.figure(figsize=(16, 12))
    for i in range(nrows * ncols):
        patient_id = pid_sample_list[i]
        img = read_img(patient_id).pixel_array
        ax = fig.add_subplot(nrows, ncols, i + 1)
        plt.imshow(img, cmap='gray')
        ax.set_title(patient_id)
        if ax_off: 
            ax.set_axis_off()
        if draw_bbox:
            bbox_rows = df[df['PatientID'] == patient_id]
            for _, row in bbox_rows.iterrows():
                x, y = row['x'], row['y']
                width, height = row['width'], row['height']
                bbox = patches.Rectangle((x, y), width, height, linewidth=.5, edgecolor='r', facecolor='none')
                ax.add_patch(bbox)
    plt.tight_layout()
    plt.subplots_adjust(wspace=.01, hspace=.01)
    return fig


# In[ ]:


pa_ids = df[df['ViewPosition']=='PA']['PatientID'].sample(20).tolist()
_ = plot_grid(df, pa_ids, nrows=2, ncols=3)


# In[ ]:


ap_ids = df[df['ViewPosition']=='AP']['PatientID'].sample(20).tolist()
_ = plot_grid(df, ap_ids, nrows=2, ncols=3)


# ## PatientAge
# 
# ### Not a whole lot to say about age other than there are 5 outliers with ages greater than 140 (likely human entry error). Otherwise it seems like the distribution of age is fairly consistent when split by target and class.

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 7))

p = sns.distplot(df[df['Target']==1]['PatientAge'], hist=True, kde=False, color='red', label='Target 1', ax=axes[0])
p = sns.distplot(df[df['Target']==0]['PatientAge'], hist=True, kde=False, color='gray', label='Target 0', ax=axes[0])
_ = p.set_ylabel('Count')
_ = p.set_title('With multiple')

p = sns.distplot(df_deduped[df_deduped['Target']==1]['PatientAge'], hist=True, kde=False, color='red', label='Target 1', ax=axes[1])
p = sns.distplot(df_deduped[df_deduped['Target']==0]['PatientAge'], hist=True, kde=False, color='gray', label='Target 0', ax=axes[1])
_ = p.set_ylabel('Count')
_ = p.set_title('Single row per patient')


# In[ ]:


fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
for i, _class in enumerate(df['class'].dropna().unique()):
    p = sns.distplot(df[df['class']==_class]['PatientAge'], hist=True, kde=False, ax=axes[i, 0])
    _ = p.set_ylabel('Count')
    _ = p.set_xlabel(f'PatientAge - {_class}')
    if i == 0: p.set_title('With multiple')
    
    p = sns.distplot(df_deduped[df_deduped['class']==_class]['PatientAge'], hist=True, kde=False, ax=axes[i, 1])
    _ = p.set_ylabel('Count')
    _ = p.set_xlabel(f'PatientAge - {_class}')
    if i == 0: p.set_title('Single row per patient')


# ## PatientSex

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 7))
p = sns.countplot(x='PatientSex', hue='Target', data=df, ax=axes[0])
_ = p.set_title('With multiple')
p = sns.countplot(x='PatientSex', hue='Target', data=df_deduped, ax=axes[1])
_ = p.set_title('Single row per patient')


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 7))
p = sns.countplot(x='PatientSex', hue='class', data=df, ax=axes[0])
_ = p.set_title('With multiple')
p = sns.countplot(x='PatientSex', hue='class', data=df_deduped, ax=axes[1])
_ = p.set_title('Single row per patient')


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 7))
p = sns.boxplot(x='PatientSex', y='PatientAge', hue='Target', data=df, ax=axes[0])
_ = p.set_title('With multiple')
p = sns.boxplot(x='PatientSex', y='PatientAge', hue='Target', data=df_deduped, ax=axes[1])
_ = p.set_title('Single row per patient')


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 7))
p = sns.boxplot(x='PatientSex', y='PatientAge', hue='class', data=df, ax=axes[0])
_ = p.set_title('With multiple')
p = sns.boxplot(x='PatientSex', y='PatientAge', hue='class', data=df_deduped, ax=axes[1])
_ = p.set_title('Single row per patient')


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 7))
p = sns.boxplot(x='PatientSex', y='PatientAge', hue='ViewPosition', data=df, ax=axes[0])
_ = p.set_title('With multiple')
p = sns.boxplot(x='PatientSex', y='PatientAge', hue='ViewPosition', data=df_deduped, ax=axes[1])
_ = p.set_title('Single row per patient')


# ## PixelSpacing

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 7))

p = sns.countplot(df['PixelSpacing_x'], hue='Target', data=df, ax=axes[0])
_ = p.set_xticklabels([x for x in p.get_xticklabels()],rotation=90)
_ = p.set_title('With multiple')
_ = p.legend(loc='upper right')

p = sns.countplot(df_deduped['PixelSpacing_x'], hue='Target', data=df_deduped, ax=axes[1])
_ = p.set_xticklabels([x for x in p.get_xticklabels()],rotation=90)
_ = p.set_title('Single row per patient')
_ = p.legend(loc='upper right')


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 7))

p = sns.countplot(df['PixelSpacing_x'], hue='class', data=df, ax=axes[0])
_ = p.set_xticklabels([x for x in p.get_xticklabels()],rotation=90)
_ = p.set_title('With multiple')
_ = p.legend(loc='upper right')

p = sns.countplot(df_deduped['PixelSpacing_x'], hue='class', data=df_deduped, ax=axes[1])
_ = p.set_xticklabels([x for x in p.get_xticklabels()],rotation=90)
_ = p.set_title('Single row per patient')
_ = p.legend(loc='upper right')


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 7))

p = sns.countplot(df['PixelSpacing_x'], hue='PatientSex', data=df, ax=axes[0])
_ = p.set_xticklabels([x for x in p.get_xticklabels()],rotation=90)
_ = p.set_title('With multiple')
_ = p.legend(loc='upper right')

p = sns.countplot(df_deduped['PixelSpacing_x'], hue='PatientSex', data=df_deduped, ax=axes[1])
_ = p.set_xticklabels([x for x in p.get_xticklabels()],rotation=90)
_ = p.set_title('Single row per patient')
_ = p.legend(loc='upper right')


# ## Initial Tabular Data

# In[ ]:


p = sns.countplot(x='Target', hue='class', data=labels_w_class)


# In[ ]:


# check every row with Target==1 has a bounding box
assert sum(labels_w_class['Target']==1) == sum(~labels_w_class['x'].isnull())

bbox_counts = labels_w_class.groupby('patientId')['Target'].sum()
labels_w_class.index = labels_w_class.patientId
labels_w_class['bbox_counts'] = bbox_counts
labels_w_class = labels_w_class.reset_index(drop=True)


# In[ ]:


p = sns.countplot(x='bbox_counts', hue='class', data=labels_w_class)


# In[ ]:


p = sns.countplot(x='bbox_counts', hue='Target', data=labels_w_class)


# In[ ]:


labels_w_class[['x', 'y', 'width', 'height']].mean()


# ## Bounding Box
# 
# ### Bounding Box Data Manipulation

# In[ ]:


def build_bbox_arrays_by_id(df):
    zeros_array_constructor = partial(np.zeros, shape=(1024,1024), dtype=np.uint8)
    arrays = defaultdict(zeros_array_constructor)
    for idx, row in df.iterrows():
        patient_id = row['patientId']
        x, y = int(row['x']), int(row['y'])
        width, height = int(row['width']), int(row['height'])
        array = arrays[patient_id]
        array[y: y + height, x: x + width] += 1
    return arrays


# In[ ]:


bbox_arrays = build_bbox_arrays_by_id(df[df['Target']==1])


# In[ ]:


groups_to_ids = {
    'pa': set(df['patientId'][df['ViewPosition']=='PA'].dropna().unique()),
    'ap': set(df['patientId'][df['ViewPosition']=='AP'].dropna().unique()),
    
    'bbox_4': set(labels_w_class['patientId'][labels_w_class['bbox_counts']==4].dropna().unique()),
    'bbox_3': set(labels_w_class['patientId'][labels_w_class['bbox_counts']==3].dropna().unique()),
    'bbox_2': set(labels_w_class['patientId'][labels_w_class['bbox_counts']==2].dropna().unique()),
    'bbox_1': set(labels_w_class['patientId'][labels_w_class['bbox_counts']==1].dropna().unique()),
    
    'f': set(df['patientId'][df['PatientSex']=='F'].dropna().unique()),
    'm': set(df['patientId'][df['PatientSex']=='M'].dropna().unique()),
    
    'age_above_60': set(df['patientId'][df['PatientAge'] > 60].dropna().unique()),
    'age_40_to_60': set(df['patientId'][(df['PatientAge'] <= 60) & (df['PatientAge'] >= 40)].dropna().unique()),
    'age_below_40': set(df['patientId'][df['PatientAge'] < 40].dropna().unique()),
}


# In[ ]:


# construct arrays representing 'density' of bounding boxes by summing the arrays
zeros_array_constructor = partial(np.zeros, shape=(1024,1024), dtype=np.uint32)
groups_to_bbox_sums = defaultdict(zeros_array_constructor)
groups_to_bbox_sums['all'] = np.zeros(shape=(1024,1024), dtype=np.uint32)

for patient_id, bbox_array in bbox_arrays.items():
    # add to all group
    groups_to_bbox_sums['all'] += bbox_array

    # add to each other group where id is in that group's id set
    for group, id_set in groups_to_ids.items():
        if patient_id in id_set:
            groups_to_bbox_sums[group] += bbox_array


# In[ ]:


def plot_density(array, ax, title, n_countour_levels=3):
    contour_set = ax.contour(
        np.arange(0, 1024, 1), 
        np.arange(1024, 0, -1),
        array, 
        n_countour_levels, 
        linewidths=.5,
        colors='black'
    )
    plt.clabel(contour_set, inline=True, fontsize=10, fmt='%.0f')

    im = ax.imshow(
        array, 
        extent=[0, 1024, 0, 1024], 
        origin='upper', 
        cmap='viridis', 
        alpha=.8
    )
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    return im


# ## Bounding Box Plots
# 
# ### Takeaways from below plots
# * Right lung tends to have more in whole group
# * Right lung tendency seems more true in Age > 60 than the other age groups
# * Right lung tendency seems more true in PA than in AP (lungs close to balanced)
# * Right lung tendency seems more true in patients with 1 bbox vs 2/3/4

# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(14, 6), sharex=True)
_ = plot_density(groups_to_bbox_sums['all'], axes, 'All Targets - Bounding Box Density')


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

_ = plot_density(groups_to_bbox_sums['pa'], axes[0], 'PA - Bounding Box Density')
_ = plot_density(groups_to_bbox_sums['ap'], axes[1], 'AP - Bounding Box Density')


# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(12, 12))

_ = plot_density(groups_to_bbox_sums['bbox_1'], axes[0, 0], '1 Box - Bounding Box Density', n_countour_levels=3)
_ = plot_density(groups_to_bbox_sums['bbox_2'], axes[0, 1], '2 Boxes - Bounding Box Density', n_countour_levels=3)
_ = plot_density(groups_to_bbox_sums['bbox_3'], axes[1, 0], '3 Boxes - Bounding Box Density', n_countour_levels=1)
_ = plot_density(groups_to_bbox_sums['bbox_4'], axes[1, 1], '4 Boxes - Bounding Box Density', n_countour_levels=1)


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

_ = plot_density(groups_to_bbox_sums['f'], axes[0], 'PatientSex - F', n_countour_levels=3)
_ = plot_density(groups_to_bbox_sums['m'], axes[1], 'PatientSex - M', n_countour_levels=3)


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(18, 6))

_ = plot_density(groups_to_bbox_sums['age_above_60'], axes[0], 'PatientAge > 60', n_countour_levels=3)
_ = plot_density(groups_to_bbox_sums['age_40_to_60'], axes[1], '40 <= PatientAge <= 60', n_countour_levels=3)
_ = plot_density(groups_to_bbox_sums['age_below_40'], axes[2], 'PatientAge < 40', n_countour_levels=3)


# ### Image Intensity Distribution

# In[ ]:


def mean_intensity(patientId):
    img = read_img(patientId)
    return np.mean(img.pixel_array)


# In[ ]:


test_patients = df['PatientID'][df['class'].isnull()].tolist()
train_patients = df['PatientID'][df['class'].isnull()].tolist()

test_mean_intensity = Parallel(n_jobs=4)(delayed(mean_intensity)(patientId) for patientId in test_patients)
train_mean_intensity = Parallel(n_jobs=4)(delayed(mean_intensity)(patientId) for patientId in train_patients)
all_mean_intensity = test_mean_intensity + train_mean_intensity

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
p = sns.distplot(all_mean_intensity, ax=axes[0], kde=False)
_ = p.set_title('All - Image Mean Intensity')
_ = p.set_xlabel('Mean Intensity')
_ = p.set_ylabel('Count')

p = sns.distplot(train_mean_intensity, ax=axes[1], kde=False)
_ = p.set_title('Train - Image Mean Intensity')
_ = p.set_xlabel('Mean Intensity')
_ = p.set_ylabel('Count')

p = sns.distplot(test_mean_intensity, ax=axes[2], kde=False)
_ = p.set_title('Test - Image Mean Intensity')
_ = p.set_xlabel('Mean Intensity')
_ = p.set_ylabel('Count')


# In[ ]:


pa_mean_intensity = Parallel(n_jobs=4)(delayed(mean_intensity)(patientId) for patientId in groups_to_ids['pa'])
ap_mean_intensity = Parallel(n_jobs=4)(delayed(mean_intensity)(patientId) for patientId in groups_to_ids['ap'])

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
p = sns.distplot(pa_mean_intensity, ax=axes[0], kde=False)
_ = p.set_title('PA - Image Mean Intensity')
_ = p.set_xlabel('Mean Intensity')
_ = p.set_ylabel('Count')
p = sns.distplot(ap_mean_intensity, ax=axes[1], kde=False)
_ = p.set_title('AP - Image Mean Intensity')
_ = p.set_xlabel('Mean Intensity')
_ = p.set_ylabel('Count')


# In[ ]:





# ## Simple LGBM Binary Classifier

# In[ ]:


df_ = df_deduped[~df_deduped['Target'].isnull()].reset_index(drop=True)
df_['patient_sex'] = df_['PatientSex'].replace({'F': 0, 'M': 1}).astype(np.int8)
df_['view_position'] = df_['ViewPosition'].replace({'AP': 0, 'PA': 1}).astype(np.int8)


# In[ ]:


cv = KFold(n_splits=10, random_state=2018, shuffle=True)
oof = np.zeros(len(df_))
feats = ['patient_sex', 'PatientAge', 'view_position', 'PixelSpacing_x']

for n_fold, (train_idx, val_idx) in enumerate(cv.split(df_[feats], df_['Target'])):
    train_x, train_y = df_[feats].loc[train_idx], df_['Target'].loc[train_idx]
    val_x, val_y = df_[feats].loc[val_idx], df_['Target'].loc[val_idx]

    clf = LGBMClassifier(
        nthread=0,
        n_estimators=10000,
        learning_rate=.001,
        num_leaves=16,
        colsample_bytree=.75,
        subsample=.75,
        max_depth=5,
        reg_alpha=3,
        reg_lambda=3,
        min_child_weight=50,
        silent=-1,
        verbose=-1,
    )

    clf.fit(
        train_x, 
        train_y, 
        eval_set=[(train_x, train_y), (val_x, val_y)], 
        eval_metric='auc', 
        verbose=500, 
        early_stopping_rounds=200,
    )

    oof[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    score = roc_auc_score(val_y, oof[val_idx])
    print(f'\nFold {n_fold + 1} AUC : {score}\n')

score = roc_auc_score(df_['Target'], oof)
print(f'\nAUC on All Folds : {score}\n')


# In[ ]:




