#!/usr/bin/env python
# coding: utf-8

# <a id="toc"></a>
# # Table of Contents
# 1. [Import libraries](#import_libraries)
# 1. [Configure hyper-parameters](#configure_hyper_parameters)
# 1. [Define helper-functions](#define_helper_functions)
# 1. [Resize images and corresponding bboxes](#resize_images_and_corresponding_bboxes)
# 1. [Save and compress the results](#save_and_compress_the_result)

# <a id="import_libraries"></a>
# # Import libraries
# [Bach to Table of Contents](#toc)

# In[ ]:


import pathlib
from pathlib import Path
import json

import numpy as np
import pandas as pd
import cv2
import albumentations as A
from tqdm import tqdm


# <a id="configure_hyper_parameters"></a>
# # Configure hyper-parameters
# [Bach to Table of Contents](#toc)

# In[ ]:


ROOT = Path('/kaggle/input/global-wheat-detection/')
TRAIN_DIR = ROOT / 'train'
TEST_DIR = ROOT / 'test'

WORKING_DIR = Path('/kaggle/working/')

IMG_SIZE = 224


# <a id="define_helper_functions"></a>
# # Define helper-functions
# [Bach to Table of Contents](#toc)

# In[ ]:


def load_dataframe(csv_path: pathlib.PosixPath, image_dir: pathlib.PosixPath) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    
    # Merge all bboxes of each corresponding image
    # Format: [[x1 y1 w1 h1], [x2 y2 w2 h2], [x3 y3 w3 h3], ...]
    df.bbox = df.bbox.apply(lambda x: ' '.join(np.array(json.loads(x), dtype=str)))
    df.bbox = df.groupby(['image_id']).bbox.transform(lambda x: '|'.join(x))
    df.drop_duplicates(inplace=True, ignore_index=True)
    df.bbox = df.bbox.apply(lambda x: np.array([item.split(' ') for item in x.split('|')], dtype=np.float32).tolist())
    
    # Create a path to each image
    df['image_path'] = df.image_id.apply(lambda x: str(image_dir / (x + '.jpg')))
    
    return df

def load_image(image_path: str) -> np.array:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    return image

def fix_out_of_range(bbox: list, max_size: int = 1024) -> list:
    bbox[2] = min(bbox[2], max_size - bbox[0])
    bbox[3] = min(bbox[3], max_size - bbox[1])

    return bbox


# In[ ]:


df = load_dataframe(ROOT / 'train.csv', TRAIN_DIR)


# In[ ]:


df


# <a id="resize_images_and_corresponding_bboxes"></a>
# # Resize images and corresponding bboxes
# [Bach to Table of Contents](#toc)

# In[ ]:


mkdir train


# In[ ]:


transform = A.Compose(
    [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
    ], 
    p=1.0, 
    bbox_params=A.BboxParams(
        format='coco',
        min_area=0, 
        min_visibility=0,
        label_fields=['labels']
    )
)

list_of_image_ids = []
list_of_bboxes = []
list_of_sources = []

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    image = load_image(row.image_path)
    bboxes = row.bbox

    # Fix "out-of-range" bboxes
    bboxes = [fix_out_of_range(bbox) for bbox in bboxes]
    
    result = transform(image=image, bboxes=bboxes, labels=np.ones(len(bboxes)))
    new_image = result['image']
    new_bboxes = np.array(result['bboxes']).tolist()
    
    # Save new image
    cv2.imwrite(str(WORKING_DIR / 'train' / (row.image_id + '.jpg')), new_image)

    for new_bbox in new_bboxes:
        list_of_image_ids.append(row.image_id)
        list_of_bboxes.append(new_bbox)
        list_of_sources.append(row.source)


# In[ ]:


new_data_dict = {
    'image_id': list_of_image_ids,
    'width': [IMG_SIZE] * len(list_of_image_ids),
    'height': [IMG_SIZE] * len(list_of_image_ids),
    'bbox': list_of_bboxes,
    'source': list_of_sources
}


# In[ ]:


new_df = pd.DataFrame(new_data_dict)


# <a id="save_and_compress_the_result"></a>
# # Save and compress the results
# [Bach to Table of Contents](#toc)

# In[ ]:


new_df.to_csv('train.csv', index=False)


# In[ ]:


get_ipython().system('cp $ROOT/sample_submission.csv ./')
get_ipython().system('cp -r $ROOT/test ./')


# In[ ]:


get_ipython().system('zip -rm -qq global-wheat-detection.zip train test train.csv sample_submission.csv')

