#!/usr/bin/env python
# coding: utf-8

# It aggregates several public tile strategies and provides pipeline for finding optimal SIZE, COLS, ROWS params form the human perspective. It also shows how to save the results.
# 
# derives from
# * https://www.kaggle.com/iafoss/panda-16x128x128-tiles
# * https://www.kaggle.com/akensert/panda-optimized-tiling-tf-data-dataset
# * https://www.kaggle.com/debanga/let-s-enhance-the-images
# * https://www.kaggle.com/harupy/visualization-panda-16x128x128-tiles
# * https://www.kaggle.com/raghaw/panda-medium-resolution-dataset-25x256x256
# 
# output dataset examples
# * 6x6 256 lafoss_tiles https://www.kaggle.com/dlarionov/prostate-cancer-tiles-medium
# * 4x4 312 akensert_tiles https://www.kaggle.com/dlarionov/akensert-1-312-4x4 (this kernel result)

# In[ ]:


import os
import numpy as np
import pandas as pd
import warnings
import math
from multiprocessing import Pool
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from openslide import OpenSlide
import cv2

IMG_DIR = '../input/prostate-cancer-grade-assessment/train_images/'

LAYER = 1 # medium
SIZE = 312
COLS = 4
ROWS = 4
N = COLS*ROWS


# In[ ]:


df = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')
df


# In[ ]:


def split_tiles(img:np.ndarray)->np.ndarray:
    reshaped = img.reshape(
        img.shape[0] // SIZE,
        SIZE,
        img.shape[1] // SIZE,
        SIZE,
        3,
    )
    transposed = reshaped.transpose(0, 2, 1, 3, 4)
    return transposed.reshape(-1, SIZE, SIZE, 3)

def join_tiles(img:np.ndarray)->np.ndarray:
    reshaped = img.reshape(
        COLS,
        ROWS,    
        img.shape[1],
        img.shape[2],
        3
    )
    transposed = reshaped.transpose(0, 2, 1, 3, 4)
    return transposed.reshape(COLS * SIZE, ROWS * SIZE, 3)

def lafoss_tiles(img:np.ndarray)->np.ndarray:
    
    # calculate paddings
    H, W, _ = img.shape
    pad_w = (SIZE - W % SIZE) % SIZE
    pad_h = (SIZE - H % SIZE) % SIZE
    
    # implement padding
    padded = np.pad(
        img,
        [[pad_h // 2, pad_h - pad_h // 2],
         [pad_w // 2, pad_w - pad_w // 2],
         [0, 0]],
        constant_values=255, # 255 - white
    )
    
    # split image into tiles
    tiles = split_tiles(padded)
    
    # calculate sums of all pixsels for each tile
    sums = tiles.reshape(tiles.shape[0], -1).sum(axis=-1)
    
    # take top N tiles by minimum sum value
    idxs_selected = np.argsort(sums)[:N]
    selected = tiles[idxs_selected]
    
    # append white tiles if necessary
    if len(selected)<N:
        selected = np.pad(
            selected,
            [[0,N-len(selected)],[0,0],[0,0],[0,0]],
            constant_values=255
        )
    
    # join selected tiles into one image
    merged = join_tiles(selected)

    return merged

def akensert_tiles(img:np.ndarray, debug=False)->np.ndarray:    
    
    # get tile coords
    img, coords = compute_coords(
        img,
        patch_size=SIZE,
        precompute=False, # returns new padded img
        min_patch_info=0.35,
        min_axis_info=0.35,
        min_consec_axis_info=0.35,
        min_decimal_keep=0.7)
    
    # sort coords (high info -> low info)
    coords = sorted(coords, key= lambda x: x[0], reverse=False)
    
    # select top N tiles
    tiles = []
    for i in range(len(coords)):
        if i == N:
            break;
        _, x, y = coords[i]
        tiles.append(img[x:x+SIZE,y:y+SIZE])
    
    # append white tiles if necessary
    selected = np.array(tiles)
    if len(selected)<N:
        selected = np.pad(
            selected,
            [[0,N-len(selected)],[0,0],[0,0],[0,0]],
            constant_values=255
        )
    
    # merge tiles to one image
    merged = join_tiles(selected)
    
    if debug:
        for (v, y, x) in coords:
            img = cv2.rectangle(img, (x, y), (x+SIZE, y+SIZE), color=(0, 0, 0), thickness=5)
            img = cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
            img = cv2.circle(img, (x+SIZE, y+SIZE), radius=5, color=(0, 255, 0), thickness=-1)
        return merged, img
    else:
        return merged


# In[ ]:


# copypaste https://www.kaggle.com/akensert/panda-optimized-tiling-tf-data-dataset
# copypaste https://www.kaggle.com/debanga/let-s-enhance-the-images

def enhance_image(image, contrast=1, brightness=15):
    """
    Enhance constrast and brightness of images
    """
    img_enhanced = cv2.addWeighted(image, contrast, image, 0, brightness)
    return img_enhanced

def unsharp_masking(img):
    """ Unsharp masking of an RGB image"""
    img_gaussian = cv2.GaussianBlur(img, (21,21), 10.0)
    return cv2.addWeighted(img, 1.8, img_gaussian, -0.8, 0, img)

def _mask_tissue(image, kernel_size=(7, 7), gray_threshold=220):
    """Masks tissue in image. Uses gray-scaled image, as well as
    dilation kernels and 'gap filling'
    """
    # Define elliptic kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    # Convert rgb to gray scale for easier masking
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Now mask the gray-scaled image (capturing tissue in biopsy)
    mask = np.where(gray < gray_threshold, 1, 0).astype(np.uint8)
    # Use dilation and findContours to fill in gaps/holes in masked tissue
    mask = cv2.dilate(mask, kernel, iterations=1)
    contour, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(mask, [cnt], 0, 1, -1)
    return mask

def _pad_image(image, pad_len, pad_val):
    """Pads inputted image, accepts both 
    2-d (mask) and 3-d (rgb image) arrays
    """
    if image is None:
        return None
    elif image.ndim == 2:
        return np.pad(
            image, ((pad_len, pad_len), (pad_len, pad_len)), pad_val)
    elif image.ndim == 3:
        return np.pad(
            image, ((pad_len, pad_len), (pad_len, pad_len), (0, 0)), pad_val)
    return None

def _transpose_image(image):
    """Inputs an image and transposes it, accepts 
    both 2-d (mask) and 3-d (rgb image) arrays
    """
    if image is None:
        return None
    elif image.ndim == 2:
        return np.transpose(image, (1, 0)).copy()
    elif image.ndim == 3:
        return np.transpose(image, (1, 0, 2)).copy()
    return None

def _get_tissue_parts_indices(tissue, min_consec_info):
    """If there are multiple tissue parts in 'tissue', 'tissue' will be 
    split. Each tissue part will be taken care of separately (later on), 
    and if the tissue part is less than min_consec_info, it's considered 
    to small and won't be returned.
    """
    split_points = np.where(np.diff(tissue) != 1)[0]+1
    tissue_parts = np.split(tissue, split_points)
    return [
        tp for tp in tissue_parts if len(tp) >= min_consec_info
    ]

def _get_tissue_subparts_coords(subtissue, patch_size, min_decimal_keep):
    """Inputs a tissue part resulting from '_get_tissue_parts_indices'.
    This tissue part is divided into N subparts and returned.
    Argument min_decimal_keep basically decides if we should reduce the
    N subparts to N-1 subparts, due to overflow.
    """
    start, end = subtissue[0], subtissue[-1]
    num_subparts = (end-start)/patch_size
    if num_subparts % 1 < min_decimal_keep and num_subparts >= 1:
        num_subparts = math.floor(num_subparts)
    else:
        num_subparts = math.ceil(num_subparts)

    excess = (num_subparts*patch_size) - (end-start)
    shift = excess // 2

    return [
        i * patch_size + start - shift 
        for i in range(num_subparts)
    ]

def _eval_and_append_xy_coords(coords,
                               image, 
                               mask, 
                               patch_size, 
                               x, y, 
                               min_patch_info,
                               transposed,
                               precompute):
    """Based on computed x and y coordinates of patch: 
    slices out patch from original image, flattens it,
    preprocesses it, and finally evaluates its mask.
    If patch contains more info than min_patch_info,
    the patch coordinates are kept, along with a value 
    'val1' that estimates how much information there 
    is in the patch. Smaller 'val1' assumes more info.
    """
    patch_1d = (
        image[y: y+patch_size, x:x+patch_size, :]
        .mean(axis=2)
        .reshape(-1)
    )
    idx_tissue = np.where(patch_1d <= 210)[0]
    idx_black = np.where(patch_1d < 5)[0]
    idx_background = np.where(patch_1d > 210)[0]

    if len(idx_tissue) > 0:
        patch_1d[idx_black] = 210
        patch_1d[idx_background] = 210
        val1 = int(patch_1d.mean())
        val2 = mask[y:y+patch_size, x:x+patch_size].mean()
        if val2 > min_patch_info:
            if precompute:
                if transposed:
                    coords = np.concatenate([
                        coords, [[val1, x-patch_size, y-patch_size]]
                    ])
                else:
                    coords = np.concatenate([
                        coords, [[val1, y-patch_size, x-patch_size]]
                    ])
            else:
                coords = np.concatenate([
                    coords, [[val1, y, x]]
                ])
               
    return coords

def compute_coords(image,
                   patch_size=256,
                   precompute=False,
                   min_patch_info=0.35,
                   min_axis_info=0.35,
                   min_consec_axis_info=0.35,
                   min_decimal_keep=0.7):

    """
    Input:
        image : 3-d np.ndarray
        patch_size : size of patches/tiles, will be of 
            size (patch_size x patch_size x 3)
        precompute : If True, only coordinates will be returned,
            these coordinates match the inputted 'original' image.
            If False, both an image and coordinates will be returned,
            the coordinates does not match the inputted image but the
            image that it is returned with.
        min_patch_info : Minimum required information in patch
            (see '_eval_and_append_xy_coords')
        min_axis_info : Minimum fraction of on-bits in x/y dimension to be 
            considered enough information. For x, this would be fraction of 
            on-bits in x-dimension of a y:y+patch_size slice. For y, this would 
            be the fraction of on-bits for the whole image in y-dimension
        min_consec_axis_info : Minimum consecutive x/y on-bits
            (see '_get_tissue_parts_indices')
        min_decimal_keep : Threshold for decimal point for removing "excessive" patch
            (see '_get_tissue_subparts_coords')
    
    Output:
        image [only if precompute is False] : similar to input image, but fits 
            to the computed coordinates
        coords : the coordinates that will be used to compute the patches later on
    """
    
    
    if type(image) != np.ndarray:
        # if image is a Tensor
        image = image.numpy()
    
    # masked tissue will be used to compute the coordinates
    mask = _mask_tissue(image)

    # initialize coordinate accumulator
    coords = np.zeros([0, 3], dtype=int)

    # pad image and mask to make sure no tissue is potentially missed out
    image = _pad_image(image, patch_size, 'maximum')
    mask = _pad_image(mask, patch_size, 'minimum')
    
    y_sum = mask.sum(axis=1)
    x_sum = mask.sum(axis=0)
    # if on bits in x_sum is greater than in y_sum, the tissue is
    # likely aligned horizontally. The algorithm works better if
    # the image is aligned vertically, thus the image will be transposed
    if len(np.where(x_sum > 0)[0]) > len(np.where(y_sum > 0)[0]):
        image = _transpose_image(image)
        mask = _transpose_image(mask)
        y_sum, _ = x_sum, y_sum
        transposed = True
    else:
        transposed = False
    
    # where y_sum is more than the minimum number of on-bits
    y_tissue = np.where(y_sum >= (patch_size*min_axis_info))[0]
    
    if len(y_tissue) < 1:
        warnings.warn("Not enough tissue in image (y-dim)", RuntimeWarning)
        if precompute: return [(0, 0, 0)]
        else: return image, [(0, 0, 0)]
    
    y_tissue_parts_indices = _get_tissue_parts_indices(
        y_tissue, patch_size*min_consec_axis_info)
    
    if len(y_tissue_parts_indices) < 1: 
        warnings.warn("Not enough tissue in image (y-dim)", RuntimeWarning)
        if precompute: return [(0, 0, 0)]
        else: return image, [(0, 0, 0)]
    
    # loop over the tissues in y-dimension
    for yidx in y_tissue_parts_indices:
        y_tissue_subparts_coords = _get_tissue_subparts_coords(
            yidx, patch_size, min_decimal_keep)
        
        for y in y_tissue_subparts_coords:
            # in y_slice, where x_slice_sum is more than the minimum number of on-bits
            x_slice_sum = mask[y:y+patch_size, :].sum(axis=0)
            x_tissue = np.where(x_slice_sum >= (patch_size*min_axis_info))[0]
            
            x_tissue_parts_indices = _get_tissue_parts_indices(
                x_tissue, patch_size*min_consec_axis_info)
            
            # loop over tissues in x-dimension (inside y_slice 'y:y+patch_size')
            for xidx in x_tissue_parts_indices:
                x_tissue_subparts_coords = _get_tissue_subparts_coords(
                    xidx, patch_size, min_decimal_keep)
                
                for x in x_tissue_subparts_coords:
                    coords = _eval_and_append_xy_coords(
                        coords, image, mask, patch_size, x, y, 
                        min_patch_info, transposed, precompute
                    )     
    
    if len(coords) < 1:
        warnings.warn("Not enough tissue in image (x-dim)", RuntimeWarning)
        if precompute: return [(0, 0, 0)]
        else: return image, [(0, 0, 0)]
    
    if precompute: return coords
    else: return image, coords


# In[ ]:


def imread(path:str, layer:int)->Image:
    if not os.path.exists(path):
        return None

    with OpenSlide(path) as slide:
        im = slide.read_region((0,0), layer, slide.level_dimensions[layer])
        im = im.convert('RGB') # drops A
        return im


# In[ ]:


for i in range(12):
    f, axarr = plt.subplots(1, 3, figsize=(20,20))
    idx = np.random.randint(0, len(df))        
    row = df.loc[idx]
    im = imread(f"{IMG_DIR}{row['image_id']}.tiff", layer=LAYER)        
    arr = np.asarray(im)
    arr = enhance_image(arr)
    arr = unsharp_masking(arr)
    tiles1, img_map = akensert_tiles(arr, debug=True)
    tiles2 = lafoss_tiles(arr)    
    axarr[0].set_title(f'id:{idx}')
    axarr[0].imshow(img_map.squeeze())
    axarr[1].set_title('akensert')
    axarr[1].imshow(tiles1.squeeze())
    axarr[2].set_title('lafoss')
    axarr[2].imshow(tiles2.squeeze())        


# Any kernel provides its own file system where you can save up to 4Tb while kernel is alive. 5Gb restriction applies onlty to /kaggle/working directory.
# Then you can use https://github.com/Kaggle/kaggle-api to create new kaggle dataset.

# In[ ]:


TILES_DIR = '/tmp/tiles'
get_ipython().system('mkdir -p {TILES_DIR}')

def process_image(idx):    
    row = df.loc[idx]
    im = imread(os.path.join(IMG_DIR, f"{row.image_id}.tiff"), layer=LAYER)
    im = np.asarray(im)
    im = akensert_tiles(im)
    im = Image.fromarray(im)
    im.save(os.path.join(TILES_DIR, f"{row.image_id}.jpg"), format='JPEG', quality=90)

batch = df#.head(100)
with Pool(processes=4) as pool:
    res = list(
        tqdm(pool.imap(process_image, list(batch.index)), total = len(batch))
    )


# In[ ]:


DS_DIR = '/tmp/dataset/'
get_ipython().system('mkdir -p {DS_DIR}/tiles')
get_ipython().system('tar czf {DS_DIR}/tiles/tiles.tar.gz -C {TILES_DIR} .')


# mykaggleapi is a private dataset. https://github.com/Kaggle/kaggle-api for more.

# In[ ]:


get_ipython().system('mkdir -p /root/.kaggle/')
get_ipython().system('cp ../input/mykaggleapi/kaggle.json /root/.kaggle/')
get_ipython().system('chmod 600 /root/.kaggle/kaggle.json')


# In[ ]:


get_ipython().system('kaggle datasets init -p {DS_DIR}')


# In[ ]:


import json

with open(f'{DS_DIR}/dataset-metadata.json', 'r+') as f:
    data = json.load(f)
    data['title'] = f'akensert {LAYER} {SIZE} {COLS}x{ROWS} '
    data['id'] = f'dlarionov/akensert-{LAYER}-{SIZE}-{COLS}x{ROWS}'
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()

get_ipython().system('cat {DS_DIR}/dataset-metadata.json')


# In[ ]:


get_ipython().system('kaggle datasets create -p {DS_DIR} -q -r tar')
get_ipython().system('rm -rf {TILES_DIR}')
get_ipython().system('rm -rf {DS_DIR}')

