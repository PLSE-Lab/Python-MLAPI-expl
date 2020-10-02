# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import os
from tensorflow.keras.preprocessing import image
import numpy as np
import scipy
from scipy.io import loadmat
import glob
import h5py
import time
from sklearn.externals.joblib import Parallel, delayed
import sys


__DATASET_ROOT = "../input/ucfcrowdcountingdataset_cvpr13/UCF_CC_50/"
__OUTPUT_NAME = "UCF_CC_50/ground-truth-h5/"

def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    pts_copy = pts.copy()
    tree = scipy.spatial.KDTree(pts_copy, leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density


def generate_density_map(img_path):
    print(img_path)
    mat_path = img_path.replace('.jpg', '_ann.mat')
    print('mat_path ', mat_path)
    mat = scipy.io.loadmat(mat_path)
    imgfile = image.load_img(img_path)
    img = image.img_to_array(imgfile)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["annPoints"]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter_density(k)
    output_path = img_path.replace(__DATASET_ROOT, __OUTPUT_NAME).replace('.jpg', '.h5')
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    print("output", output_path)
    sys.stdout.flush()
    with h5py.File(output_path, 'w') as hf:
        hf['density'] = k
    return img_path
    
def generate_path(root):
    paths = []
    for img_path in glob.glob(os.path.join(root, '*.jpg')):
        paths.append(img_path)
    return paths
    
if __name__ == "__main__":
    img_paths = generate_path(__DATASET_ROOT)
    Parallel(n_jobs=4)(delayed(generate_density_map)(p) for p in img_paths)
