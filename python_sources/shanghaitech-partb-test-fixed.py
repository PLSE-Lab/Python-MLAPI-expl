# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


__author__ = "Thai Thien"
__email__ = "tthien@apcs.vn"

import os
# from tensorflow.keras.preprocessing import image
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import glob
import h5py
import time
from sklearn.externals.joblib import Parallel, delayed
import sys
from PIL import Image
import argparse





__DATASET_ROOT = "../input/shanghaitech/ShanghaiTech/"
__OUTPUT_NAME = "ShanghaiTech_PartB_Test_fixed/"


# __DATASET_ROOT = 
# __OUTPUT_NAME =
__PART = "b_test"

def gaussian_filter_density_fixed(gt, sigma):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        density += gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density




def generate_density_map(img_path):
    print(img_path)
    mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_')
    print('mat_path ', mat_path)
    mat = scipy.io.loadmat(mat_path)
    imgfile = Image.open(img_path).convert('RGB')
    img = np.asarray(imgfile)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["image_info"][0, 0][0, 0][0]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter_density_fixed(k, 15)
    output_path = img_path.replace(__DATASET_ROOT, __OUTPUT_NAME).replace('.jpg', '.h5').replace('images','ground-truth-h5')
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    print("output", output_path)
    sys.stdout.flush()
    with h5py.File(output_path, 'w') as hf:
        hf['density'] = k
    return img_path


def generate_shanghaitech_path(root):
    # now generate the ShanghaiA's ground truth
    part_A_train = os.path.join(root, 'part_A/train_data', 'images')
    part_A_test = os.path.join(root, 'part_A/test_data', 'images')
    part_B_train = os.path.join(root, 'part_B/train_data', 'images')
    part_B_test = os.path.join(root, 'part_B/test_data', 'images')
    path_sets = [part_A_train, part_A_test, part_B_train, part_B_test]

    img_paths_a_train = []
    img_paths_a_test = []
    img_paths_b_train = []
    img_paths_b_test = []

    for img_path in glob.glob(os.path.join(part_A_train, '*.jpg')):
        img_paths_a_train.append(img_path)
    for img_path in glob.glob(os.path.join(part_B_train, '*.jpg')):
        img_paths_b_train.append(img_path)

    for img_path in glob.glob(os.path.join(part_A_test, '*.jpg')):
        img_paths_a_test.append(img_path)
    for img_path in glob.glob(os.path.join(part_B_test, '*.jpg')):
        img_paths_b_test.append(img_path)

    return img_paths_a_train, img_paths_a_test, img_paths_b_train, img_paths_b_test


if __name__ == "__main__":
    """
    TODO: this file will preprocess crowd counting dataset
    """

    start_time = time.time()
    a_train, a_test, b_train, b_test = generate_shanghaitech_path(__DATASET_ROOT)

    if __PART == "a_train":
        Parallel(n_jobs=4)(delayed(generate_density_map)(p) for p in a_train)
    elif __PART == "b_train":
        Parallel(n_jobs=4)(delayed(generate_density_map)(p) for p in b_train)
    elif __PART == "a_test":
        Parallel(n_jobs=4)(delayed(generate_density_map)(p) for p in a_test)
    elif __PART == "b_test":
        Parallel(n_jobs=4)(delayed(generate_density_map)(p) for p in b_test)

    print("--- %s seconds ---" % (time.time() - start_time))
