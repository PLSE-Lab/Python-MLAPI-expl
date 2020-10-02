# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


def featureImage(image_id, path=''):
    f = misc.imread(path+image_id+'.jpg', flatten=False)
    l = f.shape[0]
    array_image = f.flatten()

    # Les couleurs de chaque pixel sont code sur 3 nombres
    array_image_mean = []
    for i in range(0, len(array_image), 3):
        array_image_mean.append(
            (array_image[i]+array_image[i+1]+array_image[i+2])/3)

    n = len(array_image_mean)
    image_features = []
    for i in range(0, n):
        # left pixel
        if i >= l:
            left = array_image_mean[i-l]
        else:
            left = 0

        # right pixel
        if i < n-l:
            right = array_image_mean[i+l]
        else:
            right = 0

        # top pixel
        if i % l != 0:
            top = array_image_mean[i-1]
        else:
            top = 0

        # down pixel
        if (i+1) % l != 0:
            down = array_image_mean[i+1]
        else:
            down = 0

        image_features.append([array_image_mean[i], left, right, top, down])
    return np.array(image_features)


def IoU(A, B):
    """Calculate the Intersection of Union (IoU) of A and B.

    Args:
        A (set) : the first set representing predicted mask.
        B (set) : the second set representing the target mask
    Returns:
        the IoU of A and B.
    Raises:

    """
    if len(A.union(B)) != 0:
        return len(A & B)/len(A.union(B))
    else:
        return 0


def IoU_mask(mask_A, mask_B, treshold):
    """return True if the intersection between mask_A and mask_B is at least 
       equal to treshold.

    Args:
        mask_A (string) : the predicted object in run length format (predicted).
        mask_B (string) : the target object in run length format (predicted).
    Returns:
        True if the intersection between mask_A and mask_B is at least 
       equal to treshold.

    Raises:


    """
    # transform mask_A into a list
    mask_A = mask_A.split()
    mask_A = [int(i) for i in mask_A]

    # transform mask_B into a list
    mask_B = mask_B.split()
    mask_B = [int(i) for i in mask_B]

    # transform mask_A into a set of pixels
    set_A = set()
    for i in range(0, len(mask_A), 2):
        for j in range(mask_A[i], mask_A[i]+mask_A[i+1]):
            set_A.add(j)

    # transform mask_B into a set of pixels
    set_B = set()
    for i in range(0, len(mask_B), 2):
        for j in range(mask_B[i], mask_B[i]+mask_B[i+1]):
            set_B.add(j)

    return IoU(set_A, set_B) >= treshold


def average_precision(predicted_masks, true_masks, tresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)):
    """return the average precision of an image.

    Args:
        predicted_masks (list of strings) : list of string each representing a predicted mask.
        true_masks (list of strings)      : list of string each representing a true mask.
    Returns:
        the average precision of an image.

    Raises:


    """
    total_score = 0
#    tresholds =(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    true_positives = []
    false_positives = []
    false_negatives = []
    for t in tresholds:
        tp = 0
        fp = 0
        fn = 0
        # Counting true positives and false positives objects
        for w1 in predicted_masks:
            hit = False
            for w2 in true_masks:
                if IoU_mask(w1, w2, t) == True:
                    tp = tp+1
                    true_positives.append([t, w1])
                    hit = True
                    break
            if hit == False:
                fp = fp+1
                false_positives.append([t, w1])

        # Counting false negatives objects
        for w2 in true_masks:
            hit = False
            for w1 in predicted_masks:
                if IoU_mask(w2, w1, t) == True:
                    hit = True
                    break
            if hit == False:
                fn = fn+1
                false_negatives.append([t, w2])
        total_score = total_score+tp/(tp+fp+fn)

#        print('Treshold : ',t,'True positives :' ,tp, 'False positives : ', fp, 'False negatives : ',fn)
    return total_score/len(tresholds), true_positives, false_positives, false_negatives















