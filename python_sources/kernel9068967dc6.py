# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# # print(os.listdir("../input"))

import os
os.chdir('/kaggle/input')
print(os.listdir())


# import os	
import random
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf


img_h = 360
img_w = 640
batch_size = 16

def iou(box1, box2):
    xmin = K.maximum(box1[0], box2[0])
    ymin = K.maximum(box1[1], box2[1])
    xmax = K.minimum(box1[2], box2[2])
    ymax = K.minimum(box1[3], box2[3])

    w = K.maximum(0.0, xmax - xmin)
    h = K.maximum(0.0, ymax - ymin)

    intersection = w * h

    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]

    union = w1 * h1 + w2 * h2 - intersection

    return intersection/union * 100

def batch_iou():
    def batch_iou_2(y_true, y_pred):
        list_of_iou = []
        result = 0
        for i in range(batch_size):
            list_of_iou.append(iou(y_true[i],y_pred[i]))
        return K.mean(tf.convert_to_tensor(list_of_iou, dtype = tf.float32))
    return batch_iou_2

with CustomObjectScope({'GlorotUniform':glorot_uniform(), 'batch_iou_2': batch_iou(), 'BatchNormalizationV1': BatchNormalization()}):
    model = load_model('my_model_v3.h5')

model.summary()


# data
img_paths = open('/kaggle/input/img_test.txt').read().split()
label_paths = open('/kaggle/input/lab_test.txt').read().split()





def load_label(label_f):
    line = open(label_f).read().split('\n')
    
    label = line[0].split(' ')
    
    return np.asarray(label, dtype='float32')


def load_batch(img_paths, label_paths):

    images = np.zeros((batch_size, img_h, img_w, 3))
    batch_labels = np.zeros((batch_size,4))

    indx = 0
    
    for imgFile, labelFile in zip(img_paths, label_paths):
        img = cv2.imread(imgFile).astype(np.float32, copy=False)
        images[indx] = img

        label = load_label(labelFile)
        batch_labels[indx] = label

        indx +=1

    return images, batch_labels
        

def generator(img_names, gt_names, batch_size):
    # Create empty arrays to contain batch of features and labels#

    assert len(img_names) == len(gt_names), "Number of images and ground truths not equal"

    nbatches, n_skipped_per_epoch = divmod(len(img_names), batch_size)

    if True:
        #permutate images
        shuffled = list(zip(img_names, gt_names))
        random.shuffle(shuffled)
        img_names, gt_names = zip(*shuffled)

    nbatches, n_skipped_per_epoch = divmod(len(img_names), batch_size)

    count = 1
    epoch = 0

    while 1:

        epoch += 1
        i, j = 0, batch_size

        #mini batches within epoch
        mini_batches_completed = 0

        for _ in range(nbatches):
            #print(i,j)
            img_names_batch = img_names[i:j]
            gt_names_batch = gt_names[i:j]

            try:
                #get images and ground truths
                imgs, gts = load_batch(img_names_batch, gt_names_batch)

                mini_batches_completed += 1
                yield (imgs, gts)

            except IOError as err:

                count -= 1

            i = j
            j += batch_size
            count += 1


train_generator = generator(img_paths, label_paths, batch_size)
import time
start = time.time()
results = model.evaluate_generator(train_generator, steps = len(img_paths)//batch_size, verbose=1)
stop = time.time() - start
print(model.metrics_names)
print(results)
print(stop)