#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import PIL
import cv2
import sklearn
from sklearn.model_selection import train_test_split
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import tqdm
import tensorflow as tf
import re
from itertools import zip_longest
import pickle
import random
import io
import matplotlib


dirname = "/kaggle/input/prostate-cancer-grade-assessment"
yolo_max_boxes = 20
yolo_score_threshold = 0.05

generate_yolo_ds = False
use_precomputed_yolo_ds = True
train_yolo = False
use_precomputed_yolo_model = True

#yolo code from: https://github.com/zzh8829/yolov3-tf2


# In[ ]:


def read_image(filename, mask = True, series = 1, split = "train"):
    with tifffile.TiffFile(dirname + "/" + split + "_images/" + filename + ".tiff") as infile:
        img = infile.asarray(series = series)
    if mask:
        with tifffile.TiffFile(dirname + "/" + split + "_label_masks/" + filename + "_mask.tiff") as infile:
            mask = infile.asarray(series = series)[:,:,0]
        return img, mask
    return img


# In[ ]:


def generate_patches(image, mask = None, patch_size = 416):
    xs = image.shape[0] // patch_size
    ys = image.shape[1] // patch_size
    for i in range(xs):
        for j in range(ys):
            img = image[(i*patch_size):((i+1)*patch_size), (j*patch_size):((j+1)*patch_size)]
            if mask is not None:
                msk = mask[(i*patch_size):((i+1)*patch_size), (j*patch_size):((j+1)*patch_size)]
                yield img, msk
            else:
                yield img


# In[ ]:


def get_masked_dataframe():
    d = pd.read_csv(dirname + "/train.csv")
    d = d[d['data_provider'] == "radboud"]
    masklist = os.listdir(dirname + "/train_label_masks/")
    masklist = [name[:-10] for name in masklist]
    d_masked = d.merge(pd.DataFrame({'image_id': masklist}))
    return d_masked


# In[ ]:


def get_box_level_area(box, mask, level):
    box_area = box[2] * box[3]
    box_mask = mask[box[1]:(box[1] + box[3]), box[0]:(box[0]+box[2])]
    level_area = len(box_mask[box_mask == level])
    #if level > 2:
    #    print(level)
    return np.array([level, box[0], box[1], box[2], box[3], level_area, box_area])
    
def get_boxes(mask_original, resampling_factor = 8, min_box_area = 500, level_to_box_min_ratio = 0.5, box_to_image_min_ratio = 0.05, levels = [2,3,4,5]):
    results = []
    for level in levels:
        mask = mask_original.copy()
        mask = np.array(PIL.Image.fromarray(mask).resize([mask.shape[i] // resampling_factor for i in range(mask.ndim)]).resize(mask_original.shape))
        mask[mask != level] = 0
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, stats = cv2.CC_STAT_AREA)
        #if level > 3:
        #    print("initial:", len(stats))
        stats = stats[np.where(stats[:,2] * stats[:,3] > min_box_area)]
        #if level > 3:
        #    print("min box filter:", len(stats))
        mask_area = np.prod(mask.shape[:2])
        stats = stats[np.where(stats[:,2] * stats[:,3] > mask_area * box_to_image_min_ratio), :][0]
        #if level > 3:
        #    print("mask area", len(stats))
        if len(stats) > 0:
            stats = np.apply_along_axis(get_box_level_area, 1, stats, mask, level)
            #if level > 3:
            #    print("beforelevelbox:", len(stats), stats)
            stats = stats[stats[:,5] > level_to_box_min_ratio * stats[:,6]]
            #if level > 3:
            #    print("levelbox:", len(stats))
            results.append(stats)
    if len(results) >= 1:
        results = np.concatenate(results, axis = 0)
    return results


# In[ ]:


def print_plots(img, mask):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(mask)


# In[ ]:


def create_class_file(classes = ['benign', 'grade3', 'grade4', 'grade5']):
    classes = "\n".join(classes)
    with open("/kaggle/working/annotations/classes.names", 'w') as ofile:
        ofile.write(classes)


# In[ ]:


get_ipython().run_line_magic('mkdir', '/kaggle/working/annotations')
classes = ['benign', 'grade3', 'grade4', 'grade5']
create_class_file()


# In[ ]:


def create_tfrecord_example(img, boxes, classes = ['benign', 'grade3', 'grade4', 'grade5']):
    height = img.shape[0]
    width = img.shape[1]
    classes_text = [classes[i].encode('utf-8') for i in list(boxes[:,0] - 2)]
    xmin = list(boxes[:,1]/width)
    ymin = list(boxes[:,2]/height)
    xmax = list((boxes[:,1] + boxes[:,3] - 1)/width)
    ymax = list((boxes[:,2] + boxes[:,4] - 1)/height)
    img = PIL.Image.fromarray(img)
    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='JPEG')
    imgByteArr = imgByteArr.getvalue()
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgByteArr])),
            'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
            'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
            'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
            'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
            'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        }))
    return example


# In[ ]:


def process_image(name, classes, patch_size = 416, split = "train", box_to_image_min_ratio = 0.05, level_to_box_min_ratio = 0.5, min_box_area = 100, writer = None):
    image_complete, mask_complete = read_image(name)
    examples = []
    for patch_number, (img, msk) in enumerate(generate_patches(image_complete, mask_complete, patch_size)):
        means = np.mean(img, axis = 2)
        if len(means[means > 230]) > 0.8 * (patch_size * patch_size):
            continue
        boxes = get_boxes(msk, box_to_image_min_ratio = box_to_image_min_ratio, 
                          level_to_box_min_ratio = level_to_box_min_ratio, min_box_area = min_box_area)
        #if name == "8aa35680beee94df712c211da5bda211" and patch_number == 22:
            #print_plots(img, msk)
        if boxes.shape[0] > 0:
            if len(boxes) >= 100:
                raise Exception("too many boxes: " + str(len(boxes)))
            example = create_tfrecord_example(img, boxes)
            if writer:
                writer.write(example.SerializeToString())
            else:
                examples.append(example)
    return examples


# In[ ]:


def parse_example(example):
    ymins = example.features.feature['image/object/bbox/ymin'].float_list.value
    ymaxs = example.features.feature['image/object/bbox/ymax'].float_list.value
    xmins = example.features.feature['image/object/bbox/xmin'].float_list.value
    xmaxs = example.features.feature['image/object/bbox/xmax'].float_list.value
    classes = example.features.feature['image/object/class/text'].bytes_list.value
    img = example.features.feature['image/encoded'].bytes_list.value
    nboxes = len(ymins)
    boxes = [[xmins[i] * patch_size, ymins[i] * patch_size, xmaxs[i] * patch_size, ymaxs[i] * patch_size] for i in range(nboxes)]
    classes = [c.decode('utf-8') for c in classes]
    img = tf.io.decode_jpeg(tf.cast(img[0], tf.string)).numpy()
    return boxes, classes, img


# In[ ]:


box_to_image_min_ratio = 0.02
level_to_box_min_ratio = 0.3
min_box_area = 30
patch_size = 416
train_filename = f"train_ps{patch_size}bir{box_to_image_min_ratio}_lbr{level_to_box_min_ratio}_mba{min_box_area}.tfrecord"
valid_filename = f"val_ps{patch_size}bir{box_to_image_min_ratio}_lbr{level_to_box_min_ratio}_mba{min_box_area}.tfrecord"


# In[ ]:


if generate_yolo_ds:
    d_masked = get_masked_dataframe()
    train, valid = sklearn.model_selection.train_test_split(d_masked, test_size = 0.2, random_state = 0, stratify = d_masked['isup_grade'])
    print(train.shape, valid.shape)

    writer = tf.io.TFRecordWriter(train_filename)
    for fname in tqdm.tqdm(train['image_id']):
        #print(train[train['image_id'] == fname]['isup_grade'])
        process_image(fname, classes = classes, patch_size = patch_size, split = "train", 
                      box_to_image_min_ratio = box_to_image_min_ratio, 
                      level_to_box_min_ratio = level_to_box_min_ratio, 
                      min_box_area = min_box_area, writer = writer)
    writer.close()     

    upload_to_gcs(train_filename, destination = "prostate")
    del train

    writer = tf.io.TFRecordWriter(valid_filename)
    for fname in tqdm.tqdm(valid['image_id']):
        #print(train[train['image_id'] == fname]['isup_grade'])
        process_image(fname, classes = classes, patch_size = 416, split = "train", 
                      box_to_image_min_ratio = box_to_image_min_ratio, 
                      level_to_box_min_ratio = level_to_box_min_ratio, 
                      min_box_area = min_box_area, writer = writer)
    writer.close()     
    upload_to_gcs(valid_filename, destination = "prostate")


# In[ ]:


if use_precomputed_yolo_ds:
    #train_records = tf.data.TFRecordDataset(os.path.join("/kaggle", "input", "precomputedprostate", train_filename))
    valid_records = tf.data.TFRecordDataset(os.path.join("/kaggle", "input", "precomputedprostate", valid_filename))


# In[ ]:


if generate_yolo_ds or use_precomputed_yolo_ds:
    d_masked = get_masked_dataframe()
    d_masked = d_masked[d_masked['isup_grade'] > 1]
    printed = False
    while (not printed):
        sample_img=random.randint(0, d_masked.shape[0] - 1)
        examples = process_image(d_masked.iloc[sample_img]['image_id'], classes = classes, patch_size = patch_size, split = "train", 
                              box_to_image_min_ratio = box_to_image_min_ratio, 
                              level_to_box_min_ratio = level_to_box_min_ratio, 
                              min_box_area = min_box_area, writer = None)
        if len(examples) > 1:
            rands = np.random.choice(range(len(examples)), min(4, len(examples)), replace = False)
            nfig = len(rands)
            fig, ax = plt.subplots(1, nfig, figsize = (20, 10))
            for i in range(nfig):
                boxes, class_preds, img = parse_example(examples[rands[i]])
                for j in range(len(boxes)):
                    box = boxes[j]
                    xmin, ymin = box[0], box[1]
                    xmax, ymax = box[2], box[3]
                    w, h = xmax - xmin, ymax - ymin
                    rect = matplotlib.patches.Rectangle((xmin, ymin), w, h, fill = None)
                    ax[i].add_patch(rect)
                    ax[i].text(xmin, ymin, class_preds[j])
                ax[i].imshow(img)
            printed = True


# In[ ]:


if train_yolo:
    get_ipython().run_line_magic('mkdir', 'yolo')
    get_ipython().run_line_magic('cd', 'yolo')
    get_ipython().system('git clone https://github.com/zzh8829/yolov3-tf2.git')
    get_ipython().run_line_magic('cd', 'yolov3-tf2')

    get_ipython().system("sed -i '179s/Early/#Early/g' train.py")
    get_ipython().system("sed -i '180s/{epoch}/x/g' train.py")
    #!sed -i '181s/verbose/#verbose/g' train.py

    download_from_gcs("/kaggle/working/yolo/yolov3-tf2/data/yolov3.weights", directory = "prostate")
    get_ipython().system('python convert.py --weights /kaggle/working/yolo/yolov3-tf2/data/yolov3.weights --output /kaggle/working/yolo/yolov3-tf2/checkpoints/yolov3.tf')

    get_ipython().system('python train.py --dataset /kaggle/working/$train_filename --classes /kaggle/working/annotations/classes.names --num_classes 4         --mode fit --transfer darknet --batch_size 32 --epochs 7 --weights /kaggle/working/yolo/yolov3-tf2/checkpoints/yolov3.tf         --weights_num_classes 80 --size 416 --val_dataset /kaggle/working/$valid_filename')


# In[ ]:


def generate_image_features(filename, model, split = "train", patch_size = 416, max_boxes = 100):
    img = read_image(filename, split = split, mask = False)
    #print(img.shape)
    #print((img.shape[0] // patch_size) * (img.shape[1] // patch_size))
    counts = np.zeros((1,9))
    patches = []
    for i, patch in enumerate(generate_patches(img, patch_size = patch_size)):
        #print(patch.shape)
        #print(patch)
        means = np.mean(patch, axis = 2)
        if len(means[means > 230]) > 0.8 * (patch_size * patch_size):
            #print("passing")
            pass
        else:
            patches.append(patch)
            #print("staying", len(means[means == 255]), 0.3 * (patch_size * patch_size))
    if len(patches) > 0:
        patches = np.stack(patches, axis = 0)
        patches = patches/255
        batch_size = 16
        boxes, classes, scores, nums = [], [], [], []
        for i in range(len(patches) // batch_size + 1):
            batch = patches[(i*batch_size):((i+1)*batch_size)]
            batch_boxes, batch_scores, batch_classes, batch_nums = model(batch)
            boxes.append(batch_boxes)
            scores.append(batch_scores)
            classes.append(batch_classes)
            nums.append(batch_nums)
        boxes = np.concatenate(boxes, axis = 0)
        scores = np.concatenate(scores, axis = 0)
        classes = np.concatenate(classes, axis = 0)
        nums = np.concatenate(nums, axis = 0)
        boxes = boxes.reshape((boxes.shape[0] * boxes.shape[1], 4))
        classes = classes.reshape((classes.shape[0]*classes.shape[1], 1))
        boxes = np.concatenate([classes, boxes], axis = 1)
        box_areas = (boxes[:, 3] - boxes[:,1]) * (boxes[:,4] - boxes[:,2])
        boxes = boxes[box_areas > 0]
        box_areas = box_areas[box_areas > 0]
        counts = []
        for i, grade in enumerate(['benign', 'g3', 'g4', 'g5']):
            grade_boxes = boxes[boxes[:,0] == i]
            grade_areas = box_areas[boxes[:,0] == i] / patches.shape[0]
            counts.append(grade_boxes.shape[0])
            counts.append(np.sum(grade_areas))
    else:
        counts = [0] * 8
    counts.append(len(patches))
    return counts


# In[ ]:


def generate_features(d, model, patch_size = 416, split = "train", max_boxes = 100):
    results = []
    for row in tqdm.tqdm(range(d.shape[0])):
        result = generate_image_features(d.iloc[row]['image_id'], model = model, split = split, patch_size = patch_size, max_boxes = max_boxes)
        results.append(result)
    df = pd.DataFrame.from_records(results, columns = ['benign_count', 'benign_area', 'g3_count', 'g3_area', 'g4_count', 'g4_area', 'g5_count', 'g5_area', 'patch_count'])#, 
    df = pd.concat([d.reset_index(drop = True), df], axis = 1)
    return df


# In[ ]:


## code from https://github.com/zzh8829/yolov3-tf2/blob/master/train.py

yolo_iou_threshold = 0

physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print(physical_devices)
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and                     sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) *         (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) *         (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


def draw_labels(x, y, class_names):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 255), 2)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def DarknetTiny(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = DarknetConv(x, 256, 3)  # skip connection
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x, 1024, 3)
    return tf.keras.Model(inputs, (x_8, x), name=name)


def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloConvTiny(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) /         tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold
    )

    return boxes, scores, classes, valid_detections


def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


def YoloV3Tiny(size=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_8, x = DarknetTiny(name='yolo_darknet')(x)

    x = YoloConvTiny(256, name='yolo_conv_0')(x)
    output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    return Model(inputs, outputs, name='yolov3_tiny')


def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) -             tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale *             tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale *             tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss +             (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss


# In[ ]:


if use_precomputed_yolo_model:
    yolo = YoloV3(classes=4, size= patch_size)
    yolo.load_weights(os.path.join("/kaggle", "input", "precomputedprostate", "prostate_checkpoints_yolov3_train_x.tf"))


# In[ ]:


d_masked = get_masked_dataframe()
d_masked = d_masked[d_masked['isup_grade'] > 1]
printed = False
while (not printed):
    sample_img=random.randint(0, d_masked.shape[0] - 1)
    examples = process_image(d_masked.iloc[sample_img]['image_id'], classes = classes, patch_size = patch_size, split = "train", 
                          box_to_image_min_ratio = box_to_image_min_ratio, 
                          level_to_box_min_ratio = level_to_box_min_ratio, 
                          min_box_area = min_box_area, writer = None)
    if len(examples) > 1:
        rands = np.random.choice(range(len(examples)), min(4, len(examples)), replace = False)
        nfig = len(rands)
        fig, ax = plt.subplots(2, nfig, figsize = (20, 10))
        for i in range(nfig):
            boxes, class_nums, img = parse_example(examples[rands[i]])
            for j in range(len(boxes)):
                box = boxes[j]
                xmin, ymin = box[0], box[1]
                xmax, ymax = box[2], box[3]
                w, h = xmax - xmin, ymax - ymin
                rect = matplotlib.patches.Rectangle((xmin, ymin), w, h, fill = None)
                #plt.imshow(img)
                ax[0,i].add_patch(rect)
                ax[0,i].text(xmin, ymin, class_nums[j])
            ax[0,i].imshow(img)
            boxes, scores, class_preds, nums = yolo(tf.expand_dims(img/255, 0))
            img = draw_outputs(img, (boxes, scores, class_preds, nums), classes)
            ax[1,i].imshow(img)
        printed = True


# In[ ]:




