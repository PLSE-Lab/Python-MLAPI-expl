#!/usr/bin/env python
# coding: utf-8

# # Face detection model comparision
# This compares multiple face detection models, Thanks for people who shared their models, I here share a library I found too, faced, which seems pretty fast but not always accurate.
# all models suffer from imperfections as you will see below.
# 
# ## summary
# - yolo v2 in my opinion strikes a good balance between speed vs accuarcy
# 

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib notebook
import cv2 as cv
from matplotlib.patches import Rectangle
import time


# ## Yolo v2 Mobile Net
# source : https://www.kaggle.com/drjerk/detect-faces-using-yolo

# In[ ]:


from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import math
import numpy as np
import cv2


# In[ ]:


def load_mobilenetv2_224_075_detector(path):
    input_tensor = Input(shape=(224, 224, 3))
    output_tensor = MobileNetV2(weights=None, include_top=False, input_tensor=input_tensor, alpha=0.75).output
    output_tensor = ZeroPadding2D()(output_tensor)
    output_tensor = Conv2D(kernel_size=(3, 3), filters=5)(output_tensor)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.load_weights(path)
    
    return model


# In[ ]:


mobilenetv2 = load_mobilenetv2_224_075_detector("../input/facedetection-mobilenetv2/facedetection-mobilenetv2-size224-alpha0.75.h5")


# There'are 1920x1080 (16:9) and 1080x1920 (9:16) images in this competition as I can see from the samples (if you will find other, you can easily add them to SHOTS and SHOTS_T constants respectively)
# 
# Model was trained on 1:1 aspect ratio images, so if we wanna use 16:9 and 9:16 images, we need to split them into 2 pieces, also we can split them to smaller (ex 10) intersecting pieces to get more accurate predictions for smaller faces.

# In[ ]:


# Converts A:B aspect rate to B:A
def transpose_shots(shots):
    return [(shot[1], shot[0], shot[3], shot[2], shot[4]) for shot in shots]

#That constant describe pieces for 16:9 images
SHOTS = {
    # fast less accurate
    '2-16/9' : {
        'aspect_ratio' : 16/9,
        'shots' : [
             (0, 0, 9/16, 1, 1),
             (7/16, 0, 9/16, 1, 1)
        ]
    },
    # slower more accurate
    '10-16/9' : {
        'aspect_ratio' : 16/9,
        'shots' : [
             (0, 0, 9/16, 1, 1),
             (7/16, 0, 9/16, 1, 1),
             (0, 0, 5/16, 5/9, 0.5),
             (0, 4/9, 5/16, 5/9, 0.5),
             (11/48, 0, 5/16, 5/9, 0.5),
             (11/48, 4/9, 5/16, 5/9, 0.5),
             (22/48, 0, 5/16, 5/9, 0.5),
             (22/48, 4/9, 5/16, 5/9, 0.5),
             (11/16, 0, 5/16, 5/9, 0.5),
             (11/16, 4/9, 5/16, 5/9, 0.5),
        ]
    }
}

# 9:16 respectively
SHOTS_T = {
    '2-9/16' : {
        'aspect_ratio' : 9/16,
        'shots' : transpose_shots(SHOTS['2-16/9']['shots'])
    },
    '10-9/16' : {
        'aspect_ratio' : 9/16,
        'shots' : transpose_shots(SHOTS['10-16/9']['shots'])
    }
}

def r(x):
    return int(round(x))

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def non_max_suppression(boxes, p, iou_threshold):
    if len(boxes) == 0:
        return np.array([])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    indexes = np.argsort(p)
    true_boxes_indexes = []

    while len(indexes) > 0:
        true_boxes_indexes.append(indexes[-1])

        intersection = np.maximum(np.minimum(x2[indexes[:-1]], x2[indexes[-1]]) - np.maximum(x1[indexes[:-1]], x1[indexes[-1]]), 0) * np.maximum(np.minimum(y2[indexes[:-1]], y2[indexes[-1]]) - np.maximum(y1[indexes[:-1]], y1[indexes[-1]]), 0)
        iou = intersection / ((x2[indexes[:-1]] - x1[indexes[:-1]]) * (y2[indexes[:-1]] - y1[indexes[:-1]]) + (x2[indexes[-1]] - x1[indexes[-1]]) * (y2[indexes[-1]] - y1[indexes[-1]]) - intersection)

        indexes = np.delete(indexes, -1)
        indexes = np.delete(indexes, np.where(iou >= iou_threshold)[0])

    return boxes[true_boxes_indexes]

def union_suppression(boxes, threshold):
    if len(boxes) == 0:
        return np.array([])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    indexes = np.argsort((x2 - x1) * (y2 - y1))
    result_boxes = []

    while len(indexes) > 0:
        intersection = np.maximum(np.minimum(x2[indexes[:-1]], x2[indexes[-1]]) - np.maximum(x1[indexes[:-1]], x1[indexes[-1]]), 0) * np.maximum(np.minimum(y2[indexes[:-1]], y2[indexes[-1]]) - np.maximum(y1[indexes[:-1]], y1[indexes[-1]]), 0)
        min_s = np.minimum((x2[indexes[:-1]] - x1[indexes[:-1]]) * (y2[indexes[:-1]] - y1[indexes[:-1]]), (x2[indexes[-1]] - x1[indexes[-1]]) * (y2[indexes[-1]] - y1[indexes[-1]]))
        ioms = intersection / (min_s + 1e-9)
        neighbours = np.where(ioms >= threshold)[0]
        if len(neighbours) > 0:
            result_boxes.append([min(np.min(x1[indexes[neighbours]]), x1[indexes[-1]]), min(np.min(y1[indexes[neighbours]]), y1[indexes[-1]]), max(np.max(x2[indexes[neighbours]]), x2[indexes[-1]]), max(np.max(y2[indexes[neighbours]]), y2[indexes[-1]])])
        else:
            result_boxes.append([x1[indexes[-1]], y1[indexes[-1]], x2[indexes[-1]], y2[indexes[-1]]])

        indexes = np.delete(indexes, -1)
        indexes = np.delete(indexes, neighbours)

    return result_boxes

class FaceDetector():
    """
    That's API you can easily use to detect faces
    
    __init__ parameters:
    -------------------------------
    model - model to infer
    shots - list of aspect ratios that images could be (described earlier)
    image_size - model's input size (hardcoded for mobilenetv2)
    grids - model's output size (hardcoded for mobilenetv2)
    union_threshold - threshold for union of predicted boxes within multiple shots
    iou_threshold - IOU threshold for non maximum suppression used to merge YOLO detected boxes for one shot,
                    you do need to change this because there are one face per image as I can see from the samples
    prob_threshold - probability threshold for YOLO algorithm, you can balance beetween precision and recall using this threshold
    
    detect parameters:
    -------------------------------
    frame - (1920, 1080, 3) or (1080, 1920, 3) RGB Image
    returns: list of 4 element tuples (left corner x, left corner y, right corner x, right corner y) of detected boxes within [0, 1] range (see box draw code below)
    """
    def __init__(self, model=mobilenetv2, shots=[SHOTS['10-16/9'], SHOTS_T['10-9/16']], image_size=224, grids=7, iou_threshold=0.1, union_threshold=0.1):
        self.model = model
        self.shots = shots
        self.image_size = image_size
        self.grids = grids
        self.iou_threshold = iou_threshold
        self.union_threshold = union_threshold
        self.prob_threshold = 0.7
        
    
    def detect(self, frame, threshold = 0.7):
        original_frame_shape = frame.shape
        self.prob_threshold = threshold
        aspect_ratio = None
        for shot in self.shots:
            if abs(frame.shape[1] / frame.shape[0] - shot["aspect_ratio"]) < 1e-9:
                aspect_ratio = shot["aspect_ratio"]
                shots = shot
        
        assert aspect_ratio is not None
        
        c = min(frame.shape[0], frame.shape[1] / aspect_ratio)
        slice_h_shift = r((frame.shape[0] - c) / 2)
        slice_w_shift = r((frame.shape[1] - c * aspect_ratio) / 2)
        if slice_w_shift != 0 and slice_h_shift == 0:
            frame = frame[:, slice_w_shift:-slice_w_shift]
        elif slice_w_shift == 0 and slice_h_shift != 0:
            frame = frame[slice_h_shift:-slice_h_shift, :]

        frames = []
        for s in shots["shots"]:
            frames.append(cv2.resize(frame[r(s[1] * frame.shape[0]):r((s[1] + s[3]) * frame.shape[0]), r(s[0] * frame.shape[1]):r((s[0] + s[2]) * frame.shape[1])], (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST))
        frames = np.array(frames)

        predictions = self.model.predict(frames, batch_size=len(frames), verbose=0)

        boxes = []
        prob = []
        shots = shots['shots']
        for i in range(len(shots)):
            slice_boxes = []
            slice_prob = []
            for j in range(predictions.shape[1]):
                for k in range(predictions.shape[2]):
                    p = sigmoid(predictions[i][j][k][4])
                    if not(p is None) and p > self.prob_threshold:
                        px = sigmoid(predictions[i][j][k][0])
                        py = sigmoid(predictions[i][j][k][1])
                        pw = min(math.exp(predictions[i][j][k][2] / self.grids), self.grids)
                        ph = min(math.exp(predictions[i][j][k][3] / self.grids), self.grids)
                        if not(px is None) and not(py is None) and not(pw is None) and not(ph is None) and pw > 1e-9 and ph > 1e-9:
                            cx = (px + j) / self.grids
                            cy = (py + k) / self.grids
                            wx = pw / self.grids
                            wy = ph / self.grids
                            if wx <= shots[i][4] and wy <= shots[i][4]:
                                lx = min(max(cx - wx / 2, 0), 1)
                                ly = min(max(cy - wy / 2, 0), 1)
                                rx = min(max(cx + wx / 2, 0), 1)
                                ry = min(max(cy + wy / 2, 0), 1)

                                lx *= shots[i][2]
                                ly *= shots[i][3]
                                rx *= shots[i][2]
                                ry *= shots[i][3]

                                lx += shots[i][0]
                                ly += shots[i][1]
                                rx += shots[i][0]
                                ry += shots[i][1]

                                slice_boxes.append([lx, ly, rx, ry])
                                slice_prob.append(p)

            slice_boxes = np.array(slice_boxes)
            slice_prob = np.array(slice_prob)

            slice_boxes = non_max_suppression(slice_boxes, slice_prob, self.iou_threshold)

            for sb in slice_boxes:
                boxes.append(sb)


        boxes = np.array(boxes)
        boxes = union_suppression(boxes, self.union_threshold)

        for i in range(len(boxes)):
            boxes[i][0] /= original_frame_shape[1] / frame.shape[1]
            boxes[i][1] /= original_frame_shape[0] / frame.shape[0]
            boxes[i][2] /= original_frame_shape[1] / frame.shape[1]
            boxes[i][3] /= original_frame_shape[0] / frame.shape[0]

            boxes[i][0] += slice_w_shift / original_frame_shape[1]
            boxes[i][1] += slice_h_shift / original_frame_shape[0]
            boxes[i][2] += slice_w_shift / original_frame_shape[1]
            boxes[i][3] += slice_h_shift / original_frame_shape[0]

        return list(boxes)


# In[ ]:


def get_boxes_points(boxes, frame_shape):
    result = []
    for box in boxes:
        lx = int(round(box[0] * frame_shape[1]))
        ly = int(round(box[1] * frame_shape[0]))
        rx = int(round(box[2] * frame_shape[1]))
        ry = int(round(box[3] * frame_shape[0]))
        result.append((lx, ly, rx, ry))
    return result 


# That's the end of detector code

# In[ ]:


yolo_model = FaceDetector()


#    ## Faced
#    source : https://github.com/iitzco/faced

# In[ ]:


get_ipython().system('pip install /kaggle/input/facedpy/faced-0.1-py3-none-any.whl')


# In[ ]:


import faced 
from faced.utils import annotate_image
# Receives RGB numpy image (HxWxC) and
# returns (x_center, y_center, width, height, prob) tuples. 
faced_model = faced.FaceDetector()


# ## Blazeface
# source : https://www.kaggle.com/humananalog/starter-blazeface-pytorch

# In[ ]:


import sys
import torch
sys.path.insert(0, "/kaggle/input/blazeface-pytorch")

from blazeface import BlazeFace

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
blazeface = BlazeFace().to(gpu)
blazeface.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")
blazeface.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy")

# Optionally change the thresholds:
blazeface.min_score_thresh = 0.75
blazeface.min_suppression_threshold = 0.3


# In[ ]:


def get_blaze_boxes(detections, with_keypoints=False):
    result = []
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    img_shape = (128, 128)
    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img_shape[0]
        xmin = detections[i, 1] * img_shape[1]
        ymax = detections[i, 2] * img_shape[0]
        xmax = detections[i, 3] * img_shape[1]
        result.append((xmin, ymin, xmax, ymax))
    return result


# ## MTCNN
# wheel source:  https://www.kaggle.com/unkownhihi/mtcnn-package

# In[ ]:


get_ipython().system('pip install ../input/mtcnn-package/mtcnn-0.1.0-py3-none-any.whl')


# In[ ]:


from mtcnn import MTCNN
mtcnn = MTCNN()


# In[ ]:


# some helper functions
def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def scale_boxes(boxes, scale_w, scale_h):
    sb = []
    for b in boxes:
        sb.append((b[0] * scale_w, b[1] * scale_h, b[2] * scale_w, b[3] * scale_h))
    return sb


# # 1 frame test

# In[ ]:


DATA_FOLDER = '../input/deepfake-detection-challenge'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
TEST_FOLDER = 'test_videos'
videos_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'
sample_real = f'{videos_dir}/abarnvbtwb.mp4'
sample_fake = f'{videos_dir}/eepezmygaq.mp4'

train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))
json_file = [file for file in train_list if  file.endswith('json')][0]

def get_meta_from_json(path):
    df = pd.read_json(os.path.join(DATA_FOLDER, path, json_file))
    df = df.T
    return df

meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER)


# In[ ]:


def display_image_from_video(video_path, silent=False):
    ts = time.time()
    cap = cv.VideoCapture(video_path)
    _, frame = cap.read()
    assert _ == True
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    cap.release()
    frame_copy = np.array(frame)

    td = time.time()
    bboxes = faced_model.predict(frame, 0.8)
    print(f'faced time: {time.time() - td}')

    ty = time.time()
    yolo_boxes = yolo_model.detect(frame, 0.7)
    print(f'yolo time: {time.time() - ty}')
    
    # to scale boxes back since blaze only takes 128x128
    scale_w = frame.shape[1] / 128.0 
    scale_h = frame.shape[0] / 128.0
    
    tb = time.time()
    blaze_output = blazeface.predict_on_image(cv.resize(frame, (128,128)))
    print(f'blaze detection: {time.time() - tb}')
    scaled_boxes = scale_boxes(get_blaze_boxes(blaze_output), scale_w, scale_h)

    tm = time.time()
    result = mtcnn.detect_faces(frame)
    print(f'mtcnn detection: {time.time() - tm}')
    
    # Plotting boxes 
    # faced
    ann_img = annotate_image(frame_copy, bboxes)
    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111)
    ax.imshow(frame_copy)
    ax.set_title(video_path)
    # yolo boxes
    yb = get_boxes_points(yolo_boxes, frame.shape)
    
    # mtcnn
    for r in range(len(result)):
        x, y, w, h = result[r]['box']
        ax.add_patch(Rectangle((x,y),w, h, linewidth=2,edgecolor='white',facecolor='none'))
        
    # yolo
    for b in yb:
        lx, ly, rx, ry = b
        # x, y, w, h here
        ax.add_patch(Rectangle((lx,ly),rx - lx,ry - ly,linewidth=2,edgecolor='red',facecolor='none'))
    
    # blazeface
    for b in scaled_boxes:
        lx, ly, rx, ry = b
        rect = Rectangle((lx, ly), rx - lx, ry - ly, linewidth=2, edgecolor="yellow", facecolor="none")
        ax.add_patch(rect)

    print(f'time: {time.time() - ts}')


# In[ ]:


get_ipython().run_cell_magic('capture', '', '# first time run takes more time to warm up models\ndisplay_image_from_video(sample_real)')


# In[ ]:


display_image_from_video(sample_real)


# # Detection accuracy on different samples

# In[ ]:


fake_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='FAKE'].sample(50).index)


# In[ ]:


get_ipython().run_cell_magic('time', '', "for v in fake_train_sample_video:\n    print(f'processing video {v}')\n    display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, v))\n    print('====================================================================')")


# In[ ]:


real_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='REAL'].sample(30).index)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for video_file in real_train_sample_video:\n    display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))')

