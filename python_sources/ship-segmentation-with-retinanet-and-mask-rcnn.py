#!/usr/bin/env python
# coding: utf-8

# # Overview: 
# Here, we will prepeare the data needed for Keras_MaskRCNN which implements Retinanet as the object detector. The original MaskRCNN uses fasterRCNN. When we used the original MaskRCNN, the model identified more number of false postive. So we used Keras_MaskRCNN which utilizes the Retinanet as it's detector.  For the scope of this kernel, we have prepared the required data to train a beseline model. Due to the libraries requried for training the model, it was not convinient to write a kernel through which we could have used kaggle resources to train the model. However after following the steps below, you can simply start training the model in your desired machine.  For your convenience, the prepread data is already in the data section, also a trained model snaptshot has been added, so that you could visualize the result right in this Kernel.
# 
# # Result: 
# This baseline model, if you manage to train until 20 epochs will give you fairly nice public score around 67 - 68 % . 
# 
# # Beyond: 
# You can actually use it to ensemble the result from other models like MaskRcnn, whose predictions were quite fine, however with lots of false positive. This model helped us to identify the images in the test set where the ship weren't present. 
# 
# # What We Learnt: 
# This was our first competition ever in the Kaggle. We had never touched an instance segmentation problem before. We started off with Matterport's version of MaskRCNN. After post-processing the result and ensembling 3 different models, we got around 68.7% score in public leaderboard.  After several weeks of struggle with MaskRCNN, we could hardly improve the score, then we realized that our result contained many false positives. Then, we tried to use RetinaNet instead of FasterRCNN for the detector as RetinaNet is better adjusted to single out false positives. But we were already out of time, so we simply chose [Keras_MaskRCNN](https://github.com/fizyr/keras-maskrcnn/) which already had what we needed.  In short, we got to learn a lot. Thank you Kaggle. 
# 

# # Import Necessary Packages

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# # Loading the dataset and preparation 

# In[ ]:


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    :param mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    :return: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def load_mask(data):

    """Generate instance masks for an image.
    :param data: dataframe series
    :return:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    # If not a ship dataset image, delegate to parent class.
    
    height =  width =  768
    rle = [data["EncodedPixels"]]
    mask = np.zeros((height,width,len(rle)))
    for p,m in enumerate(rle):
        all_masks = np.zeros((height,width))
        all_masks += rle_decode(m)
        mask[:, :, p] = all_masks

    return mask.astype(np.bool)


def rleToMask(rleString,height,width):
    """Converts rle string to mask instance
    :param rleString: Run length encoding of a mask
    :param height: original image height
    :param width: original image width
    :return: image array of a mask
    """
    rows,cols = height,width
    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1,2)
    img = np.zeros(rows*cols,dtype=np.uint8)
    for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(cols,rows)
    img = img.T
    return img


def rle_decode(mask_rle, shape=(768, 768)):
    '''Decodes RLE string to generate binary mask
    :param mask_rle: run-length as string formated (start length)
    :param shape: (height,width) of array to return 
    :return: numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


# # Prepare the data required later for training purposes

# In[ ]:


from tqdm import tqdm_notebook
#Load Images
PATH_TO_SAVE = "DIR/TO/SAVE/MASKS/"
DATA_DIR = "PATH/FOR/CSV/"
datadir = os.listdir("../input")
masks = pd.read_csv('../input/airbus-ship-detection/train_ship_segmentations_v2.csv')
print(masks.shape[0], 'masks found')
print(masks['ImageId'].value_counts().shape[0])

images_with_ship = masks[~masks.EncodedPixels.isnull()]['ImageId'].unique().tolist()
print('There are ' +str(len(images_with_ship)) + ' image files  with masks found')

def create_data():
    """If you want to create masks and annotations, then, call this method"""
    data = pd.DataFrame(columns=["ImageId", "x1", "y1", "x2", "y2", "classname", "mask"])
    masks.fillna('', inplace=True)
    grouped = masks.groupby(["ImageId"])
    count = 0
    for index, row in tqdm_notebook(masks.iterrows()):
        if(row["EncodedPixels"]==''):
            rowtoappend = {"ImageId":row["ImageId"], "x1":'', "y1":'', "x2":'', "y2":'', "classname":'', 'mask':''}
            data.loc[count] = rowtoappend
        else:
            number_of_masks = masks[masks["ImageId"]==row["ImageId"]]
            rle = [row["EncodedPixels"]] 
            imgdata = rleToMask(row["EncodedPixels"], 768, 768)
            im = Image.fromarray(imgdata)
            i = 0
            '''While running in local mode the lines below would save the instance mask associated with a particulr image inside the desired directory
            It has been commented here because the masks dataset has been loaded as a seperate dataset in this kernel. 
            '''
            # comment starts here. Uncomment the lines below to save the masks
            # while os.path.exists(PATH_TO_SAVE + row["ImageId"] + "%s.png" % i):
                 # i += 1
            # im.save(PATH_TO_SAVE + row["ImageId"] + "%s.png" % i)
            #comment ends here
            m = load_mask(row)
            box = extract_bboxes(m)
            box = box[0]

            rowtoappend = {"ImageId":row["ImageId"], "x1":box[1], "y1":box[0], "x2":box[3], 
                           "y2":box[2], "classname":"ship", 'mask':PATH_TO_SAVE + row["ImageId"]+"%s.png" % i}
            data.loc[count] = rowtoappend
            count+=1
    data.to_csv(DATA_DIR + "annotation.csv", index=None)


# In[ ]:


# Reading the data
data = pd.read_csv('../input/maskdata/annotation_working.csv')
data.head(5)


# # The Results:
# Now we have annotation_working.csv file inside the maskdata folder, we have class label csv under the same folder, and the masks folder consists all the maks images that we will need for training the model. 

# # Actual Training
# Loading all the necessary libraries and putting the needed codes would be cumbersome for this kernel. For that simplicity, we shifted the codes into github, through the listed approach in the repo, you can train the model. 
# The major step, since you already have the masks images, annotation csv and class_ids would be just to run: 
# `./keras_maskrcnn/bin/train.py --weights=PATH/TO/COCO/WEIGHT/ --epochs=50 --steps=1000 --config=config/config.ini csv data/annotation.csv data/class_ids.csv` 
# or more precisely, follow the steps in https://github.com/vaghawan/airbus-ship-detection-using-keras-retinanet-maskrcnn#training
# 
# After you got your model trained, the `detect.py` performs the main actions.  Here we have done the modifcation of the model outputs and transformed them into the output that we would get from Matterport's version of MaskRCNN, because we were already using the inference of Matterport's version of MaskRCNN. 

# # Post-Processing Of KerasMaskRCNN Outputs:
# - The output shape of masks is always (100, ImageHeight, ImageWidth, 1), and we transformed the masks into the shape of (ImageHeight, ImageWidth, NUM_OF_DETECTED_INSTANCE) 
# - We applied Non-Max Supression to the predicted bounding boxes to remove lots of overlaps. 
# - After that, we removed the overlap in the masks by assigning the overlap pixels to the highest scored mask. 
# - We thresholded the predicted output. 
# 
# Below is the code that were used for post-processing. 

# In[ ]:


import keras
import sys
sys.path.append("../input/kerasmaskrcnn/keras-maskrcnn-master/keras-maskrcnn-master/")
# import keras_retinanet
from keras_maskrcnn import models
from keras_maskrcnn.utils.visualization import draw_mask
from keras_retinanet.utils.visualization import draw_box, draw_caption, draw_annotations
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
import skimage
from skimage.morphology import binary_opening, disk, label,binary_closing,binary_dilation

from skimage.measure import find_contours
from skimage.measure import label as label_lib
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import dilation, erosion
import scipy
import skimage.color
import skimage.io
import pandas as pd
import csv, datetime

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())



def draw_test(image, boxes, masks, scores, labels=None, color=None, binarize_threshold=0.5):
    """ Mould the keras Mask-RCNN outputs to the output given by Matterport's Mask-RCNN.
    :param image: Three dimensional image to draw on.
    :param box: Vector of at least 4 values (x1, y1, x2, y2) representing a box in the image.
    :param mask: A 2D float mask which will be reshaped to the size of the box, binarized and drawn over the image.
    :param color: Color to draw the mask with. If the box has 5 values, the last value is assumed to be the label and used to construct a default color.
    :param binarize_threshold: Threshold used for binarizing the mask.
    :param scores: A 1D Numpy array of scores
    
    return: resulting_masks: predicted masks reshaped to [num_instances, height, width, 1]
    return: kept_scores: scores corresponding to the mask
    return: kept_labels: associated labels
    return: kept_boxes: associated bounding boxes
    """
    resulting_masks = []
    kept_scores = []
    kept_labels = []
    kept_boxes = []
    
    if labels is None:
        labels = [None for _ in range(boxes.shape[0])]
    

    for box, mask, label, score in zip(boxes, masks, labels, scores):
        
        # resize to fit the box
        if label != -1.:
            kept_boxes.append(box)
            box = box.astype(int)
            mask = cv2.resize(mask, (box[2] - box[0], box[3] - box[1]))

            # binarize the mask
            mask = (mask > binarize_threshold).astype(np.uint8)
            # print(mask, mask.shape)
            # draw the mask in the image
            mask_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
            mask_image[box[1]:box[3], box[0]:box[2]] = mask
            mask = mask_image
            resulting_masks.append(mask)
            kept_scores.append(score)
            kept_labels.append(label)
            #resulting_masks = np.append(resulting_masks, mask, axis=0)
    if len(resulting_masks) <1 :
        resulting_masks = np.zeros((image.shape[0], image.shape[1], 0), np.uint8)
        kept_scores = np.array([])
        kept_labels = np.array([])
        kept_boxes = np.array([])
    else:   
        resulting_masks = np.asarray(resulting_masks)
        resulting_masks = resulting_masks.reshape(resulting_masks.shape[0],768, 768, 1 )
        kept_scores = np.asarray(kept_scores)
        kept_labels = np.asarray(kept_labels)
        kept_boxes = np.asarray(kept_boxes)
        
    print(resulting_masks.shape)
    if resulting_masks.shape[-1] == None:
        resulting_masks = resulting_masks.reshape(0, 768, 768, 1)
    
    return resulting_masks, kept_scores, kept_labels, kept_boxes


def postprocess_masks(result, image, min_pixel_size=0):

    """Clean overlaps between bounding boxes, fill small holes, smooth boundaries
    :param result: dict containing numpy arrays of masks, scores, rois and class_ids
    :param image: array of original image
    :param min_pixel_size: minimum number of pixels that the predicted mask should have to be considered as a valid result
    :return: dict containing numpy arrays of masks, scores, rois and class_ids after smoothing and removing overlaps
    """
    
    height, width = image.shape[:2]

    # If there is no mask prediction do the following
    print("inside post-process", result['masks'].shape)
    if result['masks'].shape[0] == 0:
        print("we were supposed to be here")
        result['masks'] = np.zeros([height, width, 1])
        result['masks'][0, 0, 0] = 1
        result['scores'] = np.ones(1)
        result['class_ids'] = np.zeros(1)

    keep_ind = np.where(np.sum(result['masks'], axis=(0, 1)) > min_pixel_size)[0]
    
    if len(keep_ind) < result['masks'].shape[-1]:
        # print('Deleting',len(result['masks'])-len(keep_ind), ' empty result['masks']')
        result['masks'] = result['masks'][..., keep_ind]
        result['scores'] = result['scores'][keep_ind]
        result['rois'] = result['rois'][keep_ind]
        result['class_ids'] = result['class_ids'][keep_ind]

    sort_ind = np.argsort(result['scores'])[::-1]
    
    result['masks'] = result['masks'][..., sort_ind]
    overlap = np.zeros([height, width])

    # Removes overlaps from masks with lower score
    for mm in range(result['masks'].shape[-1]):
        # Fill holes inside the mask
        mask = binary_fill_holes(result['masks'][..., mm]).astype(np.uint8)
        # Smoothen edges using dilation and erosion
        mask = erosion(dilation(mask))
        # Delete overlaps
        overlap += mask
        
        mask[overlap > 1] = 0
        
        out_label = label_lib(mask)
        
        # Remove all the pieces if there are more than one pieces
        if out_label.max() > 1:
            mask[()] = 0
            print('removed something here')
        result['masks'][..., mm] = mask
    
    keep_ind = np.where(np.sum(result['masks'], axis=(0, 1)) > min_pixel_size)[0]
    
    if len(keep_ind) < result['masks'].shape[-1]:
        result['masks'] = result['masks'][..., keep_ind]
        result['scores'] = result['scores'][keep_ind]
        result['rois'] = result['rois'][keep_ind]
        result['class_ids'] = result['class_ids'][keep_ind]
    return result


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    :param box: 1D vector [y1, x1, y2, x2]
    :param boxes: [boxes_count, (y1, x1, y2, x2)]
    :param box_area: float. the area of 'box'
    :param boxes_area: array of length boxes_count.
    
    :return: iou: Intersection over union of predicted boxes

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """ 
    y1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[3], boxes[:, 3])
    x1 = np.maximum(box[0], boxes[:, 0])
    x2 = np.minimum(box[2], boxes[:, 2])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    :param boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    :param scores: 1-D array of box scores.
    :param threshold: Float. IoU threshold to use for filtering.
    
    :return: pick: indices of boxes selected after applying non-max suppression
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
    
    y1 = boxes[:, 1]
    x1 = boxes[:, 0]
    y2 = boxes[:, 3]
    x2 = boxes[:, 2]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def color_splash(image, mask):
    """Apply color splash effect.
    :param image: RGB image [height, width, 3]
    :param mask: instance segmentation mask [height, width, instance count]

    :return: result image
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    red = image*[1,1,0]
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set

    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        mask = mask.astype(int)
        #print(rle_encode(mask))
        splash = np.where(mask, red, image).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash 


def rle_encoding(x):
    """Performs run length encoding over the array of mask instance for submission
    :param x: binary mask
    :return: run length encoding string
    """
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev+1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(str(x) for x in run_lengths)


def prob_to_rles(masks, height, width):
    """Takes post-processed binary masks as input and performs run length encoding
    :param masks: array of binary masks of shape [height, width, num_instances]
    :param height: original image height; 768 in our case
    :param width: original image width; 768 in our case
    
    :return: generator of rle encodings 
    """
    if masks.sum() < 1:
        masks = np.zeros([height, width, 1])
        # print('no masks')
        masks[0, 0, 0] = 1

    if np.any(masks.sum(axis=-1) > 1):
        print('Overlap', masks.shape)

    for mm in range(masks.shape[-1]):
        yield rle_encoding(masks[..., mm].astype(np.int32)), np.sum(masks[..., mm].astype(np.int32)==1)
        
        
def test_on_single_image(model, imagepath, labels_names:dict, SCORE_THRES= 0.2, IOU_THRES = 0.5):
    """runs inference and plot the predicted segmentation in a single image
    :param model: instance of keras Mask-RCNN model loaded with trained weight
    :param imagepath: path to image
    :param labels_name: dict containing labels eg: {0:'ship'}
    :param SCORE_THRES: score threshold 
    :param IOU_THRES: IOU threshold
    """
    image = read_image_bgr(imagepath)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    boxes  = outputs[-4][0]
    scores = outputs[-3][0]
    labels = outputs[-2][0]
    masks  = outputs[-1][0]

    # correct for image scale
    boxes /= scale

    # visualize detections
    #print(boxes.shape, scores.shape, masks.shape)
    masks, scores, labels, boxes = draw_test(draw, boxes, masks, scores, labels, color=label_color(0))
    if boxes.size !=0:
        keep_ind = non_max_suppression(boxes, scores, IOU_THRES)
        masks = masks[keep_ind, :, :]
        scores = scores[keep_ind]
        labels = labels[keep_ind]
        rois = boxes[keep_ind]
        result = {"masks":masks, "scores":scores, "class_ids":labels, "rois":rois}
        idxtokeep = np.where(result['scores']>SCORE_THRES)[0]
        result['masks'] = masks[idxtokeep,:, :]
        result['scores'] = scores[idxtokeep]
        result['class_ids'] = labels[idxtokeep]
        result['rois'] = rois[idxtokeep]

        image_arr = skimage.io.imread(imagepath)

        masks_resulted = []
        if result['masks'].size !=0:
            firstmask = result['masks'][0]

            result['masks'] = result['masks'][1:]
            for box, score, label, mask in zip(result['rois'], result['scores'], result['class_ids'], result['masks']):
                color = label_color(label)
                b = box.astype(int)
                #draw_box(draw, b, color=color)

                firstmask = np.append(firstmask, mask, axis=2)
                mask = mask[:,:,label]
                #draw_mask_overlap(draw, b, mask)

                caption = "{} {:.3f}".format(labels_to_names[label], score)
            print("concatinated mask", firstmask.shape)
            result['masks'] = firstmask
            print(result['scores'])
            splash = color_splash(image_arr, result['masks'])
            plt.figure(figsize=(15,15))
            skimage.io.imshow(splash)
            plt.show()
            plt.close()

        else:
            print("the result were removed due to the thresholding.")

    else:
        print("no instance found.")



def generate_result(model, imagedir, labels_names:dict, csv_path:str, output_image_path:str=None, SCORE_THRES= 0.2, IOU_THRES = 0.5):
    """runs inference and plot the predicted segmentation in all test images 
    :param model: instance of keras Mask-RCNN model loaded with trained weight
    :param imagedir: directory containing images
    :param labels_name: dict containing labels eg: {0:'ship'}
    :param csv_path: path to submission csv file
    :param output_image_path: path to save output images
    :param SCORE_THRES: score threshold 
    :param IOU_THRES: IOU threshold
    """
    already_tested = list(pd.read_csv(csv_path)["ImageId"].unique())
    allimagesindir = list(os.listdir(imagedir))
    yettotest = list(set(allimagesindir) - set(already_tested)) if (len(allimagesindir)>len(already_tested)) else list(set(already_tested) - set(allimagesindir))
    print(yettotest) 
    print("number of images yet to  test is:", len(yettotest), len(already_tested))

    count = 0
    for image_path in yettotest:
        img_path = imagedir+image_path
        # load image
        image = read_image_bgr(img_path)
        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)
        # process image
        start = time.time()
        outputs = model.predict_on_batch(np.expand_dims(image, axis=0))

        print("processing time: ", time.time() - start)
        

        boxes  = outputs[-4][0]
        scores = outputs[-3][0]
        labels = outputs[-2][0]
        masks  = outputs[-1][0]

        boxes /= scale

        #This is for the preparation of submission csv file.
        imagelist=[]
        encodelist=[]
        overlappedrm = []
        scorelist = []
        lengthlist = []
        
        masks, scores, labels, boxes = draw_test(draw, boxes, masks, scores, labels, color=label_color(0))
        if boxes.size !=0:
            keep_ind = non_max_suppression(boxes, scores, IOU_THRES)
            masks = masks[keep_ind, :, :]
            scores = scores[keep_ind]
            labels = labels[keep_ind]
            rois = boxes[keep_ind]
            result = {"masks":masks, "scores":scores, "class_ids":labels, "rois":rois}
            idxtokeep = np.where(result['scores']>SCORE_THRES)[0]
            result['masks'] = masks[idxtokeep,:, :]
            result['scores'] = scores[idxtokeep]
            result['class_ids'] = labels[idxtokeep]
            result['rois'] = rois[idxtokeep]

            image_arr = skimage.io.imread(img_path)

            masks_resulted = []

            if result['masks'].size != 0:
                firstmask = result['masks'][0]

                result['masks'] = result['masks'][1:]
                for box, score, label, mask in zip(result['rois'], result['scores'], result['class_ids'], result['masks']):
                    color = label_color(label)
                    b = box.astype(int)
                    firstmask = np.append(firstmask, mask, axis=2)
                    mask = mask[:,:,label]

                
                result['masks'] = firstmask
                print("kept scores.. " , result["scores"])
                if output_image_path:
                    splash = color_splash(image_arr, result['masks'])
                    file_name = image_path+"splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
                    skimage.io.imsave(output_image_path+file_name, splash)
                    
                

            else:
                print("the result were removed due to the score thresholding.")

            height, width = image_arr.shape[:2]

            if result["scores"].size == 0:
                imagelist.append(image_path)
                encodelist.append('')
                print("the mask is blank")
                scorelist.append(0.0)
                lengthlist.append(0)

            else:
                masks = result["masks"].astype(int)
                
                encode = list(prob_to_rles(masks, height, width))
                
                
                scores = list(result['scores'])
                
                if encode !=None:
                    for en, score in zip(encode, scores):
                        imagelist.append(image_path)
                        encodelist.append(en[0])
                        scorelist.append(score)
                        lengthlist.append(en[1])
                        #overlappedrm.append(rmoverlapped)
                else:
                    imagelist.append(image_path)
                    encodelist.append('')
                    scorelist.append(0.0)
                    lengthlist.append(0)

        else:
            imagelist.append(image_path)
            encodelist.append('')
            print("the mask is blank")
            scorelist.append(0.0)
            lengthlist.append(0)

        with open(csv_path, 'a') as outcsv:

            fieldnames = ['ImageId', 'EncodedPixels', 'Score', 'Length']
            writer = csv.DictWriter(outcsv, fieldnames=fieldnames)
            writer.writerows([{"ImageId":img, "EncodedPixels":enc, "Score":scr, "Length":lengt} for img, enc, scr, lengt in zip(imagelist, encodelist, scorelist, lengthlist)])


if __name__=='__main__':
    #adjust this to point to your downloaded/trained model
    model_path = '../input/snapshots/resnet50_csv_84.h5'
    #load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')
    #print(model.summary())
    SCORE_THRES = 0.2
    IOU_THRES = 0.5
    #load label to names mapping for visualization purposes
    labels_to_names = {0: 'ship'}

    img_path = '../input/airbus-ship-detection/test_v2/0a89c4e4b.jpg'
    #For test in single image
    test_on_single_image(model, img_path, labels_to_names)
    #To generate the overall Results: 
    #generate_result(model, "/var/www/mask-rcnn/data/all/test_v2/", labels_to_names, "/var/www/mask-rcnn/data/all/ensemble170-32.csv", output_image_path="/var/www/airbus-competition-using-keras-maskrcnn/examples/splash/")


# # References:
# Several references of codes and implemntation were taken from the various sources, which are listed below:
# 
# [https://github.com/fizyr/keras-maskrcnn/ ] - Original Keras Mask RCNN. 
# 
# [https://github.com/matterport/Mask_RCNN] - Many of the utility functions were actually the modified version available in the utility.py of matterport's Mask RCNN.
# 
# [https://github.com/mirzaevinom/data_science_bowl_2018] -  Some of the utility functions were actually modified/used from mirzaevinom implementation.
# 
# Kaggle Airbus Ship Detection competition, Kernels. 
# 
# **The complete implementation is in our repo:** https://github.com/vaghawan/airbus-ship-detection-using-keras-retinanet-maskrcnn

# In[ ]:




