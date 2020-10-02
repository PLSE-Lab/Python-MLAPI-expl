#!/usr/bin/env python
# coding: utf-8

# 

# # Course material from "complete guide to creating COCO datasets" by Adam Kelly
# 
# Course can be found here:
# https://www.immersivelimit.com/courses
# 
# Imported for the Here We Grow! community learning project for SingularityNet.  Virtually all of the code here is borrowed from both of these repos from github : 
# 
# https://github.com/akTwelve/cocosynth
# https://github.com/matterport/Mask_RCNN
# 

# # Mask R-CNN Training and Inference
# In this notebook we use Matterport's implementation of Mask R-CNN to train on our synthetic dataset, then use the trained weights to run inference on new images.
# 
# If you've never trained a neural network before, I wouldn't recommend starting here. Find a beginner deep learning tutorial/course and start there. I'd suggest the free course at https://course.fast.ai/. There are also lots of great free tutorials on YouTube, as well as paid courses on Udemy, Udacity, Coursera, etc.
# 
# ## TensorFlow GPU
# You definitely want TensorFlow GPU installed to run this notebook. Installation instructions are here:
# https://www.tensorflow.org/install/gpu
# 
# This includes installing CUDA, which is no small task. If you want to do serious image recognition, you just have to push through the pain.
# 
# When I created this notebook, I updated my graphics driver to the latest, then installed the CUDA Toolkit and CUDNN. Tensorflow tends to not work if you have the wrong combo of CUDA/CUDNN, so I recommend checking out the [tested build configurations](https://www.tensorflow.org/install/source#tested_build_configurations). It took me a few frustrating hours of searching and trial and error to get a combination that worked.
# 
# ## Keras
# It seems that the latest versions of Keras aren't compatible with the Mask R-CNN code. Using version 2.2.4 worked for me and other students.
# 
# ## My Combo that Worked
# - **Python 3.6** (conda create -n tf-gpu python=3.6)
# - **CUDA Toolkit 10.0** [https://developer.nvidia.com/cuda-zone](https://developer.nvidia.com/cuda-zone)
# - **CUDNN 7.4.1** [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
# - **tensorflow-gpu 1.13.1** (pip install tensorflow-gpu==1.13.1)
# - **keras 2.2.4** (pip install keras==2.2.4)

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')


# In[ ]:


get_ipython().system('nvcc --version')


# In[ ]:



get_ipython().system(' pip install -r /kaggle/input/matterhorn-mask-rcnn-for-here-we-grow/matterport_mask_rcnn/requirements_both.txt')
get_ipython().system(' pip install tensorflow==1.13.1')


# In[ ]:


#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))


# In[ ]:


import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
from pathlib import Path


# # Import Matterport's "mrcnn" libraries
# I'm using Matterport's Mask_RCNN git repo: https://github.com/matterport/Mask_RCNN
# 
# - Clone the Mask_RCNN repo to your computer.
# - Update the code cell below to point to the root directory of the repo.
# - Use pip to install everything from the requirements.txt file in that repo.
# 
# Read through their documentation and issues if you have any trouble getting started.

# In[ ]:


# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = '/kaggle/input/matterhorn-mask-rcnn-for-here-we-grow/matterport_mask_rcnn/'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'


# Import mrcnn libraries
sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib


# # Set up logging and pre-trained model paths
# This will default to sub-directories in your mask_rcnn_dir, but if you want them somewhere else, update it here.
# 
# It will also download the pre-trained coco model.

# In[ ]:


# Directory to save logs and trained model
MODEL_DIR = 'kaggle/working'

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# # Configuration
# Define configurations for training on the box_dataset_synthetic dataset.
# - Look through the code cell below and update any lines relevant to your custom dataset.
# - You may want to change:
#     - NAME (might want to be more specific)
#     - NUM_CLASSES (always 1 + the number of object categories you have)
#     - IMAGE_MIN_DIM (if you have larger training images)
#     - IMAGE_MAX_DIM (if you have larger training images)
#     - STEPS_PER_EPOCH (if you want to train on more images each epoch)
# 
# ### Note
# These are settings that worked on my machine (GTX 970 graphics card). If you are getting OOM (Out of Memory) errors, you may need to tweak the settings or your computer may not be powerful enough. If you have a better graphics card, you will want to tweak it to take advantage of that.

# In[ ]:


class CocoSynthConfig(Config):
    """Configuration for training on the box_synthetic dataset.
    Derives from the base Config class and overrides specific values.
    """
    # Give the configuration a recognizable name
    NAME = "cocosynth_dataset"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # background + 7 box types

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 50 #1000 -KYLE- I edited this to help troubleshoot the kernel : change it back for real training

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5
    
    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet50'

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 
    
config = CocoSynthConfig()
config.display()


# # Define the dataset
# I've attempted to make this generic to any COCO-like dataset. That means if you have another dataset defined in the COCO format, it should work.

# In[ ]:


class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
                
    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids


# # Create the Training and Validation Datasets
# Make sure you link to the correct locations for your training dataset in the cell below.

# In[ ]:


dataset_train = CocoLikeDataset()
dataset_train.load_data('/kaggle/input/cocosynth-for-here-we-grow/cocosynth-master/cocosynth-master/datasets/box_dataset_synthetic_complete/train/coco_instances.json',
                        '/kaggle/input/cocosynth-for-here-we-grow/cocosynth-master/cocosynth-master/datasets/box_dataset_synthetic_complete/train/images')
dataset_train.prepare()

dataset_val = CocoLikeDataset()
dataset_val.load_data('/kaggle/input/cocosynth-for-here-we-grow/cocosynth-master/cocosynth-master/datasets/box_dataset_synthetic_complete/val/coco_instances.json',
                      '/kaggle/input/cocosynth-for-here-we-grow/cocosynth-master/cocosynth-master/datasets/box_dataset_synthetic_complete/val/images')
dataset_val.prepare()


# ## Display a few images from the train and val datasets
# This will just make sure everything is set up correctly

# In[ ]:


for name, dataset in [('training', dataset_train), ('validation', dataset_val)]:
    print(f'Displaying examples from {name} dataset:')
    
    image_ids = np.random.choice(dataset.image_ids, 3)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names)


# ## Create the Training Model and Train
# This code is largely borrowed from the train_shapes.ipynb notebook in the Matterport repo.

# In[ ]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# In[ ]:


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


# ## Training
# Train in two stages:
# 
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass layers='heads' to the train() function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass layers='all' to train all layers.

# In[ ]:


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
start_train = time.time()
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, #4 -KYLE- I edited this to help troubleshoot the kernel : change it back for real training 
            layers='heads')
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')


# In[ ]:


# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
start_train = time.time()
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=1, #8 # -KYLE- I edited this to help troubleshoot the kernel : change it back for real training
            layers="all")
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')


# # Prepare to run Inference
# Create a new InferenceConfig, then use it to create a new model.

# In[ ]:


class InferenceConfig(CocoSynthConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.85
    

inference_config = InferenceConfig()


# In[ ]:


# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)


# ### Load Trained Weights
# Note: The code is set to find_last() by default, but you can also point the model to a specific pretrained weights file if you use line 3 instead of line 4.

# In[ ]:


# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = str(Path(ROOT_DIR) / "logs" / "box_synthetic20190328T2255/mask_rcnn_box_synthetic_0016.h5" )
model_path = model.find_last()

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# # Image Inference
# Run model.detect() on real images.
# 
# We get some false positives, and some misses. More training images are likely needed to improve the results.
# 
# You can adjust the "figsize" in the cell below to make the image larger or smaller.

# In[ ]:


import skimage

real_test_dir = '/kaggle/input/cocosynth-for-here-we-grow/cocosynth-master/cocosynth-master/datasets/box_dataset_synthetic_complete/test/images'
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))

for image_path in image_paths:
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=1)
    r = results[0]
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                dataset_train.class_names, r['scores'], figsize=(8,8))


# # Video Inference
# ## Prepare to run Video Inference
# Note: This code is adapted from https://www.dlology.com/blog/how-to-run-object-detection-and-segmentation-on-video-fast-for-free/ created by Chengwei Zhang.
# 
# In this section, we'll open up a video, run inference on it, and output a new video with segmentations and labels.

# In[ ]:


video_file = Path("/kaggle/input/cocosynth-for-here-we-grow/cocosynth-master/cocosynth-master/datasets/box_dataset_synthetic_complete/videotest/boxvideo_24fps.mp4")
video_save_dir = Path("kaggle/working/")
video_save_dir.mkdir(exist_ok=True)


# ### Adjust Config Parameters
# Depending on the resolution of your video, you may want to update the parameters below.
# 
# Mask R-CNN requires IMAGE_MIN_DIM to be divisible by 2, six times, so that's why I'm using 1088 instead of 1080, which is the actual height of the video.

# In[ ]:


class VideoInferenceConfig(CocoSynthConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 1088
    IMAGE_MAX_DIM = 1920
    IMAGE_SHAPE = [1920, 1080, 3]
    DETECTION_MIN_CONFIDENCE = 0.80
    

inference_config = VideoInferenceConfig()


# ### Set Up Model and Load Trained Weights

# In[ ]:


# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)


# In[ ]:


# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = str(Path(ROOT_DIR) / "logs" / "box_synthetic20190328T2255/mask_rcnn_box_synthetic_0016.h5" )
model_path = model.find_last()

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# In[ ]:


import cv2
import skimage
import random
import colorsys
from tqdm import tqdm


# In[ ]:


def random_colors(N, bright=True):
    """ Generate random colors. 
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
    Args:
        N: the number of colors to generate
        bright: whether or not to use bright colors
    Returns:
        a list of RGB colors, e.g [(0.0, 1.0, 0.0), (1.0, 0.0, 0.5), ...]
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """ Apply the given mask to the image.
    Args:
        image: a cv2 image
        mask: a mask of which pixels to color
        color: the color to use
        alpha: how visible the mask should be (0 to 1)
    Returns:
        a cv2 image with mask applied
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def display_instances(image, boxes, masks, ids, names, scores, colors):
    """ Take the image and results and apply the mask, box, and label
    Args:
        image: a cv2 image
        boxes: a list of bounding boxes to display
        masks: a list of masks to display
        ids: a list of class ids
        names: a list of class names corresponding to the ids
        scores: a list of scores of each instance detected
        colors: a list of colors to use
    Returns:
        a cv2 image with instances displayed   
    """
    n_instances = boxes.shape[0]

    if not n_instances:
        return image # no instances
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        # Check if any boxes to show
        if not np.any(boxes[i]):
            continue
        
        # Check if any scores to show
        if scores is not None:
            score = scores[i] 
        else:
            score = None

        # Add the mask
        image = apply_mask(image, masks[:, :, i], color)
        
        # Add the bounding box
        y1, x1, y2, x2 = boxes[i]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add the label
        label = names[ids[i]]
        if score:
            label = f'{label} {score:.2f}'
            
        label_pos = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2) # center of bounding box
        image = cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    return image


# ### Prepare for Inference
# Adjust the paths below to point to your video

# In[ ]:


video_file = Path("/kaggle/input/cocosynth-for-here-we-grow/cocosynth-master/cocosynth-master/datasets/box_dataset_synthetic_complete//videotest/boxvideo_24fps.mp4")
video_save_dir = Path("kaggle/working/")
video_save_dir.mkdir(exist_ok=True)
vid_name = video_save_dir / "output.mp4"
v_format="FMP4"
fourcc = cv2.VideoWriter_fourcc(*v_format)

print('Writing output video to: ' + str(vid_name))


# Random colors can be a bit intense with video, so I've set it to use the same color for each object. You can try random colors if you like.
# 
# Note: cv2 uses BGR (Blue Green Red) color representation instead of RGB (Red Green Blue), so we have to do a couple conversions here

# In[ ]:


#colors = random_colors(30)
colors = [(1.0, 1.0, 0.0)] * 30

# Change color representation from RGB to BGR before displaying instances
colors = [(color[2], color[1], color[0]) for color in colors]


# ### Perform Inference on Video
# Use the cv2 module to open the video and perform inference on each frame.
# 
# -KYLE- changed the code cell below to a markdown cell for troubleshooting purposes.  Once GPU issue is resolved I will change it back.

# input_video = cv2.VideoCapture(str(video_file))
# frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = int(input_video.get(cv2.CAP_PROP_FPS))
# output_video = None
# vid_size = None
# current_frame = 0
# 
# for i in tqdm(range(frame_count)):
#     # Read the current frame
#     ret, frame = input_video.read()
#     if not ret:
#         break
#         
#     current_frame += 1
#     
#     # Change color representation from BGR to RGB before running model.detect()
#     detect_frame = frame[:, :, ::-1]        
#     
#     # Run inference on the color-adjusted frame
#     results = model.detect([detect_frame], verbose=0)
#     r = results[0]
#     n_instances = r['rois'].shape[0]
#     
#     # Make sure we have enough colors
#     if len(colors) < n_instances:
#         # not enough colors, generate more
#         more_colors = random_colors(n_instances - len(colors))
#         
#         # Change color representation from RGB to BGR before displaying instances
#         more_colors = [(color[2], color[1], color[0]) for color in more_colors]
#         colors += more_colors
#         
#     
#     
#     # Display instances on the original frame
#     display_frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], 
#                                 dataset_train.class_names, r['scores'], colors[0:n_instances])
# 
#     # Make sure we got displayed instances
#     if display_frame is not None:
#         frame = display_frame
# 
#     # Create the output_video if it doesn't yet exist
#     if output_video is None:
#         if vid_size is None:
#             vid_size = frame.shape[1], frame.shape[0]
#         output_video = cv2.VideoWriter(str(vid_name), fourcc, float(fps), vid_size, True)
#         
#     # Resize frame if necessary
#     if vid_size[0] != frame.shape[1] and vid_size[1] != frame.shape[0]:
#         frame = cv2.resize(frame, vid_size)
#     
#     # Write the frame to the output_video
#     output_video.write(frame)
#     
# input_video.release()
# output_video.release()

# In[ ]:




