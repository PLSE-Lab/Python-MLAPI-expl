#!/usr/bin/env python
# coding: utf-8

# ## Object Detection with Detectron2 - PyTorch

# ### 1. What is Detectron2? 
# Detectron2 is a PyTorch-based modular object detection library developed by the Facebook AI Research team. It provides a large set of trained models available for download. Detectron2 includes high-quality implementations of state-of-the-art object detection algorithms, including [DensePose](http://densepose.org/), [panoptic feature pyramid networks](https://ai.facebook.com/blog/improving-scene-understanding-through-panoptic-segmentation/), and numerous variants of the pioneering [Mask R-CNN](https://research.fb.com/publications/mask-r-cnn/) model family.
# <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png" alt="Drawing" style="width: 600px;"/>

# ### 2. Installation
# Installing detectron2 is fairly simple as opposed to other object detection frameworks like the Tensorflow Object Detection API. 
# We will be installing torch, torchvision, cocoapi, and detectron2.

# In[ ]:


get_ipython().system('pip install -q -U torch torchvision -f https://download.pytorch.org/whl/torch_stable.html ')
get_ipython().system("pip install -q -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'")
get_ipython().system('pip install -q detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html')


# ### 3. Dependencies
# We are going to need the detectron2 for configuring and building the model and for viusalizing the bounding boxes. For reading and plotting images, we will be using matplotlib.

# In[ ]:


from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ### 4. Model Definition 
# 
# We will be using a pre-trained model for now. Using a pretrained model is super easy in Detectron. We only need to load in and modify some configs. Then we will load in the weights of a pretrained model. Using the configs and pretrained weights, we will create DefaultPredictor to make predictions.
# 
# We will be using a Faster R-CNN model. It uses a ResNet+FPN backbone with standard conv and FC heads for mask and box prediction, respectively. This model obtains the best speed/accuracy tradeoff. You can have a look at some of the other pretrained models available at [Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md).

# In[ ]:


# Loading the default config
cfg = get_cfg()


# Merging config from a YAML file
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))


# Downloading and loading pretrained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")


# Changing some other configs
cfg.MODEL.DEVICE = 'cpu' # setting device to CPU as no training is required as per now
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # setting threshold for this model


# Defining the Predictor
predictor = DefaultPredictor(cfg)


# ### 5. Utility Functions
# 
# We are going to define some utility functions for showing the image and the predicted labels.****

# In[ ]:


def show_image(im, height=16, width=10):
    """
    Function to display an image
    
    Args:
        im ([numpy.ndarray])
        height ([int] or None)
        width ([int] or None)
    """
    plt.figure(figsize=(16,10))
    plt.imshow(im)
    plt.axis("off")
    plt.show()


# In[ ]:


def get_predicted_labels(classes, scores, class_names):
    """
    Function to return the name of predicted classes along with accuracy scores
    
    Args:
        classes (list[int] or None)
        scores (list[float] or None)
        class_names (list[str] or None)
    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
        labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
        return labels
    else:
        return "No object identified"


# ### 6. Load Image
# We will be downloading an image from [Unsplash](http://unsplash.com) and performing object detection on it. You may provide any other image url for object detection.

# In[ ]:


# Download image as input_image.jpg
# !wget https://images.unsplash.com/photo-1585574123552-aac232a58514 -O input_image.jpg

get_ipython().system('wget https://cdn-images-1.medium.com/max/872/1*EYFejGUjvjPcc4PZTwoufw.jpeg -O input_image.jpg')

# Read image
im = mpimg.imread("input_image.jpg")

# Show image
show_image(im)


# ### 7. Prediction
# 
# Finally, we will predict the objects present in the above image using the predictor we defined earlier. Our output will be a list of predicted class labels along with the prediction score and an image with bounding box drawn over each object. 

# In[ ]:


# Predicting image
outputs = predictor(im)


# Extracting other data from the predicted image
scores = outputs["instances"].scores
classes = outputs["instances"].pred_classes
class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes


# Obtaining a list of predicted class labels using the utility function created earlier
predicted_labels = get_predicted_labels(classes, scores, class_names)


# Creating the Visualizer for visualizing the bounding boxes
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
output_im = v.get_image()[:, :, ::-1] # image with bounding box and lables defined


# Displaying the output
print(f"Predicted Objects: {predicted_labels}")
show_image(output_im, outputs)


# For now we have only used a pretrained model for detecting objects. However, I will update this notebook in near future explaining how we can train the model on a custom dataset for object detection. If you like the kernel, feel free give it an upvote.
