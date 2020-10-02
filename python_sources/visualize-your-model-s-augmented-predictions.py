#!/usr/bin/env python
# coding: utf-8

# # Augmented prediction visualizer
# 

# <img src="https://s5.gifyu.com/images/demo_trained_weights_1.gif" align="center"/>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from typing import Optional, Dict, List
from typing_extensions import TypedDict

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.style.use('seaborn')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from albumentations import Compose, Normalize, PadIfNeeded
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import zoom
from IPython.display import HTML


# In[ ]:


HEIGHT = 137
WIDTH = 236

# Preprocess
TARGET_SIZE = 128
PADDING = 8

MEAN = 0.0778441
STD = 0.216016

# You should keep this
INPUT_PATH = '/kaggle/input/bengaliai-cv19'

# Replace this to your weight dataset
DATASET_PATH = '/kaggle/input/bengali-ai-model-weights'

# Demo model's weights
MODEL_STATE_FILE = DATASET_PATH + '/exp-549--resnet34--iter-24999.pt'


# # Preprocessing
# The preprocessing script below is from Iafoss' kernel:
# 
# [https://www.kaggle.com/iafoss/image-preprocessing-128x128](https://www.kaggle.com/iafoss/image-preprocessing-128x128)
# 
# You should apply the same preprocessing steps on the sample image as you did in your training process. See the example below.

# In[ ]:


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=TARGET_SIZE, pad=64):
    # crop a box around pixels large than the threshold
    # some images contain line at the sides
    ymin, ymax, xmin, xmax = bbox(img0[5:-5, 5:-5] > 80)

    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax, xmin:xmax]

    # remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax - xmin, ymax - ymin
    ls = max(lx, ly) + pad

    # make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((ls - ly) // 2,), ((ls - lx) // 2,)], mode='constant')

    return cv2.resize(img, (size, size))


# # Transformation functions
# 
# Some transformation helper functions. You can implement your own:
# 
# ```python
# def custom_transform(image, value):
#     """Custom image transformation function.
#     
#     Args:
#         image (numpy.ndarray) Original image (without any previous transformation steps), shape: H x W
#         value (int|Tuple) Argument for the next transformation step
#         
#     Return:
#         transformed image (numpy.ndarray)
#     """
#     transformed_image = ...
# 
#     return transformed_image
# ```

# In[ ]:


def rotate_image(image, angle):
    """Rotate."""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return result

def scale_image(image, scale):
    """Zoom in/out."""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    matrix = cv2.getRotationMatrix2D(image_center, 0, scale)
    result = cv2.warpAffine(image, matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return result

def get_range(start, minval, maxval, step=1.0):
    """Dummy step generator."""
    return [x for x in np.arange(start, maxval, step)] +           [x for x in np.arange(maxval, minval, -1 * step)] +           [x for x in np.arange(minval, start, step)]


# # Prediction Visualizer
# With this simple class, you can visualize your model's output using different augmentations. It is a general script; you can use it (theoretically; I did not test it) for analyzing any image classification problem.

# In[ ]:


class LabelConfig(TypedDict):
    label_id: str
    label_name: str
    num_classes: int

class Sample(TypedDict):
    image: np.ndarray
    image_id: str
    image_mixup: Optional[np.ndarray]
    labels: Dict[str, int]


# *The code is hidden for saving some space*

# In[ ]:


class PredictionVisualizer:
    
    DARK_BLUE  = (0.1294, 0.5882, 0.9529) # True label (missclassified)
    GREEN      = (0.2980, 0.6863, 0.3137) # Predicted: correct
    RED        = (0.9869, 0.2627, 0.2118) # Predicted: incorrect
    LIGHT_BLUE = (0.7333, 0.8706, 0.9843) # other
    
    def __init__(self):
        """Prediction visualizer.
        
        Usage:
            See the example below in this notebook.
        """
        self.__model = None
        self.__labels = {}  # type: Dict[str, LabelConfig]
        self.__sample = None  # type: Sample
        self.__transform_fn = None  # type: Callable
        self.__transform_fn_steps = None  # type: List[int]
        self.__transofrm_data = []  # type: List[dict]
    
    @property
    def model(self):
        return self.__model
    
    @model.setter
    def model(self, model):
        self.__model = model

    @property
    def transform_fn(self):
        return self.__transform_fn

    def set_transform_fn(self, transform_fn, steps):
        self.__transform_fn = transform_fn
        self.__transform_fn_steps = steps
    
    @property
    def sample(self) -> Sample:
        return self.__sample

    @sample.setter
    def sample(self, sample: Sample):
        self.__sample = sample
    
    def add_label(self, label_config: LabelConfig) -> None:
        self.__labels[label_config['label_id']] = label_config
    
    @property
    def labels(self) -> Dict[str, LabelConfig]:
        return self.__labels
    
    def create(self):
        """Creates the animation."""
        fig, gs, subplots = self.__generate_figure()
        frames = self.__generate_frames()
        outputs = []
        
        for step_index, image in enumerate(frames['images']):
            output = [
                subplots['image'].imshow(image.astype(np.uint8), animated=True, cmap='Greys_r')
            ]
            
            for label_idx, (label_id, label) in enumerate(self.__labels.items()):
                colors = self.__get_colors(label['num_classes'],
                                           self.sample['labels'][label_id],
                                           np.argmax(frames[label_id], axis=1)[step_index]
                                          )
                output.append(
                    subplots[label_id].vlines(np.array([x for x in range(0, label['num_classes'])]),
                                              np.zeros(len(frames)),
                                              frames[label_id][step_index],
                                              colors
                                             )
                )
                

            outputs.append(output)

        return animation.ArtistAnimation(fig, outputs, interval=50, blit=True, repeat=True, repeat_delay=2000)    
    
    def _before_forward(self, transformed_image):
        """Before forward adapter
        
        The default implementation converts the numpy array to torch tensor and adds the missing
        `channel` and `batch` dimensions. You should update this method if your model expects a
        different input format.

        Args:
            transformed_image (numpy.ndarray): Transformed image (rotated, etc), shape: H x W

        Return:
            Prepared images (batch of image 1) for your model. The returned image's shape should
            be the shape of your model's input (For example, Pytorch: B x C x H x W)
        """        
        # Convert to float tensor
        transformed_image = torch.from_numpy(transformed_image).float()
        
        # Add 'channel' dim
        transformed_image = transformed_image.unsqueeze(0)
        
        # Add 'batch' dimension
        transformed_image = transformed_image.unsqueeze(0)
        
        return transformed_image
    
    def _forward(self, input_image):
        """You can make the forward call in here

        Args: 
            input_image (torch.Tensor | any) Prepared input for your model. Shape: B x C x H x W
        
        Return:
            You should return a dictionary of your model's predictions (logits or softmax)
            for every registered labels.
            
            ```
            with torch.no_grad():
                out_graph, out_vowel, out_conso = self.model(input_image)
            
            return {
                'grapheme_root': out_graph,
                'vowel_diacritic': out_vowel,
                'consonant_diacritic': out_conso
            }
            ```

            out_x.shape => B x label.NUM_CLASS
        """
        raise NotImplementedError

    def _softmax(self, outputs):
        """Applies a softmax function and returns the result.

        If your model has a final softmax layer, then you should override this to return
        the `outputs` argument without changes.
        
        The visualizer will call this method for every label separately.
        
        Args:
            outputs (torch.Tensor | any): Your model's output, shape: BATCH x NUM_CLASSES

        Return:
            Softmaxed values
        """
        return F.softmax(outputs, dim=1)

    def _after_forward(self, probabilities):
        """Convert the result to the required format.
        
        Args:
            probabilities (torch.Tensor | any) Your model's output after the `self._softmax` call.
            
        Return: (numpy.ndarray)
        """
        return probabilities.data.cpu().numpy()[0]
    
    def __generate_figure(self):
        """Generates the plot."""
        fig = plt.figure(constrained_layout=True, figsize=(14, 6))
        gs = fig.add_gridspec(len(self.labels), 2)
        
        subplots = {}
        subplots['image'] = fig.add_subplot(gs[:, 0], xticks=[], yticks=[])
        subplots['image'].set_title('Image id: {}'.format(self.sample['image_id']), fontsize=10)

        for label_idx, (label_id, label) in enumerate(self.__labels.items()):
            subplots[label_id] = fig.add_subplot(gs[label_idx, 1], xlim=(-1, label['num_classes']))
            subplots[label_id].set_title('{} (label: {})'.format(label['label_name'], self.sample['labels'][label_id]), fontsize=10)
    
        return fig, gs, subplots
    
    def __generate_frames(self):
        """Generates the frames."""
        
        assert self.model is not None
        assert self.sample is not None
        assert self.transform_fn is not None
        
        h, w = self.sample['image'].shape
        steps = len(self.__transform_fn_steps)
        
        frames = {}
        
        # Placeholder for the transformed images
        frames['images'] = np.zeros((steps, h, w))
        
        # Create placeholders for the labels
        for label_idx, (label_id, label) in enumerate(self.__labels.items()):
            frames[label_id] = np.zeros((steps, label['num_classes']))
            
        for step, transform_step_value in enumerate(self.__transform_fn_steps):
            
            # Transform the original image
            transformed_image = self.__transform_fn(self.sample['image'], transform_step_value)
            
            # Save the transformed image as a new frame
            frames['images'][step, ...] = transformed_image
            
            # Prepare the image for the model
            input_image = self._before_forward(transformed_image.copy())
            
            # Predict
            model_output = self._forward(input_image)
            
            # Add the results to the frames
            for label_id, output_logits in model_output.items():
                frames[label_id][step, ...] = self._after_forward(self._softmax(output_logits))
                
        return frames

    def __get_colors(self, size, target, pred):
        """Generates the colors of the vlines."""
        gra_color = [self.LIGHT_BLUE for _ in range(size)]

        if pred == target:
            gra_color[pred] = self.GREEN
        else:
            gra_color[pred] = self.RED
            gra_color[target] = self.DARK_BLUE

        return gra_color    


# # Model
# Replace this with your model. This is a Pytorch model, but you can use any other framework as well (I did not test it!)

# In[ ]:


class BengaliModel(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()

        self.backbone = torchvision.models.resnet34(pretrained=pretrained)
        num_bottleneck_filters = self.backbone.fc.in_features
        self.head_dropout = 0.1

        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.last = nn.Sequential(
            nn.BatchNorm2d(num_bottleneck_filters),
            nn.ReLU(inplace=True)
        )

        self.fc_graph = nn.Linear(num_bottleneck_filters, 168)
        self.fc_vowel = nn.Linear(num_bottleneck_filters, 11)
        self.fc_conso = nn.Linear(num_bottleneck_filters, 7)        

    def forward(self, x):
        batch_size, C, H, W = x.shape
        
        x = (x - MEAN * 255.0) / (STD * 255.0)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.last(x)        

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = F.dropout(x, self.head_dropout, self.training)

        fc_graph = self.fc_graph(x)
        fc_vowel = self.fc_vowel(x)
        fc_conso = self.fc_conso(x)

        return fc_graph, fc_vowel, fc_conso


# # Prediction analysis

# ### Loading samples from parquet

# In[ ]:


train_df = pd.read_csv(INPUT_PATH + '/train.csv')

# bengali_sample_id = 0
bengali_sample_id = 13241
bengali_sample = train_df.loc[bengali_sample_id].to_dict()
bengali_sample


# In[ ]:


datafile = INPUT_PATH + '/train_image_data_{}.parquet'.format(0)
parq = pq.read_pandas(datafile, columns=[str(x) for x in range(32332)]).to_pandas()

# I trained my models using this inverted pixels, make sure this is matches with your training.
parq = 255 - parq.iloc[:, :].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)


# ### Your custom visualizer
# To make this work with your model, you have to update some of the visualizer's methods.
# The default implementation works with Pytorch models, but you still may have to modify it.

# In[ ]:


class MyVisualzier(PredictionVisualizer):
    
    def __init__(self):
        super().__init__()

    # The implementation of the `_forward` method is required.
    # ---------------------------------------------------------
    def _forward(self, input_image):

        if torch.cuda.is_available():
            input_image = input_image.cuda()
        
        with torch.no_grad():
            out_graph, out_vowel, out_conso = self.model(input_image)

        return {
            'grapheme_root': out_graph,
            'vowel_diacritic': out_vowel,
            'consonant_diacritic': out_conso
        }
        
        
    # Implementation below this is optional
    # ----------------------------------------------
    def _before_forward(self, transformed_image):
        """Before forward adapter
        
        The default implementation converts the numpy array to torch tensor and adds the missing
        `channel` and `batch` dimensions. You should update this method if your model expects a
        different input format.

        Args:
            transformed_image (numpy.ndarray): Transformed image (rotated, etc), shape: H x W

        Return:
            Prepared images (a one element batch) for your model. The returned image's shape should
            be the shape of your model's input (For example, Pytorch: B x C x H x W)
        """
        return super()._before_forward(transformed_image)

    def _softmax(self, outputs):
        """Applies a softmax function and returns the result.

        If your model has a final softmax layer, then this method should return
        the `outputs` argument without changes.
        
        The visualizer will call this method for every label separately.
        
        Args:
            outputs (torch.Tensor | any): Your model's output, shape: BATCH x NUM_CLASSES

        Return:
            Softmaxed values
        """
        return super()._softmax(outputs)
    
    def _after_forward(self, probabilities):
        """Convert the result to the required format.
        
        Args:
            probabilities (torch.Tensor | any) Your model's output after the `self._softmax` call.
            
        Return: (numpy.ndarray)
        """
        return super()._after_forward(probabilities)


# ### Testing with random weights

# In[ ]:


model = BengaliModel(pretrained=False)

# If you'd like to use these demo weights, you have to
# enable GPU first

# My custom model state
model_state = torch.load(MODEL_STATE_FILE)

# Loading the weights
model.load_state_dict(model_state['model_state_dict'])

# Do not forget this!
x = model.eval()

# Move to the GPU if available
if torch.cuda.is_available():
    model.cuda()


# In[ ]:


visualizer = MyVisualzier()

# Set your model
visualizer.model = model

# Add the labels to the visualizer
visualizer.add_label({'label_id': 'grapheme_root', 'label_name': 'Grapheme Root', 'num_classes': 168})
visualizer.add_label({'label_id': 'vowel_diacritic', 'label_name': 'Vowel Diacritic', 'num_classes': 11})
visualizer.add_label({'label_id': 'consonant_diacritic', 'label_name': 'Consonant Diacritic', 'num_classes': 7})

# Create a new sample
sample: Sample = {
    # You should preprocess the image for your model
    'image': crop_resize(parq[bengali_sample_id], size=TARGET_SIZE, pad=PADDING),
    
    # Image id
    'image_id': 'Test_{}'.format(bengali_sample_id),

    # True labels
    'labels': {
        'grapheme_root': bengali_sample['grapheme_root'],
        'vowel_diacritic': bengali_sample['vowel_diacritic'],
        'consonant_diacritic': bengali_sample['consonant_diacritic']
    }
}
    
# Set the new sample
visualizer.sample = sample

# Set rotation.
visualizer.set_transform_fn(rotate_image, get_range(1, -45, 45, 1))
# visualizer.set_transform_fn(scale_image, get_range(1, 0.25, 1.5, 0.02))

# Create animation
anim = visualizer.create()

# Show the JS animation
HTML(anim.to_jshtml())


# In[ ]:





# # More examples
# (Animated gifs only)

# <img src="https://s5.gifyu.com/images/demo_trained_weights_1.gif" width="600" align="center"/>

# <img src="https://s5.gifyu.com/images/demo_trained_weights_misscls_2.gif" width="600" align="center"/>

# <img src="https://s5.gifyu.com/images/demo_trained_weights_misscls_scale_2.gif" width="600" align="center"/>

# ---------------
# **Thanks for reading.** If you find this notebook helpful (or interesting) please vote!

# In[ ]:




