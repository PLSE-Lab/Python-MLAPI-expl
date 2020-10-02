#!/usr/bin/env python
# coding: utf-8

# # Face detection with BlazeFace
# 
# This notebook shows how to use the model from [BlazeFace PyTorch](https://www.kaggle.com/humananalog/blazeface-pytorch) for detecting faces in images.

# In[ ]:


import os, sys
import numpy as np
import torch
import cv2

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# For the best results, enable GPU in the notebook. But CPU should work fine as well.

# In[ ]:


print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())


# In[ ]:


gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu


# Helper code for making plots:

# In[ ]:


def plot_detections(img, detections, with_keypoints=True):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(img)
    
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d faces" % detections.shape[0])
        
    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor="r", facecolor="none", 
                                 alpha=detections[i, 16])
        ax.add_patch(rect)

        if with_keypoints:
            for k in range(6):
                kp_x = detections[i, 4 + k*2    ] * img.shape[1]
                kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
                circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1, 
                                        edgecolor="lightskyblue", facecolor="none", 
                                        alpha=detections[i, 16])
                ax.add_patch(circle)
        
    plt.show()


# ## Load the model

# In[ ]:


import sys
sys.path.insert(0, "/kaggle/input/blazeface-pytorch")


# In[ ]:


from blazeface import BlazeFace

net = BlazeFace().to(gpu)
net.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")
net.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy")

# Optionally change the thresholds:
net.min_score_thresh = 0.75
net.min_suppression_threshold = 0.3


# ## Load an image
# 

# In[ ]:


input_dir = "/kaggle/input/deepfake-detection-challenge/test_videos"
video_path = os.path.join(input_dir, np.random.choice(os.listdir(input_dir)))
video_path


# In[ ]:


video_path = "/kaggle/input/deepfake-detection-challenge/test_videos/uhrqlmlclw.mp4"


# The input image should be 128x128. BlazeFace will not automatically resize the image, you have to do this yourself!
# 
# The images from the Deepfake Detection Challenge dataset are 1920x1080, so resizing to 128x128 will squash them a bit. The version of BlazeFace we're using here doesn't work very well on small faces, so you may need to be more clever about how you resize/crop the input images.

# In[ ]:


def read_frame(video_path):
    capture = cv2.VideoCapture(video_path)
    ret, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128))
    capture.release()
    return frame


# In[ ]:


frame = read_frame(video_path)


# In[ ]:


plt.imshow(frame)


# ## Make the prediction

# In[ ]:


get_ipython().run_line_magic('time', 'detections = net.predict_on_image(frame)')
detections.shape


# This returns a PyTorch tensor of shape `(num_faces, 17)`. 
# 
# How to interpret these 17 numbers:
# 
# - The first 4 numbers describe the bounding box corners: 
#     - `ymin, xmin, ymax, xmax`
#     - These are normalized coordinates (between 0 and 1).
#     - Note that y comes before x here!
# - The next 12 numbers are the x,y-coordinates of the 6 facial landmark keypoints:
#     - `right_eye_x, right_eye_y`
#     - `left_eye_x, left_eye_y`
#     - `nose_x, nose_y`
#     - `mouth_x, mouth_y`
#     - `right_ear_x, right_ear_y`
#     - `left_ear_x, left_ear_y`
#     - Tip: these labeled as seen from the perspective of the person, so their right is your left.
# - The final number is the confidence score that this detection really is a face.
# 
# If no faces are found, the tensor has shape `(0, 17)`.

# In[ ]:


detections


# In[ ]:


plot_detections(frame, detections)


# ## Prediction on a batch

# In[ ]:


frame1 = read_frame("/kaggle/input/deepfake-detection-challenge/test_videos/jyfvaequfg.mp4")
frame2 = read_frame("/kaggle/input/deepfake-detection-challenge/test_videos/gkutjglghz.mp4")

batch = np.stack([frame1, frame2])
batch.shape


# In[ ]:


get_ipython().run_line_magic('time', 'detections = net.predict_on_batch(batch)')


# The batch prediction returns a list of PyTorch tensors, one for each image in the batch.

# In[ ]:


len(detections)


# In[ ]:


[x.shape[0] for x in detections]


# In[ ]:


plot_detections(frame1, detections[0])


# In[ ]:


plot_detections(frame2, detections[1])


# In[ ]:




