#!/usr/bin/env python
# coding: utf-8

# # Working with facenet-pytorch and decord
# 
# As of version 2.2, the MTCNN module of facenet-pytorch can work directly with images represented as numpy arrays. This change achieves higher performance when reading video frames with either `cv2.VideoCapture` or `decord.VideoReader` as it avoids conversion to PIL format. A number of additional enhancements have been added to improve detection efficiency.
# 
# **This notebook demonstrates how to detect every face in every frame in every video of the dataset at full resolution in approximately 3 hours.**
# 
# ---
# 
# **UPDATE (2020-03-04):** Video reading has been switched from cv2 to decord for improved performance.
# 
# ---

# In[ ]:


get_ipython().run_cell_magic('capture', '', '# Install facenet-pytorch (with internet use "pip install facenet-pytorch")\n!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.2.9-py3-none-any.whl\n!cp /kaggle/input/decord/install.sh . && chmod  +x install.sh && ./install.sh')


# ## Imports

# In[ ]:


import sys, os
sys.path.insert(0,'/kaggle/working/reader/python')

from facenet_pytorch import MTCNN
import torch
import cupy
from decord import VideoReader, gpu
import glob
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ## The FastMTCNN Class
# 
# The following class implements a strided version of MTCNN. See [here](https://www.kaggle.com/timesler/fast-mtcnn-detector-55-fps-at-full-resolution) for the original implementation.

# In[ ]:


class FastMTCNN(object):
    """Fast MTCNN implementation."""
    
    def __init__(self, stride, *args, **kwargs):
        """Constructor for FastMTCNN class.
        
        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.
        
        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.mtcnn = MTCNN(*args, **kwargs)
        
    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
                      
        boxes, probs = self.mtcnn.detect(frames[::self.stride])

        faces = []
        probs_out = []
        frame_index = []
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            for box, prob in zip(boxes[box_ind], probs[box_ind]):
                box = [int(b) for b in box]
                faces.append(frame[box[1]:box[3], box[0]:box[2]].copy())
                probs_out.append(prob)
                frame_index.append(i)
                
        
        return faces, probs, frame_index


# ## Define face detector
# 
# The following face detector can detect all faces in a video in approximately 2.8 seconds, allowing all videos in the public test set to be processed in 2.8 * 4000 = 11200 seconds = 3.1 hours.

# In[ ]:


fast_mtcnn = FastMTCNN(
    stride=10,
    margin=20,
    factor=0.6,
    keep_all=True,
    device=device,
    thresholds=[0.6, 0.7, 0.98]
)


# ## Process all videos

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef mean_detection_prob(prob):\n    cnt_p = 0\n    sum_p = 0\n    for p in prob:\n        for pp in p:\n            if pp is not None:\n                cnt_p += 1\n                sum_p += pp\n    return sum_p / cnt_p\n\n\ndef get_frames(filename, batch_size=30):\n    v_cap = VideoReader(filename, ctx=gpu())\n    v_len = len(v_cap)\n\n    frames = []\n    for i in range(0, v_len, batch_size):\n        batch = v_cap.get_batch(range(i, min(i + batch_size, v_len - 1))).asnumpy()\n        frames.extend(batch.copy())\n    \n    frames = np.array(frames)\n    \n    del v_cap, v_len, batch\n    \n    return frames\n\n\nfilenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')\n\nnum_faces = 0\nprobs = []\nindexes = []\npbar = tqdm(filenames)\nfor filename in pbar:\n    frames = get_frames(filename)\n\n    faces, prob, index = fast_mtcnn(frames)        \n    probs.append(mean_detection_prob(prob))\n\n    num_faces += len(faces)\n    pbar.set_description(f'Faces found: {num_faces}')\n\n    del frames")


# In[ ]:


probs = np.asarray(probs)
probs = np.clip((1 - probs) ** (1 / 6) * 1.7, 0.0, 1.0)
plt.hist(probs, 40);

filenames = [os.path.basename(f) for f in filenames]

submission = pd.DataFrame({'filename': filenames, 'label': probs})
submission.to_csv('submission.csv', index=False)
submission

