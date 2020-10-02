#!/usr/bin/env python
# coding: utf-8

# # Fast MTCNN detector
# 
# This notebook demonstrates how to achieve 45 frames per second speeds for loading frames and detecting faces on full resolution videos.
# 
# ## Algorithm
# 
# **Striding**: The algorithm used is a strided modification of MTCNN in which face detection is performed on only every _N_ frames, and applied to all frames. For example, with a batch of 9 frames, we could pass frames 0, 3, and 6 to MTCNN. Then, the bounding boxes (and potentially landmarks) returned for frame 0 would be naively applied to frames 1 and 2. Similarly, the detections for frame 3 are applied to frames 4 and 5, and the detections for frames 6 are applied to frames 7 and 8.
# 
# Although this assume that faces do not move between frames significantly, this is generally a good approximation for low stride numbers. If the stride is 3, we are assuming that the face does not significantly alter position for an additional 2 frames, or ~0.07 seconds. If faces are moving faster than this, they are likely to be extremely blurry anyway. Furthermore, ensuring that faces are cropped with a small margin mitigates the impact of face drift.
# 
# **Scale pyramid**: The algorithm uses a slightly smaller scaling factor (0.6 vs 0.709) than the original MTCNN algorithm to construct the scaling pyramid applied to input images. For details of the scaling pyramid, see the [original paper](https://arxiv.org/abs/1604.02878) for details of the scaling pyramid approach.
# 
# **Multi-threading**: A modest performance gain comes from loading video frames (with `cv2.VideoCapture`) using threading. This functionality is provided by the `FileVideoStream` class of the imutils package.
# 
# ## Other resources
# 
# See the following kernel for a guide to using the MTCNN functionality of facenet-pytorch: https://www.kaggle.com/timesler/guide-to-mtcnn-in-facenet-pytorch

# In[ ]:


get_ipython().run_cell_magic('capture', '', '# Install facenet-pytorch (with internet use "pip install facenet-pytorch")\n!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.2.7-py3-none-any.whl\n!pip install /kaggle/input/imutils/imutils-0.5.3')


# ## Imports

# In[ ]:


from facenet_pytorch import MTCNN
from PIL import Image
import torch
from imutils.video import FileVideoStream
import cv2
import time
import glob
from tqdm.notebook import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')[:100]


# ## The FastMTCNN class
# 
# The class below is a thin wrapper for the MTCNN implementation in the `facenet-pytorch` package that implements the algorithm described above.

# In[ ]:


class FastMTCNN(object):
    """Fast MTCNN implementation."""
    
    def __init__(self, stride, resize=1, *args, **kwargs):
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
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        
    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                    for f in frames
            ]
                      
        boxes, probs = self.mtcnn.detect(frames[::self.stride])

        faces = []
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            for box in boxes[box_ind]:
                box = [int(b) for b in box]
                faces.append(frame[box[1]:box[3], box[0]:box[2]])
        
        return faces


# ## Full resolution detection
# 
# In this example, we demonstrate how to detect faces using full resolution frames (i.e., `resize=1`).

# In[ ]:


fast_mtcnn = FastMTCNN(
    stride=4,
    resize=1,
    margin=14,
    factor=0.6,
    keep_all=True,
    device=device
)


# In[ ]:


def run_detection(fast_mtcnn, filenames):
    frames = []
    frames_processed = 0
    faces_detected = 0
    batch_size = 60
    start = time.time()

    for filename in tqdm(filenames):

        v_cap = FileVideoStream(filename).start()
        v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

        for j in range(v_len):

            frame = v_cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            if len(frames) >= batch_size or j == v_len - 1:

                faces = fast_mtcnn(frames)

                frames_processed += len(frames)
                faces_detected += len(faces)
                frames = []

                print(
                    f'Frames per second: {frames_processed / (time.time() - start):.3f},',
                    f'faces detected: {faces_detected}\r',
                    end=''
                )

        v_cap.stop()

run_detection(fast_mtcnn, filenames)


# ## Half resolution detection
# 
# In this example, we demonstrate how to detect faces using half resolution frames (i.e., `resize=0.5`).

# In[ ]:


fast_mtcnn = FastMTCNN(
    stride=4,
    resize=0.5,
    margin=14,
    factor=0.5,
    keep_all=True,
    device=device
)


# In[ ]:


run_detection(fast_mtcnn, filenames)


# In[ ]:




