#!/usr/bin/env python
# coding: utf-8

# # Video reading speed test
# 
# See also [this discussion topic](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/122328).
# 
# I currently have a face detector that takes about 30 ms for a batch of 30 images, but loading that batch from the video takes 330 ms (on my local computer). So video loading is 10x slower than doing face detection. Because we're dealing with a LOT of data in this competition, it's worthwhile to reduce any overhead where possible.
# 
# > Note that loading speeds appear to be slower on Kaggle than on my local machine (1.2 seconds instead of 330 ms for the same video). Not sure why that is. On my local machine the videos are stored on HDD, not SSD.

# In[ ]:


import os
import cv2
import numpy as np


# Pick a video at random.

# In[ ]:


train_dir = "/kaggle/input/deepfake-detection-challenge/train_sample_videos"
video_path = os.path.join(train_dir, np.random.choice(os.listdir(train_dir)))
video_path


# This version uses OpenCV. It looks at all the frames in the video but only decodes the ones we're interested in:

# In[ ]:


def grab_frames_from_video(path, num_frames=10):
    capture = cv2.VideoCapture(path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, frame_count, num_frames, endpoint=False, dtype=np.int)

    i = 0
    for frame_idx in range(int(frame_count)):
        # Get the next frame, but don't decode if we're not using it.
        ret = capture.grab()
        if not ret: 
            print("Error grabbing frame %d from movie %s" % (frame_idx, path))

        # Need to look at this frame?
        if frame_idx >= frame_idxs[i]:
            ret, frame = capture.retrieve()
            if not ret or frame is None:
                print("Error retrieving frame %d from movie %s" % (frame_idx, path))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Do something with `frame`

            i += 1
            if i >= len(frame_idxs):
                break

    capture.release()


# In[ ]:


get_ipython().run_line_magic('time', 'grab_frames_from_video(video_path, num_frames=10)')


# 1.2 seconds for reading a single video. Ouch. With 4000 videos that is 1.5 hours to read the entire test set.

# In[ ]:


get_ipython().run_line_magic('time', 'grab_frames_from_video(video_path, num_frames=50)')


# At least reading more frames doesn't make the time much worse...

# The next version jumps directly to the frame you want to read. You might expect this to be faster but it actually isn't. My guess is that a "streaming" approach is more efficient than a "random access" approach because, unless you happen to grab a keyframe, the decoder still needs to read all the previous frames in order to reconstruct the one you're asking for.

# In[ ]:


# This version.

def grab_frames_from_video(path, num_frames=10):
    capture = cv2.VideoCapture(path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, frame_count, num_frames, endpoint=False, dtype=np.int)

    for i, frame_idx in enumerate(frame_idxs):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = capture.read()
        if not ret or frame is None:
            print("Error retrieving frame %d from movie %s" % (frame_idx, path))
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    capture.release()


# In[ ]:


get_ipython().run_line_magic('time', 'grab_frames_from_video(video_path, num_frames=10)')


# Yeah that's 4x slower than the other method.

# In[ ]:


get_ipython().run_line_magic('time', 'grab_frames_from_video(video_path, num_frames=50)')


# And unlike the previous method, it gets way worse the more frames you want to look at.

# In[ ]:




