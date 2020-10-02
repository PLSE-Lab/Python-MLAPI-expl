#!/usr/bin/env python
# coding: utf-8

# In this kernel goal is to simply extract video batches to be fed to a model. For this we will compare vanilla open-cv. FileVideoStream from imutils and excellent latest contribution: Decord from https://www.kaggle.com/leighplt/decord-videoreader/notebook.
# 
# ### Steps
# 
# - Read frames with sample rate 10 (every 10th starting from 0th index) ~ batch of 30 frames per video
# - Resize frames
# - Convert to pytorch batch 

# TODO: Add DALI as a dataset, or if it's available let me know :)

# In[ ]:


get_ipython().system('pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali')


# ### Intro

# In[ ]:


from fastai.vision import *
import cv2 as cv


# In[ ]:


train_sample_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T.reset_index()
train_sample_metadata.columns = ['fname','label','split','original']
train_sample_metadata.head()


# In[ ]:


fake_sample_df = train_sample_metadata[train_sample_metadata.label == 'FAKE']
real_sample_df = train_sample_metadata[train_sample_metadata.label == 'REAL']


# In[ ]:


train_dir = Path('/kaggle/input/deepfake-detection-challenge/train_sample_videos/')
test_dir = Path('/kaggle/input/deepfake-detection-challenge/test_videos/')
train_video_files = get_files(train_dir, extensions=['.mp4'])
test_video_files = get_files(test_dir, extensions=['.mp4'])


# In[ ]:


len(train_video_files), len(test_video_files)


# ### Nvidia DALI

# In[ ]:


from nvidia.dali.pipeline import Pipeline
from nvidia.dali import ops


# In[ ]:


fname = train_video_files[0]


# In[ ]:


batch_size=1
sequence_length=30
initial_prefetch_size=16

class VideoPipe(Pipeline):
    "video pipeline for a single video with 30 frames"
    def __init__(self, batch_size, num_threads, device_id, data, shuffle):
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=16)
        self.input = ops.VideoReader(device="gpu", filenames=data, sequence_length=sequence_length,
                                     shard_id=0, num_shards=1,
                                     random_shuffle=shuffle, initial_fill=initial_prefetch_size)
    def define_graph(self):
        output = self.input(name="Reader")
        return output


# In[ ]:


def dali_batch(fname):
    pipe = VideoPipe(batch_size=batch_size, num_threads=defaults.cpus, device_id=0, data=[fname], shuffle=False)
    pipe.build()
    pipe_out = pipe.run()
    sequences_out = pipe_out[0].as_cpu().as_array()
    data = torch.from_numpy(sequences_out[0])
    data = data.permute(0,3,1,2).cuda()
    return F.interpolate(data.to(torch.float32), (640,640))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'data = dali_batch(train_video_files[0])')


# In[ ]:


img0 = Image(data[0]/255)
img0.show(figsize=(10,10))


# ### Vanilla opencv

# In[ ]:


def frame_img_generator(path, freq=None):
    "frame generator for a given video file"
    vidcap = cv.VideoCapture(str(path))
    n_frames = 0
    while True:
        success = vidcap.grab()
        if not success: 
            vidcap.release()
            break   
            
        if (freq is None) or (n_frames % freq == 0):
            _, frame = vidcap.retrieve()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#             frame = cv.resize(frame, (640,640))
            yield frame    
        
        n_frames += 1
        
    vidcap.release()


# In[ ]:


get_ipython().run_cell_magic('time', '', '# CPU warm up\nframes = list(frame_img_generator(train_video_files[0], 10)); len(frames)')


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'frames = list(frame_img_generator(train_video_files[0], freq=10)); len(frames)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'frames = frame_img_generator(train_video_files[0], 10)\ndata = torch.from_numpy(array(frames))\ndata = data.permute(0,3,1,2).cuda()\ndata = F.interpolate(data.to(torch.float32), (640,640))')


# In[ ]:


img1 = Image(data[0]/255)
img1.show(figsize=(10,10))


# In[ ]:


del frames
del data; gc.collect()


# ### imutils: FileVideoStream

# In[ ]:


get_ipython().system('pip install -q /kaggle/input/imutils/imutils-0.5.3')


# In[ ]:


from imutils.video import FileVideoStream


# In[ ]:


def fvs_img_generator(path, freq=None):
    "frame generator for a given video file"
    fvs = FileVideoStream(str(path)).start()
    n_frames = 0
    while fvs.more():
        frame = fvs.read()
        if frame is None: break # https://github.com/jrosebr1/imutils/pull/119
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        if (freq is None) or (n_frames % freq == 0):
            yield frame
        
        n_frames += 1
    fvs.stop()


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'frames = list(fvs_img_generator(str(train_video_files[0]), 10))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'frames = list(fvs_img_generator(str(train_video_files[0]), 10))\ndata = torch.from_numpy(array(frames))\ndata = data.permute(0,3,1,2).cuda()\ndata = F.interpolate(data.to(torch.float32), (640,640))')


# In[ ]:


img2 = Image(data[0]/255)
img2.show(figsize=(10,10))


# In[ ]:


del frames
del data; gc.collect()


# In[ ]:


assert torch.all(img1.data == img2.data)


# ### Decord Reader GPU
# 
# Thanks to: https://www.kaggle.com/leighplt/decord-videoreader/data

# In[ ]:


get_ipython().system('cp /kaggle/input/decord/install.sh . && chmod  +x install.sh && ./install.sh ')


# In[ ]:


sys.path.insert(0,'/kaggle/working/reader/python')

from decord import VideoReader
from decord import cpu, gpu
from decord.bridge import set_bridge
set_bridge('torch')


# In[ ]:


# GPU warm up
video = VideoReader(str(train_video_files[0]), ctx=gpu())
del video; gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'video = VideoReader(str(train_video_files[0]), ctx=gpu())\ndata = video.get_batch(range(0, len(video), 10))\ndata = F.interpolate(data.to(torch.float32), (640,640))')


# In[ ]:


img3 = Image(data[0]/255)
img3.show(figsize=(10,10))


# In[ ]:


del video
del data; gc.collect()


# One thing we can notice is that Decord GPU is not given exactly same results whereas previous 2 methods give exact same pixel level results. Let's check how close both results are.
# 
# 98% of the pixels are within -+0.01 difference.

# In[ ]:


torch.mean(torch.isclose(img1.data, img3.data, atol=0.01).float())


# ### Decord Reader CPU
# 
# On CPU we don't see any pixel level difference but it's slower.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'video = VideoReader(str(train_video_files[0]), ctx=cpu())\ndata = video.get_batch(range(0, len(video), 10)).cuda()\ndata = F.interpolate(data.to(torch.float32), (640,640))')


# In[ ]:


img4 = Image(data[0]/255)
img4.show(figsize=(10,10))


# In[ ]:


del video
del data; gc.collect()


# In[ ]:


assert torch.all(img1.data == img4.data)


# NVIDIA Dali seems to be more accurate

# In[ ]:


torch.mean(torch.isclose(img0.data, img1.data, atol=0.01).float())


# ### Conclusion
# 
# - Decoder GPU gives a really good boost with litle cost of ~%3 of deviated pixels within (-0.01, 0.01) range. That's a risk I am willing to take :)
# - FileVideoStream isn't much different than open-cv probably we do resizing on a GPU and don't have much CPU bound processing to get full power of threading
# - Let me know if there are anything I am mising
# - As discussed on https://www.kaggle.com/leighplt/decord-videoreader/notebook memory leaks can occur, so garbage collection is important. I also recommend using https://github.com/stas00/ipyexperiments/blob/master/docs/ipyexperiments.md
# - Feedback are welcome!

# In[ ]:




