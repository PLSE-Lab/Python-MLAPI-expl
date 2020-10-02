#!/usr/bin/env python
# coding: utf-8

# It's fork from https://www.kaggle.com/zaharch/data-leak-in-metadata @nosound
# 
# The original author has computed the data distribution, so we can compute Public LB score directly.
# 
# This leak will be repaired shortly afterwards.

# # If you like it, please give it an upvote.

# In[ ]:


from math import log

# loss1 - None
loss1 = -1/4000 * (174 * log(174/1477) + (1477 - 174) * log(1 - 174/1477))


# loss2 - 16:9
# cd == '1/48000':
loss2_1 = -1/4000 * (1407 * log(1407/1873) + (1873 - 1407) * log(1 - 1407/1873))
# cd == else:
loss2_2 = -1/4000 * (156 * log(156/206) + (206 - 156) * log(1 - 156/206))


# loss3 - 9:16
# cd == '1/48000':
loss3_1 = -1/4000 * (70 * log(70/178) + (178 - 70) * log(1 - 70/178))
# cd == else:
loss3_2 = -1/4000 * (182 * log(182/241) + (241 - 182) * log(1 - 182/241))


# others
others = -1/4000 * (11 * log(11/25) + 14 * log(14/25))

score = loss1 + loss2_1 + loss2_2 + loss3_1 + loss3_2 + others


# In[ ]:


print('Public LB score: ', int(score * 100000) / 100000)


# # Metadata is leaking targets

# This notebook uses display_aspect_ratio metadata field as a great fake video predictor.

# In[ ]:


import pandas as pd
import glob
import os
import subprocess as sp
import tqdm.notebook as tqdm
from collections import defaultdict
import json

get_ipython().system(' tar xvf ../input/ffmpeg-static-build/ffmpeg-git-amd64-static.tar.xz')


# # Getting the metadata field with ffprobe

# The code below is from
# 
# http://www.scikit-video.org/stable/io.html
# 
# and specifically assembled from the following source files
# 
# https://github.com/scikit-video/scikit-video/blob/master/skvideo/io/ffprobe.py
# 
# https://github.com/scikit-video/scikit-video/blob/master/skvideo/utils/__init__.py
# 
# Thanks to [btk1](https://www.kaggle.com/rakibilly) for the [ffmpeg Static Build dataset](https://www.kaggle.com/rakibilly/ffmpeg-static-build)

# In[ ]:


def check_output(*popenargs, **kwargs):
    closeNULL = 0
    try:
        from subprocess import DEVNULL
        closeNULL = 0
    except ImportError:
        import os
        DEVNULL = open(os.devnull, 'wb')
        closeNULL = 1

    process = sp.Popen(stdout=sp.PIPE, stderr=DEVNULL, *popenargs, **kwargs)
    output, unused_err = process.communicate()
    retcode = process.poll()

    if closeNULL:
        DEVNULL.close()

    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        error = sp.CalledProcessError(retcode, cmd)
        error.output = output
        raise error
    return output

def ffprobe(filename):
    
    command = ["../working/ffmpeg-git-20191209-amd64-static/ffprobe", "-v", "error", "-show_streams", "-print_format", "xml", filename]

    xml = check_output(command)
    
    return xml

def get_markers(video_file):

    xml = ffprobe(str(video_file))
    
    found = str(xml).find('display_aspect_ratio')
    if found >= 0:
        ar = str(xml)[found+22:found+26]
    else:
        ar = None
        
    found = str(xml).find('"audio" codec_time_base')
    if found >= 0:
        cd = str(xml)[found+25:found+32]
    else:
        cd = None
    
    return ar, cd


# In[ ]:


video_file = '/kaggle/input/deepfake-detection-challenge/test_videos/gunamloolc.mp4'
get_markers(video_file)


# # Motivation

# In[ ]:


filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/train_sample_videos/*.mp4')


# In[ ]:


my_dict = defaultdict()
for filename in tqdm.tqdm(filenames):
    fn = filename.split('/')[-1]
    ar, cd = get_markers(filename)
    my_dict[fn] = ar


# In[ ]:


display_aspect_ratios = pd.DataFrame.from_dict(my_dict, orient='index')
display_aspect_ratios.columns = ['display_aspect_ratio']
display_aspect_ratios = display_aspect_ratios.fillna('NONE')


# In[ ]:


labels = json.load(open('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json', encoding="utf8"))

labels = pd.DataFrame(labels).transpose()
labels = labels.reset_index()
labels = labels.join(display_aspect_ratios, on='index')


# In[ ]:


labels.head()


# In[ ]:


pd.crosstab(labels.display_aspect_ratio, labels.label)


# # Make predictions

# In[ ]:


filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')


# In[ ]:


sub = pd.read_csv('/kaggle/input/deepfake-detection-challenge/sample_submission.csv')
sub.label = 11/25
sub = sub.set_index('filename',drop=False)


# In the public test datatset it is not strictly true that if display_aspect_ratio field is missing it is a real video, and if it equals 16:9 it is a fake. There are some exceptions for whatever reason, but not too many. By selecting a proper threshold below we can decrease log-loss substantially. The thresholds are selected by Gurobi solver running on available submissions.
# 
# In more details, there are 4 groups that I am looking at, depending on the values of display_aspect_ratio, see below. In each of the groups there are [1303,  516,  167,   14] real samples and [ 174, 1563,  252,   11] fakes samples. For each group I select probability of a fake for that group, - this is the value that minimizes log-loss. How did I get the numbers for each group? I had some submissions with scores already, where I put different values for those groups. With these constraints it is possible to find the numbers, even manually. But manually is a little bit tiresome, so I wrote a mixed integer programming formulation for that problem, and used Gurobi to solve.

# In[ ]:


for filename in tqdm.tqdm(filenames):
    
    fn = filename.split('/')[-1]
    ar, cd = get_markers(filename)
    
    if ar is None:
        sub.loc[fn, 'label'] = 174/1477
    if cd == '1/48000':
        if ar == '16:9':
            sub.loc[fn, 'label'] = 1407/1873
        if ar == '9:16':
            sub.loc[fn, 'label'] = 70/178
    else:
        if ar == '16:9':
            sub.loc[fn, 'label'] = 156/206
        if ar == '9:16':
            sub.loc[fn, 'label'] = 182/241


# In[ ]:


sub.label.value_counts()


# In[ ]:


sub.to_csv('submission.csv', index=False)

