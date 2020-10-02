#!/usr/bin/env python
# coding: utf-8

# Access to original videos to give a simple impression on annotations...

# ## Loading libraries & datasets

# In[ ]:


import tensorflow as tf
import numpy as np
from IPython.display import YouTubeVideo


# ## Validation: Collect Video-level information

# In[ ]:


vid_ids = []
labels = []
seg_start = []
seg_end = []
seg_label = []
seg_scores = []
validate_record = "../input/validate-sample/validate/validate00.tfrecord"
for example in tf.python_io.tf_record_iterator(validate_record):
    tf_example = tf.train.Example.FromString(example)
    vid_ids.append(tf_example.features.feature['id']
                   .bytes_list.value[0].decode(encoding='UTF-8'))
    labels.append(tf_example.features.feature['labels'].int64_list.value)
    seg_start.append(tf_example.features.feature['segment_start_times'].int64_list.value)
    seg_end.append(tf_example.features.feature['segment_end_times'].int64_list.value)
    seg_label.append(tf_example.features.feature['segment_labels'].int64_list.value)
    seg_scores.append(tf_example.features.feature['segment_scores'].float_list.value)


# In[ ]:


import pandas as pd
vocab = pd.read_csv('../input/vocabulary.csv')
label_mapping =  vocab[['Index', 'Name']].set_index('Index', drop=True).to_dict()['Name']


# ## Play the first video
# 
# Goto http://data.yt8m.org/2/j/i/Iv/Iv00.js we can find the video id is **DxWJGOZL1co**

# In[ ]:


print('The first video id:',vid_ids[0])
print('Label of this video:',labels[0])
print('Segment start of this video:',seg_start[0])
print('Segment label of this video:',seg_label[0])
print('Segment Score of this video:',seg_scores[0])
print('Segment names of this video:',[label_mapping[x] for x in list(set(seg_label[0]))])


# ### Translation from annotation
# 2:25 - Not Laser lighting display; 1:50 - Not Laser lighting display; 2:15 - Laser lighting display; 
# 
# 2-35 - Not Laser lighting display; 1:10 - Not Laser lighting display

# In[ ]:


YouTubeVideo('DxWJGOZL1co')


# ## Watch the next video
# Find i("ww00","JdYkqQFprUI") from the URL

# In[ ]:


print('The 2nd video id:',vid_ids[1])
print('Label of this video:',labels[1])
print('Segment start of this video:',seg_start[1])
print('Segment label of this video:',seg_label[1])
print('Segment Score of this video:',seg_scores[1])
print('Segment names of this video:',[label_mapping[x] for x in list(set(seg_label[1]))])


# In[ ]:


YouTubeVideo('JdYkqQFprUI')


# ## One more for video *f9wADEgGuH8*

# In[ ]:


print('The next video id:',vid_ids[2])
print('Label of this video:',labels[2])
print('Segment start of this video:',seg_start[2])
print('Segment label of this video:',seg_label[2])
print('Segment Score of this video:',seg_scores[2])
print('Segment names of this video:',[label_mapping[x] for x in list(set(seg_label[2]))])


# ### Translation from annotation
# 1:55 - Mountain; 3:10 - Not Mountain; 0:40 - Not Mountain; 1:00 - Mountain; 1:50 - Not Mountain

# In[ ]:


YouTubeVideo('f9wADEgGuH8')


# ## One more: 4C8kuTvHXqQ

# In[ ]:


print('The next video id:',vid_ids[3])
print('Label of this video:',labels[3])
print('Segment start of this video:',seg_start[3])
print('Segment label of this video:',seg_label[3])
print('Segment Score of this video:',seg_scores[3])
print('Segment names of this video:',[label_mapping[x] for x in list(set(seg_label[3]))])


# ### Translation from annotation
# 2:05 - Spinach; 2:15 - Spinach; 4:05 - Spinach; 3:30 - Not Spinach; 1:30 - Not Spinach

# In[ ]:


YouTubeVideo('4C8kuTvHXqQ')

