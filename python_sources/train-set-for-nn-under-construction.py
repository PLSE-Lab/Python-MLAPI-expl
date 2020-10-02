#!/usr/bin/env python
# coding: utf-8

# ****This notebook is based on @inversion's Starter Kernel. It extracts the data from TFRecord format using a subsample of the YouTube-8M `frame-level` and `validate` data. 
# 
# Corrections made over prev version.

# In[ ]:


import numpy as np 
import pandas as pd

import os
print(os.listdir("../input"))


# In[ ]:


print(os.listdir("../input/frame-sample/frame"))


# In[ ]:


# Loading libraries & datasets
import tensorflow as tf
import numpy as np
from IPython.display import YouTubeVideo

video_no1_record = "../input/frame-sample/frame/train00.tfrecord"
validate_no1_record = "../input/validate-sample/validate/validate00.tfrecord"
video_no2_record = "../input/frame-sample/frame/train01.tfrecord"
validate_no2_record = "../input/validate-sample/validate/validate01.tfrecord"

vid_file_list = [video_no1_record, video_no2_record]
val_file_list = [validate_no1_record, validate_no2_record]


# for vid in tf.python_io.tf_record_iterator(video_no1_record):
#     tf_seq_example = tf.train.SequenceExample.FromString(vid)
#     print(tf_seq_example)
#     break

# In[ ]:


vid_numb = 0
for vid_file in vid_file_list:
    for video in tf.python_io.tf_record_iterator(vid_file):
        tf_seq_example = tf.train.SequenceExample.FromString(video)
        tf_example = tf.train.Example.FromString(video)
        segment_start_times = tf_example.features.feature['segment_start_times'].int64_list.value
        segment_end_times = tf_example.features.feature['segment_end_times'].int64_list.value
        seg_rgb = tf_seq_example.feature_lists.feature_list['rgb'].feature[0].bytes_list.value[0]
        vid_numb += 1
        if vid_numb == 20:
            #print(tf_seq_example)
            break


# In[ ]:


print(vid_numb)
print(segment_start_times)
print(segment_end_times)
print(seg_rgb)


# In[ ]:


for val_file in val_file_list: 
    for example in tf.python_io.tf_record_iterator(val_file):
        tf_example = tf.train.Example.FromString(example)
        #print(tf_example)
        break


# In[ ]:


val_vid_ids = []
val_vid_labels = []
vid_segs_dict = {}

num_of_videos = 0
for val_file in val_file_list: 
    for example in tf.python_io.tf_record_iterator(val_file):
        tf_example = tf.train.Example.FromString(example)
        vid_id = tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8')
        val_vid_ids.append(vid_id)
        val_vid_labels.append(tf_example.features.feature['labels'].int64_list.value)

        seg_info_dict = {}

        segment_start_times = tf_example.features.feature['segment_start_times'].int64_list.value
        segment_end_times = tf_example.features.feature['segment_end_times'].int64_list.value
        segment_labels = tf_example.features.feature['segment_labels'].int64_list.value
        segment_scores = tf_example.features.feature['segment_scores'].float_list.value

        segment_start_times, segment_end_times, segment_labels, segment_scores            = (list(t) for t in zip(*sorted(zip(segment_start_times, segment_end_times, segment_labels, segment_scores),reverse=True)))

        seg_info_dict['seg_start_times'] = segment_start_times
        seg_info_dict['seg_end_times'] = segment_end_times
        seg_info_dict['seg_labels'] = segment_labels
        seg_info_dict['seg_scores'] = segment_scores

        vid_segs_dict[vid_id] = seg_info_dict

        num_of_videos += 1
        if num_of_videos == 20:
            #print(tf_example)
            break


# In[ ]:


print(vid_segs_dict['ww00']['seg_start_times'])


# Here is the code to break the TEST set into tfrecords of 5 sec segments. You can just use this to break the test set and then use Github starter code as explained. The dynamic RNN should yield results no matter what video length. I have not tried this myself yet, but the below code works. 

# In[ ]:


for test_file in test_files:
    new_test_filename = test_file[:8] + "NEW" + test_file[8:]
    writer = tf.python_io.TFRecordWriter(new_test_filename)
    for vid in tf.python_io.tf_record_iterator(test_file):
        tf_seq_example = tf.train.SequenceExample.FromString(vid)
        n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)
        # iterate through frames
        n_segments = n_frames // 5
        
        tf_example = tf.train.Example.FromString(vid)
        vid_id = tf_example.features.feature['id']                       .bytes_list.value[0].decode(encoding='UTF-8')
        
        for seg_no in range(n_segments):
            seg_id = vid_id + ":{}".format(seg_no*5)
            seg_id_tf = tf.train.Feature(bytes_list=tf.train.BytesList(value=[seg_id.encode('utf-8')]))
            
            sess = tf.InteractiveSession()
            
            rgb1 = tf_seq_example.feature_lists.feature_list['rgb'].                            feature[seg_no*5]
            rgb2 = tf_seq_example.feature_lists.feature_list['rgb'].                            feature[seg_no*5 + 1]
            rgb3 = tf_seq_example.feature_lists.feature_list['rgb'].                            feature[seg_no*5 + 2]
            rgb4 = tf_seq_example.feature_lists.feature_list['rgb'].                            feature[seg_no*5 + 3]
            rgb5 = tf_seq_example.feature_lists.feature_list['rgb'].                            feature[seg_no*5 + 4]
                
            aud1 = tf_seq_example.feature_lists.feature_list['audio'].                            feature[seg_no*5]
            aud2 = tf_seq_example.feature_lists.feature_list['audio'].                            feature[seg_no*5 + 1]
            aud3 = tf_seq_example.feature_lists.feature_list['audio'].                            feature[seg_no*5 +2]
            aud4 = tf_seq_example.feature_lists.feature_list['audio'].                            feature[seg_no*5 + 3]
            aud5 = tf_seq_example.feature_lists.feature_list['audio'].                            feature[seg_no*5 + 4]
            
            sess.close()
            
            rgb_list_tf = [rgb1, rgb2, rgb3, rgb4, rgb5]
            aud_list_tf =[aud1, aud2, aud3, aud4, aud5]
            
            rgb = tf.train.FeatureList(feature=rgb_list_tf)
            audio = tf.train.FeatureList(feature=aud_list_tf)
            
            seg_tf_dict = {'rgb': rgb, 'audio': audio}
            
            seg_tf_features = tf.train.FeatureLists(feature_list=seg_tf_dict)

            seg_context = tf.train.Features(feature={'id': seg_id_tf})
            
            example = tf.train.SequenceExample(context=seg_context, feature_lists=seg_tf_features)
            
            writer.write(example.SerializeToString())
            
    writer.close()


# In[ ]:


import time
start_t = time.time()


# In[ ]:


print(vid_segs_dict.keys())


# The starter data they gave us is comprised of only 30 something validation examples and over 2000 videos. Below is a script to find validation data for the video files they gave. Turns out none are there. This is only 0.4% of the entire dataset. The validation file is even smaller, like 0.01% (I don't get why).

# In[ ]:


train_vid_ids = []
train_seg_rgb = []
train_seg_aud = []
train_seg_labels = [] #label is not the target. we will train 1000 different nets for each label
train_seg_targets = []

video_num = 0
for vid_file in vid_file_list:
    for video in tf.python_io.tf_record_iterator(vid_file):

        video_num += 1

        tf_example = tf.train.Example.FromString(video)

        vid_id = tf_example.features.feature['id']                       .bytes_list.value[0].decode(encoding='UTF-8')

        if vid_id in vid_segs_dict.keys():
            vid_segment_start_times = vid_segs_dict[vid_id]['seg_start_times']
            vid_segment_end_times = vid_segs_dict[vid_id]['seg_end_times']
            vid_segment_labels = vid_segs_dict[vid_id]['seg_labels']
            vid_segment_scores = vid_segs_dict[vid_id]['seg_scores']
            print(video_num)
        else:
            continue

        tf_seq_example = tf.train.SequenceExample.FromString(video)
        n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)

        seg_rgb_record = []
        seg_audio_record = []
        recording = "no"

        sess = tf.InteractiveSession()

        # iterate through frames

        for i in range(n_frames):

            if i in vid_segment_start_times:
                vid_segment_start_times.pop()
                recording = "yes"
                train_seg_labels.append(vid_segment_labels.pop())
                train_seg_targets.append(vid_segment_scores.pop())
                train_vid_ids.append(vid_id)

            elif i in vid_segment_end_times:
                vid_segment_end_times.pop()
                recording = "no"
                train_seg_rgb.append(seg_rgb_record)
                train_seg_aud.append(seg_audio_record)
                seg_rgb_record = []
                seg_audio_record = []

            if recording == "yes":
                seg_rgb_record.append(tf.cast(tf.decode_raw(
                    tf_seq_example.feature_lists.feature_list['rgb']
                      .feature[i].bytes_list.value[0],tf.uint8)
                           ,tf.float32).eval())
                seg_audio_record.append(tf.cast(tf.decode_raw(
                    tf_seq_example.feature_lists.feature_list['audio']
                      .feature[i].bytes_list.value[0],tf.uint8)
                           ,tf.float32).eval())

        sess.close()

        if video_num == 16:
            break


# In[ ]:


print("total time: ", (time.time() - start_t)) 


# In[ ]:


print(video_num)
print(len(train_vid_ids))
print(len(train_seg_labels))
print(len(train_seg_targets))
print(len(train_seg_rgb))
print(len(train_seg_aud))
print(len(train_seg_rgb[0]))
print(len(train_seg_aud[0]))
print(len(train_seg_rgb[4]))
print(len(train_seg_aud[4]))


# Why is the validation file 3.5MB with so little info, I cannot understand. Must have some other information in there.
