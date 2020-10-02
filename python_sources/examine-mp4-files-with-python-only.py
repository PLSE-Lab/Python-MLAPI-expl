#!/usr/bin/env python
# coding: utf-8

# # Examine MP4 files
# 
# I thought it would be fun to write some Python code that digs into the MP4 files, without using any external dependencies such as `ffprobe`.
# 
# More information about the MP4 file format at the following links:
# 
# - http://xhelmboyx.tripod.com/formats/mp4-layout.txt
# - https://github.com/OpenAnsible/rust-mp4/raw/master/docs/ISO_IEC_14496-14_2003-11-15.pdf
# - https://developer.apple.com/library/archive/documentation/QuickTime/QTFF/QTFFPreface/qtffPreface.html

# In[ ]:


import os, sys
import struct
import numpy as np


# An MP4 file is a container that consists of several sections, also known as boxes, chunks, or atoms. Boxes can be nested. Getting data out of an MP4 file consists of finding the box(es) you're interested in, then digging into those.

# In[ ]:


def find_boxes(f, start_offset=0, end_offset=float("inf")):
    """Returns a dictionary of all the data boxes and their absolute starting
    and ending offsets inside the mp4 file.

    Specify a start_offset and end_offset to read sub-boxes.
    """
    s = struct.Struct("> I 4s") 
    boxes = {}
    offset = start_offset
    f.seek(offset, 0)
    while offset < end_offset:
        data = f.read(8)               # read box header
        if data == b"": break          # EOF
        length, text = s.unpack(data)
        f.seek(length - 8, 1)          # skip to next box
        boxes[text] = (offset, offset + length)
        offset += length
    return boxes


# An interesting box is **mvhd**, the "movie header", which is inside the **moov** box. Among other things, the movie header contains the duration of the video.

# In[ ]:


def scan_mvhd(f, offset):
    f.seek(offset, 0)
    f.seek(8, 1)            # skip box header

    data = f.read(1)        # read version number
    version = int.from_bytes(data, "big")
    word_size = 8 if version == 1 else 4

    f.seek(3, 1)            # skip flags
    f.seek(word_size*2, 1)  # skip dates

    timescale = int.from_bytes(f.read(4), "big")
    if timescale == 0: timescale = 600

    duration = int.from_bytes(f.read(word_size), "big")

    print("Duration (sec):", duration / timescale)


# The main function for examining an MP4 file. Right now it just prints out the file offsets of some of the more interesting boxes, such as **udta** which contains metadata and **trak** that describes the tracks in the movie file.

# In[ ]:


def examine_mp4(filename):
    print("Examining:", filename)
    
    with open(filename, "rb") as f:
        boxes = find_boxes(f)
        print(boxes)

        # Sanity check that this really is a movie file.
        assert(boxes[b"ftyp"][0] == 0)

        moov_boxes = find_boxes(f, boxes[b"moov"][0] + 8, boxes[b"moov"][1])
        print(moov_boxes)

        trak_boxes = find_boxes(f, moov_boxes[b"trak"][0] + 8, moov_boxes[b"trak"][1])
        print(trak_boxes)

        udta_boxes = find_boxes(f, moov_boxes[b"udta"][0] + 8, moov_boxes[b"udta"][1])
        print(udta_boxes)

        scan_mvhd(f, moov_boxes[b"mvhd"][0])


# Let's try it on some of the videos:

# In[ ]:


test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"
test_files = [x for x in os.listdir(test_dir) if x[-4:] == ".mp4"]

train_dir = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/"
train_files = [x for x in os.listdir(train_dir) if x[-4:] == ".mp4"]


# In[ ]:


examine_mp4(os.path.join(train_dir, np.random.choice(train_files)))


# In[ ]:


examine_mp4(os.path.join(test_dir, np.random.choice(test_files)))


# I originally wrote this code to quickly get the durations from the mp4 files but it turns out they're all about the same length (10 seconds).
# 
# I also used this to see if there was any leakage inside the mp4 files themselves, such as in the metadata, but so far I haven't found anything. ;-)
# 
# Anyway, I just wanted to show that you don't necessarily need external tools to read information from mp4 files. :-)

# In[ ]:




