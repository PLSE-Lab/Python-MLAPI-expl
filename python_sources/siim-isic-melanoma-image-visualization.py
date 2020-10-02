#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import pydicom as dicom
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import cv2

print("default backend: ", matplotlib.get_backend())
matplotlib.use('Agg')


# In[ ]:


# path
df = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")
df = df.sort_values(by="patient_id").reset_index(drop=True)

test_df = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")
test_df = test_df.sort_values(by="patient_id").reset_index(drop=True)

TEST_IMG_PATH = "/kaggle/input/siim-isic-melanoma-classification/test/"
TRAIN_IMG_PATH = "/kaggle/input/siim-isic-melanoma-classification/train/"

# image resolution
WIDTH = 500
HEIGHT = 320
N_CHANNELS = 3

# framrate
FRAMERATE = 2


# plot and save np.array as image file with meta annotation

# In[ ]:


def plot(img_path, img_name, annotate,
         w=WIDTH, h=HEIGHT, dpi=80):
    
    # data
    ds = dicom.dcmread(img_path)
    image = cv2.resize(ds.pixel_array, (w, h))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    annotate = [str(key) + ": " + str(value)+"\n"
                    for key, value in annotate.items()]

    figsize = w / float(dpi), h / float(dpi)

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
  
    ax.imshow(image, interpolation='nearest')
    ax.axis('off')
    
    # annotate
    ax.text(280, 75, "".join(annotate), 
            color="black", weight="bold",
            fontsize=7,
            bbox=dict(boxstyle="round",
                        facecolor='#BF02F6', 
                        alpha=0.3, 
                        edgecolor='black'
                        )
                )
    # save
    plt.savefig(img_name, bbox_inches='tight', 
                transparent=False, pad_inches=0.006, dpi=dpi)
    
    return plt.close()


# Writing Video from images(np arrays)

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain_video_name = \'train_data.avi\'\ntest_video_name = \'test_data.avi\'\n\ndef make_video(df, data_path, \n               video_name, framerate=FRAMERATE,\n               width=WIDTH, height=HEIGHT):\n    \n    # Define the codec and create VideoWrite object\n    fourcc = cv2.VideoWriter_fourcc(*\'XVID\')\n    video = cv2.VideoWriter(video_name, fourcc, framerate, (width, height))\n\n    for i in range(len(df)):\n            \n        meta_data = df.iloc[i, :].to_dict()\n        img_path = data_path + meta_data["image_name"] + ".dcm"\n\n        img_name = "skin.png"\n\n        # create image file\n        plot(img_path, img_name, meta_data)\n        \n        # write as np array remove as image\n        video.write(cv2.imread(img_name))\n        os.remove(img_name)\n        \n\n    cv2.destroyAllWindows()\n    return video.release()\n\nmake_video(df, TRAIN_IMG_PATH, train_video_name)\n#make_video(test_df, TEST_IMG_PATH, test_video_name)')


# adding sound

# In[ ]:


get_ipython().system('pip install youtube_dl -qq ')


# In[ ]:


# https://stackoverflow.com/questions/27473526/download-only-audio-from-youtube-video-using-youtube-dl-in-python-script
from __future__ import unicode_literals
import youtube_dl


ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=DnCJBZVTiJ4']);


# In[ ]:


# convert from avi to mp4 
get_ipython().system('ffmpeg -i train_data.avi -strict -2 train_data.mp4')

# adding sound
get_ipython().system('ffmpeg  -stream_loop -1 -i "Deus Ex - Human Revolution - Detroit Limb Clinic (1 Hour of Music & Ambience)-DnCJBZVTiJ4.mp3" -c copy -v 0 -f nut - | ffmpeg -thread_queue_size 10K -i - -i train_data.mp4 -c copy -map 1:v -map 0:a -shortest -y train_data_with_sound.mp4')
              
# piece of video
#!ffmpeg -ss 00:00:00 -i stats_with_sound.mp4 -to 00:01:00 -c copy stats_clip.mp4   


# time codes

# In[ ]:


# from num_images to timecode
def get_timecode(n_images, framerate=FRAMERATE):
    seconds = n_images / framerate
    return '{h:02d}:{m:02d}:{s:02d}'             .format(h=int(seconds/3600),
                    m=int(seconds/60%60),
                    s=int(seconds%60))

unique_patients = np.unique(df["patient_id"].values, 
                             return_index=True)

timecodes = [f"{n} - {get_timecode(t)}" for n, t 
                    in zip(unique_patients[0], unique_patients[1])]
#timecodes[:10]

with open('timecodes.txt', 'w') as f:
    f.write(f"Timestamps:\nPatient id    Time\n")
    for item in timecodes:
        f.write(f"\n\n{item}")        


# In[ ]:


get_ipython().run_cell_magic('HTML', '', '<video width="800" height="600" controls>\n  <source src="train_data_with_sound.mp4" type="video/mp4">\n</video>')


# In[ ]:


get_ipython().system('head -n 20 timecodes.txt')


# ### [High resolution video](https://www.youtube.com/watch?v=CZmCkOhjv14)

# ### Full list of time codes
# 

# In[ ]:


get_ipython().system('cat timecodes.txt')

