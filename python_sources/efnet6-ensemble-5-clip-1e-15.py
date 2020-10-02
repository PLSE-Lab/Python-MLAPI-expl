#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import numpy as np
import pandas as pd
import cv2
from keras import layers, Model


# In[ ]:


get_ipython().system('pip install ../input/efnet6weights/efficientnet-1.1.0-py3-none-any.whl')
import efficientnet.keras as efn


# In[ ]:


# Taken from here: https://github.com/1adrianb/face-alignment
# Thanks to Adrian Bulat (https://github.com/1adrianb)
# Licensed under BSD 3-Clause, which allows modification, distribution, commercial and private use
# Imported as package because pip installation requires internet to download weights for detector
# I've modified the code to remove all stuff related to landmarks extraction, so I've used only s3fd detection module
sys.path.insert(0, "../input")
import s3fdfacedetector

fd = s3fdfacedetector.S3FDFaceDetector()


# In[ ]:


# Big thanks to Human Analog
# Taken from here: https://www.kaggle.com/humananalog/deepfakes-inference-demo
class VideoReader:
    """Helper class for reading one or more frames from a video file."""

    def __init__(self, verbose=True):
        """Creates a new VideoReader.

        Arguments:
            verbose: whether to print warnings and error messages
            insets: amount to inset the image by, as a percentage of 
                (width, height). This lets you "zoom in" to an image 
                to remove unimportant content around the borders. 
                Useful for face detection, which may not work if the 
                faces are too small.
        """
        self.verbose = verbose

    def read_frames_at_indices(self, path, frame_idxs):
        """Reads frames from a video and puts them into a NumPy array.

        Arguments:
            path: the video file
            frame_idxs: a list of frame indices. Important: should be
                sorted from low-to-high! If an index appears multiple
                times, the frame is still read only once.

        Returns:
            - a NumPy array of shape (num_frames, height, width, 3)
            - a list of the frame indices that were read

        Reading stops if loading a frame fails, in which case the first
        dimension returned may actually be less than num_frames.

        Returns None if an exception is thrown for any reason, or if no
        frames were read.
        """
        assert len(frame_idxs) > 0
        capture = cv2.VideoCapture(path)
        result = self._read_frames_at_indices(path, capture, frame_idxs)
        capture.release()
        return result

    def _read_frames_at_indices(self, path, capture, frame_idxs):
        try:
            frames = []
            idxs_read = []
            for frame_idx in range(frame_idxs[0], frame_idxs[-1] + 1):
                ret = capture.grab()
                if not ret:
                    if self.verbose:
                        print("Error grabbing frame %d from movie %s" % (frame_idx, path))
                    break

                current = len(idxs_read)
                if frame_idx == frame_idxs[current]:
                    ret, frame = capture.retrieve()
                    if not ret or frame is None:
                        if self.verbose:
                            print("Error retrieving frame %d from movie %s" % (frame_idx, path))
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    idxs_read.append(frame_idx)

            if len(frames) > 0:
                return np.stack(frames), idxs_read
            if self.verbose:
                print("No frames read from movie %s" % path)
            return None
        except:
            if self.verbose:
                print("Exception while reading movie %s" % path)
            return None
        
video_reader = VideoReader()


# In[ ]:


test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"
test_videos = sorted([x for x in os.listdir(test_dir)])


# In[ ]:


weights = ["efnet_6_2020-03-26_14-08-45_epoch_06",
           "efnet_6_2020-03-27_21-28-54_epoch_06",
           "efnet_6_2020-03-29_21-28-21_epoch_06",
           "efnet_6_2020-03-29_21-29-50_epoch_06",
           "efnet_6_2020-03-30_00-26-00_epoch_06"
          ]


# In[ ]:


def efnet(backbone='6', input_shape=(224,224,3), num_classes=1):
    
    DENSE_KERNEL_INITIALIZER = {
        'class_name': 'VarianceScaling',
        'config': {
            'scale': 1. / 3.,
            'mode': 'fan_out',
            'distribution': 'uniform'
        }
    }
    
    if backbone == '0':
        EFNet = efn.EfficientNetB0
    elif backbone == '1':
        EFNet = efn.EfficientNetB1
    elif backbone == '2':
        EFNet = efn.EfficientNetB2
    elif backbone == '3':
        EFNet = efn.EfficientNetB3
    elif backbone == '4':
        EFNet = efn.EfficientNetB4
    elif backbone == '5':
        EFNet = efn.EfficientNetB5
    elif backbone == '6':
        EFNet = efn.EfficientNetB6
    elif backbone == '7':
        EFNet = efn.EfficientNetB7

    base_model = EFNet(input_shape=input_shape, weights=None, include_top=False, classes=num_classes)
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(num_classes, activation='sigmoid', kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)
        
    return Model(inputs=[base_model.input], outputs=[output])


# In[ ]:


models = []

for i in range(len(weights)):
    models.append(efnet())


# In[ ]:


for model, weight in zip(models, weights):
    model.load_weights('/kaggle/input/efnet6weights/' + weight + '.hdf5')


# In[ ]:


faces_batch_size = 55

frames_per_video = 55

face_size = 224

clip = 1e-15

ws = [1/(len(models))]*len(models)


# In[ ]:


out_probs = []

for video in test_videos:
    
    fake_probs = []
    faces_batch = []
    
    try:
        capture = cv2.VideoCapture(f'{test_dir+video}')
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs = np.linspace(0, frame_count - 1, frames_per_video, endpoint=True, dtype=np.int)
        result = video_reader.read_frames_at_indices(f'{test_dir+video}', frame_idxs=frame_idxs)
        
        size_multiplier = 2 if min(result[0][0][:,:,0].shape) < 1080 else 4
        resize_size = tuple(int(ti/size_multiplier) for ti in result[0][0][:,:,0].shape[::-1])

        smalls = min(result[0][0][:,:,0].shape)//20
        multiplier = 1080/min(result[0][0][:,:,0].shape)

        for original_image in result[0]:

            resized_image = cv2.resize(original_image, resize_size)
            predictions = fd.get_detections(resized_image)

            try:
                prediction = predictions[0]
            except:
                print("Unable to find face")
            else:
                for prediction in predictions:
                        
                    left, right, top, bottom = size_multiplier*int(prediction[0]), size_multiplier*int(prediction[2]), size_multiplier*int(prediction[1]), size_multiplier*int(prediction[3])
                    
                    if left < 0 or right < 0 or top < 0 or bottom < 0 or left > original_image.shape[1] or right > original_image.shape[1] or top > original_image.shape[0] or bottom > original_image.shape[0]:
                        continue

                    top += int((bottom - top)/4)
                    left += int((right - left)/12)
                    right -= int((right - left)/12)
                    
                    if (right-left < smalls) or (bottom - top < smalls):
                        continue
                        
                    face_max_side = max(bottom - top, right-left)

                    if face_max_side <= 64/multiplier:
                        face_offset = 8/multiplier
                    elif face_max_side <= 128/multiplier:
                        face_offset = 12/multiplier
                    elif face_max_side <= 192/multiplier:
                        face_offset = 16/multiplier
                    elif face_max_side <= 256/multiplier:
                        face_offset = 20/multiplier
                    else:
                        face_offset = 24/multiplier

                    face_offset = int(face_offset)

                    face_image = cv2.resize(original_image[max(top-face_offset, 0):min(bottom+face_offset, resize_size[1]*size_multiplier), max(left-face_offset, 0):min(right+face_offset, resize_size[0]*size_multiplier)], (face_size, face_size))
                    faces_batch.append(face_image)
                    
        while len(faces_batch) > 0:

            batch_size = min(len(faces_batch), faces_batch_size)
            x = np.zeros((batch_size, face_size, face_size, 3))

            for b in range(batch_size):
                x[b] = faces_batch[b]

            del(faces_batch[:batch_size])

            x = efn.preprocess_input(x)
            
            ys = []
            for model in models:
                ys.append(model.predict(x))

            for (y0, y1, y2, y3, y4) in zip(ys[0], ys[1], ys[2], ys[3], ys[4]):
                fake_probs.append(ws[0]*y0[-1] + ws[1]*y1[-1] + ws[2]*y2[-1] + ws[3]*y3[-1] + ws[4]*y4[-1])

    except Exception as e:
        print("Prediction error on video %s: %s" % (video, str(e)))
        
    if len(fake_probs) == 0:
        out_prob = 0.5
    else:
        out_prob = np.median(fake_probs).clip(clip, 1-clip)
        
    out_probs.append(out_prob)


# In[ ]:


submission_df = pd.DataFrame({"filename": test_videos, "label": out_probs})
submission_df.to_csv("submission.csv", index=False)

