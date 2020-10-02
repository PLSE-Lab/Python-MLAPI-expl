'''
Below we use Keras' VGG-16 model, pre-traied on a large, 1000-classes dataset. We then
see how to input frames from a video into the model for inference. If a frame
contains one of the objects from the original dataset, the model should "predict" it.
We will see that the video frame has to be reshaped and scaled to be as like the
training data as possible.
This works great on my machine (every programmer's excuse), but will not run in
this Kaggle Docker container because it can't find the path to ffmpeg required by scikit-video.
Too bad since scikit-video is great. It reads the frames directly as Numpy arrays. We can
then check the number and shape of the frames with Numpy's shape() method. I really wanted to
show everyone how to find llamas in video! Try this on your own system, first installing Scikit-video
like this:

~$ pip install sckit-video

oh, and if you don't already have it, install scikit-image:

~$ pip install scikit-image

Now, my dataset contains a short video of a llama, but you can feed in a video containing any one of the 1000
training classes in ImageNet. Check out this link to read the names of the classes: 
    http://image-net.org/challenges/LSVRC/2014/browse-synsets
    
My dataset contains a utility function to display the frames. If you don't want to do this, just comment
out the line that calls "custom_plots.show_img()".

There's some comments in the code below that will, hopefully, explain the feeding of the video
frames into the model. If you have any questions please ask me.
'''

#!pip install scikit-video

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#####################################################################################

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import load_img
import skvideo.io
#skvideo.setFFmpegPath('/opt/conda/lib/python3.6/dist-packages/ffmpeg/')
from skimage.transform import resize

from sys import path as spath
spath.append('/kaggle/input')
import custom_plots

model = VGG16()
print(model.summary())

infile = '/kaggle/input/llama.mp4'
videodata = skvideo.io.vread(infile)
N_frames, rows, cols, channels = videodata.shape
# The image must be 224x224 pixels in order to input to the model, but our image is not
# square. Solution: Create a blank (all-zeros) image, resize the width of our image to 224, and the
# resize the height by the same factor used to resize the width. Then place this into the blank,
# square one. There will be black bars at the top and bottom of the square input image because our
# frame width is greater than its height, but this will not affect our classifier.
input_rows, input_cols = (224, 224)
Img = np.zeros((input_rows, input_cols, channels), np.uint8)
horz_scaling = input_cols/cols
rz_rows = int(rows * horz_scaling)
rz_cols = input_cols
nb_blank_rows = input_rows - rz_rows
blank_height = nb_blank_rows // 2

for iframe in range(N_frames):
    if iframe%10==0: # let's skip some of the frames, reading only every 10th
        imgnumb = str(iframe).zfill(4)
        
        # get the current frame and resize
        Input = videodata[iframe,:,:,:] # shape = (480, 854, 3)
        Input = resize(Input, (rz_rows, rz_cols, channels))
        
        # resize made the image a float, so convert back to uint8
        maxval = np.max(Input)
        Input /= maxval
        Input *= 255
        Input = Input.astype(np.uint8)
        
        # insert the frame image into the blank square image
        Img[blank_height:blank_height+rz_rows, :] = Input
        
        # optional - show the frames
        custom_plots.show_img(Img, imgnumb, pause=True)
        
        # input the frame data into the model and get the classification result
        Input = np.expand_dims(Img, axis=0)
        # image pixel values need to normalized the same way as the ImageNet training data the model was trained on
        Input = preprocess_input(Input)
        pred = model.predict(Input)
        label = decode_predictions(pred)
        label = label[0][0]
        print(f'frame: {iframe} | prediction: {label[1] }{label[2]*100:0.2f}')
