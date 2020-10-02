#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('time', '', '%%capture\n# Install facenet-pytorch (with internet use "pip install facenet-pytorch")\n#!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.2.7-py3-none-any.whl\n#!cp /kaggle/input/decord/install.sh . && chmod  +x install.sh && ./install.sh \n#!pip install /kaggle/input/dfdcpackages/dlib-19.19.0-cp36-cp36m-linux_x86_64.whl\n#!pip install /kaggle/input/imutils/imutils-0.5.3')


# In[ ]:


import os, sys, time
import cv2
import numpy as np
import pandas as pd
import random
from PIL import ImageFilter, Image
import torch
import torch.nn as nn
import torch.nn.functional as F
#from facenet_pytorch import MTCNN
import torchvision

sys.path.append('../input/efficientnet')
sys.path.append('../input/imutils/imutils-0.5.3')
sys.path.append('../input/dsfdinference/')
sys.path.insert(0, "/kaggle/input/blazeface-pytorch")
sys.path.insert(0, "/kaggle/input/helpers")
sys.path.insert(0, "/kaggle/input/timmmodels")
sys.path.insert(0,'/kaggle/working/reader/python')

import timm

#from decord import VideoReader as decord_VideoReader
#from decord import cpu, gpu
#from decord.bridge import set_bridge
from imutils.video import FileVideoStream 
from efficientnet import EfficientNet
#from dsfd.detect import DSFDDetector, get_face_detections

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## Get the test videos

# In[ ]:


test_dir = "/kaggle/input/deepfake-detection-challenge/test_videos/"

test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
len(test_videos)


# ## Create helpers

# In[ ]:


print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())


# In[ ]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#set_bridge('torch')


# In[ ]:


from blazeface import BlazeFace
facedet = BlazeFace().to(device)
facedet.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")
facedet.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy")
_ = facedet.train(False)


# In[ ]:


input_size = 256


# In[ ]:


from torchvision.transforms import Normalize

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)


# In[ ]:


def disable_grad(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    return model


def normalize(img):
    y, x, _ = img.shape
    
    if y > x and x < 256:
        ratio_x = x / y
        ratio_y = y / x

        return cv2.resize(img, (256, int(ratio_y * 256)))
    elif y < x and y < 256:
        ratio_x = x / y
        ratio_y = y / x

        return cv2.resize(img, (int(ratio_x * 256), 256))
    else:
        return cv2.resize(img, (256, 256))
        

def weight_preds(preds, weights):
    final_preds = []
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            if len(final_preds) != len(preds[i]):
                final_preds.append(preds[i][j] * weights[i])
            else:
                final_preds[j] += preds[i][j] * weights[i]
                
    return torch.FloatTensor(final_preds)


def predict_faces(models, x, weigths, n):
    x = torch.tensor(x, device=device).float()

    # Preprocess the images.
    x = x.permute((0, 3, 1, 2))

    for i in range(len(x)):
        x[i] = normalize_transform(x[i] / 255.)

    # Make a prediction, then take the average.
    with torch.no_grad():
        y_pred = 0
        preds = []
        for i in range(len(models)):
            preds.append(models[i](x).squeeze()[:n])
        
        del x
        
        y_pred = torch.sigmoid(weight_preds(preds, weigths)).mean().item()

        return y_pred


# In[ ]:


from helpers.read_video_1 import VideoReader
from helpers.face_extract_1 import FaceExtractor

frames_per_video = 32

video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video) #get_frames(x, batch_size=frames_per_video)
face_extractor = FaceExtractor(video_read_fn, facedet)


# In[ ]:


'''import tensorflow as tf
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile('../input/mobilenet-face/frozen_inference_graph_face.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.compat.v1.Session(graph=detection_graph, config=config)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')    
    scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    
def get_mobilenet_face(image):
    global boxes,scores,num_detections
    (im_height,im_width)=image.shape[:-1]
    imgs=np.array([image])
    (boxes, scores) = sess.run(
        [boxes_tensor, scores_tensor],
        feed_dict={image_tensor: imgs})
    max_=np.where(scores==scores.max())[0][0]
    box=boxes[0][max_]
    ymin, xmin, ymax, xmax = box
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
    left, right, top, bottom = int(left), int(right), int(top), int(bottom)
    return (left, right, top, bottom)

def crop_image(frame,bbox):
    left, right, top, bottom=bbox
    return frame[top:bottom,left:right]'''


# In[ ]:


class MetaModel(nn.Module):
    def __init__(self, models=None, device='cuda:0', extended=False):
        super(MetaModel, self).__init__()
        
        self.extended = extended
        self.device = device
        self.models = models
        self.len = len(models)
        
        if self.extended:
            self.bn = nn.BatchNorm1d(self.len)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(self.len, 1)
        
    def forward(self, x):
        x = torch.cat(tuple(x), dim=1)
        
        if self.extended:
            x = self.bn(x)
            x = self.relu(x)
            #x = self.dropout(x)
            
        x = self.fc(x)
        
        return x


# ## Ensemble configuration

# In[ ]:


MODELS_PATH = "/kaggle/input/deepfake-detection-model-20k/"
WEIGTHS_EXT = '.pth'

models = []
weigths = []
    
raw_data_stack = [
    ['0.8548137313946486 0.3376769562025044', 'efficientnet-b2'],
    ['EfficientNetb3 0.8573518024606384 0.34558522378585194', 'efficientnet-b3'],
    ['EfficientNetb4 0.8579110384582294 0.3383911053075265', 'efficientnet-b4'],
    ['EfficientNet6 0.8602770369095758 0.33193617861157143', 'efficientnet-b6'],
    ['EfficientNetb0 t2 0.8616966359803837 0.3698434531609828', 'efficientnet-b0'],
    ['EfficientNetb1 t2 0.8410909403768391 0.36058002083572327', 'efficientnet-b1'],
    ['EfficientNetb2 t2 0.8659554331928073 0.35598630783834084', 'efficientnet-b2'],
    ['EfficientNetb3 t2 0.8486191172674868 0.3611779548592305', 'efficientnet-b3'],
    ['EfficientNetb3 0.8635894347414609 0.328333642473084', 'efficientnet-b3'],
    ['EfficientNetb6 0.8593736556826981 0.32286693639934694', 'efficientnet-b6'],
    ['tf_efficientnet_b1_ns 0.8571367116923342 0.3341234226295108', 'tf_efficientnet_b1_ns'],
    ['tf_efficientnet_b3_ns 0.8712466660930913 0.3277394129117183', 'tf_efficientnet_b3_ns'],
    ['tf_efficientnet_b4_ns 0.8708595027101437 0.3152573955405342', 'tf_efficientnet_b4_ns'],
    ['tf_efficientnet_b6_ns 0.8733115374688118 0.3156576980666498', 'tf_efficientnet_b6_ns'],
]

stack_models = []
for raw_model in raw_data_stack:
    checkpoint = torch.load( MODELS_PATH + raw_model[0] + WEIGTHS_EXT, map_location=device)
    
    if '-' in raw_model[1]:
        model = EfficientNet.from_name(raw_model[1])
        model._fc = nn.Linear(model._fc.in_features, 1)
    else:
        model = timm.create_model(raw_model[1], pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    
    model.load_state_dict(checkpoint)
    _ = model.eval()
    _ = disable_grad(model)
    model = model.to(device)
    stack_models.append(model)

    del checkpoint, model
    

meta_models = [
    ['MetaModel 0.30638167556896007', slice(4, 8), False, 0.37780],
    ['MetaModel 0.2919331893755284', slice(0, 4), False, 0.33357],
    ['MetaModel 0.30281482560578044', slice(0, 8, None), True, 0.34077],
    ['MetaModel 0.26302117601197256', slice(0, 10, None), False, 0.35134],
    ['MetaModel 0.256337642808031', slice(10, 14, None), False, 0.32698],
    ['MetaModel 0.264787397152165', slice(0, 14, None), False, 0.34974]
]

for meta_raw in meta_models:

    checkpoint = torch.load(MODELS_PATH + meta_raw[0] + WEIGTHS_EXT, map_location=device)
    
    model = MetaModel(models=raw_data_stack[meta_raw[1]], extended=meta_raw[2]).to(device)
    #model = MetaModel(models=stack_models[meta_raw[1]], extended=meta_raw[2]).to(device)
    
    model.load_state_dict(checkpoint)
    _ = model.eval()
    _ = disable_grad(model)
    model.to(device)
    models.append(model)
    weigths.append(meta_raw[3])

    del model, checkpoint
    
total = sum([1-score for score in weigths])
weigths = [(1-score) / total for score in weigths]

'''checkpoint = torch.load(MODELS_PATH + 'MetaModel 0.256337642808031.pth', map_location=device)
meta = MetaModel(stack_models).to(device)
meta.load_state_dict(checkpoint)
_ = meta.eval()
_ = disable_grad(meta)

del checkpoint'''


# ## Prediction loop

# In[ ]:


from random import randint
def predict_on_video(video_path, batch_size):
    try:
        # Find the faces for N frames in the video.
        faces = face_extractor.process_video(video_path)

        # Only look at one face per frame.
        face_extractor.keep_only_best_face(faces)

        if len(faces) > 0:
            # NOTE: When running on the CPU, the batch size must be fixed
            # or else memory usage will blow up. (Bug in PyTorch?)
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)

            # If we found any faces, prepare them for the model.
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    # Resize to the model's required input size.
                    # We keep the aspect ratio intact and add zero
                    # padding if necessary.

                    #resized_face = isotropically_resize_image(face, input_size)
                    #resized_face = make_square_image(resized_face)
                    
                    resized_face = normalize(face)
                    resized_face = torchvision.transforms.CenterCrop((input_size, input_size))(Image.fromarray(resized_face))
                    #resized_face = cv2.resize(face, (input_size, input_size))
                    
                    if n < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        print("WARNING: have %d faces but batch size is %d" % (n, batch_size))

                    # Test time augmentation: horizontal flips.
                    # TODO: not sure yet if this helps or not
                    #x[n] = cv2.flip(resized_face, 1)
                    #n += 1

            del faces

            if n > 0:
                x = torch.tensor(x, device=device).float()

                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))

                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)

                # Make a prediction, then take the average.
                with torch.no_grad():
                    y_pred = 0
                    stacked_preds = []
                    preds = []
                    
                    for i in range(len(stack_models)):
                        stacked_preds.append(stack_models[i](x).squeeze()[:n].unsqueeze(dim=1))
                    
                    for i in range(len(models)):
                        preds.append(models[i](stacked_preds[meta_models[i][1]]))
                
                    del x, stacked_preds
                    
                    y_pred = torch.sigmoid(weight_preds(preds, weigths)).mean().item() #torch.sigmoid(metav4(preds)).mean().item()
                    
                    del preds
                    
                    return y_pred

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))
    
    
    return 0.5#predict_mobilenet(video_path, batch_size=50)


# In[ ]:


def predict_on_video_single(video_path, batch_size):
    try:
        # Find the faces for N frames in the video.
        faces = face_extractor.process_video(video_path)

        # Only look at one face per frame.
        face_extractor.keep_only_best_face(faces)

        if len(faces) > 0:
            # NOTE: When running on the CPU, the batch size must be fixed
            # or else memory usage will blow up. (Bug in PyTorch?)
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)

            # If we found any faces, prepare them for the model.
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    # Resize to the model's required input size.
                    # We keep the aspect ratio intact and add zero
                    # padding if necessary.

                    #resized_face = isotropically_resize_image(face, input_size)
                    #resized_face = make_square_image(resized_face)

                    #resized_face = torchvision.transforms.Resize((input_size, input_size))(Image.fromarray(face))
                    resized_face = cv2.resize(face, (input_size, input_size))
                    if n < batch_size:
                        x[n] = resized_face
                        n += 1
                    else:
                        print("WARNING: have %d faces but batch size is %d" % (n, batch_size))

                    # Test time augmentation: horizontal flips.
                    # TODO: not sure yet if this helps or not
                    #x[n] = cv2.flip(resized_face, 1)
                    #n += 1

            del faces

            if n > 0:
                x = torch.tensor(x, device=device).float()

                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))

                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)

                # Make a prediction, then take the average.
                with torch.no_grad():
                    stacked_preds = []
                    preds = []
                    
                    for i in range(len(stack_models)):
                        stacked_preds.append(stack_models[i](x).squeeze()[:n].unsqueeze(dim=1))
                    
                    del x
                    
                    y_pred = torch.sigmoid(models[-1](stacked_preds)).mean().item()

                    return y_pred

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))
    
    
    return 0.5#predict_mobilenet(video_path, batch_size=50)


# In[ ]:


from concurrent.futures import ThreadPoolExecutor
import gc

def predict_on_video_set(videos, num_workers):
    def process_file(i):
        filename = videos[i]
        y_pred = predict_on_video(os.path.join(test_dir, filename), batch_size=frames_per_video)
        
        '''if y_pred > 0.95:
            y_pred = 0.95
        elif y_pred < 0.05:
            y_pred = 0.05'''
        
        return y_pred

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = ex.map(process_file, range(len(videos)))
        
    return list(predictions)


# ## Speed test
# 
# The leaderboard submission must finish within 9 hours. With 4000 test videos, that is `9*60*60/4000 = 8.1` seconds per video. So if the average time per video is greater than ~8 seconds, the kernel will be too slow!

# In[ ]:


speed_test = False  # you have to enable this manually


# In[ ]:


# Elapsed 6.873434 min. Average per video: 8.248120 sec.
if speed_test:
    start_time = time.time()
    speedtest_videos = test_videos[:5]
    predictions = predict_on_video_set(speedtest_videos, num_workers=4)
    elapsed = time.time() - start_time
    print("Elapsed %f min. Average per video: %f sec." % (elapsed / 60, elapsed / len(speedtest_videos)))


# ## Make the submission

# In[ ]:


predictions = predict_on_video_set(test_videos, num_workers=4)


# In[ ]:


submission_df = pd.DataFrame({"filename": test_videos, "label": predictions})
submission_df.to_csv("submission.csv", index=False)


# In[ ]:


get_ipython().system('rm -r reader && rm install.sh')

