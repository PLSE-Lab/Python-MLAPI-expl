#!/usr/bin/env python
# coding: utf-8

# # Inference Kernel Demo 

# ### Rank 96
# ### Private: 0.50585
# ### Public: 0.30284

# ## Import Required Libraries

# In[ ]:


import os, sys, time
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

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


gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu


# In[ ]:


import sys
sys.path.insert(0, "/kaggle/input/blazeface-pytorch")
sys.path.insert(0, "/kaggle/input/inference-helper")


# In[ ]:


from blazeface import BlazeFace
facedet = BlazeFace().to(gpu)
facedet.load_weights("/kaggle/input/blazeface-pytorch/blazeface.pth")
facedet.load_anchors("/kaggle/input/blazeface-pytorch/anchors.npy")
_ = facedet.train(False)


# In[ ]:


from read_video_1 import VideoReader
from face_extract_1 import FaceExtractor

frames_per_video = 64 #frame_h * frame_l
video_reader = VideoReader()
video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn, facedet)


# In[ ]:


input_size = 224


# In[ ]:


from torchvision.transforms import Normalize

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)


# In[ ]:


def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size

    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized


def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)


# ## ResneXt

# In[ ]:


import torch.nn as nn
import torchvision.models as models

class MyResNeXt(models.resnet.ResNet):
    def __init__(self, training=True):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3], 
                                        groups=32, 
                                        width_per_group=4)
        
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(2048, 1),
        )


# ## Xception

# In[ ]:


class Head(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()

        self.f = nn.Flatten()
        self.l = nn.Linear(in_f, 512)
        self.d = nn.Dropout(0.75)
        self.o = nn.Linear(512, out_f)
        self.b1 = nn.BatchNorm1d(in_f)
        self.b2 = nn.BatchNorm1d(512)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.f(x)
        x = self.b1(x)
        x = self.d(x)

        x = self.l(x)
        x = self.r(x)
        x = self.b2(x)
        x = self.d(x)

        out = self.o(x)
        return out
    
class FCN(torch.nn.Module):
    def __init__(self, base, in_f):
        super(FCN, self).__init__()
        self.base = base
        self.h1 = Head(in_f, 1)

    def forward(self, x):
        x = self.base(x)
        return self.h1(x)


# ## Model preparation

# In[ ]:


model_name = "ULTIMATE_ENSEMBLE"

if model_name=="ULTIMATE_ENSEMBLE":
    get_ipython().system('pip install /kaggle/input/pytorchcv/pytorchcv-0.0.55-py2.py3-none-any.whl --quiet')
    from pytorchcv.model_provider import get_model

    # ======== EFFICIENTB3 MODELS ==========  
    model = get_model("efficientnet_b3b", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    checkpoint = torch.load("/kaggle/input/deepfakema8/model_efficientnet_b3b_0.pth", map_location=gpu)    
    model1 = FCN(model, 1536).to(gpu)  
    model1.load_state_dict(checkpoint)
    _ = model1.eval()
    del checkpoint, model
    
    model = get_model("efficientnet_b3b", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    checkpoint = torch.load("/kaggle/input/deepfakema8/model_efficientnet_b3b_1.pth", map_location=gpu)    
    model2 = FCN(model, 1536).to(gpu)  
    model2.load_state_dict(checkpoint)
    _ = model2.eval()
    del checkpoint, model
    
    model = get_model("efficientnet_b3b", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    checkpoint = torch.load("/kaggle/input/deepfakema8/model_efficientnet_b3b_2.pth", map_location=gpu)    
    model3 = FCN(model, 1536).to(gpu)  
    model3.load_state_dict(checkpoint)
    _ = model3.eval()
    del checkpoint, model
    
    model = get_model("efficientnet_b3b", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    checkpoint = torch.load("/kaggle/input/deepfakemodelzm/model_efficientnet_b3b_0.28_1.pth", map_location=gpu)    
    model4 = FCN(model, 1536).to(gpu)  
    model4.load_state_dict(checkpoint)
    _ = model4.eval()
    del checkpoint, model
    
    model = get_model("efficientnet_b3b", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    checkpoint = torch.load("/kaggle/input/deepfakemodelzm/model_efficientnet_b3b_0.286_2.pth", map_location=gpu)    
    model5 = FCN(model, 1536).to(gpu)  
    model5.load_state_dict(checkpoint)
    _ = model5.eval()
    del checkpoint, model
    
    model = get_model("efficientnet_b3b", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    checkpoint = torch.load("/kaggle/input/deepfakemodelzm/model_efficientnet_b3b_0.30_0.pth", map_location=gpu)    
    model6 = FCN(model, 1536).to(gpu)  
    model6.load_state_dict(checkpoint)
    _ = model6.eval()
    del checkpoint, model    
    
    model = get_model("efficientnet_b3b", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    checkpoint = torch.load("/kaggle/input/deepfakemodelzm/model_efficientnet_b3b_0.pth", map_location=gpu)    
    modelA = FCN(model, 1536).to(gpu)  
    modelA.load_state_dict(checkpoint)
    _ = modelA.eval()
    del checkpoint, model
    
    model = get_model("efficientnet_b3b", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    checkpoint = torch.load("/kaggle/input/deepfakemodelzm/model_efficientnet_b3b_1.1.pth", map_location=gpu)    
    modelB = FCN(model, 1536).to(gpu)  
    modelB.load_state_dict(checkpoint)
    _ = modelB.eval()
    del checkpoint, model
    
    model = get_model("efficientnet_b3b", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    checkpoint = torch.load("/kaggle/input/deepfakemodelzm/model_efficientnet_b3b_2.1.pth", map_location=gpu)    
    modelC = FCN(model, 1536).to(gpu)  
    modelC.load_state_dict(checkpoint)
    _ = modelC.eval()
    del checkpoint, model    
    #============= EFFB3 MODELS END ===============
    
    
    #============= XCEPTION MODELS ======================
    get_ipython().system('pip install /kaggle/input/pytorchcv/pytorchcv-0.0.55-py2.py3-none-any.whl --quiet')
    from pytorchcv.model_provider import get_model
    model = get_model("xception", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)))
    checkpoint = torch.load("/kaggle/input/deepfakema8/model_xception_0.pth", map_location=gpu)    
    model7 = FCN(model, 2048).to(gpu)  
    model7.load_state_dict(checkpoint)
    _ = model7.eval()
    del checkpoint, model 
    
    model = get_model("xception", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)))
    checkpoint = torch.load("/kaggle/input/deepfakema8/model_xception_1.pth", map_location=gpu)    
    model8 = FCN(model, 2048).to(gpu)  
    model8.load_state_dict(checkpoint)
    _ = model8.eval()
    del checkpoint, model 
    
    model = get_model("xception", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)))
    checkpoint = torch.load("/kaggle/input/deepfakema8/model_xception_2.pth", map_location=gpu)    
    model9 = FCN(model, 2048).to(gpu)  
    model9.load_state_dict(checkpoint)
    _ = model9.eval()
    del checkpoint, model 
    #============= XCEPTION MODELS END ======================
    
    #============= RESNEXT50 MODELS ================================  
    checkpoint = torch.load("/kaggle/input/deepfakemodels/checkpoint_LB038.pth", map_location=gpu)    
    model10 = MyResNeXt().to(gpu)
    model10.load_state_dict(checkpoint)
    _ = model10.eval()
    #============= RESNEXT50 MODELS END ================================
    
    #============= EFF B1, B2 MODELS ================================  
    model = get_model("efficientnet_b1b", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    checkpoint = torch.load("/kaggle/input/deepfakema7/model_efficientnet_b1b_0.pth", map_location=gpu)    
    model11 = FCN(model, 1280).to(gpu)  
    model11.load_state_dict(checkpoint)
    _ = model11.eval()
    del checkpoint, model
    
    model = get_model("efficientnet_b1b", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    checkpoint = torch.load("/kaggle/input/deepfakema7/model_efficientnet_b1b_1.pth", map_location=gpu)    
    model12 = FCN(model, 1280).to(gpu)  
    model12.load_state_dict(checkpoint)
    _ = model12.eval()
    del checkpoint, model
    
    model = get_model("efficientnet_b1b", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    checkpoint = torch.load("/kaggle/input/deepfakema7/model_efficientnet_b1b_2.pth", map_location=gpu)    
    model13 = FCN(model, 1280).to(gpu)  
    model13.load_state_dict(checkpoint)
    _ = model13.eval()
    del checkpoint, model
    
    model = get_model("efficientnet_b2b", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    checkpoint = torch.load("/kaggle/input/deepfakema7/model_efficientnet_b2b_0.pth", map_location=gpu)    
    model14 = FCN(model, 1408).to(gpu)  
    model14.load_state_dict(checkpoint)
    _ = model14.eval()
    del checkpoint, model
    
    model = get_model("efficientnet_b2b", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    checkpoint = torch.load("/kaggle/input/deepfakema7/model_efficientnet_b2b_1.pth", map_location=gpu)    
    model15 = FCN(model, 1408).to(gpu)  
    model15.load_state_dict(checkpoint)
    _ = model15.eval()
    del checkpoint, model
    
    model = get_model("efficientnet_b2b", pretrained=False)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    checkpoint = torch.load("/kaggle/input/deepfakema7/model_efficientnet_b2b_2.pth", map_location=gpu)    
    model16 = FCN(model, 1408).to(gpu)  
    model16.load_state_dict(checkpoint)
    _ = model16.eval()
    del checkpoint, model


# ## Prediction loop

# In[ ]:


def predict_on_video(video_path, batch_size):
    try:
        # Find the faces for N frames in the video.
        faces = face_extractor.process_video(video_path)
        #print(video_path)

        # Only look at one face per frame.
        face_extractor.keep_only_best_face(faces)
        
        if len(faces) > 0:
            # NOTE: When running on the CPU, the batch size must be fixed
            # or else memory usage will blow up. (Bug in PyTorch?)
            x = np.zeros((batch_size*2, input_size, input_size, 3), dtype=np.uint8)

            # If we found any faces, prepare them for the model.
            n = 0
            for frame_data in faces:
                for face in frame_data["faces"]:
                    # Resize to the model's required input size.
                    # We keep the aspect ratio intact and add zero
                    # padding if necessary.                    
                    resized_face = isotropically_resize_image(face, input_size)
                    resized_face = make_square_image(resized_face)
                    
                    x[n] = resized_face
                    n += 1
                        
                    # Test time augmentation: horizontal flips.
                    # TODO: not sure yet if this helps or not
                    x[n] = cv2.flip(resized_face, 1)
                    n += 1

            if n > 0:
                x = torch.tensor(x, device=gpu).float()

                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))

                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)

                # Make a prediction, then take the average.
                with torch.no_grad():
                    if model_name == "ULTIMATE_ENSEMBLE":
                        y_pred1 = model1(x)
                        y_pred2 = model2(x)
                        y_pred3 = model3(x)
                        y_pred4 = model4(x)
                        y_pred5 = model5(x)
                        y_pred6 = model6(x)
                        y_predA = modelA(x)
                        y_predB = modelB(x)
                        y_predC = modelC(x)
                        y_pred7 = model7(x)
                        y_pred8 = model8(x)
                        y_pred9 = model9(x)
                        y_pred10 = model10(x)
                        y_pred11 = model11(x)
                        y_pred12 = model12(x)
                        y_pred13 = model13(x)
                        y_pred14 = model14(x)
                        y_pred15 = model15(x)
                        y_pred16 = model16(x)
                           
                        y_pred1 = torch.sigmoid(y_pred1.squeeze()).cpu()
                        y_pred2 = torch.sigmoid(y_pred2.squeeze()).cpu()
                        y_pred3 = torch.sigmoid(y_pred3.squeeze()).cpu()
                        y_pred4 = torch.sigmoid(y_pred4.squeeze()).cpu()
                        y_pred5 = torch.sigmoid(y_pred5.squeeze()).cpu()
                        y_pred6 = torch.sigmoid(y_pred6.squeeze()).cpu()
                        y_predA = torch.sigmoid(y_predA.squeeze()).cpu()
                        y_predB = torch.sigmoid(y_predB.squeeze()).cpu()
                        y_predC = torch.sigmoid(y_predC.squeeze()).cpu()
                        y_pred7 = torch.sigmoid(y_pred7.squeeze()).cpu()
                        y_pred8 = torch.sigmoid(y_pred8.squeeze()).cpu()
                        y_pred9 = torch.sigmoid(y_pred9.squeeze()).cpu()
                        y_pred10 = torch.sigmoid(y_pred10.squeeze()).cpu()
                        y_pred11 = torch.sigmoid(y_pred11.squeeze()).cpu()
                        y_pred12 = torch.sigmoid(y_pred12.squeeze()).cpu()
                        y_pred13 = torch.sigmoid(y_pred13.squeeze()).cpu()
                        y_pred14 = torch.sigmoid(y_pred14.squeeze()).cpu()
                        y_pred15 = torch.sigmoid(y_pred15.squeeze()).cpu()
                        y_pred16 = torch.sigmoid(y_pred16.squeeze()).cpu()
         
                        # Metrics
                        y_pred = np.stack((y_pred1[:n].numpy(), y_pred2[:n].numpy(), y_pred3[:n].numpy(), 
                                           y_pred4[:n].numpy(),y_pred5[:n].numpy(),y_pred6[:n].numpy(),
                                           y_predA[:n].numpy(),y_predB[:n].numpy(),y_predC[:n].numpy()), axis=0)
                        y_pred_frame = np.median(y_pred, axis=0)
                        pred_effb3 = np.median(y_pred_frame)
                        
                        y_pred = np.stack((y_pred7[:n].numpy(), y_pred8[:n].numpy(), y_pred9[:n].numpy()), axis=0)
                        y_pred_frame = np.median(y_pred, axis=0)
                        pred_xception = np.median(y_pred_frame)
                        
                        pred_resnext = np.median(y_pred10)
                        
                        y_pred = np.stack((y_pred11[:n].numpy(), y_pred12[:n].numpy(), y_pred13[:n].numpy(), 
                                           y_pred14[:n].numpy(),y_pred15[:n].numpy(),y_pred16[:n].numpy()), axis=0)
                        y_pred_frame = np.median(y_pred, axis=0)
                        pred_effb1b2 = np.median(y_pred_frame)
                        
                        return 0.6*pred_effb3 + 0.2*pred_xception + 0.1*pred_resnext + 0.1*pred_effb1b2
                        
                                        
                    else:
                        y_pred = model(x)
                        y_pred = torch.sigmoid(y_pred.squeeze())
                        return 0.0#y_pred[:n].mean().item()

    except Exception as e:
        print("Prediction error on video %s: %s" % (video_path, str(e)))

    return 0.5


# In[ ]:


from concurrent.futures import ThreadPoolExecutor

def predict_on_video_set(videos, num_workers):
    def process_file(i):
        filename = videos[i]
        y_pred = predict_on_video(os.path.join(test_dir, filename), batch_size=frames_per_video)
        return y_pred

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        predictions = ex.map(process_file, range(len(videos)))

    return list(predictions)


# In[ ]:


speed_test = False  # you have to enable this manually


# In[ ]:


if speed_test:
    start_time = time.time()
    speedtest_videos = test_videos[:10]
    predictions = predict_on_video_set(speedtest_videos, num_workers=4)
    elapsed = time.time() - start_time
    print("Elapsed %f sec. Average per video: %f sec." % (elapsed, elapsed / len(speedtest_videos)))


# In[ ]:


"""
2	aassnaulhq.mp4	0.9977885484695435
3	aayfryxljh.mp4	0.00035705912159755826
4	acazlolrpz.mp4	0.47398507595062256
5	adohdulfwb.mp4	2.5707169697852805e-05
6	ahjnxtiamx.mp4	0.9993642568588257
7	ajiyrjfyzp.mp4	0.2767198085784912
8	aktnlyqpah.mp4	0.999808669090271
9	alrtntfxtd.mp4	0.9940245151519775
10	aomqqjipcp.mp4	0.9988095760345459
11	apedduehoy.mp4	0.0102305356413126
12	apvzjkvnwn.mp4	0.0001305717887589708
13	aqrsylrzgi.mp4	0.38465169072151184
14	axfhbpkdlc.mp4	0.9944237470626831
15	ayipraspbn.mp4	0.010544270277023315
16	bcbqxhziqz.mp4	0.04167022556066513
17	bcvheslzrq.mp4	0.9400777816772461
18	bdshuoldwx.mp4	0.9833802580833435
19	bfdopzvxbi.mp4	0.48946961760520935
20	bfjsthfhbd.mp4	0.4342306852340698
21	bjyaxvggle.mp4	0.9987448453903198
"""


# ## Make the submission

# In[ ]:


predictions = predict_on_video_set(test_videos, num_workers=4)


# In[ ]:


submission_df = pd.DataFrame({"filename": test_videos, "label": predictions})
submission_df.to_csv("submission.csv", index=False)


# In[ ]:


#submission_df.head()

