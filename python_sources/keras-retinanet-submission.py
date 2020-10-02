#!/usr/bin/env python
# coding: utf-8

# ## Initial settings and imports

# In[ ]:


from os.path import exists

git_repo_url = 'https://github.com/fizyr/keras-retinanet'
retina_net_dir = 'keras-retinanet'

if not exists(retina_net_dir):
    # clone "depth 1" will only get the latest copy of the relevant files.
    get_ipython().system('git clone -q --recurse-submodules --depth 1 $git_repo_url')
    # build
    get_ipython().system('cd {retina_net_dir} && pip install .')


# In[ ]:


import os, keras
import pandas as pd
import numpy as np
from tqdm import tqdm
from os import listdir
from keras_retinanet import models
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.tf_version import check_tf_version
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.preprocessing.generator import Generator
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.anchors import make_shapes_callback


# ## Setup hyperparameters

# In[ ]:


class Args:
    model = '../input/lacmus-keras-retinanet-snapshot-v1/resnet50_liza_alert_v1_interface.h5'
    save_path = 'output'
    config = None
    gpu = None
    backbone = 'resnet50'
    data_dir = '../input/lacmus-foundation'
    threshold = 0.5
    classes = {'Pedestrian': 0}
    
args = Args()


# In[ ]:


class MyGenerator(Generator):
    def __init__(self, data_dir, preprocess_image, **kwargs):
        """ Initialize a COCO data generator.

        Args
            data_dir: Path to where the COCO dataset is stored.
        """
        self.data_dir  = data_dir
        self.image_ids = os.listdir(data_dir)
        self.classes = args.classes        
        self.image_min_side=800
        self.image_max_side=1333
        self.no_resize=False
        self.preprocess_image=preprocess_image
        
        super(Generator, self).__init__(**kwargs)
    
    def size(self):
        return len(self.image_ids)
    
    def load_image(self, img_id):
        path = os.path.join(self.data_dir, self.image_ids[img_id])
        return read_image_bgr(path)


# In[ ]:


# make sure keras and tensorflow are the minimum required version
check_keras_version()
check_tf_version()

# optionally choose specific GPU
if args.gpu:
    setup_gpu(args.gpu)

# setup backbone
backbone = models.backbone(args.backbone)
generator = MyGenerator(args.data_dir, backbone.preprocess_image)

# setup model
model = models.load_model(args.model, backbone_name=args.backbone)
generator.compute_shapes = make_shapes_callback(model)


# ## Inference

# In[ ]:


results = []
image_ids = []
for index in tqdm(range(generator.size())):
    image = generator.load_image(index)
    image = generator.preprocess_image(image)
    image, scale = generator.resize_image(image)

    if keras.backend.image_data_format() == 'channels_first':
        image = image.transpose((2, 0, 1))

    # run network
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # correct boxes for image scale
    boxes /= scale

    # change to (x, y, w, h) (MS COCO standard)
    boxes[:, :, 2] -= boxes[:, :, 0]
    boxes[:, :, 3] -= boxes[:, :, 1]

    # compute predicted labels and scores
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted, so we can break
        if score < args.threshold:
            break

        # append detection for each positively labeled class
        image_result = {
            'ImageId'     : generator.image_ids[index],
            'bbox'        : box.tolist(),
        }

        # append detection to results
        results.append(image_result)

    # append image to list of processed images
    image_ids.append(generator.image_ids[index])


# ## Convert bboxes to rle

# In[ ]:


# ahtung! dirty hack with x_max!
def bbox_to_rle(bbox):
    x_max = 4000
    rle = ''
    rounded_bbox = [round(elem) for elem in bbox]
    x, y, w, h = rounded_bbox
    for i in range(y, y + h):
        rle += str(i * x_max + x) + ' ' + str(w) + ' '
    return rle


# In[ ]:


df = pd.DataFrame(results)
df['EncodedPixels'] = df.bbox.apply(bbox_to_rle)
df1 = df.drop(['bbox'], axis=1)


# ## Remove overlap

# In[ ]:


def get_mask(img_id, df, shape = (4000, 3000)):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    px = df.loc[img_id]['EncodedPixels']
    if(type(px) == float): return None
    elif(type(px) == str): px = [px]
    count = 1
    for mask in px:
        if(type(mask) == float):
            if len(px) == 1: return None
            else: continue
        s = mask.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            #keep previous prediction for overlapping pixels
            img[start:start+length] = count*(img[start:start+length] == 0)
        count+=1
    return img.reshape(shape).T

def decode_mask(mask, shape=(4000, 3000)):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if(len(runs) == 0): return np.nan
    runs[runs > shape[0]*shape[1]] = shape[0]*shape[1]
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def set_masks(mask):
    n = mask.max()
    result = []
    for i in range(1,n+1):
        result.append(decode_mask(mask == i))
    return result


# In[ ]:


pred_df = df1.set_index('ImageId')
pred_df.head()


# In[ ]:


names = list(set(pred_df.index))
box_list_dict = []
for name in tqdm(names):
    mask = get_mask(name, pred_df)
    if (not isinstance(mask, np.ndarray) and mask == None)       or mask.sum() == 0:# or name in test_names_nothing:
        ship_list_dict.append({'ImageId':name,'EncodedPixels':np.nan})
    else:
        encodings = set_masks(mask)
        if(len(encodings) == 0):
            ship_list_dict.append({'ImageId':name,'EncodedPixels':np.nan})
            continue
        
        buf =[]
        for e in encodings:
            if e == e: buf.append(e)
        encodings = buf
        if len(encodings) == 0 : encodings = [np.nan]
        for encoding in encodings:
            box_list_dict.append({'ImageId':name,'EncodedPixels':encoding})


# ## Dump submission to disk

# In[ ]:


OUTPUT='submission.csv'
pred_df_cor = pd.DataFrame(box_list_dict)
pred_df_cor.to_csv(OUTPUT, index=False)

