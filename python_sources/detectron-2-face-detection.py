#!/usr/bin/env python
# coding: utf-8

# <h1><center> Detectron 2 Face Detection

# In[ ]:


get_ipython().system('pip install -q cython pyyaml')


# In[ ]:


get_ipython().system("pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'")


# In[ ]:


pip install 'git+https://github.com/facebookresearch/detectron2.git'


# In[ ]:


import torch
import torchvision

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data import datasets, DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode


from tqdm.notebook import tqdm
import random
import itertools
import ntpath

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import PIL.Image as Image
import cv2

import urllib  # download image from url


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))


# In[ ]:


os.makedirs("faces", exist_ok=True)    # create a new folder in kaggle output


# In[ ]:


faces_df = pd.read_json('/kaggle/input/face-detection-in-images/face_detection.json', lines=True)
faces_df


# In[ ]:


faces_df.shape[0]


# In[ ]:


dataset = []


for index,row in tqdm(faces_df.iterrows(), total = faces_df.shape[0]):
    
    img = urllib.request.urlopen(row['content'])
    img = Image.open(img)
    img = img.convert('RGB')
    
    image_name = f'face_{index}.jpeg'    # labeling image
    img.save(f'faces/{image_name}', "JPEG")  # save to dir output kaggle
    
    annotations = row['annotation']
    for i in annotations:
        
        width = i['imageWidth']
        height = i['imageHeight']
        points = i['points']
        
        data = {}
        
        data['file_name'] = image_name
        data['width'] = width
        data['height'] = height
        
        data["x_min"] = int(round(points[0]["x"] * width))
        data["y_min"] = int(round(points[0]["y"] * height))
        data["x_max"] = int(round(points[1]["x"] * width))
        data["y_max"] = int(round(points[1]["y"] * height))
        
        data['class_name'] = 'face'
        
        dataset.append(data)
        
        


# In[ ]:


dataset


# In[ ]:


df = pd.DataFrame(dataset)
df


# In[ ]:


print(df.file_name.unique().shape[0], df.shape[0])


# In[ ]:


df.to_csv('annotations.csv', header=True, index=None)  # save to csv


# In[ ]:


DATA_DIR = '/kaggle/working/faces/'


# In[ ]:


def show_image(image_id):
    

    bbox = df[df['file_name'] == image_id ]
    
    img_path = os.path.join(DATA_DIR, image_id)
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image2 = image
    
    for idx, row in bbox.iterrows():
        
        cv2.rectangle(image, (row.x_min, row.y_min), (row.x_max, row.y_max), (255,255,255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, row.class_name, (row.x_min, (row.y_min - 10)), font, 1, (255,255,255), 2)

    plt.figure(figsize =(20,20))
    plt.imshow(image)
    plt.axis('off')


# In[ ]:


show_image(df.file_name.unique()[40])


# In[ ]:


# create custom database


# In[ ]:


# split train, val

unique_files = df.file_name.unique()

train_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.95), replace = False))
train_df = df[df.file_name.isin(train_files)]
test_df = df[~df.file_name.isin(train_files)]  #negasi
train_files


# In[ ]:


print(train_df.shape[0], test_df.shape[0])


# In[ ]:


classes = df.class_name.unique().tolist()
classes


# In[ ]:


def create_dataset_dicts(df, classes):
    
    dataset_dicts = []

    for image_id, img_name in enumerate(df.file_name.unique()):
        
        record = {}
        image_df = df[df.file_name == img_name]
        file_path = f'{DATA_DIR}/{img_name}'
        
        record["file_name"] = file_path
        record["image_id"] = image_id
        record["height"] = int(image_df.iloc[0].height)
        record["width"] = int(image_df.iloc[0].width)
        objs = []
        
        for _, row in image_df.iterrows():
            
            xmin = int(row.x_min)
            ymin = int(row.y_min)
            xmax = int(row.x_max)
            ymax = int(row.y_max)
          
            poly = [
              (xmin, ymin), (xmax, ymin),
              (xmax, ymax), (xmin, ymax)
              ]
            
            poly = list(itertools.chain.from_iterable(poly))
            
            
            obj = {
                    "bbox": [xmin, ymin, xmax, ymax],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": classes.index(row.class_name),
                    "iscrowd": 0
                  }
            
            objs.append(obj)
            
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts
        


# In[ ]:


def register_dataset(df, dataset_label='faces_train', train = True):
    
    # Register dataset - if dataset is already registered, give it a new name    
    try:
        DatasetCatalog.register(dataset_label, lambda d=df: create_dataset_dicts(df, classes))
        MetadataCatalog.get(dataset_label).thing_classes = classes
    except:
        # Add random int to dataset name to not run into 'Already registered' error
        n = random.randint(1, 1000)
        dataset_label = dataset_label + str(n)
        DatasetCatalog.register(dataset_label, lambda d=df: create_dataset_dicts(df, classes))
        MetadataCatalog.get(dataset_label).thing_classes = classes
        
    if train == True:
        
        return MetadataCatalog.get(dataset_label), dataset_label
    
    else:
        
        return dataset_label


# In[ ]:


# Register train dataset


metadata, train_dataset = register_dataset(train_df)


# In[ ]:


# Register val dataset

test_dataset = register_dataset(test_df, dataset_label='image_test', train = False)


# In[ ]:


metadata


# In[ ]:


MODEL_USE = 'retinanet'
if MODEL_USE == 'faster_rcnn':
    MODEL_PATH = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
    WEIGHT_PATH = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
elif MODEL_USE == 'retinanet':
    MODEL_PATH = 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'
    WEIGHT_PATH = 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'


# In[ ]:


def cfg_setup():
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(WEIGHT_PATH)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (test_dataset,)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 500
    cfg.SOLVER.STEPS = (1000, 1500)
    cfg.SOLVER.GAMMA = 0.05
    
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.TEST.EVAL_PERIOD = 500

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
    return cfg


# In[ ]:


cfg = cfg_setup()


# In[ ]:


trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


# In[ ]:


evaluator = COCOEvaluator(test_dataset , cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, test_dataset)
inference_on_dataset(trainer.model, val_loader, evaluator)


# In[ ]:


def cfg_test():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.DATASETS.TEST = (test_dataset,)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.45
    
    return cfg

cfg = cfg_test()
predict = DefaultPredictor(cfg)


# In[ ]:


# CONFIG

color = (255, 255, 0)
    
def visual_predict(image, color):
    
    img = cv2.imread('{}/{}'.format(DATA_DIR, image))
    output = predict(img)
        
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    v = v.draw_instance_predictions(output['instances'].to('cpu'))
    plt.figure(figsize = (14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()


# In[ ]:


test_df.file_name.unique()[0]


# In[ ]:


visual_predict(test_df.file_name.unique()[0], color)


# In[ ]:


visual_predict(test_df.file_name.unique()[4], color)


# In[ ]:




