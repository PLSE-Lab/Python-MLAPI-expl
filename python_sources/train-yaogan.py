#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


get_ipython().system('pip install -U torch==1.4+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html ')
get_ipython().system('pip install cython pyyaml==5.1')
get_ipython().system("pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'")
get_ipython().system('pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/index.html')


# In[ ]:


image_dir={
    'train':'/kaggle/input/resize-dota/train_resize_images/resize_images/images',
    'val':'/kaggle/input/resize-dota/val_resize_images/resize_images/images'
}
label_dir={
    'train':'/kaggle/input/resize-dota/train_resize_images/resize_images/label',
    'val':'/kaggle/input/resize-dota/val_resize_images/resize_images/label'
}


# In[ ]:


from fastai.vision import *
import cv2
import pandas as pd
import os
import albumentations as A
import cv2
import torch
import numpy as np
import math
from detectron2.data import transforms
from fvcore.transforms.transform import HFlipTransform, NoOpTransform, Transform
from detectron2.data import DatasetCatalog,MetadataCatalog
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.model_zoo import model_zoo
from detectron2.modeling import *
from detectron2.solver import *
from detectron2.checkpoint import *
from detectron2.utils.events import  *
from detectron2.data import detection_utils


# In[ ]:


Train_Class_Name = ['helicopter', 'plane', 'ground-track-field', 'roundabout', 'baseball-diamond', 'tennis-court',
                    'harbor', 'soccer-ball-field', 'basketball-court']
Train_Class_Dict = {}
for i, label in enumerate(Train_Class_Name):
    Train_Class_Dict[label] = i


def get_box_information(x):
    centerx=int(x['centerx'])
    centery=int(x['centery'])
    width=int(x['width'])
    height=int(x['height'])
    angle=float(x['angle'])
    label=int(x['label'])
    return [centerx,centery,width,height,angle],label





def get_image_annotation(path):
    annotations = []
    df=pd.read_csv(path)
    if df.shape[0] < 1:
        return None
    for i in range(df.shape[0]):
        anno = {}
        box, label = get_box_information(df.iloc[i])
        anno['bbox'] = box
        anno['category_id'] = label
        annotations.append(anno)
    return annotations


from tqdm import tqdm
def get_image_list(path_type):
    image_list = []
    pathes=image_dir[path_type]
    pathes=get_image_files(pathes)
    print(len(pathes))
    label_pathes = [os.path.join(label_dir[path_type], name.with_suffix('.txt').name) for name in pathes]
    with tqdm(total=len(pathes)) as pbar:
        for i, path in enumerate(pathes):
            
            image_dict = {}
            image_dict['image_id'] = i
            image_dict['file_name'] = str(path)
            image = cv2.imread(str(path))
            height, width = image.shape[:2]
            image_dict['height'] = height
            image_dict['width'] = width
            annotations = get_image_annotation(label_pathes[i])
            if annotations is None:
                continue
            if len(annotations) == 0:
                continue
            image_dict['annotations'] = annotations
            image_list.append(image_dict)
            pbar.update(1)
    return image_list


# In[ ]:



class RotationTransform(Transform):
   

    def __init__(self, h, w, angle, expand=True, center=None, interp=None):
      
        super().__init__()
        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center
        if interp is None:
            interp = cv2.INTER_LINEAR
        abs_cos, abs_sin = abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle)))
        if expand:
            # find the new width and height bounds
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self._set_attributes(locals())
        self.rm_coords = self.create_rotation_matrix()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None):
        
        if len(img) == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        interp = interp if interp is not None else self.interp
        return cv2.warpAffine(img, self.rm_image, (self.bound_w, self.bound_h), flags=interp)

    def apply_coords(self, coords):
      
        if len(coords) == 0:
            return coords
        coords = np.asarray(coords, dtype=float)
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation

    def create_rotation_matrix(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm

import inspect
import pprint
from abc import ABCMeta, abstractmethod
class TransformGen(metaclass=ABCMeta):
 

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def get_transform(self, img):
        pass

    def _rand_range(self, low=1.0, high=None, size=None):
      
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
       
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), 'a'
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(name)
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                argstr.append("{}={}".format(name, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()



class RandomRotation(TransformGen):
   

    def __init__(self, angle, expand=True, center=None, sample_style="range", interp=None):
       
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        self.is_range = sample_style == "range"
        if isinstance(angle, (float, int)):
            angle = (angle, angle)
        if center is not None and isinstance(center[0], (float, int)):
            center = (center, center)
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        center = None
        if self.is_range:
            angle = np.random.uniform(self.angle[0], self.angle[1])
            if self.center is not None:
                center = (
                    np.random.uniform(self.center[0][0], self.center[1][0]),
                    np.random.uniform(self.center[0][1], self.center[1][1]),
                )
        else:
            angle = np.random.choice(self.angle)
            if self.center is not None:
                center = np.random.choice(self.center)

        if center is not None:
            center = (w * center[0], h * center[1])  # Convert to absolute coordinates

        return RotationTransform(h, w, angle, expand=self.expand, center=center, interp=self.interp),angle


class Rotate_Rotation:
    def __init__(self):
        self.tfms_gen = RandomRotation([-45., 45.])

    def __call__(self, datadict):
        image = datadict['image']
        tfms, angle = self.tfms_gen.get_transform(image)
        image = tfms.apply_image(image)
        datadict['image'] = image
        for anno in datadict['annotations']:
            box = (anno['bbox'])

            centeryx = np.array([(box[0], box[1])])

            new_centeryx = tfms.apply_coords(centeryx)
            new_angle = self.get_new_angle(box[-1], angle)
            box[0] = new_centeryx[0][0]
            box[1] = new_centeryx[0][1]
            box[-1] = new_angle
            anno['bbox'] = box

    def get_new_angle(self, angle, rotate_angle):
        new_angle = angle + rotate_angle
        if new_angle > 90:
            new_angle = -(180 - new_angle)
        elif new_angle < -90:
            new_angle = -90 - (new_angle)

        return new_angle


class Hlip_Rotation:
    def __init__(self):
        pass

    def __call__(self, datadict):
        image = datadict['image']
        w = datadict['width']
        new_image = A.HorizontalFlip(p=1)(image=image)['image']
        datadict['image'] = new_image
        for anno in datadict['annotations']:
            box = anno['bbox']
            new_box = self.hlip_box(w, box)
            anno['bbox'] = new_box

    def hlip_box(self, w, box):
        box[0] = w - box[0]
        box[4] = -box[4]
        return box


class Contrast:
    def __init__(self):
        pass

    def __call__(self, datadict):
        tfms = A.RandomBrightnessContrast(p=1)
        image = tfms(image=datadict['image'])['image']
        datadict['image'] = image




def get_transform_list(min_angle=-45, max_angle=45):
    transorm_list = []
    rotate = Rotate_Rotation()
    hlip = Hlip_Rotation()
    contrast = Contrast()
    transorm_list.append(rotate)
    transorm_list.append(hlip)
    transorm_list.append(contrast)
    return transorm_list
    
        


# In[ ]:



class myDatasetMapper:
    def __init__(self,cfg,is_train=True):
        self.img_format = cfg.INPUT.FORMAT
        self.is_train=is_train
        self.tfms=get_transform_list()

    def __call__(self,dataset_dict):
        dataset_dict=copy.deepcopy(dataset_dict)
        image=detection_utils.read_image(dataset_dict['file_name'],format=self.img_format)
        detection_utils.check_image_size(dataset_dict,image)
        dataset_dict['image']=image
        image_shape=image.shape[:2]


        if not self.is_train:
            dataset_dict.pop('annotations',None)
        for tfm in self.tfms:
            tfm(dataset_dict)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        anno=dataset_dict['annotations']
        dataset_dict.pop('annotations')
        instance=detection_utils.annotations_to_instances_rotated(anno,image_shape)
        dataset_dict['instances']=detection_utils.filter_empty_instances(instance)
        return dataset_dict


# In[ ]:


datas=get_image_list('train')
cfg=get_cfg()
mapper=myDatasetMapper(cfg)
totol=0
totol_a=0
for data in datas:
    data=mapper(data)
    instances=data['instances']
    if instances.


# In[ ]:


import copy
for d in ['val']:
    DatasetCatalog.register('yaogan1'+'_'+d,lambda d=d:get_image_list(d))
    MetadataCatalog.get('yaogan1'+'_'+d).set(thing_classes=Train_Class_Name)

cfg=get_cfg()
cfg.DATASETS.TRAIN=('yaogan1_val',)
cfg.DATALOADER.NUM_WORKERS=1
cfg.SOLVER.IMS_PER_BATCH=2
cfg.SOLVER.MAX_ITER=5000
cfg.SOLVER.CHECKPOINT_PERIOD=500
mapper=myDatasetMapper(cfg)
dataloader=build_detection_train_loader(cfg,mapper)

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.PROPOSAL_GENERATOR.NAME='RRPN'
cfg.MODEL.ANCHOR_GENERATOR.NAME='RotatedAnchorGenerator'
cfg.MODEL.ANCHOR_GENERATOR.ANGLES=[(-90,-60,-30,0,30,60,90)]
cfg.MODEL.ANCHOR_GENERATOR.SIZE=[[100,200,250,300,350]]
cfg.MODEL.RPN.BBOX_REG_WEIGHTS=(1.0,1.0,1.0,1.0,1.0)
cfg.MODEL.ROI_HEADS.NAME='RROIHeads'
cfg.MODEL.ROI_HEADS.NUM_CLASS=len(Train_Class_Name)
cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE='ROIAlignRotated'
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS=(10.0,10.0,5.0,5.0,5.0)
os.makedirs('result')
cfg.OUTPUT_DIR='result'

model=build_model(cfg)
optimizer=build_optimizer(cfg,model)
scheduler=build_lr_scheduler(cfg,optimizer)
checkpointer=DetectionCheckpointer(model,cfg.OUTPUT_DIR,optimizer=optimizer,scheduler=scheduler)
start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False).get("iteration", -1) + 1
)
max_iter = cfg.SOLVER.MAX_ITER
periodic_checkpointer=PeriodicCheckpointer(checkpointer,cfg.SOLVER.CHECKPOINT_PERIOD,max_iter=max_iter)
writers = (
    [
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
        TensorboardXWriter(cfg.OUTPUT_DIR),
    ]

)
from tqdm import tqdm
with EventStorage(start_iter) as storage:
    with tqdm(total=max_iter) as pbar:

        for data,iteration in zip(dataloader,range(start_iter,max_iter)):

            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            scheduler.step()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
            pbar.update(1)


# In[ ]:




