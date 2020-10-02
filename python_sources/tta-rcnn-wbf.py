#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import gc

sys.path.insert(0, "../input/weightedboxesfusion/")
gc.collect()


# In[ ]:


import glob
from itertools import product
import os
import random
from ensemble_boxes import *
import torch
from  torch.utils.data import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
import albumentations as A
from albumentations import (
    BboxParams,
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Crop,
    Compose
)
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import cv2
from matplotlib import pyplot as plt
import time
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm


# In[ ]:


def visualize(**images):
       """PLot images in one row."""
       n = len(images)
       plt.figure(figsize=(16, 5))
       for i, (name, image) in enumerate(images.items()):
           plt.subplot(1, n, i + 1)
           plt.xticks([])
           plt.yticks([])
           plt.title(' '.join(name.split('_')).title())
           plt.imshow(image)
       plt.show()


# In[ ]:


class Metrics:
    @staticmethod
    def calculate_iou(gt, pr, form='pascal_voc') -> float:
        """Calculates the Intersection over Union.

        Args:
            gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box
            pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box
            form: (str) gt/pred coordinates format
                - pascal_voc: [xmin, ymin, xmax, ymax]
                - coco: [xmin, ymin, w, h]
        Returns:
            (float) Intersection over union (0.0 <= iou <= 1.0)
        """
        if form == 'coco':
            gt = gt.copy()
            pr = pr.copy()

            gt[2] = gt[0] + gt[2]
            gt[3] = gt[1] + gt[3]
            pr[2] = pr[0] + pr[2]
            pr[3] = pr[1] + pr[3]

        # Calculate overlap area
        dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1

        if dx < 0:
            return 0.09
        dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

        if dy < 0:
            return 0.0

        overlap_area = dx * dy

        # Calculate union area
        union_area = (
                (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +
                (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -
                overlap_area
        )

        return overlap_area / union_area

    @staticmethod
    def find_best_match(gts, pred, pred_idx, threshold=0.5, form='pascal_voc', ious=None) -> int:
        """Returns the index of the 'best match' between the
        ground-truth boxes and the prediction. The 'best match'
        is the highest IoU. (0.0 IoUs are ignored).

        Args:
            gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
            pred: (List[Union[int, float]]) Coordinates of the predicted box
            pred_idx: (int) Index of the current predicted box
            threshold: (float) Threshold
            form: (str) Format of the coordinates
            ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

        Return:
            (int) Index of the best match GT box (-1 if no match above threshold)
        """
        best_match_iou = -np.inf
        best_match_idx = -1
        for gt_idx in range(len(gts)):

            if gts[gt_idx][0] < 0:
                # Already matched GT-box
                continue

            iou = -1 if ious is None else ious[gt_idx][pred_idx]

            if iou < 0:
                iou = Metrics.calculate_iou(gts[gt_idx], pred, form=form)

                if ious is not None:
                    ious[gt_idx][pred_idx] = iou

            if iou < threshold:
                continue

            if iou > best_match_iou:
                best_match_iou = iou
                best_match_idx = gt_idx

        return best_match_idx

    @staticmethod
    def calculate_precision(gts, preds, threshold=0.5, form='coco', ious=None) -> float:
        """Calculates precision for GT - prediction pairs at one threshold.

        Args:
            gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
            preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
                   sorted by confidence value (descending)
            threshold: (float) Threshold
            form: (str) Format of the coordinates
            ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.

        Return:
            (float) Precision
        """
        n = len(preds)
        tp = 0
        fp = 0

        for pred_idx in range(n):

            best_match_gt_idx = Metrics.find_best_match(gts, preds[pred_idx], pred_idx,
                                                        threshold=threshold, form=form, ious=ious)

            if best_match_gt_idx >= 0:
                # True positive: The predicted box matches a gt box with an IoU above the threshold.
                tp += 1
                # Remove the matched GT box
                gts[best_match_gt_idx] = -1
            else:
                # No match
                # False positive: indicates a predicted box had no associated gt box.
                fp += 1

        # False negative: indicates a gt box had no associated predicted box.
        fn = (gts.sum(axis=1) > 0).sum()

        return tp / (tp + fp + fn)

    @staticmethod
    def calculate_image_precision(gts, preds, thresholds=(0.5,), form='coco') -> float:
        """Calculates image precision.

        Args:
            gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes
            preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,
                   sorted by confidence value (descending)
            thresholds: (float) Different thresholds
            form: (str) Format of the coordinates

        Return:
            (float) Precision
        """
        n_threshold = len(thresholds)
        image_precision = 0.0

        ious = np.ones((len(gts), len(preds))) * -1
        # ious = None

        for threshold in thresholds:
            precision_at_threshold = Metrics.calculate_precision(gts.copy(), preds, threshold=threshold,
                                                                 form=form, ious=ious)
            image_precision += precision_at_threshold / n_threshold

        return image_precision


# In[ ]:


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class TrainUtils:
    @staticmethod
    def trainModels(models, train_dataloader, valid_dataloader, device, num_epochs=1, valid_pred_min=0.65):
        for model in models:
            print("Starting train model ", str(model.__name__))
            model.to(device)
            # construct an optimizer
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(params, lr=0.005,
                                        momentum=0.9, weight_decay=0.0005)

            train_hist = Averager()
            t = 0
            for epoch in range(num_epochs):
                model.train()
                train_hist.reset()
                for images, targets,image_path in train_dataloader:
                    model.train()
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    train_loss = losses.item()
                    train_hist.send(train_loss)
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    t += 1

                model.eval()
                validation_image_precisions = []
                iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]
                for images, targets,image_path in valid_dataloader:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    with torch.no_grad():
                        outputs = model(images)

                    for i, image in enumerate(images):
                        boxes = outputs[i]['boxes'].data.cpu().numpy()
                        scores = outputs[i]['scores'].data.cpu().numpy()
                        gt_boxes = targets[i]['boxes'].cpu().numpy()
                        preds_sorted_idx = np.argsort(scores)[::-1]
                        preds_sorted = boxes[preds_sorted_idx]
                        image_precision = Metrics.calculate_image_precision(preds_sorted,
                                                                            gt_boxes,
                                                                            thresholds=iou_thresholds,
                                                                            form='coco')
                        validation_image_precisions.append(image_precision)

                valid_prec = np.mean(validation_image_precisions)
                print("Validation Precision: {0:.4f}".format(valid_prec))

                # print training/validation statistics
                print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                    epoch,
                    train_loss
                ))

                if valid_prec >= valid_pred_min:
                    print('Validation precision increased({:.6f} --> {:.6f}).  Saving model ...'.format(
                        valid_pred_min,
                        valid_prec))
                    torch.save(model.state_dict(), 'faster_rrcnn_' + str(model.__name__) + '.pth')
                    valid_pred_min = valid_prec
            torch.save(model.state_dict(), 'faster_rrcnn_' + str(model.__name__) + '_' + str(time.time()) + '.pth')


# In[ ]:


class BaseWheatTTA:
    """ author: @shonenkov """
    image_size = 512

    def augment(self, image):
        raise NotImplementedError
    
    def batch_augment(self, images):
        raise NotImplementedError
    
    def deaugment_boxes(self, boxes):
        raise NotImplementedError

class TTAHorizontalFlip(BaseWheatTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(1)
    
    def batch_augment(self, images):
        return images.flip(2)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]
        return boxes

class TTAVerticalFlip(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return image.flip(2)
    
    def batch_augment(self, images):
        return images.flip(3)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]
        return boxes
    
class TTARotate90(BaseWheatTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = self.image_size - boxes[:, [1,3]]
        res_boxes[:, [1,3]] = boxes[:, [2,0]]
        return res_boxes

class TTACompose(BaseWheatTTA):
    """ author: @shonenkov """
    def __init__(self, transforms):
        self.transforms = transforms
        
    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image
    
    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images
    
    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)
        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)
        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)
        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)
        return result_boxes
    
    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)


# In[ ]:


class FpnResenet50:
    @staticmethod
    def getNet():
        fpn_resnet = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=False)
        num_classes = 2  # 1 class (wheat) + background
        # get number of input features for the classifier
        in_features = fpn_resnet.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        fpn_resnet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        fpn_resnet.__name__ = "fpn_resnet"

        return fpn_resnet


# In[ ]:


class DatasetUtils:
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    @staticmethod
    def preprocessing_csv(data_frame):
        # seperate bbox to x,y,w,h columns
        seperator = lambda x: np.fromstring(x[1:-1], sep=',')
        bbox = np.stack(data_frame['bbox'].apply(seperator))
        for i, dim in enumerate(['x', 'y', 'w', 'h']):
            data_frame[dim] = bbox[:, i]
        data_frame.drop(columns='bbox', inplace=True)

    @staticmethod
    def splitData(all_data_records: pd.DataFrame, test_size=0.33):
        image_ids = all_data_records['image_id'].unique()
        train_ids, valid_ids = train_test_split(image_ids, test_size=test_size, random_state=42)
        valid_df = all_data_records[all_data_records['image_id'].isin(valid_ids)]
        train_df = all_data_records[all_data_records['image_id'].isin(train_ids)]
        return train_df, valid_df


# In[ ]:


# Functions to visualize bounding boxes and class labels on an image. 
# Based on https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max =  int(x_min), int(y_min), int(x_max), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualizeTarget(image,target, category_id_to_name={1 : "Wheat"}):
    img = image.copy()
    for idx, bbox in enumerate(target['boxes']):
        img = visualize_bbox(img, bbox, target['labels'][idx], category_id_to_name)
#     plt.figure(figsize=(12, 12))
#     plt.imshow(img)
    return img
    
def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params=BboxParams(format='pascal_voc', min_area=min_area, 
                                               min_visibility=min_visibility, label_fields=['labels']))


# In[ ]:


class WheatDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)  # reading an image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)  # changing color space BGR --> RGB
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].to_numpy()
        
        area = (boxes[:, 3]) * (boxes[:, 2])  # Calculating area of boxes W * H
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # upper coordinate X + W
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # lower coordinate Y + H
                
        # xmin,ymin,xmax,ymax
        
        area = torch.as_tensor(area, dtype=torch.float32)
        
        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {'bboxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]), 'area': area, 'iscrowd': iscrowd}
                
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['bboxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            
            image = sample["image"]
            
            target['bboxes'] = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
            target['bboxes'] =  target['bboxes'].reshape(-1, 4)
            target["boxes"] = target["bboxes"]
            
            del target['bboxes']
            return image,target, f'{self.image_dir}/{image_id}.jpg'  # image Tensor , target with boxes , path to image

    def __len__(self):
        return self.image_ids.shape[0]


# In[ ]:


DATA_ROOT_PATH = '../input/global-wheat-detection'

all_wheat_dataset_train = pd.read_csv('../input/global-wheat-detection/train.csv')

all_wheat_dataset_train.head()


# In[ ]:


DatasetUtils.preprocessing_csv(all_wheat_dataset_train)
all_wheat_dataset_train.head()


# In[ ]:


train_df,valid_df = DatasetUtils.splitData(all_data_records=all_wheat_dataset_train)
train_df.shape,valid_df.shape


# In[ ]:


aug_trans = get_aug([HorizontalFlip(p=0.5),
                     VerticalFlip(p=0.5),
                     A.ToGray(p=0.3),
                     A.GaussianBlur(p=0.3),
                     A.RandomBrightnessContrast(p=0.7),
                     RandomCrop(p=0.5,height=512,width=512),
                     Resize(width=512,height=512),
                     ToTensorV2(p=1)])


# In[ ]:


#train_dataset = WheatDataset(train_df,DATA_ROOT_PATH+"/train",transforms=aug_trans)
valid_dataset = WheatDataset(valid_df,DATA_ROOT_PATH+"/train",transforms=aug_trans)


# In[ ]:


#train_dataloader = DataLoader(train_dataset,batch_size=8,collate_fn=DatasetUtils.collate_fn,num_workers=8)
valid_dataloader = DataLoader(valid_dataset,batch_size=8,collate_fn=DatasetUtils.collate_fn,num_workers=8)


# In[ ]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device


# In[ ]:


for data in valid_dataloader:
    images, targets, image_path = data
    targets = [{k: v.to(device).detach().cpu().numpy() for k, v in t.items()} for t in targets]
    images = list(image.to(device) for image in images)
    with torch.no_grad():
        for image, target in zip(images, targets):
            image = image.squeeze(0).permute(1,2,0)
            image = image.detach().cpu().numpy()
            img = visualizeTarget(image,target) 
            visualize(image =img)
    break


# # Models

# In[ ]:


resnet50 = FpnResenet50.getNet()


# In[ ]:


#%time TrainUtils.trainModels([resnet50], train_dataloader, valid_dataloader, device, num_epochs=20, valid_pred_min=0.55)


# In[ ]:


#torch.save(resnet50.state_dict(), 'resnet_taa_finish_20_full_data_v1_' + str(resnet50.__name__) + '_' + str(time.time()) + '.pth')


# # Loading pretrain model

# In[ ]:


resnet50.load_state_dict(torch.load('../input/resnet50-tta/resnet_taa_finish_20_full_data_v1_fpn_resnet_1593370667.8672676.pth'))


# # Pseudo Labeling 

# # let predict test dataset & retrain 

# In[ ]:


class TestDataset(Dataset):
    def __init__(self, path, transform=None):
        self.image_paths = glob.glob(os.path.join(path, '*.jpg'))
        self.transform = transform

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)  # reading an image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)  # changing color space BGR --> RGB
        image /= 255.0
        if self.transform:
            data = { 'image' : image}
            data = self.transform(**data)
            image = data['image']
        return image, self.image_paths[index]  # return image tensor , path to image

    def __len__(self):
        return len(self.image_paths)


# In[ ]:


def run_wbf(predictions,weights=None,image_size=512,iou_thr=0.5,skip_box_thr=0.43):
    boxes_list = [(pred["boxes"] / (image_size-1)).tolist() for pred in predictions]
    scores_list = [pred["scores"].tolist() for pred in predictions]
    labels_list = [np.ones(len(score)).astype(int).tolist() for score in scores_list]
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    return boxes, scores, labels


# In[ ]:


# create array of tta_transforms randomly 

tta_transform = TTACompose([
    TTARotate90(),
    TTAVerticalFlip(),
])

tta_transforms = []
for tta_combination in product([TTAHorizontalFlip(), None], 
                               [TTAVerticalFlip(), None],
                               [TTARotate90(), None]):
    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))


# In[ ]:


resnet50.cuda()
resnet50.eval()
iou_thr = 0.4
detection_threshold = 0.9

test_images_paths = glob.glob(os.path.join('/kaggle/input/global-wheat-detection/test/*'))
test_df = []

for image_path in tqdm(test_images_paths):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # reading an image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)  # changing color space BGR --> RGB
    image /= 255.0
    t = Compose([Resize(width=512,height=512),ToTensorV2()])
    data = { "image": image}
    data = t(**data)
    image = data["image"]
    image = image.squeeze(0)
   

    predictions = []
    n = 5
    
    for i in range(n):
        ## create tta image
        selected_tta = random.choice(tta_transforms)
        tta_image = selected_tta.augment(image) ## need to be random
        outputs = resnet50(tta_image.unsqueeze(0).cuda())
        boxes = outputs[0]['boxes'].data.detach().cpu().numpy()
        scores = outputs[0]['scores'].data.detach().cpu().numpy()
        boxes = boxes[scores >= detection_threshold]
        scores = scores[scores >= detection_threshold]
        original_boxes  = selected_tta.deaugment_boxes(boxes)
        predictions.append({"boxes"  : original_boxes,'scores': scores})
    
    boxes, scores, labels = run_wbf(predictions,iou_thr=iou_thr,image_size=512)
    
    boxes = boxes * 1024
    
    image_name = image_path.split("/")[-1].split(".")[0]
    new_boxes = []
    for (b, s) in zip(boxes,scores):
        x_min , y_min , x_max,y_max = b
        
        x_min = round(x_min)
        y_min = round(y_min)
        
        x_max = round(x_max)
        y_max = round(y_max)
        

        w = round(x_max-x_min) # width
        h = round(y_max-y_min) # height

        if(x_min + w > 1024):
            x_max -= (x_min + w  - 1024)
        if(y_min + h > 1024):
            y_max -= (y_min + h  - 1024)
        b = (x_min , y_min , x_max,y_max)
        new_boxes.append(b)
        
        test_df.append([image_name,1024,1024,x_min,y_min,x_max - x_min,y_max - y_min])

        
    
    boxes = new_boxes
    image = image.permute(1,2,0).detach().cpu().numpy()

#     image = cv2.resize(image,(1024,1024))
#     original_with_boxes = visualizeTarget(image
#                               ,{"boxes": boxes,'labels' : np.ones(len(boxes))})

#     visualize(original=original_with_boxes)
    


        #image_id	width	height	source	x	y	w	h
test_df = pd.DataFrame(test_df,columns=["image_id","width","height","x","y","w","h"])


# # pretrain model on pesudo labeling

# In[ ]:


test_dataset = WheatDataset(test_df,DATA_ROOT_PATH+"/test",transforms=aug_trans)


# In[ ]:


test_dataloader = DataLoader(test_dataset,batch_size=8,collate_fn=DatasetUtils.collate_fn,num_workers=8)


# In[ ]:


# for data in test_dataloader:
#     images, targets, image_path = data
#     targets = [{k: v.to(device).detach().cpu().numpy() for k, v in t.items()} for t in targets]
#     images = list(image.to(device) for image in images)
#     with torch.no_grad():
#         for image, target in zip(images, targets):
#             image = image.squeeze(0).permute(1,2,0)
#             image = image.detach().cpu().numpy()
#             img = visualizeTarget(image,target) 
#             visualize(image =img)


# # train on test pesudo labels

# In[ ]:


get_ipython().run_line_magic('time', 'TrainUtils.trainModels([resnet50], test_dataloader, valid_dataloader, device, num_epochs=3, valid_pred_min=0.55)')


# # Tesing with TTA 

# # Submission

# In[ ]:


# scores :
# iou_thr = 0.4
# detection_threshold = 0.7  = 0.54
# n = 5 

# iou_thr = 0.4
# detection_threshold = 0.5  = 0.56
# n = 20 


# full data 20 epoch + TTA + WBF
# iou_thr = 0.4
# detection_threshold = 0.5  = 0.67
# n = 20 



# In[ ]:


resnet50.cuda()
resnet50.eval()
iou_thr = 0.4
detection_threshold = 0.5

test_images_paths = glob.glob(os.path.join('/kaggle/input/global-wheat-detection/test/*'))
submission = []

for image_path in tqdm(test_images_paths):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # reading an image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)  # changing color space BGR --> RGB
    image /= 255.0
    t = Compose([Resize(width=512,height=512),ToTensorV2()])
    data = { "image": image}
    data = t(**data)
    image = data["image"]
    image = image.squeeze(0)
   

    predictions = []
    n = 20
    
    for i in range(n):
        ## create tta image
        selected_tta = random.choice(tta_transforms)
        tta_image = selected_tta.augment(image) ## need to be random
        outputs = resnet50(tta_image.unsqueeze(0).cuda())
        boxes = outputs[0]['boxes'].data.detach().cpu().numpy()
        scores = outputs[0]['scores'].data.detach().cpu().numpy()
        boxes = boxes[scores >= detection_threshold]
        scores = scores[scores >= detection_threshold]
        original_boxes  = selected_tta.deaugment_boxes(boxes)
        predictions.append({"boxes"  : original_boxes,'scores': scores})
    
    boxes, scores, labels = run_wbf(predictions,iou_thr=iou_thr,image_size=512)
    
    boxes = boxes * 1024
        
    
    ##
    prediction_string = []
    
    new_boxes = []
    for (b, s) in zip(boxes,scores):
        x_min , y_min , x_max,y_max = b
              
        x_min = round(x_min)
        y_min = round(y_min)
        
        x_max = round(x_max)
        y_max = round(y_max)
        

        w = round(x_max-x_min) # width
        h = round(y_max-y_min) # height

        if(x_min + w > 1024):
            x_max -= (x_min + w  - 1024)
        if(y_min + h > 1024):
            y_max -= (y_min + h  - 1024)
        

        # after recalculate
        w = round(x_max-x_min) # width
        h = round(y_max-y_min) # height
        
        b = (x_min , y_min , x_max,y_max)
        new_boxes.append(b)
        
        

        prediction_string.append(f"{s} {x_min} {y_min} {h} {w}")
    prediction_string = " ".join(prediction_string)
    
    image_name = image_path.split("/")[-1].split(".")[0]
    submission.append([image_name, prediction_string])
    
    
    image_numpy = image.permute(1,2,0).detach().cpu().numpy()
    
    image_numpy = cv2.resize(image_numpy,(1024,1024))
    original_with_boxes = visualizeTarget(image_numpy
                              ,{"boxes": new_boxes,'labels' : np.ones(len(new_boxes))})
    
    visualize(original=original_with_boxes)

 


# In[ ]:


SUBMISSION_PATH = '/kaggle/working'
submission_id = 'submission'
cur_submission_path = os.path.join(SUBMISSION_PATH, '{}.csv'.format(submission_id))
sample_submission = pd.DataFrame(submission, columns=["image_id","PredictionString"])
sample_submission.to_csv(cur_submission_path, index=False)
submission_df = pd.read_csv(cur_submission_path)


# In[ ]:


submission_df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




