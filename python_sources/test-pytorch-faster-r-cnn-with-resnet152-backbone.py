#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np

from PIL import Image
from enum import Enum
from pathlib import Path

from dataclasses import dataclass

import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from albumentations import *
from albumentations.pytorch import ToTensorV2
import cv2


# In[ ]:


device = torch.device('cuda')


# In[ ]:


def fasterrcnn_resnet101_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=False,
                             trainable_backbone_layers=3, **kwargs):
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet152', pretrained_backbone)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model


# In[ ]:


def initialize_model():
    model = fasterrcnn_resnet101_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


# In[ ]:


model = initialize_model()


# In[ ]:


save_model_path = "/kaggle/input/resnet152modelepoch20"


# In[ ]:


model.load_state_dict(torch.load(os.path.join(save_model_path, f"best_model_epoch_{20}.pth")))
model.to(device)
model.eval()


# In[ ]:


@dataclass
class DatasetArguments:
    data_dir: Path
    images_lists_dict: dict
    labels_csv_file_name: str

@dataclass
class DataLoaderArguments:
    batch_size: int
    num_workers: int
    dataset_arguments: DatasetArguments


# In[ ]:


phase="test"


# In[ ]:


root_data_dir = Path("/kaggle/input/global-wheat-detection/")
unlabeled_generated_images_path = Path(f"/kaggle/input/global-wheat-detection/{phase}/")


# In[ ]:


def get_images_file_names(directory):
    _, _, files = next(os.walk(directory))
    return files
test_file_names = get_images_file_names(unlabeled_generated_images_path)
test_file_names = [x.split(".")[0] for x in test_file_names]


# In[ ]:


images_lists_dict = {
    "test": test_file_names
}


# In[ ]:


prediction_dataset_arguments = DatasetArguments(
    data_dir=root_data_dir,
    images_lists_dict=images_lists_dict,
    labels_csv_file_name="sample_submission.csv",
)
predict_dataloaders_arguments = DataLoaderArguments(
    batch_size=4,
    num_workers=0,
    dataset_arguments=prediction_dataset_arguments
)


# In[ ]:


def transform_set():
    transforms_dict = {
        'test': get_test_transforms()
    }
    return transforms_dict


def get_test_transforms():
    return Compose(
        [
            ToTensorV2(p=1.0),
        ]
    )


# In[ ]:


class ObjectDetectionDataset(Dataset):
    def __init__(self, images_root_directory, images_list, labels_csv_file_name, phase, transforms):
        super(ObjectDetectionDataset).__init__()
        self.images_root_directory = images_root_directory
        self.phase = phase
        self.transforms = transforms
        self.images_list = images_list
        if self.phase in ["train", "val"]:
            self.labels_dataframe = pd.read_csv(os.path.join(images_root_directory, labels_csv_file_name))

    def __getitem__(self, item):
        sample = {
            "local_image_id": None,
            "image_id": None,
            "labels": None,
            "boxes": None,
            "area": None,
            "iscrowd": None
        }

        image_id = self.images_list[item]
        image_path = os.path.join(self.images_root_directory,
                                  "train" if self.phase in ["train", "val"] else "test",
                                  image_id + ".jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        sample["local_image_id"] = image_id
        sample["image_id"] = torch.tensor([item])
        if self.phase in ["train", "val"]:
            boxes = self.labels_dataframe[self.labels_dataframe.image_id == image_id].bbox.values.tolist()
            boxes = [eval(box_i) for box_i in boxes]
            areas = _areas(boxes)
            boxes = _adjust_boxes_format(boxes)

            sample["labels"] = torch.ones((len(boxes),), dtype=torch.int64)
            sample["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            sample["area"] = torch.as_tensor(areas, dtype=torch.float32)
            sample["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        if self.transforms is not None:
            sample["image"] = image
            if self.phase in ["train", "val"]:
                transformed_sample = self.transforms(image=sample["image"],
                                                     bboxes=sample["boxes"],
                                                     labels=sample["labels"])
                sample["boxes"] = torch.as_tensor(transformed_sample["bboxes"], dtype=torch.float32)
            else:
                transformed_sample = self.transforms(image=sample["image"])
            image = transformed_sample["image"]
            del sample["image"]
        return image, sample
    def __len__(self):
        return len(self.images_list)


# In[ ]:


def create_dataset(arguments):
    dataset = ObjectDetectionDataset(arguments.data_dir,
                                     arguments.images_lists_dict[arguments.phase],
                                     arguments.labels_csv_file_name,
                                     arguments.phase,
                                     arguments.transforms)
    return dataset


# In[ ]:


def create_prediction_dataloader(arguments, input_size):
    data_transforms = transform_set()
    batch_size = arguments.batch_size
    num_workers = arguments.num_workers
    arguments.dataset_arguments.phase = phase
    arguments.dataset_arguments.transforms = data_transforms["test"]
    image_datasets = create_dataset(arguments.dataset_arguments)
    dataloader = DataLoader(image_datasets, batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
    return dataloader


def collate_fn(batch):
    return tuple(zip(*batch))


# In[ ]:


dataloader = create_prediction_dataloader(predict_dataloaders_arguments, None)


# In[ ]:


detection_threshold=0.45
def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


# In[ ]:


results = []


# In[ ]:


for images, sample in dataloader:
    image_ids = [x["local_image_id"] for x in sample]
    images = list(image.to(device) for image in images)
    outputs = model(images)

    for i, image in enumerate(images):

        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()
        
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        image_id = image_ids[i]
        
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        
        result = {
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores) if boxes.shape[0] > 0 else ""
        }

        
        results.append(result)


# In[ ]:


test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])


# In[ ]:


test_df.to_csv('submission.csv', index=False)


# In[ ]:


test_df.head()


# In[ ]:




