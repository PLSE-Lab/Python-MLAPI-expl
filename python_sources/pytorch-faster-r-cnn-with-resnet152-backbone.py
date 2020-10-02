#!/usr/bin/env python
# coding: utf-8

# In this kernel, I show how I changed the backbone of the Faster-R-CNN model from ResNet50 to ResNet152. To achieve that, I used some of the source code of the torchvision and changed it manually.

# The following points are covered:
# * Create dataset
# * Create dataloader
# * Prepare the model
# * Training the model

# ## Load Libraries

# In[ ]:


import os
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from albumentations import *
from albumentations.pytorch import ToTensorV2
import cv2


# ## Prepare Input variables

# In[ ]:


class Device(Enum):
    GPU = "GPU"
    TPU = "TPU"


# In[ ]:


root_data_dir = Path("/kaggle/input/global-wheat-detection/")
test_train_ratio = 0.1
batch_size=8
seed = 0
train_device = Device.GPU
number_of_epochs = 2
learning_rate = 0.0001
weight_decay = 1e-5


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


# ## Split Data to train and val datasets

# In[ ]:


def _get_images_file_names_from_csv(directory):
    dataframe = pd.read_csv(os.path.join(directory, "train.csv"))
    files = dataframe["image_id"].unique().tolist()
    return files


# In[ ]:


def _choose_train_valid_file_names(file_names, valid_numbers, seed):
    np.random.seed(seed)
    valid_file_names = np.random.choice(file_names, valid_numbers, replace=False).tolist()
    train_file_names = [file_name_i for file_name_i in file_names if file_name_i not in valid_file_names]
    return train_file_names, valid_file_names


# In[ ]:


#split data
file_names = _get_images_file_names_from_csv(root_data_dir)
valid_numbers = round(len(file_names) * test_train_ratio)
train_file_names, valid_file_names = _choose_train_valid_file_names(file_names, valid_numbers, seed)


# In[ ]:


images_lists_dict = {
    "train": train_file_names,
    "val": valid_file_names,
}


# In[ ]:


dataset_arguments = DatasetArguments(
    data_dir=root_data_dir,
    images_lists_dict=images_lists_dict,
    labels_csv_file_name="train.csv",
)

dataloaders_arguments = DataLoaderArguments(
    batch_size=batch_size,
    num_workers=1,
    dataset_arguments=dataset_arguments
)


# ## Prepare the transforms:
# I chose some of the transforms from this [notebook](https://www.kaggle.com/shonenkov/training-efficientdet)

# In[ ]:


def transform_set():
    transforms_dict = {
        'train': get_train_transforms(),
        'val': get_valid_transforms()
    }
    return transforms_dict


def get_train_transforms():
    return Compose(
        [OneOf([HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
                                   val_shift_limit=0.2, p=0.9),
                RandomBrightnessContrast(brightness_limit=0.2,
                                         contrast_limit=0.2, p=0.9)],
               p=0.9),
            ToGray(p=0.01),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0)],
        p=1.0,
        bbox_params=BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_valid_transforms():
    return Compose(
        [
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


# ## Create the pytorch dataset

# In[ ]:


def _adjust_boxes_format(boxes):
    # original format [xmin, ymin, width, height]
    # new format [xmin, ymin, xmax, ymax]
    adjusted_boxes = []
    for box_i in boxes:
        adjusted_box_i = [0, 0, 0, 0]
        adjusted_box_i[0] = box_i[0]
        adjusted_box_i[1] = box_i[1]
        adjusted_box_i[2] = box_i[0] + box_i[2]
        adjusted_box_i[3] = box_i[1] + box_i[3]
        adjusted_boxes.append(adjusted_box_i)
    return adjusted_boxes


# In[ ]:


def _areas(boxes):
    # original format [xmin, ymin, width, height]
    areas = []
    for box_i in boxes:
        areas.append(box_i[2] * box_i[3])
    return areas


# In[ ]:


# dataset
class ObjectDetectionDataset(Dataset):
    def __init__(self, images_root_directory,
                 images_list,
                 labels_csv_file_name,
                 phase,
                 transforms):
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
            transformed_sample = self.transforms(image=sample["image"],
                                                 bboxes=sample["boxes"],
                                                 labels=sample["labels"])
            image = transformed_sample["image"]
            sample["boxes"] = torch.as_tensor(transformed_sample["bboxes"], dtype=torch.float32)
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


def create_datasets_dictionary(arguments, input_size):
    data_transforms = transform_set()
    image_datasets = {
        'train': None,
        'val': None
    }
    for phase in ['train', 'val']:
        arguments.phase = phase
        arguments.transforms = data_transforms[phase]
        image_datasets[phase] = create_dataset(arguments)
    return image_datasets


# ## Create the pytorch dataloaders

# In[ ]:


def collate_fn(batch):
    return tuple(zip(*batch))


# In[ ]:


def create_dataloaders_dictionary(arguments, input_size):
    batch_size = arguments.batch_size
    num_workers = arguments.num_workers
    image_datasets = create_datasets_dictionary(arguments.dataset_arguments, input_size)
    dataloaders_dict = {x: DataLoader(image_datasets[x],
                                      batch_size=batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=num_workers,
                                      collate_fn=collate_fn) for x in ['train', 'val']}
    return dataloaders_dict


# ## Prepare the model
# I used the code here:
# [torchvision source code](https://github.com/pytorch/vision/blob/3d65fc6723f1e0709916f24d819d6e17a925b394/torchvision/models/detection/backbone_utils.py#L44)

# In[ ]:


def fasterrcnn_resnet101_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True,
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


def get_training_device(train_device):
    if train_device == Device.GPU:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            raise ValueError("No GPU was found")
    else:
        device = torch.device("cpu")
    return device


# In[ ]:


device = get_training_device(train_device)


# In[ ]:


model = initialize_model()


# In[ ]:


model = model.to(device)


# In[ ]:


dataloaders = create_dataloaders_dictionary(dataloaders_arguments,input_size=None)


# ### Some basic calculations that could be useful later

# In[ ]:


train_dataset_size = len(dataloaders["train"].dataset)
number_of_iteration_per_epoch = int(train_dataset_size / dataloaders_arguments.batch_size)
total_number_of_iteration = number_of_epochs * number_of_iteration_per_epoch
learning_rate_step_size = 2 * number_of_iteration_per_epoch


# ## Prepare for training

# In[ ]:


def get_learnable_parameters(model, feature_extract):
    params_to_update = model.parameters()

    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    return params_to_update


# In[ ]:


params_to_update = get_learnable_parameters(model, feature_extract=False)
optimizer = optim.Adam(params_to_update, lr=learning_rate, weight_decay=weight_decay)


# In[ ]:


lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                              T_0=learning_rate_step_size,
                                                              T_mult=1)


# In[ ]:


def _save_model(model, model_path):
    torch.save(model, model_path)


# In[ ]:


# Saving the checkpoint helps to starting training from a certain point.
def _save_checkpoint(epoch, model, optimizer, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)


# In[ ]:


def save_model(epoch,model, optimizer):
    model_path = f"best_model_epoch_{epoch}.pth"
    _save_model(model.state_dict(), model_path)
    checkpoint_path = f"checkpoint_{epoch}.pth"
    _save_checkpoint(epoch, model, optimizer, checkpoint_path)


# In[ ]:


class Detector:
    def fit_model(self):
        start_epoch = 0
        iteration_i = 0
        for current_epoch in range(start_epoch, number_of_epochs):
            running_loss = 0
            print(f"Starting Epoch: {current_epoch}")
            progress_bar = tqdm(dataloaders["train"])
            for inputs, labels in  progress_bar:
                running_loss_i = self.training_round(inputs, labels)
                running_loss += running_loss_i
                current_running_error = running_loss/((iteration_i - 
                                                      current_epoch * 
                                                      number_of_iteration_per_epoch + 1)*batch_size)
                progress_bar.set_description(f"Running train loss: {current_running_error}")
                iteration_i += 1
            epoch_loss = running_loss / len(dataloaders["train"].dataset)
            print(f"Finishing Current epoch: {current_epoch} ... training loss: {epoch_loss}")
            print("saving the model and checkpoint: ")
            save_model(current_epoch, model, optimizer)
            for inputs, labels in tqdm(dataloaders["val"]):
                self.validation_round(inputs, labels)

    def training_round(self, inputs, labels):
        inputs = list(image.to(device) for image in inputs)
        inputs = torch.stack(inputs)
        labels = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in labels]
        model.train()
        loss_dict = model(inputs, labels)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        running_loss_i = loss.item() * inputs.size(0) 
        return running_loss_i

    def validation_round(self, inputs, labels):
        model.eval()
        inputs = list(image.to(device) for image in inputs)
        inputs = torch.stack(inputs)
        labels = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in labels]
        outputs = model(inputs)
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        # Note: I used here MSCOCO evaluation metric locally. Unfortunately, I could not run in this kernel.
        # I appreciate it if you can help here


# ## Start training

# In[ ]:


detector  =  Detector()


# In[ ]:


## The model will be saved for each epoch
detector.fit_model()


# **I appreciate your feedback and upvote if you think it was useful**

# In[ ]:




