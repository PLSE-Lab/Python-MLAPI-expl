# import libraries
import os
os.system("pip install catalyst")
os.system("pip install segmentation_models_pytorch")
from catalyst.utils import set_global_seed, prepare_cudnn
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, OptimizerCallback, CriterionCallback, AUCCallback
import segmentation_models_pytorch as smp
import datetime
import argparse
import warnings
import gc
import json
from catalyst import utils
from catalyst.utils import set_global_seed, prepare_cudnn
import os
import pretrainedmodels
import segmentation_models_pytorch as smp
import torch.nn as nn
import torchvision
import cv2
import albumentations as albu
import matplotlib.pyplot as plt
import cv2
import numpy as np
from catalyst.dl.callbacks import InferCallback, CheckpointCallback
import pandas as pd
from typing import Optional, Type, Dict
from catalyst.dl.utils import plot_metrics, save_checkpoint
import numpy as np
import pandas as pd
from torch.optim.optimizer import Optimizer
import math
import itertools as it
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import albumentations as albu
import warnings
from sklearn import metrics
from inspect import signature
import tqdm

warnings.filterwarnings("once")

# get loss
def get_loss(loss: str = 'BCE'):
    if loss == 'BCEDiceLoss':
        criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    elif loss == 'BCEJaccardLoss':
        criterion = smp.utils.losses.BCEJaccardLoss(eps=1.)
    elif loss == 'FocalLoss':
        criterion = FocalLoss()
    elif loss == 'BCEMulticlassDiceLoss':
        criterion = BCEMulticlassDiceLoss()
    elif loss == 'MulticlassDiceMetricCallback':
        criterion = MulticlassDiceMetricCallback()
    elif loss == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    return criterion

# get model
def get_model(model_type: str = 'Unet',
              encoder: str = 'Resnet18',
              encoder_weights: str = 'imagenet',
              activation: str = None,
              n_classes: int = 4,
              task: str = 'segmentation',
              source: str = 'pretrainedmodels',
              head: str = 'simple'):
    """
    Get model for training or inference.

    Returns loaded models, which is ready to be used.

    Args:
        model_type: segmentation model architecture
        encoder: encoder of the model
        encoder_weights: pre-trained weights to use
        activation: activation function for the output layer
        n_classes: number of classes in the output layer
        task: segmentation or classification
        source: source of model for classification
        head: simply change number of outputs or use better output head

    Returns:

    """
    if task == 'segmentation':
        if model_type == 'Unet':
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=n_classes,
                activation=activation
            )

        elif model_type == 'Linknet':
            model = smp.Linknet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=n_classes,
                activation=activation
            )

        elif model_type == 'FPN':
            model = smp.FPN(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=n_classes,
                activation=activation
            )

        else:
            model = None

    elif task == 'classification':
        if source == 'pretrainedmodels':
            model_fn = pretrainedmodels.__dict__[encoder]
            model = model_fn(num_classes=1000, pretrained=encoder_weights)
        elif source == 'torchvision':
            model = torchvision.models.__dict__[encoder](pretrained=encoder_weights)

        if head == 'simple':
            model.last_linear = nn.Linear(model.last_linear.in_features, n_classes)
        else:
            model = Net(net=model)

    return model


class Flatten(nn.Module):
    """
    Simple class for flattening layer.

    """
    def forward(self, x):
        return x.view(x.size()[0], -1)

    
class AdaptiveConcatPool2d(nn.Module):
    # https://github.com/fastai/fastai/blob/e8c855ac70d9d1968413a75c7e9a0f149d28cab3/fastai/layers.py#L171

    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self,
                 size: Optional[int] = None):
        "Output will be 2*size or 2 if size is None"
        super().__init__()
        size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x: Type[torch.Tensor]) -> Type[torch.Tensor]:
        return torch.cat([self.mp(x), self.ap(x)], 1)
    

class Net(nn.Module):
    def __init__(
            self,
            num_classes: int = 4,
            p: float = 0.2,
            net = None) -> None:
        """
        Custom head architecture

        Args:
            num_classes: number of classes
            p: dropout probability
            net: original model
        """
        super().__init__()
        modules = list(net.children())[:-1]
        n_feats = list(net.children())[-1].in_features
        # add custom head
        modules += [nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(81536),
            nn.Dropout(p),
            nn.Linear(81536, n_feats),
            nn.Linear(n_feats, num_classes),
            nn.Sigmoid()
        )]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        logits = self.net(x)
        return logits


#general utils
def get_img(x: str = 'img_name', folder: str = 'train_images'):
    """
    Return image based on image name and folder.

    Args:
        x: image name
        folder: folder with images

    Returns:

    """
    image_path = os.path.join(folder, x)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    """
    Decode rle encoded mask.

    Args:
        mask_rle: encoded mask
        shape: final shape

    Returns:

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order='F')


def make_mask(df: pd.DataFrame, image_name: str = 'img.jpg', shape: tuple = (1400, 2100)):
    """
    Create mask based on df, image name and shape.

    Args:
        df: dataframe with cloud dataset
        image_name: image name
        shape: final shape

    Returns:

    """

    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label)
            masks[:, :, idx] = mask

    return masks


def mask2rle(img):
    """
    Convert mask to rle.

    Args:
        img:

    Returns:

    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# segmentation dataset
class CloudDataset(Dataset):

    def __init__(self, path: str = '',
                 df: pd.DataFrame = None,
                 datatype: str = 'train',
                 img_ids: np.array = None,
                 transforms=albu.Compose([albu.HorizontalFlip()]),
                 preprocessing=None,
                 preload: bool = False,
                 image_size: tuple = (320, 640),
                 augmentation: str = 'default',
                 filter_bad_images: bool = False):
        """

        Args:
            path: path to data
            df: dataframe with data
            datatype: train|valid|test
            img_ids: list of imagee ids
            transforms: albumentation transforms
            preprocessing: preprocessing if necessary
            preload: whether to preload data
            image_size: image size for resizing
            augmentation: name of augmentation settings
            filter_bad_images: to filter out bad images
        """

        self.df = df
        self.path = path
        self.datatype = datatype if datatype == 'test' else 'train'
        if self.datatype != 'test':
            self.data_folder = f"{path}/train_images"
        else:
            self.data_folder = f"{path}/test_images"
        self.img_ids = img_ids
        # list of bad images from discussions
        self.bad_imgs = ['046586a.jpg', '1588d4c.jpg', '1e40a05.jpg', '41f92e5.jpg', '449b792.jpg', '563fc48.jpg',
                         '8bd81ce.jpg', 'c0306e5.jpg', 'c26c635.jpg', 'e04fea3.jpg', 'e5f2f24.jpg', 'eda52f2.jpg',
                         'fa645da.jpg']
        if filter_bad_images:
            self.img_ids = [i for i in self.img_ids if i not in self.bad_imgs]
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.dir_name = f"{self.path}/preload_{augmentation}_{image_size[0]}_{image_size[1]}"

        self.preload = preload
        self.preloaded = False
        if self.preload:
            self.save_processed_()
            self.preloaded = True

    def save_processed_(self):
        """
        Saves train images with augmentations, to speed up training.

        Returns:

        """
        os.makedirs(self.dir_name, exist_ok=True)
        self.dir_name += f"/{self.datatype}"
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
            for i, e in enumerate(self.img_ids):
                img, mask = self.__getitem__(i)
                np.save(f"{self.dir_name}/{e}_mask.npy", mask)
                np.save(f"{self.dir_name}/{e}_img.npy", img)

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        if self.preloaded and self.datatype != 'valid':
            img = np.load(f"{self.dir_name}/{image_name}_img.npy")
            mask = np.load(f"{self.dir_name}/{image_name}_mask.npy")

        else:
            mask = make_mask(self.df, image_name)
            image_path = os.path.join(self.data_folder, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            if self.preprocessing:
                preprocessed = self.preprocessing(image=img, mask=mask)
                img = preprocessed['image']
                mask = preprocessed['mask']

        return img, mask

    def __len__(self):
        return len(self.img_ids)

# classification dataset
class CloudDatasetClassification(Dataset):

    def __init__(self, path: str = '',
                 df: pd.DataFrame = None,
                 datatype: str = 'train',
                 img_ids: np.array = None,
                 transforms=albu.Compose([albu.HorizontalFlip()]),
                 preprocessing=None,
                 preload: bool = False,
                 image_size: tuple = (320, 640),
                 augmentation: str = 'default',
                 one_hot_labels: dict = None,
                 filter_bad_images: bool = False):
        """

        Args:
            path: path to data
            df: dataframe with data
            datatype: train|valid|test
            img_ids: list of imagee ids
            transforms: albumentation transforms
            preprocessing: preprocessing if necessary
            preload: whether to preload data
            image_size: image size for resizing
            augmentation: name of augmentation settings
            one_hot_labels: dictionary with labels for images
            filter_bad_images: to filter out bad images
        """
        self.df = df
        self.path = path
        self.datatype = datatype if datatype == 'test' else 'train'
        if self.datatype != 'test':
            self.data_folder = f"{path}/train_images"
        else:
            self.data_folder = f"{path}/test_images"
        self.img_ids = img_ids
        self.bad_imgs = ['046586a.jpg', '1588d4c.jpg', '1e40a05.jpg', '41f92e5.jpg', '449b792.jpg', '563fc48.jpg',
                         '8bd81ce.jpg', 'c0306e5.jpg', 'c26c635.jpg', 'e04fea3.jpg', 'e5f2f24.jpg', 'eda52f2.jpg',
                         'fa645da.jpg']
        if filter_bad_images:
            self.img_ids = [i for i in self.img_ids if i not in self.bad_imgs]
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.dir_name = f"{self.path}/preload_{augmentation}_{image_size[0]}_{image_size[1]}"
        self.one_hot_labels = one_hot_labels

        self.preload = preload
        self.preloaded = False
        if self.preload:
            self.save_processed_()
            self.preloaded = True

    def save_processed_(self):

        os.makedirs(self.dir_name, exist_ok=True)
        self.dir_name += f"/{self.datatype}"
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
            for i, e in enumerate(self.img_ids):
                img, mask = self.__getitem__(i)
                np.save(f"{self.dir_name}/{e}_img.npy", img)

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        if self.preloaded and self.datatype != 'valid':
            img = np.load(f"{self.dir_name}/{image_name}_img.npy")

        else:
            image_path = os.path.join(self.data_folder, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            augmented = self.transforms(image=img)
            img = augmented['image']
            if self.preprocessing:
                preprocessed = self.preprocessing(image=img)
                img = preprocessed['image']

            label = self.one_hot_labels[image_name]

        return img, label

    def __len__(self):
        return len(self.img_ids)


def prepare_loaders(path: str = '',
                    bs: int = 4,
                    num_workers: int = 0,
                    preprocessing_fn=None,
                    preload: bool = False,
                    image_size: tuple = (320, 640),
                    augmentation: str = 'default',
                    task: str = 'segmentation'):
    """
    Prepare dataloaders for catalyst.

    At first reads dataframe with the data and prepares it to be used in dataloaders.
    Creates dataloaders and returns them.

    Args:
        path: path to data
        bs: batch size
        num_workers: number of workers
        preprocessing_fn: preprocessing
        preload: whether to save augmented data on disk
        image_size: image size to resize
        augmentation: augmentation name
        task: segmentation or classification

    Returns:

    """

    train = pd.read_csv(f'{path}/train.csv')
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    id_mask_count = train.loc[~train['EncodedPixels'].isnull(), 'Image_Label'].apply(
        lambda x: x.split('_')[0]).value_counts(). \
        reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
    train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42,
                                            stratify=id_mask_count['count'], test_size=0.1)

    if task == 'classification':
        train_df = train[~train['EncodedPixels'].isnull()]
        classes = train_df['label'].unique()
        train_df = train_df.groupby('im_id')['label'].agg(set).reset_index()
        for class_name in classes:
            train_df[class_name] = train_df['label'].map(lambda x: 1 if class_name in x else 0)

        img_2_ohe_vector = {img: np.float32(vec) for img, vec in zip(train_df['im_id'], train_df.iloc[:, 2:].values)}

    sub = pd.read_csv(f'{path}/sample_submission.csv')
    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
    test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values

    if task == 'segmentation':
        if preload:
            _ = CloudDataset(path=path, df=train, datatype='train', img_ids=id_mask_count['img_id'].values,
                             transforms=get_training_augmentation(augmentation=augmentation, image_size=image_size),
                             preprocessing=get_preprocessing(preprocessing_fn),
                             preload=preload, image_size=(320, 640))

        train_dataset = CloudDataset(path=path, df=train, datatype='train', img_ids=train_ids,
                                     transforms=get_training_augmentation(augmentation=augmentation, image_size=image_size),
                                     preprocessing=get_preprocessing(preprocessing_fn),
                                     preload=preload, image_size=(320, 640))
        valid_dataset = CloudDataset(path=path, df=train, datatype='valid', img_ids=valid_ids,
                                     transforms=get_validation_augmentation(image_size=image_size),
                                     preprocessing=get_preprocessing(preprocessing_fn),
                                     preload=preload, image_size=(320, 640))

    elif task == 'classification':
        if preload:
            _ = CloudDatasetClassification(path=path, df=train, datatype='train', img_ids=id_mask_count['img_id'].values,
                             transforms=get_training_augmentation(augmentation=augmentation, image_size=image_size),
                             preprocessing=get_preprocessing(preprocessing_fn),
                             preload=preload, image_size=(320, 640), one_hot_labels=img_2_ohe_vector)

        train_dataset = CloudDatasetClassification(path=path, df=train, datatype='train', img_ids=train_ids,
                                     transforms=get_training_augmentation(augmentation=augmentation,
                                                                          image_size=image_size),
                                     preprocessing=get_preprocessing(preprocessing_fn),
                                     preload=preload, image_size=(320, 640), one_hot_labels=img_2_ohe_vector)
        valid_dataset = CloudDatasetClassification(path=path, df=train, datatype='valid', img_ids=valid_ids,
                                     transforms=get_validation_augmentation(image_size=image_size),
                                     preprocessing=get_preprocessing(preprocessing_fn),
                                     preload=preload, image_size=(320, 640), one_hot_labels=img_2_ohe_vector)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_dataset = CloudDataset(path=path, df=sub, datatype='test', img_ids=test_ids,
                                transforms=get_validation_augmentation(image_size=image_size),
                                preprocessing=get_preprocessing(preprocessing_fn), preload=preload,
                                image_size=(320, 640))
    test_loader = DataLoader(test_dataset, batch_size=bs // 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    loaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader
    }

    return loaders



def to_tensor(x, **kwargs):
    """
    Convert image or mask.

    Args:
        x:
        **kwargs:

    Returns:

    """

    return x.transpose(2, 0, 1).astype('float32')

# easily choose augmentations
def get_training_augmentation(augmentation: str='default', image_size: tuple = (320, 640)):
    """
    Get augmentations
    There is a dictionary where values are different augmentation functions, so it easy to
    switch between augmentations;

    Args:
        augmentation:
        image_size:

    Returns:

    """
    LEVELS = {
        'default': get_training_augmentation0,
        '1': get_training_augmentation1,
        '2': get_training_augmentation2
    }

    assert augmentation in LEVELS.keys()
    return LEVELS[augmentation](image_size)


def get_training_augmentation0(image_size: tuple = (320, 640)):
    """

    Args:
        image_size:

    Returns:

    """
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.5),
        albu.RandomGamma(),
        albu.Resize(*image_size)
    ]
    return albu.Compose(train_transform)


def get_training_augmentation1(image_size: tuple = (320, 640)):
    """

    Args:
        image_size:

    Returns:

    """
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.2),
        albu.Resize(*image_size),
    ]
    return albu.Compose(train_transform)


def get_training_augmentation2(image_size: tuple = (320, 640)):
    """

    Args:
        image_size:

    Returns:

    """
    train_transform = [
        albu.Resize(*image_size),
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=0.1, shift_limit=0.2),
        albu.Blur(),
        albu.RandomBrightnessContrast()
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation(image_size: tuple = (320, 640)):
    """

    Args:
        image_size:

    Returns:

    """
    test_transform = [
        albu.Resize(*image_size)
    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)



def predict(loaders=None,
            runner=None,
            class_params: dict = None,
            path: str = '',
            sub_name: str = ''):
    """

    Args:
        loaders:
        runner:
        class_params:
        path:
        sub_name:

    Returns:

    """
    encoded_pixels = []
    image_id = 0
    for _, test_batch in enumerate(loaders['test']):
        runner_out = runner.predict_batch({"features": test_batch[0].cuda()})['logits']
        for _, batch in enumerate(runner_out):
            for probability in batch:

                probability = probability.cpu().detach().numpy()
                if probability.shape != (350, 525):
                    probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                    prediction, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0],
                                                       class_params[image_id % 4][1])
                if num_predict == 0:
                    encoded_pixels.append('')
                else:
                    r = mask2rle(prediction)
                    encoded_pixels.append(r)
                image_id += 1

    sub = pd.read_csv(f'{path}/sample_submission.csv')
    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv(f'submissions/submission_{sub_name}.csv', columns=['Image_Label', 'EncodedPixels'], index=False)
    

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def visualize(image, mask, original_image=None, original_mask=None, fontsize: int = 14):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.

    Args:
        image: transformed image
        mask: transformed mask
        original_image:
        original_mask:
        fontsize:

    Returns:

    """
    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 5, figsize=(24, 24))

        ax[0].imshow(image)
        for i in range(4):
            ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].set_title(f'Mask {class_dict[i]}', fontsize=fontsize)
    else:
        f, ax = plt.subplots(2, 5, figsize=(24, 12))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        for i in range(4):
            ax[0, i + 1].imshow(original_mask[:, :, i])
            ax[0, i + 1].set_title(f'Original mask {class_dict[i]}', fontsize=fontsize)

        ax[1, 0].imshow(image)
        ax[1, 0].set_title('Transformed image', fontsize=fontsize)

        for i in range(4):
            ax[1, i + 1].imshow(mask[:, :, i])
            ax[1, i + 1].set_title(f'Transformed mask {class_dict[i]}', fontsize=fontsize)


def visualize_with_raw(image, mask, original_image=None, original_mask=None, raw_image=None, raw_mask=None):
    """
    Similar to visualize function, but with post-processed image, mask.

    Args:
        image:
        mask:
        original_image:
        original_mask:
        raw_image:
        raw_mask:

    Returns:

    """
    fontsize = 14
    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}

    f, ax = plt.subplots(3, 5, figsize=(24, 12))

    ax[0, 0].imshow(original_image)
    ax[0, 0].set_title('Original image', fontsize=fontsize)

    for i in range(4):
        ax[0, i + 1].imshow(original_mask[:, :, i])
        ax[0, i + 1].set_title(f'Original mask {class_dict[i]}', fontsize=fontsize)

    ax[1, 0].imshow(raw_image)
    ax[1, 0].set_title('Original image', fontsize=fontsize)

    for i in range(4):
        ax[1, i + 1].imshow(raw_mask[:, :, i])
        ax[1, i + 1].set_title(f'Raw predicted mask {class_dict[i]}', fontsize=fontsize)

    ax[2, 0].imshow(image)
    ax[2, 0].set_title('Transformed image', fontsize=fontsize)

    for i in range(4):
        ax[2, i + 1].imshow(mask[:, :, i])
        ax[2, i + 1].set_title(f'Predicted mask with processing {class_dict[i]}', fontsize=fontsize)


def plot_with_augmentation(image, mask, augment):
    """
    Wrapper for `visualize` function.

    Args:
        image:
        mask:
        augment:

    Returns:

    """
    augmented = augment(image=image, mask=mask)
    image_flipped = augmented['image']
    mask_flipped = augmented['mask']
    visualize(image_flipped, mask_flipped, original_image=image, original_mask=mask)


def post_process(probability: np.array = None, threshold: float = 0.5, min_size: int = 10):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored

    Args:
        probability: mask
        threshold: threshold for processing
        min_size: min_size for processing

    Returns:

    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def dice(img1: np.array, img2: np.array) -> float:
    """
    Calculate dice of two images
    Args:
        img1:
        img2:

    Returns:

    """
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())


def get_optimal_postprocess(loaders=None,
                            runner=None,
                            logdir: str = ''
                            ):
    """
    Calculate optimal thresholds for validation data.

    Args:
        loaders: loaders with necessary datasets
        runner: runner
        logdir: directory with model checkpoints

    Returns:

    """
    loaders['infer'] = loaders['valid']

    runner.infer(
        model=runner.model,
        loaders=loaders,
        callbacks=[
            CheckpointCallback(
                resume=f"{logdir}/checkpoints/best.pth"),
            InferCallback()
        ],
    )
    valid_masks = []
    probabilities = np.zeros((2220, 350, 525))
    for i, (batch, output) in enumerate(zip(
            loaders['infer'].dataset, runner.callbacks[0].predictions["logits"])):
        image, mask = batch
        for m in mask:
            if m.shape != (350, 525):
                m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            valid_masks.append(m)

        for j, probability in enumerate(output):
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            probabilities[i * 4 + j, :, :] = probability

    class_params = {}
    for class_id in range(4):
        print(class_id)
        attempts = []
        for t in range(0, 100, 10):
            t /= 100
            for ms in [0, 100, 1000, 5000, 10000, 11000, 14000, 15000, 16000, 18000, 19000, 20000, 21000, 23000, 25000, 27000, 30000, 50000]:
                masks = []
                for i in range(class_id, len(probabilities), 4):
                    probability = probabilities[i]
                    predict, num_predict = post_process(sigmoid(probability), t, ms)
                    masks.append(predict)

                d = []
                for i, j in zip(masks, valid_masks[class_id::4]):
                    if (i.sum() == 0) & (j.sum() == 0):
                        d.append(1)
                    else:
                        d.append(dice(i, j))

                attempts.append((t, ms, np.mean(d)))

        attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

        attempts_df = attempts_df.sort_values('dice', ascending=False)
        print(attempts_df.head())
        best_threshold = attempts_df['threshold'].values[0]
        best_size = attempts_df['size'].values[0]

        class_params[class_id] = (best_threshold, best_size)

    print(class_params)
    return class_params

# optimizers
class Ranger(Optimizer):

    # https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/ranger.py

    def __init__(self, params, lr=1e-3, alpha=0.5, k=6,
                 N_sma_threshhold=5, betas=(.95, 0.999), eps=1e-5, weight_decay=0):
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params
        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None,None,None] for ind in range(10)]

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  #get state dict for this param

                if len(state) == 0:   #if first time to run...init dictionary with our desired entries
                    #if self.first_run_check==0:
                        #self.first_run_check=1
                        #print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    #look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                #begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                #compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                #compute mean moving avg
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1


                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

                #integrated look ahead...
                #we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer'] #get access to slow param tensor
                    slow_p.add_(self.alpha, p.data - slow_p)  #(fast weights - slow weights) * alpha
                    p.data.copy_(slow_p)  #copy interpolated weights to RAdam param tensor

        return loss


class RAdam(Optimizer):
    """
    https://github.com/LiyuanLucasLiu/RAdam
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

    
# https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py
class Lookahead(Optimizer):
    def __init__(self, base_optimizer,alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        self.optimizer = base_optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        for group in self.param_groups:
            group["step_counter"] = 0
        self.slow_weights = [[p.clone().detach() for p in group['params']]
                                for group in self.param_groups]

        for w in it.chain(*self.slow_weights):
            w.requires_grad = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        loss = self.optimizer.step()
        for group,slow_weights in zip(self.param_groups,self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] % self.k != 0:
                continue
            for p,q in zip(group['params'],slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(self.alpha,p.data - q.data)
                p.data.copy_(q.data)
        return loss


class Ralamb(Optimizer):
    '''
    Ralamb optimizer (RAdam + LARS trick)
    https://gist.github.com/redknightlois/c4023d393eb8f92bb44b2ab582d7ec20
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(Ralamb, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Ralamb, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ralamb does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, radam_step = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        radam_step = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        radam_step = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = radam_step

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                radam_norm = p_data_fp32.pow(2).sum().sqrt()
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm

                state['weight_norm'] = weight_norm
                state['adam_norm'] = radam_norm
                state['trust_ratio'] = trust_ratio

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-radam_step * trust_ratio, exp_avg, denom)
                else:
                    p_data_fp32.add_(-radam_step * trust_ratio, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


def get_optimizer(optimizer: str = 'Adam',
                  lookahead: bool = False,
                  model=None,
                  separate_decoder: bool = True,
                  lr: float = 1e-3,
                  lr_e: float = 1e-3):
    """
    # https://github.com/lonePatient/lookahead_pytorch/blob/master/run.py

    :param optimizer:
    :param lookahead:
    :param model:
    :param separate_decoder:
    :param lr:
    :param lr_e:
    :return:
    """

    if separate_decoder:
        params = [
                    {'params': model.decoder.parameters(), 'lr': lr
                     },
                    {'params': model.encoder.parameters(), 'lr': lr_e},
                ]
    else:
        params = [{'params': model.parameters(), 'lr': lr}]

    if optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=lr)
    elif optimizer == 'RAdam':
        optimizer = RAdam(params, lr=lr)
    elif optimizer == 'Ralamb':
        optimizer = Ralamb(params, lr=lr)
    else:
        raise ValueError('unknown base optimizer type')

    if lookahead:
        optimizer = Lookahead(base_optimizer=optimizer, k=5, alpha=0.5)

    return optimizer


def find_threshold(y_valid: np.array = None, y_pred: np.array = None, eval_metric: str = 'f1',
                   print_scores: bool = True):
    """
    Find threshold for predictions based on metric

    Args:
        y_valid:
        y_pred:
        eval_metric:
        print_scores:

    Returns:

    """

    metrics_dict = {'f1': metrics.f1_score,
                    'recall': metrics.recall_score,
                    'precision': metrics.precision_score}

    metric_value = 0
    best_t = 0
    for t in np.arange(0.1, 1, 0.05):
        valid_pr = (y_pred > t).astype(int)
        valid_metric = metrics_dict[eval_metric](y_valid, valid_pr)
        if valid_metric > metric_value:
            metric_value = valid_metric
            best_t = t

    t = best_t
    if print_scores:
        valid_pr = (y_pred > t).astype(int)
        valid_f1 = metrics.f1_score(y_valid, valid_pr)
        valid_r = metrics.recall_score(y_valid, valid_pr)

        valid_p = metrics.precision_score(y_valid, valid_pr)

        valid_roc = metrics.roc_auc_score(y_valid, y_pred)

        valid_apc = metrics.average_precision_score(y_valid, y_pred)
        print(
            f"""Best threshold: {t:.2f}. Valid f1: {valid_f1:.4f}. Valid recall: {valid_r:.4f}.\nValid precision: {valid_p:.4f}. Valid rocauc: {valid_roc:.4f}. Valid apc: {valid_apc:.4f}.""")

        print(metrics.confusion_matrix(y_valid, valid_pr))

    return t


def plot_precision_recall(y_true, y_pred, title=''):
    """

    Args:
        y_true:
        y_pred:
        title:

    Returns:

    """
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    average_precision = metrics.average_precision_score(y_true, y_pred)
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'2-class Precision-Recall curve: AP={average_precision:0.2f} {title}')


class CustomCheckpointCallback(CheckpointCallback):
    def process_checkpoint(
        self,
        logdir: str,
        checkpoint: Dict,
        is_best: bool,
        main_metric: str = "loss",
        minimize_metric: bool = True
    ):

        exclude = ["criterion", "optimizer", "scheduler"]
        checkpoint = {
            key: value
            for key, value in checkpoint.items()
            if all(z not in key for z in exclude)
        }
        suffix = self.get_checkpoint_suffix(checkpoint)
        suffix = f"{suffix}.exception_"
        
        filepath = save_checkpoint(
            checkpoint=checkpoint,
            logdir=f"{logdir}/checkpoints/",
            suffix=suffix,
            is_best=is_best,
            is_last=True
        )

        valid_metrics = checkpoint["valid_metrics"]
        checkpoint_metric = valid_metrics[main_metric]
        self.top_best_metrics.append(
            (filepath, checkpoint_metric, valid_metrics)
        )
        self.truncate_checkpoints(minimize_metric=minimize_metric)

        metrics = self.get_metric(valid_metrics)
        self.save_metric(logdir, metrics)
