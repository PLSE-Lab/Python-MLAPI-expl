#!/usr/bin/env python
# coding: utf-8

# # Birdsong Pytorch Baseline: ResNeSt50-fast (Training)

# ## About
# 
# In this notebook, I try ResNeSt, which is the one of state of the art in image recognition.  
# 
# For the fair comparison with @hidehisaarai1213 's [great baseline](https://www.kaggle.com/hidehisaarai1213/inference-pytorch-birdcall-resnet-baseline), I used a model with the same depth and as the same experimental settings as possible. But There are some differences mainly because of the GPU resource limitation.
# 
# The experimental settings are as follows:
# 
# * Randomly crop 5 seconds for each train audio clip each epoch.
# * No augmentation.
# * Used pretrained weight of _`ResNeSt50-fast-1s1x64d`_ provided by the authors at [their repository](https://github.com/zhanghang1989/ResNeSt).
# * Used `BCELoss`
# * Trained **_50_** epoch and saved the weight which got best **_loss_** (this is because f1 score relies on thresholds.)
# * `Adam` optimizer (`lr=0.001`) with `CosineAnnealingLR` (`T_max=10`).
# * Used `StratifiedKFold(n_splits=5)` to split dataset and used only first fold
# * `batch_size`: **_50_**
# * melspectrogram parameters
#   - `n_mels`: 128
#   - `fmin`: 20
#   - `fmax`: 16000
# * image size: 224x547
# 
# I forked a lot of codes such as preprocessing from @hidehisaarai1213 's [notebook](https://www.kaggle.com/hidehisaarai1213/inference-pytorch-birdcall-resnet-baseline) and [GitHub repository](https://github.com/koukyo1994/kaggle-birdcall-resnet-baseline-training). Many thanks!!!
# 
# 
# ### Note
# 
# #### about dataset
# I prepared resmpaled train dataset for this notebook, see more details in:
# https://www.kaggle.com/c/birdsong-recognition/discussion/164197
# 
# 
# #### about custom packages
# In this **_training notebook_**, I used two custom packages, `pytorch-pfn-extras` for training and the authors' official implementation of `ResNeSt` for building model.  
# On the other hand, as stated in [code requirements](https://www.kaggle.com/c/birdsong-recognition/overview/code-requirements), participants are **not allowed** to use custom packages in **_submission notebook_**.
# 
# If you fork this notebook, keep the above things in mind.
# 
# 
# ### Reference
# 
# #### ResNeSt: Split-Attention Networks
# * author: Hang Zhang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Muller, R. Manmatha, Mu Li and Alex Smola 
# * paper: [arXiv 2004.08955](https://arxiv.org/abs/2004.08955)
# * code: [GitHub](https://github.com/koukyo1994/kaggle-birdcall-resnet-baseline-training)
# 
# #### pytorch-pfn-extras
# * author: Preferred Networks, Inc.
# * code: [GitHub](https://github.com/pfnet/pytorch-pfn-extras)

# ## Prepare

# ### import libraries

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'pip install ../input/pytorch-pfn-extras/pytorch-pfn-extras-0.2.1/')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'pip install ../input/resnest50-fast-package/resnest-0.0.6b20200701/resnest/')


# In[ ]:


import os
import gc
import time
import shutil
import random
import warnings
import typing as tp
from pathlib import Path
from contextlib import contextmanager

import yaml
from joblib import delayed, Parallel

import cv2
import librosa
import audioread
import soundfile as sf

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import resnest.torch as resnest_torch

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions as ppe_extensions

pd.options.display.max_rows = 500
pd.options.display.max_columns = 500


# In[ ]:


Path("/root/.cache/torch/checkpoints").mkdir(parents=True)


# In[ ]:


get_ipython().system('cp ../input/resnest50-fast-package/resnest50_fast_*.pth /root/.cache/torch/checkpoints/')


# ### define utilities

# In[ ]:


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
#     torch.backends.cudnn.deterministic = True  # type: ignore
#     torch.backends.cudnn.benchmark = True  # type: ignore
    

@contextmanager
def timer(name: str) -> None:
    """Timer Util"""
    t0 = time.time()
    print("[{}] start".format(name))
    yield
    print("[{}] done in {:.0f} s".format(name, time.time() - t0))


# ### read data

# In[ ]:


ROOT = Path.cwd().parent
INPUT_ROOT = ROOT / "input"
RAW_DATA = INPUT_ROOT / "birdsong-recognition"
TRAIN_AUDIO_DIR = RAW_DATA / "train_audio"
TRAIN_RESAMPLED_AUDIO_DIRS = [
  INPUT_ROOT / "birdsong-resampled-train-audio-{:0>2}".format(i)  for i in range(5)
]
TEST_AUDIO_DIR = RAW_DATA / "test_audio"


# In[ ]:


# train = pd.read_csv(RAW_DATA / "train.csv")
train = pd.read_csv(TRAIN_RESAMPLED_AUDIO_DIRS[0] / "train_mod.csv")


# In[ ]:


train.head().T


# In[ ]:


if not TEST_AUDIO_DIR.exists():
    TEST_AUDIO_DIR = INPUT_ROOT / "birdcall-check" / "test_audio"
    test = pd.read_csv(INPUT_ROOT / "birdcall-check" / "test.csv")
else:
    test = pd.read_csv(RAW_DATA / "test.csv")


# In[ ]:


test.head().T


# ### settings

# In[ ]:


settings_str = """
globals:
  seed: 1213
  device: cuda
  num_epochs: 50
  output_dir: /kaggle/training_output/
  use_fold: 0
  target_sr: 32000

dataset:
  name: SpectrogramDataset
  params:
    img_size: 224
    melspectrogram_parameters:
      n_mels: 128
      fmin: 20
      fmax: 16000
    
split:
  name: StratifiedKFold
  params:
    n_splits: 5
    random_state: 42
    shuffle: True

loader:
  train:
    batch_size: 50
    shuffle: True
    num_workers: 2
    pin_memory: True
    drop_last: True
  val:
    batch_size: 100
    shuffle: False
    num_workers: 2
    pin_memory: True
    drop_last: False

model:
  name: resnest50_fast_1s1x64d
  params:
    pretrained: True
    n_classes: 264

loss:
  name: BCEWithLogitsLoss
  params: {}

optimizer:
  name: Adam
  params:
    lr: 0.001

scheduler:
  name: CosineAnnealingLR
  params:
    T_max: 10
"""


# In[ ]:


settings = yaml.safe_load(settings_str)


# In[ ]:


# if not torch.cuda.is_available():
#     settings["globals"]["device"] = "cpu"


# In[ ]:


for k, v in settings.items():
    print("[{}]".format(k))
    print(v)


# ### preprocess audio data
# 
# Code is forked from: https://github.com/koukyo1994/kaggle-birdcall-resnet-baseline-training/blob/master/input/birdsong-recognition/prepare.py
# 
# I modified this partially. 
# 
# However, in this notebook, I used uploaded resampled audio because this preprocessing is too heavy for kaggle notebook.

# In[ ]:


def resample(ebird_code: str,filename: str, target_sr: int):    
    audio_dir = TRAIN_AUDIO_DIR
    resample_dir = TRAIN_RESAMPLED_DIR
    ebird_dir = resample_dir / ebird_code
    
    try:
        y, _ = librosa.load(
            audio_dir / ebird_code / filename,
            sr=target_sr, mono=True, res_type="kaiser_fast")

        filename = filename.replace(".mp3", ".wav")
        sf.write(ebird_dir / filename, y, samplerate=target_sr)
    except Exception as e:
        print(e)
        with open("skipped.txt", "a") as f:
            file_path = str(audio_dir / ebird_code / filename)
            f.write(file_path + "\n")


# In[ ]:


# train_org = train.copy()
# TRAIN_RESAMPLED_DIR = Path("/kaggle/processed_data/train_audio_resampled")
# TRAIN_RESAMPLED_DIR.mkdir(parents=True)

# for ebird_code in train.ebird_code.unique():
#     ebird_dir = TRAIN_RESAMPLED_DIR / ebird_code
#     ebird_dir.mkdir()

# warnings.simplefilter("ignore")
# train_audio_infos = train[["ebird_code", "filename"]].values.tolist()
# Parallel(n_jobs=NUM_THREAD, verbose=10)(
#     delayed(resample)(ebird_code, file_name, TARGET_SR) for ebird_code, file_name in train_audio_infos)

# train["resampled_sampling_rate"] = TARGET_SR
# train["resampled_filename"] = train["filename"].map(
#     lambda x: x.replace(".mp3", ".wav"))
# train["resampled_channels"] = "1 (mono)"


# ## Definition

# ### Dataset
# * forked from: https://github.com/koukyo1994/kaggle-birdcall-resnet-baseline-training/blob/master/src/dataset.py
# * modified partialy
# 

# In[ ]:


BIRD_CODE = {
    'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4,
    'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9,
    'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14,
    'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19,
    'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24,
    'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29,
    'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34,
    'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39,
    'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44,
    'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49,
    'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54,
    'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59,
    'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64,
    'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69,
    'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74,
    'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79,
    'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84,
    'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89,
    'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94,
    'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99,
    'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104,
    'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109,
    'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114,
    'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119,
    'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124,
    'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129,
    'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134,
    'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139,
    'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144,
    'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149,
    'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154,
    'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159,
    'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164,
    'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169,
    'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174,
    'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179,
    'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184,
    'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189,
    'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194,
    'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199,
    'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204,
    'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209,
    'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214,
    'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219,
    'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224,
    'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229,
    'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234,
    'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239,
    'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244,
    'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249,
    'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254,
    'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259,
    'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263
}

INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}


# In[ ]:


PERIOD = 5

def mono_to_color(
    X: np.ndarray, mean=None, std=None,
    norm_max=None, norm_min=None, eps=1e-6
):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

class SpectrogramDataset(data.Dataset):
    def __init__(
        self,
        file_list: tp.List[tp.List[str]], img_size=224,
        waveform_transforms=None, spectrogram_transforms=None, melspectrogram_parameters={}
    ):
        self.file_list = file_list  # list of list: [file_path, ebird_code]
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        wav_path, ebird_code = self.file_list[idx]

        y, sr = sf.read(wav_path)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)
        else:
            len_y = len(y)
            effective_length = sr * PERIOD
            if len_y < effective_length:
                new_y = np.zeros(effective_length, dtype=y.dtype)
                start = np.random.randint(effective_length - len_y)
                new_y[start:start + len_y] = y
                y = new_y.astype(np.float32)
            elif len_y > effective_length:
                start = np.random.randint(len_y - effective_length)
                y = y[start:start + effective_length].astype(np.float32)
            else:
                y = y.astype(np.float32)

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(melspec)
        else:
            pass

        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

#         labels = np.zeros(len(BIRD_CODE), dtype="i")
        labels = np.zeros(len(BIRD_CODE), dtype="f")
        labels[BIRD_CODE[ebird_code]] = 1

        return image, labels


# ### Training Utility

# In[ ]:


def get_loaders_for_training(
    args_dataset: tp.Dict, args_loader: tp.Dict,
    train_file_list: tp.List[str], val_file_list: tp.List[str]
):
    # # make dataset
    train_dataset = SpectrogramDataset(train_file_list, **args_dataset)
    val_dataset = SpectrogramDataset(val_file_list, **args_dataset)
    # # make dataloader
    train_loader = data.DataLoader(train_dataset, **args_loader["train"])
    val_loader = data.DataLoader(val_dataset, **args_loader["val"])
    
    return train_loader, val_loader


# In[ ]:


def get_model(args: tp.Dict):
    model =getattr(resnest_torch, args["name"])(pretrained=args["params"]["pretrained"])
    del model.fc
    # # use the same head as the baseline notebook.
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, args["params"]["n_classes"]))
    
    return model


# In[ ]:


def train_loop(
    manager, args, model, device,
    train_loader, optimizer, scheduler, loss_func
):
    """Run minibatch training loop"""
    while not manager.stop_trigger:
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            with manager.run_iteration():
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_func(output, target)
                ppe.reporting.report({'train/loss': loss.item()})
                loss.backward()
                optimizer.step()
                scheduler.step()

def eval_for_batch(
    args, model, device,
    data, target, loss_func, eval_func_dict={}
):
    """
    Run evaliation for valid
    
    This function is applied to each batch of val loader.
    """
    model.eval()
    data, target = data.to(device), target.to(device)
    output = model(data)
    # Final result will be average of averages of the same size
    val_loss = loss_func(output, target).item()
    ppe.reporting.report({'val/loss': val_loss})
    
    for eval_name, eval_func in eval_func_dict.items():
        eval_value = eval_func(output, target).item()
        ppe.reporting.report({"val/{}".format(eval_aame): eval_value})


# In[ ]:


def set_extensions(
    manager, args, model, device, test_loader, optimizer,
    loss_func, eval_func_dict={}
):
    """set extensions for PPE"""
        
    my_extensions = [
        # # observe, report
        ppe_extensions.observe_lr(optimizer=optimizer),
        # ppe_extensions.ParameterStatistics(model, prefix='model'),
        # ppe_extensions.VariableStatisticsPlot(model),
        ppe_extensions.LogReport(),
        ppe_extensions.PlotReport(['train/loss', 'val/loss'], 'epoch', filename='loss.png'),
        ppe_extensions.PlotReport(['lr',], 'epoch', filename='lr.png'),
        ppe_extensions.PrintReport([
            'epoch', 'iteration', 'lr', 'train/loss', 'val/loss', "elapsed_time"]),
#         ppe_extensions.ProgressBar(update_interval=100),

        # # evaluation
        (
            ppe_extensions.Evaluator(
                test_loader, model,
                eval_func=lambda data, target:
                    eval_for_batch(args, model, device, data, target, loss_func, eval_func_dict),
                progress_bar=True),
            (1, "epoch"),
        ),
        # # save model snapshot.
        (
            ppe_extensions.snapshot(
                target=model, filename="snapshot_epoch_{.updater.epoch}.pth"),
            ppe.training.triggers.MinValueTrigger(key="val/loss", trigger=(1, 'epoch'))
        ),
    ]
           
    # # set extensions to manager
    for ext in my_extensions:
        if isinstance(ext, tuple):
            manager.extend(ext[0], trigger=ext[1])
        else:
            manager.extend(ext)
        
    return manager


# ## Training

# ### prepare data

# #### get wav file path

# In[ ]:


tmp_list = []
for audio_d in TRAIN_RESAMPLED_AUDIO_DIRS:
    if not audio_d.exists():
        continue
    for ebird_d in audio_d.iterdir():
        if ebird_d.is_file():
            continue
        for wav_f in ebird_d.iterdir():
            tmp_list.append([ebird_d.name, wav_f.name, wav_f.as_posix()])
            
train_wav_path_exist = pd.DataFrame(
    tmp_list, columns=["ebird_code", "resampled_filename", "file_path"])

del tmp_list

train_all = pd.merge(
    train, train_wav_path_exist, on=["ebird_code", "resampled_filename"], how="inner")

print(train.shape)
print(train_wav_path_exist.shape)
print(train_all.shape)


# In[ ]:


train_all.head()


# #### split data

# In[ ]:


# # for test run
# test_run_idx = sorted(np.random.choice(len(train_all), len(train_all) // 10, replace=False))
# train_all = train_all.iloc[test_run_idx, :].reset_index(drop=True)
# settings["globals"]["num_epochs"] = 20
# settings["scheduler"]["params"]["T_max"] = 4


# In[ ]:


skf = StratifiedKFold(**settings["split"]["params"])

train_all["fold"] = -1
for fold_id, (train_index, val_index) in enumerate(skf.split(train_all, train_all["ebird_code"])):
    train_all.iloc[val_index, -1] = fold_id
    
# # check the propotion
fold_proportion = pd.pivot_table(train_all, index="ebird_code", columns="fold", values="xc_id", aggfunc=len)
print(fold_proportion.shape)


# In[ ]:


fold_proportion


# In[ ]:


use_fold = settings["globals"]["use_fold"]
train_file_list = train_all.query("fold != @use_fold")[["file_path", "ebird_code"]].values.tolist()
val_file_list = train_all.query("fold == @use_fold")[["file_path", "ebird_code"]].values.tolist()

print("[fold {}] train: {}, val: {}".format(use_fold, len(train_file_list), len(val_file_list)))


# ## run training

# In[ ]:


set_seed(settings["globals"]["seed"])
device = torch.device(settings["globals"]["device"])
output_dir = Path(settings["globals"]["output_dir"])

# # # get loader
train_loader, val_loader = get_loaders_for_training(
    settings["dataset"]["params"], settings["loader"], train_file_list, val_file_list)

# # # get model
model = get_model(settings["model"])
model = model.to(device)

# # # get optimizer
optimizer = getattr(
    torch.optim, settings["optimizer"]["name"]
)(model.parameters(), **settings["optimizer"]["params"])

# # # get scheduler
scheduler = getattr(
    torch.optim.lr_scheduler, settings["scheduler"]["name"]
)(optimizer, **settings["scheduler"]["params"])

# # # get loss
loss_func = getattr(nn, settings["loss"]["name"])(**settings["loss"]["params"])

# # # create training manager
trigger = None

manager = ppe.training.ExtensionsManager(
    model, optimizer, settings["globals"]["num_epochs"],
    iters_per_epoch=len(train_loader),
    stop_trigger=trigger,
    out_dir=output_dir
)

# # # set manager extensions
manager = set_extensions(
    manager, settings, model, device,
    val_loader, optimizer, loss_func,
)


# In[ ]:


# # runtraining
train_loop(
    manager, settings, model, device,
    train_loader, optimizer, scheduler, loss_func)


# In[ ]:


del train_loader
del val_loader
del model
del optimizer
del scheduler
del loss_func
del manager

gc.collect()


# ## save results

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'ls /kaggle/training_output')


# In[ ]:


for f_name in ["log","loss.png", "lr.png"]:
    shutil.copy(output_dir / f_name, f_name)


# In[ ]:


log = pd.read_json("log")
best_epoch = log["val/loss"].idxmin() + 1
log.iloc[[best_epoch - 1],]


# In[ ]:


shutil.copy(output_dir / "snapshot_epoch_{}.pth".format(best_epoch), "best_model.pth")


# In[ ]:


m = get_model({
    'name': settings["model"]["name"],
    'params': {'pretrained': False, 'n_classes': 264}})
state_dict = torch.load('best_model.pth')
m.load_state_dict(state_dict)

