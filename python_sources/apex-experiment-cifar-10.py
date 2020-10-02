import os

os.environ['SEED'] = '1309'

# -----------------------------------
#         Environment Setup      
# -----------------------------------
from subprocess import run
from pathlib import Path

run(["pip install delegator.py"], shell=True, check=True)

import delegator

c = delegator.run('nvidia-smi', block=True)
print(c.out)

c = delegator.run('git clone https://github.com/NVIDIA/apex /tmp/apex', block=True)
assert c.return_code == 0
c = delegator.run('pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" /tmp/apex/.', block=True)
assert c.return_code == 0

c = delegator.run('pip install https://github.com/ceshine/pytorch_helper_bot/archive/master.zip', block=True)
assert c.return_code == 0

c = delegator.run('git clone https://github.com/ceshine/apex_pytorch_cifar_experiment /tmp/src', block=True)
assert c.return_code == 0
c = delegator.run('cp /tmp/src/*.py .', block=True)
assert c.return_code == 0

c = delegator.run('pip install python-telegram-bot pretrainedmodels', block=True)
assert c.return_code == 0

import sys
sys.path.append("../input/")
sys.path.append(".")

# -----------------------------------
#             Training
# -----------------------------------
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
from helperbot import (
    TriangularLR, BaseBot, WeightDecayOptimizerWrapper,
    LearningRateSchedulerCallback
)
from helperbot.metrics import SoftmaxAccuracy
from apex import amp

from baseline import (
    CifarBot, get_cifar10_dataset,
    get_wide_resnet, get_se_resnext,
    get_gpu_memory_map
)
from telegram_tokens import BOT_TOKEN, CHAT_ID
from telegram_sender import telegram_sender

DEVICE = torch.device("cuda")
EPOCHS = 10
MODEL_FUNC = get_wide_resnet

@telegram_sender(token=BOT_TOKEN, chat_id=CHAT_ID)
def train_apex(level):
    train_dl, valid_dl = get_cifar10_dataset(batch_size=128)
    steps_per_epoch = len(train_dl)

    model = MODEL_FUNC()
    
    optimizer = optim.SGD(
        model.parameters(), lr=0.1,
        momentum=0.9, weight_decay=5e-4)  
    if level != "O0":
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=level
        )
    
    n_epochs = EPOCHS
    n_steps = n_epochs * steps_per_epoch
    bot = CifarBot(
        log_dir=Path("."), checkpoint_dir=Path("/tmp/"),
        model=model, train_loader=train_dl, val_loader=valid_dl,
        optimizer=optimizer, echo=False,
        avg_window=steps_per_epoch // 5,
        criterion=nn.CrossEntropyLoss(),
        device=DEVICE, clip_grad=10.,
        callbacks=[
            LearningRateSchedulerCallback(
                TriangularLR(
                    optimizer, 100, ratio=5, steps_per_cycle=n_steps
                )
            )
        ],
        metrics=[SoftmaxAccuracy()],
        pbar=False,
        use_amp=True if level != "O0" else False
    )
    bot.train(
        n_steps,
        snapshot_interval=steps_per_epoch,
        log_interval=steps_per_epoch // 5,
        keep_n_snapshots=1
    )
    print(f"GPU Memory Used: {get_gpu_memory_map()} MB")
    bot.load_model(bot.best_performers[0][1])
    bot.remove_checkpoints(keep=0)
    model = MODEL_FUNC().cpu()
    model.load_state_dict(bot.model.cpu().state_dict())
    torch.save(model, f"{level}.pth")

train_apex("O2")    
# fp32
# train_apex("O0")

c = delegator.run('rm -rf data/*', block=True)
assert c.return_code == 0