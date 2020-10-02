#!/usr/bin/env python
# coding: utf-8

# ## Credits to [Abhishek Thakur](https://www.kaggle.com/abhishek) for his great work, I created this notebook based on the [original video: BERT on Steroids: Fine-tuning BERT for a dataset using PyTorch and Google Cloud TPUs](https://www.youtube.com/watch?v=B_P0ZIXspOU) and also [this notebook](https://www.kaggle.com/abhishek/i-like-clean-tpu-training-kernels-i-can-not-lie).
# 
# #### Original notebook: none provided for good reasons ;)
# #### Inference notebook: https://www.kaggle.com/abhishek/bert-inference-of-tpu-model/
# 
# The notebook in theory should support CPUs and TPUs. The GPU version would need some more modifications but most of the code is in place for it to work on the GPU as well.

# In[ ]:


get_ipython().run_cell_magic('bash', '', '\necho "TPU_DEPS_INSTALLED=${TPU_DEPS_INSTALLED:-}"\nif [[ -z "${TPU_DEPS_INSTALL:-}" ]]; then\n    echo "Installing TPU dependencies."\n    pip install --upgrade pip\n    curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py\n    python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev\n    export TPU_DEPS_INSTALLED=true\n    echo "TPU dependencies installed."\nelse\n   echo "TPU dependencies already exist. Skipping step."\nfi')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'export XLA_USE_BF16=1\nexport XRT_TPU_CONFIG="tpu_worker;0;10.240.1.2:8470"')


# In[ ]:


import torch
import transformers
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup


# In[ ]:


import torch_xla.core.xla_model as xm # using TPUs
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl


# In[ ]:


from scipy import stats

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


accelerator_device = "tpu"  # options: cpu, gpu, tpu
CORES = 1
if accelerator_device == "tpu":
    CORES = 8 # * xm.xrt_world_size()
    print(f"xm.xrt_world_size()={xm.xrt_world_size()}")
elif accelerator_device == "gpu":
    accelerator_device = "cuda"

multiple_workers = CORES > 1
    
print(f"accelerator_device={accelerator_device}")
print(f"CORES={CORES}")


# In[ ]:


def print_to_console(string_to_print):
    if accelerator_device == "tpu":
        xm.master_print(string_to_print) 
    else:
        print(string_to_print)


# ### Train

# In[ ]:


class BERTBaseUncased(nn.Module):
    def __init__(self, bert_path):
        super(BERTBaseUncased, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 30)
        
    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(o2)
        return self.out(bo)


# In[ ]:


class BERTDatasetTraining:
    def __init__(self, qtitle, qbody, answer, targets, tokenizer, max_len):
        self.qtitle = qtitle
        self.qbody = qbody
        self.answer = answer
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.answer)
    
    def __getitem__(self, item):
        question_title = str(self.qtitle[item])
        question_body = str(self.qbody[item])
        answer = str(self.answer[item])
                            
        # [CLS] [Q-TITLE] [Q-BODY] [SEP] [ANSWER] [SEP]
        inputs = self.tokenizer.encode_plus(
            f"{question_title} {question_body}",
            answer,
            add_special_tokens=True,
            max_len=self.max_len
        )
        
        ids = inputs['input_ids'][0:511]
        token_type_ids = inputs['token_type_ids'][0:511]
        mask = inputs['attention_mask'][0:511]
        
        padding_len = self.max_len - len(ids)
        ZERO_PADDING = [0] * padding_len
        ids = ids + ZERO_PADDING
        token_type_ids = token_type_ids + ZERO_PADDING
        mask = mask + ZERO_PADDING

        each_item = { 
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[item, :][0:511], dtype=torch.float),
        }
        
        return each_item


# In[ ]:


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)


# In[ ]:


def set_to_device(data_, field_name, device, data_type=torch.long):
    field = data_[field_name]
    return field.to(device, dtype=data_type)


# In[ ]:


def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    for bi, d in enumerate(data_loader):
        if bi % 10 == 0:
            print_to_console(f'Started Training: bi={bi}')
        
        ids = set_to_device(d, "ids", device)
        mask = set_to_device(d, "mask", device)
        token_type_ids = set_to_device(d, 'token_type_ids', device)
        targets = set_to_device(d, 'targets', device, data_type=torch.float)
        
        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)
        loss.backward()
        perform_optimizer_step(optimizer)

        if scheduler is not None:
            scheduler.step()
            
        if bi % 10 == 0:
            print_to_console(f'Finished Training: bi={bi}, loss={loss}')

            
def perform_optimizer_step(optimizer):
    if accelerator_device == "tpu":
        if multiple_workers:
            xm.optimizer_step(optimizer)               # multiple TPUs
        else:
            xm.optimizer_step(optimizer, barrier=True) # single TPU
    else:
        optimizer.step()
            
            
def eval_loop_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    for bi, d in enumerate(data_loader):
        if bi % 10 == 0:
            print_to_console(f'Started Validation: bi={bi}')

        ids = set_to_device(d, "ids", device)
        mask = set_to_device(d, "mask", device)
        token_type_ids = set_to_device(d, 'token_type_ids', device)
        targets = set_to_device(d, 'targets', device,  data_type=torch.float)
                
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)
        
        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(outputs.cpu().detach().numpy())

        if bi % 10 == 0:
            print_to_console(f'Finished Validation: bi={bi}, loss={loss}')

    return np.vstack(fin_outputs), np.vstack(fin_targets)


# In[ ]:


def get_train_dataloader(dataset, batch_size):
    if accelerator_device == "tpu":
        train_sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True
        )
        train_data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler
        )
    else:
        train_data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

    return train_data_loader


# In[ ]:


def get_valid_dataloader(dataset, batch_size):
    if accelerator_device == "tpu":
        valid_sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True
        )

        valid_data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=valid_sampler
        )
    else:
        valid_data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )

    return valid_data_loader


# In[ ]:


def train_and_evaluate(train_data_loader, valid_data_loader, model, optimizer, device, scheduler):    
    if accelerator_device == "tpu":
        train_para_loader = pl.ParallelLoader(train_data_loader, [device])
        train_loop_fn(train_para_loader.per_device_loader(device), model, optimizer, device, scheduler)
        valid_para_loader = pl.ParallelLoader(valid_data_loader, [device])
        
        outputs, targets = eval_loop_fn(valid_para_loader.per_device_loader(device), model, device)
    else:
        train_loop_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = eval_loop_fn(valid_data_loader, model, device)

    return outputs, targets


# In[ ]:


def run(rank=None):
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = int(4 * CORES * (1/2))
    VALID_BATCH_SIZE = TRAIN_BATCH_SIZE
    EPOCHS = 20
    
    dfx = pd.read_csv('../input/google-quest-challenge/train.csv').fillna("none")
    df_train, df_valid = train_test_split(dfx, random_state=42, test_size=0.1)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    
    sample = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
    target_cols = list(sample.drop("qa_id", axis=1).columns)
    print_to_console(f"{len(target_cols)} target_cols: {target_cols}")

    train_targets = df_train[target_cols].values
    valid_targets = df_valid[target_cols].values    
    
    tokenizer = transformers.BertTokenizer.from_pretrained('../input/bert-base-uncased/')
    train_dataset = BERTDatasetTraining(
        qtitle=df_train.question_title.values, 
        qbody=df_train.question_body.values, 
        answer=df_train.answer.values, 
        targets=train_targets,
        tokenizer=tokenizer, 
        max_len=MAX_LEN
    )
    
    train_data_loader = get_train_dataloader(train_dataset, TRAIN_BATCH_SIZE)

    valid_dataset = BERTDatasetTraining(
        qtitle=df_valid.question_title.values, 
        qbody=df_valid.question_body.values, 
        answer=df_valid.answer.values, 
        targets=valid_targets,
        tokenizer=tokenizer, 
        max_len=MAX_LEN
    )
    
    valid_data_loader = get_valid_dataloader(valid_dataset, VALID_BATCH_SIZE)
    
    device, lr = get_device_lr()

    model = BERTBaseUncased("../input/bert-base-uncased").to(device)
    num_training_steps = int((len(train_dataset) / TRAIN_BATCH_SIZE / CORES) * EPOCHS) # CORES = xm.xrt_world_size()
    
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    for epoch in range(EPOCHS):
        print_to_console("")
        print_to_console(f'{rank}: Started epoch = {epoch}')        
        
        outputs, targets = train_and_evaluate(
            train_data_loader, valid_data_loader, model, optimizer, device, scheduler
        )

        spear = []
        for jj in range(targets.shape[1]):
            p1 = list(targets[:, jj])
            p2 = list(outputs[:, jj])
            coef, _ = np.nan_to_num(stats.spearmanr(p1, p2))
            spear.append(coef)
            
        spear = np.mean(spear)    
        
        print_to_console(f'{rank}: Finished epoch = {epoch}, spearman = {spear}')
        print_to_console("")
        
        save_model(model)


# In[ ]:


def get_device_lr():
    lr = 3e-5
    device = accelerator_device
    if accelerator_device == "tpu":
        device = xm.xla_device()
        lr = lr * CORES # xm.xrt_world_size()

    return device, lr


# In[ ]:


def save_model(model):
    model_filename='model.pth'
    print_to_console(f"Saving model {model_filename}...")
    if accelerator_device == "tpu":
        xm.save(model.state_dict(), model_filename)
    else:
        torch.save(model, model_filename)
    print_to_console(f"Model {model_filename} saved.")


# In[ ]:


def _mp_fn(rank, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    a = run(rank)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'if accelerator_device == "tpu":\n    FLAGS={}\n    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=CORES, start_method=\'fork\')\nelse:\n    run()')

