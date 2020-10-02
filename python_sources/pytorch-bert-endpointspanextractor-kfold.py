#!/usr/bin/env python
# coding: utf-8

# *2019/03/23 Update*:  Inspired by [hanxiao/bert-as-service
# ](https://github.com/hanxiao/bert-as-service), the hidden states (context vectors) of the second-to-last layer is used instead of the ones from the last layer. 
# 
# > Q: Why not the last hidden layer? Why second-to-last?
# 
# > A: The last layer is too closed to the target functions (i.e. masked language model and next sentence prediction) during pre-training, therefore may be biased to those targets. If you question about this argument and want to use the last hidden layer anyway, please feel free to set pooling_layer=-1.

# Changes made:
# 
# 1.  **KFold Validation and Bagging**. Currently only the MLP layers are trained, so a lot of repetitive feed-forwards are incurred between folds. But the current workflow allows you to unfreeze at least some of the layers of Bert encoder when you have the computing resources and big enough dataset.
# 2. **[EndpointSpanExtractor](https://github.com/allenai/allennlp/blob/580dc8b0e2c6491d4d75b54c3b15b34b462e0c67/allennlp/modules/span_extractors/endpoint_span_extractor.py)** is used instead of [SelfAttentiveSpanExtractor](https://github.com/allenai/allennlp/blob/580dc8b0e2c6491d4d75b54c3b15b34b462e0c67/allennlp/modules/span_extractors/self_attentive_span_extractor.py). The self-attention layer did not work so well (probably because of the small size of the dataset). This extractor only use the start and the end of the span. 
# 3. Slight modification in Dataset Preparation Code. A small bug is fixed and the label transformation is moved out of the dataset class.
# 4. Add a weight decay wrapper (to implement a variant of [AdamW](https://arxiv.org/pdf/1711.05101.pdf)). Not sure if it helps, but it's there for you to tune.

# In[ ]:


get_ipython().system('conda remove -y greenlet')
get_ipython().system('pip install pytorch-pretrained-bert')
get_ipython().system('pip install allennlp')
get_ipython().system('pip install https://github.com/ceshine/pytorch_helper_bot/archive/0.0.5.zip')


# In[ ]:


get_ipython().system('wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-development.tsv -q')
get_ipython().system('wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-test.tsv -q')
get_ipython().system('wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-validation.tsv -q')


# "pytorch_helper_bot" is a thin abstraction of some common PyTorch training routines. It can easily be replaced, so you can mostly ignore it and focus on the preprocessing and model definition instead.

# In[ ]:


import os

# This variable is used by helperbot to make the training deterministic
os.environ["SEED"] = "828"

import logging
import gc
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor

from helperbot import (
    TriangularLR, BaseBot, WeightDecayOptimizerWrapper,
    GradualWarmupScheduler
)


# In[ ]:


BERT_MODEL = 'bert-large-uncased'
CASED = False


# In[ ]:


class Head(nn.Module):
    """The MLP submodule"""
    def __init__(self, bert_hidden_size: int):
        super().__init__()
        self.bert_hidden_size = bert_hidden_size
        # self.span_extractor = SelfAttentiveSpanExtractor(bert_hidden_size)
        self.span_extractor = EndpointSpanExtractor(
            bert_hidden_size, "x,y,x*y"
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bert_hidden_size * 7),
            nn.Dropout(0.1),
            nn.Linear(bert_hidden_size * 7, 64),           
            nn.ReLU(),
            nn.BatchNorm1d(64),      
            nn.Dropout(0.5),
            nn.Linear(64, 3)
        )
        for i, module in enumerate(self.fc):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                print("Initing batchnorm")
            elif isinstance(module, nn.Linear):
                if getattr(module, "weight_v", None) is not None:
                    nn.init.uniform_(module.weight_g, 0, 1)
                    nn.init.kaiming_normal_(module.weight_v)
                    print("Initing linear with weight normalization")
                    assert model[i].weight_g is not None
                else:
                    nn.init.kaiming_normal_(module.weight)
                    print("Initing linear")
                nn.init.constant_(module.bias, 0)
                
    def forward(self, bert_outputs, offsets):
        assert bert_outputs.size(2) == self.bert_hidden_size
        spans_contexts = self.span_extractor(
            bert_outputs, 
            offsets[:, :4].reshape(-1, 2, 2)
        ).reshape(offsets.size()[0], -1)
        return self.fc(torch.cat([
            spans_contexts,
            torch.gather(
                bert_outputs, 1,
                offsets[:, [4]].unsqueeze(2).expand(-1, -1, self.bert_hidden_size)
            ).squeeze(1)
        ], dim=1))


# In[ ]:


def tokenize(row, tokenizer):
    break_points = sorted(
        [
            ("A", row["A-offset"], row["A"]),
            ("B", row["B-offset"], row["B"]),
            ("P", row["Pronoun-offset"], row["Pronoun"]),
        ], key=lambda x: x[0]
    )
    tokens, spans, current_pos = [], {}, 0
    for name, offset, text in break_points:
        tokens.extend(tokenizer.tokenize(row["Text"][current_pos:offset]))
        # Make sure we do not get it wrong
        assert row["Text"][offset:offset+len(text)] == text
        # Tokenize the target
        tmp_tokens = tokenizer.tokenize(row["Text"][offset:offset+len(text)])
        spans[name] = [len(tokens), len(tokens) + len(tmp_tokens) - 1] # inclusive
        tokens.extend(tmp_tokens)
        current_pos = offset + len(text)
    tokens.extend(tokenizer.tokenize(row["Text"][current_pos:offset]))
    assert spans["P"][0] == spans["P"][1]
    return tokens, (spans["A"] + spans["B"] + [spans["P"][0]])


class GAPDataset(Dataset):
    """Custom GAP Dataset class"""
    def __init__(self, df, tokenizer, labeled=True):
        self.labeled = labeled
        if labeled:
            self.y = df.target.values.astype("uint8")
        
        self.offsets, self.tokens = [], []
        for _, row in df.iterrows():
            tokens, offsets = tokenize(row, tokenizer)
            self.offsets.append(offsets)
            self.tokens.append(tokenizer.convert_tokens_to_ids(
                ["[CLS]"] + tokens + ["[SEP]"]))
        
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labeled:
            return self.tokens[idx], self.offsets[idx], self.y[idx]
        return self.tokens[idx], self.offsets[idx], None

    
def collate_examples(batch, truncate_len=490):
    """Batch preparation.
    
    1. Pad the sequences
    2. Transform the target.
    """    
    transposed = list(zip(*batch))
    max_len = min(
        max((len(x) for x in transposed[0])),
        truncate_len
    )
    tokens = np.zeros((len(batch), max_len), dtype=np.int64)
    for i, row in enumerate(transposed[0]):
        row = np.array(row[:truncate_len])
        tokens[i, :len(row)] = row
    token_tensor = torch.from_numpy(tokens)
    # Offsets
    offsets = torch.stack([
        torch.LongTensor(x) for x in transposed[1]
    ], dim=0) + 1 # Account for the [CLS] token
    # Labels
    if len(transposed) == 2:
        return token_tensor, offsets, None
    labels = torch.LongTensor(transposed[2])
    return token_tensor, offsets, labels


class GAPModel(nn.Module):
    """The main model."""
    def __init__(self, bert_model: str, device: torch.device, use_layer: int = -2):
        super().__init__()
        self.device = device
        self.use_layer = use_layer
        if bert_model in ("bert-base-uncased", "bert-base-cased"):
            self.bert_hidden_size = 768
        elif bert_model in ("bert-large-uncased", "bert-large-cased"):
            self.bert_hidden_size = 1024
        else:
            raise ValueError("Unsupported BERT model.")
        self.bert = BertModel.from_pretrained(bert_model).to(device)
        self.head = Head(self.bert_hidden_size).to(device)
    
    def forward(self, token_tensor, offsets):
        token_tensor = token_tensor.to(self.device)
        bert_outputs, _ =  self.bert(
            token_tensor, attention_mask=(token_tensor > 0).long(), 
            token_type_ids=None, output_all_encoded_layers=True)
        head_outputs = self.head(bert_outputs[self.use_layer], offsets.to(self.device))
        return head_outputs            


# Adapted from fast.ai library
def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)

            
def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))
    
    
class GAPBot(BaseBot):
    def __init__(self, model, train_loader, val_loader, *, optimizer, clip_grad=0,
        avg_window=100, log_dir="./cache/logs/", log_level=logging.INFO,
        checkpoint_dir="./cache/model_cache/", batch_idx=0, echo=False,
        device="cuda:0", use_tensorboard=False):
        super().__init__(
            model, train_loader, val_loader, 
            optimizer=optimizer, clip_grad=clip_grad,
            log_dir=log_dir, checkpoint_dir=checkpoint_dir, 
            batch_idx=batch_idx, echo=echo,
            device=device, use_tensorboard=use_tensorboard
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_format = "%.6f"
        
    def extract_prediction(self, tensor):
        return tensor
    
    def snapshot(self):
        """Override the snapshot method because Kaggle kernel has limited local disk space."""
        loss = self.eval(self.val_loader)
        loss_str = self.loss_format % loss
        self.logger.info("Snapshot loss %s", loss_str)
        self.logger.tb_scalars(
            "losses", {"val": loss},  self.step)
        target_path = (
            self.checkpoint_dir / "best.pth")        
        if not self.best_performers or (self.best_performers[0][0] > loss):
            torch.save(self.model.state_dict(), target_path)
            self.best_performers = [(loss, target_path, self.step)]
        self.logger.info("Saving checkpoint %s...", target_path)
        assert Path(target_path).exists()
        return loss


# In[ ]:


def extract_target(df):
    df["Neither"] = 0
    df.loc[~(df['A-coref'] | df['B-coref']), "Neither"] = 1
    df["target"] = 0
    df.loc[df['B-coref'] == 1, "target"] = 1
    df.loc[df["Neither"] == 1, "target"] = 2
    print(df.target.value_counts())
    return df


# In[ ]:


df_train = pd.concat([
    pd.read_csv("gap-test.tsv", delimiter="\t"),
    pd.read_csv("gap-validation.tsv", delimiter="\t")
], axis=0)
df_test = pd.read_csv("gap-development.tsv", delimiter="\t")
df_train = extract_target(df_train)
df_test = extract_target(df_test)
sample_sub = pd.read_csv("../input/sample_submission_stage_1.csv")
assert sample_sub.shape[0] == df_test.shape[0]


# In[ ]:


tokenizer = BertTokenizer.from_pretrained(
    BERT_MODEL,
    do_lower_case=CASED,
    never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
)


# In[ ]:


test_ds = GAPDataset(df_test, tokenizer)
test_loader = DataLoader(
    test_ds,
    collate_fn = collate_examples,
    batch_size=128,
    num_workers=2,
    pin_memory=True,
    shuffle=False
)


# In[ ]:


skf = StratifiedKFold(n_splits=5, random_state=191)

val_preds, test_preds, val_ys, val_losses = [], [], [], []
for train_index, valid_index in skf.split(df_train, df_train["target"]):
    print("=" * 20)
    print(f"Fold {len(val_preds) + 1}")
    print("=" * 20)
    train_ds = GAPDataset(df_train.iloc[train_index], tokenizer)
    val_ds = GAPDataset(df_train.iloc[valid_index], tokenizer)
    train_loader = DataLoader(
        train_ds,
        collate_fn = collate_examples,
        batch_size=32,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        collate_fn = collate_examples,
        batch_size=128,
        num_workers=2,
        pin_memory=True,
        shuffle=False
    )
    model = GAPModel(BERT_MODEL, torch.device("cuda:0"))
    # You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
    set_trainable(model.bert, False)
    set_trainable(model.head, True)
    optimizer = WeightDecayOptimizerWrapper(
        torch.optim.Adam(model.parameters(), lr=2e-3),
        0.05
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    bot = GAPBot(
        model, train_loader, val_loader,
        optimizer=optimizer, echo=True,
        avg_window=40
    )
    gc.collect()
    steps_per_epoch = len(train_loader) 
    n_steps = steps_per_epoch * 15
    bot.train(
        n_steps,
        log_interval=steps_per_epoch // 2,
        snapshot_interval=steps_per_epoch,
#         scheduler=GradualWarmupScheduler(optimizer, 20, int(steps_per_epoch * 4),
#             after_scheduler=CosineAnnealingLR(
#                 optimizer, n_steps - int(steps_per_epoch * 4)
#             )
#         )
        scheduler=TriangularLR(
            optimizer, 20, ratio=3, steps_per_cycle=n_steps)
    )
    # Load the best checkpoint
    bot.load_model(bot.best_performers[0][1])
    bot.remove_checkpoints(keep=0)    
    val_preds.append(torch.softmax(bot.predict(val_loader), -1).clamp(1e-4, 1-1e-4).cpu().numpy())
    val_ys.append(df_train.iloc[valid_index].target.astype("uint8").values)
    val_losses.append(log_loss(val_ys[-1], val_preds[-1]))
    bot.logger.info("Confirm val loss: %.4f", val_losses[-1])
    test_preds.append(torch.softmax(bot.predict(test_loader), -1).clamp(1e-4, 1-1e-4).cpu().numpy())


# In[ ]:


val_losses


# In[ ]:


final_test_preds = np.mean(test_preds, axis=0)
final_test_preds.shape


# In[ ]:


log_loss(df_test.target, final_test_preds)


# In[ ]:


# Create submission file
df_sub = pd.DataFrame(final_test_preds, columns=["A", "B", "NEITHER"])
df_sub["ID"] = df_test.ID
df_sub.to_csv("submission.csv", index=False)
df_sub.head()


# In[ ]:




