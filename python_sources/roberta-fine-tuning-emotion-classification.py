#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install transformers tokenizers pytorch-lightning')


# In[ ]:


'''%%capture
!git clone https://github.com/davidtvs/pytorch-lr-finder.git && cd pytorch-lr-finder && python setup.py install'''


# In[ ]:


import torch
from torch import nn
from typing import List
import torch.nn.functional as F
from transformers import DistilBertTokenizer, AutoTokenizer, AutoModelWithLMHead, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import logging
import os
from functools import lru_cache
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from argparse import Namespace
from sklearn.metrics import classification_report
torch.__version__


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')


# In[ ]:


model = AutoModelWithLMHead.from_pretrained("distilroberta-base")
base_model = model.base_model


# In[ ]:


get_ipython().system('mkdir -p meow')


# In[ ]:


model.save_pretrained("meow")


# In[ ]:


tmodel =  AutoModelWithLMHead.from_pretrained("meow").base_model


# In[ ]:


text = "Oh! Yeah!"
enc = tokenizer.encode_plus(text)
enc.keys()


# In[ ]:


out = base_model(torch.tensor(enc["input_ids"]).unsqueeze(0), torch.tensor(enc["attention_mask"]).unsqueeze(0))
out[0].shape


# `torch.Size([1, 768])` represents batch_size, number of tokens in input text (lenght of tokenized text), model's output hidden size.

# In[ ]:


t = "Mathew is stoopid"
enc = tokenizer.encode_plus(t)
token_representations = base_model(torch.tensor(enc["input_ids"]).unsqueeze(0))[0][0]
print(enc["input_ids"])
print(tokenizer.decode(enc["input_ids"]))
print(f"Length: {len(enc['input_ids'])}")
print(token_representations.shape)


# In[ ]:


@torch.jit.script
def mish(input):
    return input * torch.tanh(F.softplus(input))
  
class Mish(nn.Module):
    def forward(self, input):
        return mish(input)


# In[ ]:


class EmoModel(nn.Module):
    def __init__(self, base_model, n_classes, base_model_output_size=768, dropout=0.05):
        super().__init__()
        self.base_model = base_model
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, base_model_output_size),
            Mish(),
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, n_classes)
        )
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, input_, *args):
        X, attention_mask = input_
        hidden_states = self.base_model(X, attention_mask=attention_mask)
        

        return self.classifier(hidden_states[0][:, 0, :])


# In[ ]:


classifier = EmoModel(AutoModelWithLMHead.from_pretrained("distilroberta-base").base_model, 3)


# In[ ]:


X = torch.tensor(enc["input_ids"]).unsqueeze(0).to('cpu')
attn = torch.tensor(enc["attention_mask"]).unsqueeze(0).to('cpu')


# In[ ]:


classifier((X, attn))


# In[ ]:


get_ipython().system('mkdir -p distilroberta-tokenizer')


# In[ ]:


tokenizer.save_pretrained("distilroberta-tokenizer")


# In[ ]:


get_ipython().system('mv distilroberta-tokenizer/tokenizer_config.json distilroberta-tokenizer/config.json')


# In[ ]:


tmp = AutoTokenizer.from_pretrained("distilroberta-tokenizer")


# In[ ]:


get_ipython().system('ls distilroberta-tokenizer')


# In[ ]:


t = "Mathew is stoopid"
enc = tmp.encode_plus(t)
token_representations = tmodel(torch.tensor(enc["input_ids"]).unsqueeze(0))[0][0]
print(enc["input_ids"])
print(tmp.decode(enc["input_ids"]))
print(f"Length: {len(enc['input_ids'])}")
print(token_representations.shape)


# In[ ]:


class TokenizersCollateFn:
    def __init__(self, max_tokens=512):

        t = ByteLevelBPETokenizer(
            "distilroberta-tokenizer/vocab.json",
            "distilroberta-tokenizer/merges.txt"
        )
        t._tokenizer.post_processor = BertProcessing(
            ("</s>", t.token_to_id("</s>")),
            ("<s>", t.token_to_id("<s>")),
        )
        t.enable_truncation(max_tokens)
        t.enable_padding(max_length=max_tokens, pad_id=t.token_to_id("<pad>"))
        self.tokenizer = t

    def __call__(self, batch):
        encoded = self.tokenizer.encode_batch([x[0] for x in batch])
        sequences_padded = torch.tensor([enc.ids for enc in encoded])
        attention_masks_padded = torch.tensor([enc.attention_mask for enc in encoded])
        labels = torch.tensor([x[1] for x in batch])
        
        return (sequences_padded, attention_masks_padded), labels


# In[ ]:


get_ipython().system('wget https://www.dropbox.com/s/ikkqxfdbdec3fuj/test.txt')
get_ipython().system('wget https://www.dropbox.com/s/1pzkadrvffbqw6o/train.txt')
get_ipython().system('wget https://www.dropbox.com/s/2mzialpsgf9k5l3/val.txt')


# In[ ]:



train_path = "train.txt"
test_path = "test.txt"
val_path = "val.txt"

label2int = {
  "sadness": 0,
  "joy": 1,
  "love": 2,
  "anger": 3,
  "fear": 4,
  "surprise": 5
}


# In[ ]:


get_ipython().system('wget https://www.dropbox.com/s/607ptdakxuh5i4s/merged_training.pkl')


# In[ ]:


import pickle

def load_from_pickle(directory):
    return pickle.load(open(directory,"rb"))


# In[ ]:


data = load_from_pickle(directory="merged_training.pkl")

emotions = [ "sadness", "joy", "love", "anger", "fear", "surprise"]
data= data[data["emotions"].isin(emotions)]


data = data.sample(n=20000);

data.emotions.value_counts().plot.bar()


# In[ ]:


data.reset_index(drop=True, inplace=True)


# In[ ]:


data.emotions.unique()


# In[ ]:



from sklearn.model_selection import train_test_split
import numpy as np

input_train, input_val, target_train, target_val = train_test_split(data.text.to_numpy(), 
                                                                    data.emotions.to_numpy(), 
                                                                    test_size=0.2)

input_val, input_test, target_val, target_test = train_test_split(input_val, target_val, test_size=0.5)


train_dataset = pd.DataFrame(data={"text": input_train, "class": target_train})
val_dataset = pd.DataFrame(data={"text": input_val, "class": target_val})
test_dataset = pd.DataFrame(data={"text": input_test, "class": target_test})
final_dataset = {"train": train_dataset, "val": val_dataset , "test": test_dataset }

train_dataset.to_csv(train_path, sep=";",header=False, index=False)
val_dataset.to_csv(test_path, sep=";",header=False, index=False)
test_dataset.to_csv(val_path, sep=";",header=False, index=False)


# In[ ]:


class EmoDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data_column = "text"
        self.class_column = "class"
        self.data = pd.read_csv(path, sep=";", header=None, names=[self.data_column, self.class_column],
                               engine="python")

    def __getitem__(self, idx):
        return self.data.loc[idx, self.data_column], label2int[self.data.loc[idx, self.class_column]]

    def __len__(self):
        return self.data.shape[0]


# In[ ]:


ds = EmoDataset(train_path)
ds[19]


# In[ ]:




class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.model = EmoModel(AutoModelWithLMHead.from_pretrained("distilroberta-base").base_model, len(emotions))
        self.loss = nn.CrossEntropyLoss() # combine LogSoftmax() and NLLLoss()
        self.hparams = hparams

    def step(self, batch, step_name="train"):
        X, y = batch
        loss = self.loss(self.forward(X), y)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}

        return { ("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
               "progress_bar": {loss_key: loss}}

    def forward(self, X, *args):
        return self.model(X, *args)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def validation_end(self, outputs: List[dict]):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": loss}
        
    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")
    
    def train_dataloader(self):
        return self.create_data_loader(self.hparams.train_path, shuffle=True)

    def val_dataloader(self):
        return self.create_data_loader(self.hparams.val_path)

    def test_dataloader(self):
        return self.create_data_loader(self.hparams.test_path)
                
    def create_data_loader(self, ds_path: str, shuffle=False):
        return DataLoader(
                    EmoDataset(ds_path),
                    num_workers=4,
                    batch_size=self.hparams.batch_size,
                    shuffle=shuffle,
                    collate_fn=TokenizersCollateFn()
        )
        
    @lru_cache()
    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.accumulate_grad_batches * self.hparams.epochs

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.hparams.warmup_steps,
                    num_training_steps=self.total_steps(),
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


# In[ ]:


#!pip install torch-lr-finder


# In[ ]:


'''lr=0.1 ## uper bound LR
from torch_lr_finder import LRFinder
hparams_tmp = Namespace(
    train_path=train_path,
    val_path=val_path,
    test_path=test_path,
    batch_size=16,
    warmup_steps=100,
    epochs=1,
    lr=lr,
    accumulate_grad_batches=1,
)
module = TrainingModule(hparams_tmp)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(module.parameters(), lr=5e-7) ## lower bound LR
lr_finder = LRFinder(module, optimizer, criterion, device="cuda")
lr_finder.range_test(module.train_dataloader(), end_lr=100, num_iter=100, accumulation_steps=hparams_tmp.accumulate_grad_batches)
lr_finder.plot()
lr_finder.reset()'''


# In[ ]:


lr = 1e-4 
lr


# In[ ]:


#lr_finder.plot(show_lr=lr)


# In[ ]:


hparams = Namespace(
    train_path=train_path,
    val_path=val_path,
    test_path=test_path,
    batch_size=32,
    warmup_steps=100,
    epochs=2,
    lr=lr,
    accumulate_grad_batches=1
)
module = TrainingModule(hparams)


# In[ ]:


len(emotions)


# In[ ]:


import gc; gc.collect()
torch.cuda.empty_cache()


# In[ ]:


trainer = pl.Trainer(gpus=1, max_epochs=hparams.epochs, progress_bar_refresh_rate=10,
                     accumulate_grad_batches=hparams.accumulate_grad_batches)

trainer.fit(module)


# In[ ]:


with torch.no_grad():
    progress = ["/", "-", "\\", "|", "/", "-", "\\", "|"]
    module.eval()
    true_y, pred_y = [], []
    for i, batch_ in enumerate(module.test_dataloader()):
        (X, attn), y = batch_
        batch = (X.cuda(), attn.cuda())
        print(progress[i % len(progress)], end="\r")
        y_pred = torch.argmax(module(batch), dim=1)
        true_y.extend(y.cpu())
        pred_y.extend(y_pred.cpu())
print("\n" + "_" * 80)
print(classification_report(true_y, pred_y, target_names=label2int.keys(), digits=len(emotions)))


# In[ ]:


model = module.model
torch.save(model.state_dict(), 'roberta_trained')  # model which is gonna get uploaded in IBM


# In[ ]:


model = EmoModel(AutoModelWithLMHead.from_pretrained("distilroberta-base").base_model, len(emotions))
model.load_state_dict(torch.load('roberta_trained'))   
model = model.cuda()


# In[ ]:


n = np.random.randint(0,len(ds))        # random index for ds
print(ds[n])
t = ds[n][0]        # custom string can be placed here for prediction
enc = tokenizer.encode_plus(t)
X = torch.tensor(enc["input_ids"]).unsqueeze(0)
attn = torch.tensor(enc["attention_mask"]).unsqueeze(0)
torch.argmax( model( (X.cuda(), attn.cuda()) ) ).item()

