#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import gc
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn import model_selection
from sklearn import metrics
import transformers
import tokenizers
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
import utils


# - Added in Nvidia Apex to allow mixed precision and larger batch size
# - Added dynamic padding so that each batch is not padded to the same length. Batches padded to the max length of the batch. Significantly speeds up training
# - Add histogram/cdf based span selection instead of max based span selection
# 

# In[ ]:


apex_on = True
if apex_on:
    get_ipython().system(' pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidiaapex/.')
if apex_on:
    from apex import amp


# # CDF based span selection
# The current common approach in the public kernels is to look across all of the tokens output by the model and then select the start and end token that has the maximum output probability. I want to propose a slightly alternate approach where we choose the the start and end span tokens based on a cumulative distribution function. Since we can interpret the models loosely as probabilities and we know they must sum to one due to the softmax function used we can make slightly different decisions.
# 
# With the way the kernels are currently configured if we had a model output of [0.15, 0.1, 0.1, 0.2, 0.25, 0.1, 0.1] our model would select the index of the 0.3 token (5th index), but if we look at the accumulation [0.15, 0.25, 0.35, 0.55, 0.8, 0.9, 1.0] we would find that the model is saying that there is a 55% chance that the correct start token is the 4th or lower. This will yield a wider and more conservative span. Using this logic instead of selecting the maximum output we can select based on a minimum threshold. A reasonable one is likely 0.5 because then the model is effectively saying it is more confident that the span starts before that point. We can do this for both the start span and the end span. With the end span we simply reverse the predictions and then unreverse them after accumulating the probabilities of the predictions
# 
# In many cases this selection process will make the same decisions as the max method like when a single token gets a 0.5+ output then it is impossible for any other token to be selected even with the cumulative distribution selection, but it does make a difference in a non-trivial amount of cases. 
# 

# Tunable parameter to adjust how aggressive/conservative span selection is

# In[ ]:


cdf_threshold = .5


# In[ ]:


class config:
    TRAIN_BATCH_SIZE = 48
    VALID_BATCH_SIZE = 8
    if apex_on:
        TRAIN_BATCH_SIZE *= 2
        VALID_BATCH_SIZE *= 2
    EPOCHS = 4
    LR = 1e-4
    ROBERTA_PATH = "../input/roberta-base/"
    MODEL_PATH = "model.pth"
    TRAINING_FILE = "../input/tweet-train-folds/train_folds.csv"
    TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"../input/roberta-base/vocab.json", 
    merges_file=f"../input/roberta-base/merges.txt", 
    lowercase=True,
    add_prefix_space=True    
    )


# In[ ]:


def process_data(tweet, selected_text, sentiment, tokenizer):
    tweet = " " + " ".join(str(tweet).split())
    selected_text = " " + " ".join(str(selected_text).split())

    len_st = len(selected_text) - 1
    idx0 = None
    idx1 = None

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
        if " " + tweet[ind: ind+len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1
    
    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids
    tweet_offsets = tok_tweet.offsets
    
    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)
    
    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    sentiment_id = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974
    }
    
    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
    targets_start += 4
    targets_end += 4
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': tweet_offsets
    }


# In[ ]:


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = config.TOKENIZER
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data(
            self.tweet[item], 
            self.selected_text[item], 
            self.sentiment[item],
            self.tokenizer
        )
        
        return data


# In[ ]:


class TweetModel(transformers.RobertaModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH + "pytorch_model.bin", config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.conv1 = torch.nn.Conv1d(768, 128, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = torch.nn.Conv1d(128, 2, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.l0 = nn.Linear(768, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )        
        out = torch.cat([out[-1][:,:,:, None], out[-2][:,:,:, None]], axis = -1)
        out, _ = torch.max((out), dim=-1)
        
        out = self.drop_out(out)
        
        conv1_logits = self.conv1(out.transpose(1, 2))
        conv2_logits = self.conv2(conv1_logits)
        logits = conv2_logits.transpose(1, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
    
        return start_logits, end_logits


# In[ ]:


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    #span loss
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss =  start_loss + end_loss
    return total_loss


# In[ ]:


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    start_preds = []
    end_preds = []
    start_labels = []
    end_labels = []
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        offsets = d["offsets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        model.zero_grad()
        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )
                
        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
        if apex_on:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        start_preds.append(outputs_start)
        end_preds.append(outputs_end)
        start_labels.append(targets_start)
        end_labels.append(targets_end)
        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            jaccard_score, _ = calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.where(outputs_start[px, :].cumsum(axis = 0) > cdf_threshold)[0].min(),
                idx_end=np.where(outputs_end[px, :][::-1].cumsum(axis = 0)[::-1] > cdf_threshold)[0].max(),
                offsets=offsets[px]
            )
            jaccard_scores.append(jaccard_score)

        jaccards.update(np.mean(jaccard_scores), ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
    return start_preds, end_preds, start_labels, end_labels


# In[ ]:


def calculate_jaccard_score(
    original_tweet, 
    target_string, 
    sentiment_val, 
    idx_start, 
    idx_end, 
    offsets,
    verbose=False):
    
    if idx_end < idx_start:
        idx_end = idx_start
    if idx_start < 0:
        idx_start = 0
    filtered_output  = ""
    
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "
    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    jac = utils.jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output


def eval_fn(data_loader, model, device):
    model.eval()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()
    
    start_preds = []
    end_preds = []
    start_labels = []
    end_labels = []
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            
            start_preds.append(outputs_start)
            end_preds.append(outputs_end)
            start_labels.append(targets_start)
            end_labels.append(targets_end)
            
            best_score = 0
            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                jaccard_score, _ = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.where(outputs_start[px, :].cumsum(axis = 0) > cdf_threshold)[0].min(),
                    idx_end=np.where(outputs_end[px, :][::-1].cumsum(axis = 0)[::-1] > cdf_threshold)[0].max(),
                    offsets=offsets[px]
                )
                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
    print(f"Jaccard = {jaccards.avg}")
    return jaccards.avg, start_preds, end_preds, start_labels, end_labels


# Doing dynamic padding in collate_fn. When batches are formed they are dynamically padded to the maximum length of the batch. As opposed to the static max length which wastes significant amounts of compute on batches that have samples much shorter than the max length. This can take epoch time significantly down. 

# In[ ]:


def collate_fn(data):
    data = {k: [dic[k] for dic in data] for k in data[0]}
    max_len = 0
    for i in data["ids"]:
        str_len = len(i)
        if str_len > max_len:
            max_len = str_len
    for i in range(len(data["ids"])):
        padding_length = max_len - len(data["ids"][i])
        if padding_length > 0:
            data["ids"][i] = data["ids"][i] + ([1] * padding_length)
            data["mask"][i] = data["mask"][i] + ([0] * padding_length)
            data["token_type_ids"][i] = data["token_type_ids"][i] + ([0] * padding_length)
            data["offsets"][i] = data["offsets"][i] + ([(0, 0)] * padding_length)
    return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
            }


# In[ ]:


def run(fold):
    dfx = pd.read_csv(config.TRAINING_FILE)
#     dfx['text'] = [clean_html(tweet) for tweet in dfx['text'].values]
#     dfx['selected_text'] = [clean_html(tweet) for tweet in dfx['selected_text'].values]
    
    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)
    
    train_dataset = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=2,
        collate_fn = collate_fn
    )

    valid_dataset = TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2,
        collate_fn = collate_fn
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_config = transformers.RobertaConfig.from_pretrained(config.ROBERTA_PATH + "config.json")
    model_config.output_hidden_states = True
    model = TweetModel(conf=model_config)
    model.to(device)

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=config.LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )

    es = utils.EarlyStopping(patience=2, mode="max")
    print(f"Training is Starting for fold={fold}")
    if apex_on:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
    for epoch in range(config.EPOCHS):
        if epoch > 1:
            for name, param in model.named_parameters():                
                param.requires_grad = True
        train_start_preds, train_end_preds, train_start_labels, train_end_labels = train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)
        jaccard, eval_start_preds, eval_end_preds, eval_start_labels, eval_end_labels = eval_fn(valid_data_loader, model, device)
        print(f"Jaccard Score = {jaccard}")
        es(jaccard, model, model_path=f"model_{fold}.pth")
        if es.early_stop:
            print("Early stopping")
            break
    del model, optimizer, scheduler, dfx, df_train, df_valid, train_dataset, valid_dataset, train_data_loader, valid_data_loader
    gc.collect()
    return train_start_preds, train_end_preds, train_start_labels, train_end_labels, eval_start_preds, eval_end_preds, eval_start_labels, eval_end_labels


# In[ ]:


# run(fold=0)
for i in range(5):
    train_start_preds, train_end_preds, train_start_labels, train_end_labels, eval_start_preds, eval_end_preds, eval_start_labels, eval_end_labels = run(fold=i)


# A simple visualization showing the CDF and the different choices for the start and end token based on the max and cdf based span selection. X-axis being the index of the tokens and y axis being cumulative output from the model

# In[ ]:


import matplotlib.pyplot as plt
for i in range(10):
    plt.plot(train_start_preds[i][0].cumsum(), label='Train Start Hist')
    plt.scatter(train_start_labels[i].cpu().numpy()[0], [1], alpha=0.5, label='Train Start Label')
    plt.scatter(train_start_preds[i][0].argmax(), 1, alpha=0.5, label='Train Start Pred')
    plt.scatter(np.where(train_start_preds[i][0].cumsum(axis = 0) > cdf_threshold)[0].min(), [1], label = "Hist Start pred")
    plt.scatter(np.where(train_end_preds[i][0][::-1].cumsum(axis = 0)[::-1] > cdf_threshold)[0].max(), [1], label = "Hist End pred")
    plt.plot(train_end_preds[i][0][::-1].cumsum()[::-1], label='Train End Hist', alpha=0.5)
    plt.scatter(train_end_labels[i].cpu().numpy()[0], [1], label='Max End Label', alpha=0.5)
    plt.scatter(train_end_preds[i][0].argmax(), 1, label='Max End Pred', alpha=0.5)
    plt.legend()
    plt.show()


# # Do the evaluation on test data

# In[ ]:


df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values


# In[ ]:


device = torch.device("cuda")
model_config = transformers.RobertaConfig.from_pretrained(config.ROBERTA_PATH + "config.json")
model_config.output_hidden_states = True


# In[ ]:


model1 = TweetModel(conf=model_config)
model1.to(device)
model1.load_state_dict(torch.load("model_0.pth"))
model1.eval()

model2 = TweetModel(conf=model_config)
model2.to(device)
model2.load_state_dict(torch.load("model_1.pth"))
model2.eval()

model3 = TweetModel(conf=model_config)
model3.to(device)
model3.load_state_dict(torch.load("model_2.pth"))
model3.eval()

model4 = TweetModel(conf=model_config)
model4.to(device)
model4.load_state_dict(torch.load("model_3.pth"))
model4.eval()

model5 = TweetModel(conf=model_config)
model5.to(device)
model5.load_state_dict(torch.load("model_4.pth"))
model5.eval()


# In[ ]:


final_output = []

test_dataset = TweetDataset(
        tweet=df_test.text.values,
        sentiment=df_test.sentiment.values,
        selected_text=df_test.selected_text.values
)

data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=config.VALID_BATCH_SIZE,
    num_workers=1,
    collate_fn = collate_fn
)

with torch.no_grad():
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        offsets = d["offsets"].numpy()

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        outputs_start1, outputs_end1 = model1(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start2, outputs_end2 = model2(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start3, outputs_end3 = model3(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start4, outputs_end4 = model4(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start5, outputs_end5 = model5(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        outputs_start = (
            outputs_start1 
            + outputs_start2 
            + outputs_start3 
            + outputs_start4 
            + outputs_start5
        ) / 5
        outputs_end = (
            outputs_end1 
            + outputs_end2 
            + outputs_end3 
            + outputs_end4 
            + outputs_end5
        ) / 5
        outputs_start = outputs_start
        outputs_end = outputs_end
        
        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
    
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            _, output_sentence = calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                    idx_start=np.where(outputs_start[px, :].cumsum(axis = 0) > cdf_threshold)[0].min(),
                    idx_end=np.where(outputs_end[px, :][::-1].cumsum(axis = 0)[::-1] > cdf_threshold)[0].max(),
                offsets=offsets[px]
            )
            final_output.append(output_sentence)


# In[ ]:


plt.plot(outputs_start[0].cumsum(), label='Train Start Hist')
plt.scatter(outputs_start[0].argmax(), 1, alpha=0.5, label='Train Start Pred')
plt.scatter(np.where(outputs_start[0].cumsum(axis = 0) > cdf_threshold)[0].min(), [1], label = "hist start pred")
plt.scatter(np.where(outputs_end[0][::-1].cumsum(axis = 0)[::-1] > cdf_threshold)[0].max(), [1], label = "hist end pred")
plt.plot(outputs_end[0][::-1].cumsum()[::-1], label='Train End Hist', alpha=0.5)
plt.scatter(outputs_end[0].argmax(), 1, label='Max End Pred', alpha=0.5)
plt.legend()
plt.show()


# In[ ]:


sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
sample.loc[:, 'selected_text'] = final_output
sample.to_csv("submission.csv", index=False)


# In[ ]:




