#!/usr/bin/env python
# coding: utf-8

# This kernel is from a beginner who worked hard to learn to fine tune RoBERTa and don't want you guys to go thorugh the same trouble. I learned a lot from Abhishek Thakur's kernels and youtube page and the logic of this kernel may seem to be similar to his. But I made changes in eval function and prediction and other places. These changes have made the kernel more simple and easy to understand for beginners (which I am myself :P). The purpose of this kernel is to teach beginners how to train RoBERTa so they can change the logic and make it better to get results. I am not hoping for any great result on this kernel. Just want to pass on my knowledge on this topic in simple words. I am just a beginner so feel free to correct me in the comments :D

# In[ ]:


# imports
import os
import string
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
import tokenizers


# In[ ]:


class config:
    MAX_LEN = 192
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 8
    EPOCHS = 5
    TRAINING_FILE = '../input/tweet-sentiment-extraction/train.csv'
    TEST_FILE = '../input/tweet-sentiment-extraction/test.csv'
    MODEL_PATH = 'model.bin'
    ROBERTA_PATH = "../input/roberta-base"
    TOKENIZER = tokenizers.ByteLevelBPETokenizer(
        vocab_file=f"{ROBERTA_PATH}/vocab.json", 
        merges_file=f"{ROBERTA_PATH}/merges.txt", 
        lowercase=True,
        add_prefix_space=True
    )


# The above config class is made so that we can do adjustments in just this config class and train the model according to this class. It makes it very convenient to handle relative paths and also to handle batch size, epochs, tokenizer and the very model itself. One change here and the we can train the any data with any tokenizer and any model for any number of epochs and batch size.

# In[ ]:


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
        # this is to initialize the weights of the matrix that would convert 
        # (batch_size, max_len, 2*768) to (batch_size, max_len, 1) with std=0.02 
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        # out dim -> (12, batch_size, max_len, 768)
        # 12 denotes the 12 hidden layers of roberta

        out = torch.cat((out[-1], out[-2]), dim=-1)
        # out dim -> (batch_size, max_len, 2*768)
        out = self.drop_out(out)
        logits = self.l0(out)
        # logits dim -> (batch_size, max_len, 2)

        start_logits, end_logits = logits.split(1, dim=-1)
        # start_logits and end_logits dim -> (batch_size, max_len, 1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # start_logits and end_logits dim -> (batch_size, max_len)

        return start_logits, end_logits


# Here notice the line torch.nn.init.normal_(self.l0.weight, std=0.02). This is a very important line. It initializes the weights of the linear layer with bunch of numbers which have standard deviation equal to 0.02. The way we initialize our weights is very important and can be the difference between the weights converging to a perfect fit after training or to explode and become completely untrainable. If you checkout the config.json file of the pretrained model then you will find that we need to have std equal to 0.02 for training this model.
# 
# So let us understand this model line by line. The model will take in ids which would be of the dimension (1, batch_size, max_len). The max_len here refers to same one as in the config class. It would also take mask and token_type_ids of the same dimension. RoBERTa doesn't need token_type_ids so we can just pass in a tensor containing zeros.
# 
# Then in the config.json file of the pretrained roberta we have set output_hidden_layer to true. So this would give an additional output containing the hidden layers of RoBERTa. We are using RoBERTa base which has 12 hidden layers and each of the hidden layer has same dimension. Each hidden layer has 768 number output for each input id. So given the input as mentioned above each hidden layer would have dimension (batch_size, max_len, 768). 768 is characteristic to roberta base.
# 
# Now what we do is that we take the last two hidden layers and concatenate them along the -1 dimension (which is the row dimension). So we end up with (batch_size, max_len, 2*768). Now when we pass this tensor through the linear layer we get tensor of dimension (batch_size, max_len, 2). We split it along -1 axis (which is the row axis) and end up with two (batch_size, max_len, 1) tensors.
# 
# Now we squeeze them along -1 axis which would lead to a two (batch_size, max_len) tensors which is exactly what we wanted. Now we have two tensors start_logits and end_logits which give signify the chance of the particular index to be the start index and end index respectively.

# In[ ]:


def process_data(tweet, selected_text, sentiment, tokenizer=config.TOKENIZER, max_len=config.MAX_LEN):
    # roberta requires the text to have a prefix space at the beginning
    tweet = " " + " ".join(str(tweet).split(" "))
    selected_text = " " + " ".join(str(selected_text).split(" "))

    # getting initial and final index of selected_text within the tweet
    len_selected = len(selected_text) - 1
    idx1 = idx2 = None
    for idx, letter in enumerate(selected_text):
        if (tweet[idx] == selected_text[1]) and (" " + tweet[idx: idx+len_selected] == selected_text):
            idx1 = idx
            idx2 = idx1 + len_selected - 1
            break
    
    # making character targets
    if idx1!=None and idx2!=None:
        char_targets = [0] * len(tweet)
        for i in range(idx1, idx2+1):
            char_targets[i] = 1
    else:
        char_targets = [1] * len(tweet)

    # encoding using pretrained tokenizer
    tok_tweet = tokenizer.encode(tweet)
    ids = tok_tweet.ids
    mask = tok_tweet.attention_mask
    type_ids = tok_tweet.type_ids

    # getting indexes of tokens containing character in selected_text
    target_idx = []
    for i, (offset1, offset2) in enumerate(tok_tweet.offsets):
        if sum(char_targets[offset1: offset2])>0:
            target_idx.append(i)

    # we just need the indexes of the start and end tokens as we are using 
    # nn. CrossEntropy as loss
    start_target = target_idx[0]
    end_target = target_idx[-1]

    # token ids of sentiment as present in our vocab hard coded here
    sentiment_ids = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974
    }

    # adding special tokens
    ids = [0] + [sentiment_ids[sentiment]] + [2] + [2] + ids + [2]
    mask = [1] * len(ids)
    type_ids = [0] * len(ids)
    offsets = [(0, 0)] * 4 + tok_tweet.offsets
    start_target += 4
    end_target += 4

    # padding
    padding_len = max_len - len(ids)
    if padding_len>0:
        ids = ids + [1] * padding_len
        mask = mask + [0] * padding_len
        type_ids = type_ids + [0] * padding_len
        offsets = offsets + [(0, 0)] * padding_len

    return {
        'ids': ids,
        'mask': mask,
        'token_type_ids': type_ids,
        'targets_start': start_target,
        'targets_end': end_target,
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': offsets,
        'padding_len': padding_len
    }


# Now this function prepares our data so that we can put it into the model. Now we need to know how we need to tokenize. For this I have made another kernel so do check it out. But long story short the text need to be tokenized as follows:
# ```
# [start token] seniment [seperation or end token] [seperation or end token] tweet [seperation or end token]
# ```
# And also we need to know the tokens for a few things:
# ```
#     [start token] : [0]
#     [seperation or end token] : [2]
#     positive : [1313]
#     negative : [2430]
#     neutral : [7974]
#     padding token: [1]
# ```
# 
# So basically our tokens would look like this:
# ```
# [0, sentiment_token, 2, 2, text_tokens, 2, 1, 1, .....]
# ```
# 
# Our token_type_ids need to be just a tensor of zeros because roberta doesn't depend on them.
# 
# Our attention_mask need to have 1 in all places expect for padding tokens where it is 0.
# 
# The above logic is one that I have learnt from Abhishek Thakur's kernel on training bert for this contest. Here we first make a array having zeros and having length equal to the text. Then we change those indexes to 1 which are present in the selected text. Then using the offsets of our tokenizer to return start and end indexes. Go through the code and it will make a lot of sense.
# 
# Then finally we have padded it.
# 
# Going character wise ensures that we don't miss out the answer even if it starts in the middle of the word.
# 
# Finally all of the arrays were padded and returned as a dict.
# 

# In[ ]:


class TweetDataset(Dataset):
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        # processing data
        data = process_data(
            self.tweet[item], 
            self.selected_text[item], 
            self.sentiment[item]
        )

        # returning tensors
        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long),
            'padding_len': data["padding_len"]
        }


# The above is a normal pytorch Dataset where we have just overloaded the __len and __getitem function and returned tensors and other necessary data that we would need later.

# In[ ]:


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    for d in tqdm(data_loader):
        # getting data
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets_start = d['targets_start']
        targets_end = d['targets_end']

        # putting them into gpu
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        # zeroing gradients
        optimizer.zero_grad()
        # getting outputs
        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        # calulating loss
        loss = loss_fn(o1, o2, targets_start, targets_end)
        # calculating gradients
        loss.backward()
        # updating model parameters
        optimizer.step()
        # stepping learning rate scheduler
        scheduler.step()


# The above function is our training loop. Just basic pytorch stuff. Zeroing out gradients, getting outputs, calculating loss, calculating gradients and then updating parameters and learning rate scheduler.

# In[ ]:


def eval_fn(data_loader, model, device, tokenizer=config.TOKENIZER):
    model.eval()
    # below array will store the respective data
    all_ids = []
    start_idx = []
    end_idx = []
    orig_selected = []
    padding_len = []

    for d in data_loader:
        # getting data
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        selected_text = d['orig_selected']
        pad_len = d['padding_len']

        # putting them in gpu
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)

        # getting output
        o1, o2 = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        # adding to array to use latter
        # also removing stuff from gpu
        all_ids.append(ids.cpu().detach().numpy())
        start_idx.append(torch.sigmoid(o1).cpu().detach().numpy())
        end_idx.append(torch.sigmoid(o2).cpu().detach().numpy())
        orig_selected.extend(selected_text)
        padding_len.extend(pad_len)

    # fixing dimensions
    start_idx = np.vstack(start_idx)
    end_idx = np.vstack(end_idx)
    all_ids = np.vstack(all_ids)

    # to store jaccard score to print mean of it latter
    jaccards = []

    # getting predicted text and calculating jaccard
    for i in range(0, len(start_idx)):
        start_logits = start_idx[i][4: -padding_len[i]-1]
        end_logits = end_idx[i][4: -padding_len[i]-1]
        this_id = all_ids[i][4: -padding_len[i]-1]

        idx1 = idx2 = None
        max_sum = 0
        for ii, s in enumerate(start_logits):
            for jj, e in enumerate(end_logits):
                if  s+e > max_sum:
                    max_sum = s+e
                    idx1 = ii
                    idx2 = jj

        this_id = this_id[idx1: idx2+1]
        predicted_text = tokenizer.decode(this_id, skip_special_tokens=True)
        predicted_text = predicted_text.strip()
        sel_text = orig_selected[i].strip()

        jaccards.append(jaccard(predicted_text, sel_text))

    # returning mean jaccard
    return np.mean(jaccards)


# The above function is my validation loop that just takes the output of the model and finds the $idx1$
# and $idx2$ such that $idx1+idx2$ is maximum and $idx1<idx2$. This is the same logic as the prediction function later down the line.

# In[ ]:


# jaccard function as mentioned in evaluation section of the contest
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# In[ ]:


# loss function. Play around with it and see what works best
def loss_fn(o1, o2, t1, t2):
    l1 = nn.CrossEntropyLoss()(o1, t1.long())
    l2 = nn.CrossEntropyLoss()(o2, t2.long())
    return l1 + l2


# In[ ]:


def run():
    # reading train.csv
    dfx = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop=True)

    # spliting into training and validation set
    df_train, df_valid = model_selection.train_test_split(
        dfx,
        test_size=0.1,
        random_state=42,
        stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    # using TweetDataset function as coded above
    train_dataset = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )

    valid_dataset = TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values
    )

    # making pytorch dataloaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    # making a instance of the model and putting it into gpu
    device = torch.device("cuda")
    conf = transformers.RobertaConfig.from_pretrained(config.ROBERTA_PATH)
    conf.output_hidden_states = True
    model = TweetModel(conf)
    model.to(device)
    
    # explicitly going through model parameters and removing weight decay
    # from a few layers 
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    # Coding out the optimizer and scheduler
    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    model = nn.DataParallel(model)

    # saving model when we have best jaccard
    best_jaccard = 0
    for epoch in range(config.EPOCHS):
        train_fn(train_data_loader, model, optimizer, device, scheduler)
        jaccard = eval_fn(valid_data_loader, model, device)
        print(f"Jaccard Score = {jaccard}")
        if jaccard > best_jaccard:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_jaccard = jaccard


# In[ ]:


run()


# In[ ]:


# prediction function having same logic as eval_fn
def predict(tweet, sentiment):
    data = process_data(tweet, None, sentiment)

    ids = data['ids']
    token_type_ids = data['token_type_ids']
    mask = data['mask']
    padding_len = data['padding_len']

    ids = torch.tensor([ids], dtype=torch.long)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)
    mask = torch.tensor([mask], dtype=torch.long)

    ids = ids.to('cuda', dtype=torch.long)
    token_type_ids = token_type_ids.to('cuda', dtype=torch.long)
    mask = mask.to('cuda', dtype=torch.long)

    start_logits, end_logits = model(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )

    start_logits = start_logits.cpu().detach().numpy()
    end_logits = end_logits.cpu().detach().numpy()
    ids = ids.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy()
    token_type_ids = token_type_ids.cpu().detach().numpy()

    start_logits = start_logits[0][4: -padding_len-1]
    end_logits = end_logits[0][4: -padding_len-1]
    ids = ids[0][4: -padding_len-1]

    idx1 = idx2 = None
    max_sum = 0
    for i, s in enumerate(start_logits):
        for j, e in enumerate(end_logits):
            if  s+e > max_sum:
                max_sum = s+e
                idx1 = i
                idx2 = j
    
    if idx1==None or idx2==None:
        return tweet

    ids = ids[idx1: idx2+1]
    predicted_text = config.TOKENIZER.decode(ids, skip_special_tokens=True)
    predicted_text = predicted_text.strip()
    
    return predicted_text


# In[ ]:


# loading the model and putting it into gpu
conf = transformers.RobertaConfig.from_pretrained(config.ROBERTA_PATH)
conf.output_hidden_states = True
model = TweetModel(conf)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.eval()


# In[ ]:


# making submission
test = pd.read_csv(config.TEST_FILE)
test['selected_text'] = [predict(test.text.values[i], test.sentiment.values[i]) for i in tqdm(range(len(test)))]
submission = test.drop(columns=['text', 'sentiment'])
submission.to_csv('submission.csv', index=False)

