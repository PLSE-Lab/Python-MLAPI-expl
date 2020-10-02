#!/usr/bin/env python
# coding: utf-8

# ## 1. About this kernel
# 
# * In this kernel i have shown how we can create more data by replacing polar words in selected_text with their synonyms.I have used SentimentIntensityAnalyzer from nltk to extract polar words and i have used http://paraphrase.org for phrase matching. For each polar word in selected_text i have created upto 3 variants.
# 
# * The dataset can be found at https://www.kaggle.com/rohitsingh9990/tse-augmented
# 
# 
# ## Whats New
# 
# * In this version i am creating a training pipeline to show how we can use this augmented data for training.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import nltk
import re

from nltk import word_tokenize
from nltk.corpus import stopwords

stoplist = stopwords.words('english')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import requests
import json

url = "http://paraphrase.org/api/en/search/"


def get_synonyms(word):
    results = []
    querystring = {"batchNumber":"0","needsPOSList":"true","q":word}
    headers = {
        'cache-control': "no-cache",
        'postman-token': "2d3d31e7-b571-f4ae-d69b-8116ff64d752"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    response_js = response.json()
    
    res_count = response_js['hits']['found']
    if res_count > 0:
        res_count = min(3, res_count )
        hits = response_js['hits']['hit'][:res_count]
        results = [ hit['target'] for hit in hits]
    return results


# ## 2. Few samples:

# In[ ]:


# get_synonyms('so sick')  
# get_synonyms('gutted') 
# get_synonyms('lovely')  


# ## 3. Reading Data

# In[ ]:


train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').dropna().reset_index(drop=True)
train = train[train.sentiment != 'neutral'].dropna().reset_index(drop=True).head(100)


# In[ ]:


vader = SentimentIntensityAnalyzer()

def sentiment_scores(word):
    score = vader.polarity_scores(word)
    return score

def sort_by_len(lst): 
    lst.sort(key=len) 
    return lst 


# In[ ]:


def find_synonym( text, selected_text, sentiment):
    
    selected_texts = []
    texts = []
    
    orig_words = selected_text.split()
    words = [ word_tokenize(str(word)) for word in orig_words]
    words = [ word[0] for word in words if len(word) > 0]

    polar_words = []
    if sentiment == 'positive':
        polar_words = [ word for word in words if sentiment_scores(word)['pos'] > 0]
    elif sentiment == 'negative':
        polar_words = [ word for word in words if sentiment_scores(word)['neg'] > 0]

    if len(polar_words) == 0:
        b=orig_words[0]
        b=re.sub(r'\W+','',b).lower()
        for word in orig_words:
            if(len(word)>len(b)):
                b=word
        polar_words = [b]

    polar_word = sort_by_len(polar_words)[-1]

    try:        
        similar_words = get_synonyms(polar_word) 
        
        for similar in similar_words:
            selected_texts.append(re.sub(polar_word,similar,selected_text))
            texts.append(re.sub(polar_word,similar,text))
        
    except Exception as e:
        print(e)
        if texts == [] and selected_texts == []:
            return ('','')
            
    return (texts,selected_texts)


# > Note: **<span style="color:Red">If you want to recreate the aug dataset from scratch, simply uncomment the below code and enable internet connection**  
# 

# In[ ]:


# generated = train.progress_apply(lambda x : find_synonym(x.text,x.selected_text, x.sentiment),axis=1)
# x,y = list(map(list,zip(*generated.values.tolist())))

# new_df=pd.DataFrame({"textID": train.textID.values,"text":x,"sentiment":train.sentiment.values,'selected_text':y})
# new_df.to_csv('twitter_augmented.csv',index=False)
# new_df.head()


# In[ ]:


aug_df = pd.read_csv("../input/tse-augmented/twitter_augmented.csv")
aug_df.head(10)


# ## 4. Creating Training Pipeline

# In[ ]:


import os
import pandas as pd
from sklearn import model_selection
import tokenizers
import albumentations
import random
import transformers
import torch.nn as nn
import torch
from ast import literal_eval
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from albumentations.core.transforms_interface import DualTransform, BasicTransform


# In[ ]:


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[ ]:


seed_torch(seed=42)


# In[ ]:


class config:
    MAX_LEN = 168
    TRAIN_BATCH_SIZE = 48
    VALID_BATCH_SIZE = 16
    EPOCHS = 4
    BASE_PATH = "../input/tweet-sentiment-extraction"
    ROBERTA_PATH = '../input/roberta-base'
    TRAINING_FILE = 'train_folds.csv'
    AUG_FILE = '../input/tse-augmented/twitter_augmented.csv'
    device = torch.device('cuda')
    
    TOKENIZER = tokenizers.ByteLevelBPETokenizer(
        vocab_file=f"{ROBERTA_PATH}/vocab.json", 
        merges_file=f"{ROBERTA_PATH}/merges.txt", 
        lowercase=True,
        add_prefix_space=True
    )


# ### 4.1 create folds

# In[ ]:


df = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
df = df.dropna().reset_index(drop=True)

df["kfold"] = -1

kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.sentiment.values)):
    print(len(trn_), len(val_))
    df.loc[val_, 'kfold'] = fold

df.to_csv("train_folds.csv", index=False)


# ### 4.2 Creating NLP Augmentation pipeline similar to Albumentations in Deep Learning

# In[ ]:


class NLPTransform(BasicTransform):
    """ Transform for nlp task."""

    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params

    def get_sentences(self, text, lang='en'):
        return sent_tokenize(text, self.LANGS.get(lang, 'english'))


class AddSynonymTransform(NLPTransform):
    """ Add random non toxic statement """
    def __init__(self, always_apply=False, p=0.5):
        super(AddSynonymTransform, self).__init__(always_apply, p)


    def apply(self, data, **params):
        # text, lang = data
        text, selected_text = data

        rand_ind = random.randint(0,2)
        text =  text[rand_ind] if len(text) > rand_ind else text[0]
        selected_text =  selected_text[rand_ind] if len(selected_text) > rand_ind else selected_text[0]
        return text, selected_text


def get_train_transforms():
    return albumentations.Compose([
        AddSynonymTransform(p=0.5),
        # you can add more transforms here
    ])


# ### 4.3 creating dataset

# In[ ]:



def process_data(tweet, selected_text, tweet_aug, selected_text_aug, sentiment, tokenizer, max_len, train_transforms):

    if train_transforms and type(tweet_aug) != float and len(tweet_aug) > 0:
        tweet_aug, selected_text_aug = train_transforms(data=(tweet_aug, selected_text_aug))['data']
        if type(tweet_aug) != list:
            tweet = tweet_aug
            selected_text = selected_text_aug

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

    input_ids = [0] + [sentiment_id[sentiment]]  + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4  +  tweet_offsets + [(0, 0)]
    targets_start += 4
    targets_end += 4


    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
    
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


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text, tweet_aug, selected_text_aug, train_transforms=None):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tweet_aug = tweet_aug
        self.selected_text_aug = selected_text_aug
        self.tokenizer = config.TOKENIZER
        self.train_transforms = train_transforms
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):

        tweet_aug = self.tweet_aug[item] if len(self.tweet_aug) > item else []
        selected_text_aug = self.selected_text_aug[item] if len(self.selected_text_aug) > item else []
        

        data = process_data(
            self.tweet[item],
            self.selected_text[item],
            tweet_aug,
            selected_text_aug,
            self.sentiment[item],
            self.tokenizer,
            self.max_len,
            self.train_transforms
        )

        return {
            'ids': torch.tensor(data['ids'], dtype=torch.long),
            'mask': torch.tensor(data['mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(data['token_type_ids'], dtype=torch.long),
            'targets_start': torch.tensor(data['targets_start'], dtype=torch.long),
            'targets_end': torch.tensor(data['targets_end'], dtype=torch.long),
            'orig_tweet': data['orig_tweet'],
            'orig_selected': data['orig_selected'],
            'sentiment': data['sentiment'],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }


# ### 4.3 train and eval functions

# In[ ]:


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets_start = d['targets_start']
        targets_end = d['targets_end']
        sentiment = d['sentiment']
        orig_selected = d['orig_selected']
        orig_tweet = d['orig_tweet']
        targets_start = d['targets_start']
        targets_end = d['targets_end']
        offsets = d['offsets']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        model.zero_grad()
        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = utils.loss_fn(outputs_start, outputs_end, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            jaccard_score, _ = utils.calculate_jaccard_score(
                original_tweet = tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start = np.argmax(outputs_start[px,:]),
                idx_end=np.argmax(outputs_end[px,:]),
                offsets=offsets[px]
            )
            jaccard_scores.append(jaccard_score)
        jaccards.update(np.mean(jaccard_scores), ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)



def eval_fn(data_loader, model, device):
    model.eval()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d['ids']
            token_type_ids = d['token_type_ids']
            mask = d['mask']
            targets_start = d['targets_start']
            targets_end = d['targets_end']
            sentiment = d['sentiment']
            orig_selected = d['orig_selected']
            orig_tweet = d['orig_tweet']
            targets_start = d['targets_start']
            targets_end = d['targets_end']
            offsets = d['offsets'].numpy()

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

            loss = utils.loss_fn(outputs_start, outputs_end, targets_start, targets_end)

            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()

            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                jaccard_score, _ = utils.calculate_jaccard_score(
                    original_tweet = tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start = np.argmax(outputs_start[px,:]),
                    idx_end=np.argmax(outputs_end[px,:]),
                    offsets=offsets[px]
                )
                jaccard_scores.append(jaccard_score)
            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
    return jaccards.avg, losses.avg


# ### 4.4 Model

# In[ ]:


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, model_config):
        super(TweetModel, self).__init__(model_config)
        self.roberta = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=model_config)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768 , 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask, token_type_ids):
        seq, pooled = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = self.drop_out(seq)
        logits = self.l0(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


# ### 4.5 utils

# In[ ]:


class utils:
    
    def loss_fn(start_logits, end_logits, start_positions, end_positions):
        loss_func = nn.CrossEntropyLoss()
        start_loss = loss_func(start_logits, start_positions)
        end_loss = loss_func(end_logits, end_positions)
        total_loss = (start_loss + end_loss)
        return total_loss



    class AverageMeter():
        """Computes and stores the average and current value"""
        def __init__(self):
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


    def jaccard(str1, str2): 
        a = set(str1.lower().split()) 
        b = set(str2.lower().split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))


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

        filtered_output  = ""
        for ix in range(idx_start, idx_end + 1):
            filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
            if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
                filtered_output += " "

        if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
            filtered_output = original_tweet

        jac = utils.jaccard(target_string.strip(), filtered_output.strip())
        return jac, filtered_output


    class EarlyStopping:
        def __init__(self, patience=7, mode="max", delta=0.0003):
            self.patience = patience
            self.counter = 0
            self.mode = mode
            self.best_score = None
            self.early_stop = False
            self.delta = delta
            if self.mode == "min":
                self.val_score = np.Inf
            else:
                self.val_score = -np.Inf

        def __call__(self, epoch_score, model, model_path):

            if self.mode == "min":
                score = -1.0 * epoch_score
            else:
                score = np.copy(epoch_score)

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(epoch_score, model, model_path)
            elif score < self.best_score + self.delta:
                self.counter += 1
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(epoch_score, model, model_path)
                self.counter = 0
            print(f'Best score till now {self.best_score}')

        def save_checkpoint(self, epoch_score, model, model_path):
            if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
                print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
                # print('')
                torch.save(model.state_dict(), model_path)
            self.val_score = epoch_score


# ### 4.6 Training Function

# In[ ]:


def run(fold):

    dfx = pd.read_csv(config.TRAINING_FILE)

    aug_df = pd.read_csv(
        config.AUG_FILE,
        converters={"text": literal_eval, "selected_text": literal_eval}
    )
    aug_df.columns = ['textID', 'text_aug', 'sentiment', 'selected_text_aug']
    dfx = dfx.merge(aug_df, how='left')

    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_train = df_train.sample(frac=1).reset_index(drop=True)

    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)


    train_dataset = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values,
        tweet_aug=df_train.text_aug.values,
        selected_text_aug=df_train.selected_text_aug.values,
        train_transforms= get_train_transforms()
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    valid_dataset = TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values,
        tweet_aug=[],
        selected_text_aug=[],
        train_transforms = None
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=0
    )

    device = config.device

    model_config = transformers.RobertaConfig.from_pretrained(config.ROBERTA_PATH)
    model = TweetModel(model_config=model_config)

    model.to(device)

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=1, 
        num_training_steps=num_train_steps
    )

    es = utils.EarlyStopping(patience=2, mode='max')
    print('*'* 100)
    print(f"Training is Starting for fold={fold}")
    best_jac = 0

    for epoch in range(config.EPOCHS):
        train_fn(train_data_loader, model, optimizer, device, scheduler)
        jaccard, loss = eval_fn(valid_data_loader, model, device)
        print(f'Jaccard Score = {jaccard}')
        print(f'Loss = {loss}')
        es(jaccard, model, model_path=f'model_{fold}.bin')
        if es.early_stop:
            print('Early Stopping')
            break
        if jaccard > best_jac:
            best_jac = jaccard


# In[ ]:


for i in range(0, 5):
    run(i)


# ### 4.7 Inference

# In[ ]:


df_test = pd.read_csv(f"{config.BASE_PATH}/test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values


# In[ ]:


df_test.head()


# In[ ]:


model_config = transformers.RobertaConfig.from_pretrained(config.ROBERTA_PATH)

ENSEMBLES = [
    {'model': TweetModel(model_config=model_config), 'state_dict': f"model_0.bin", 'weight': 1 }, # CV: ??
    {'model': TweetModel(model_config=model_config), 'state_dict': f"model_1.bin", 'weight': 1}, # CV: ??
    {'model': TweetModel(model_config=model_config), 'state_dict': f"model_2.bin", 'weight': 1}, # CV: ??
    {'model': TweetModel(model_config=model_config), 'state_dict': f"model_3.bin", 'weight': 1}, # CV: ??
    {'model': TweetModel(model_config=model_config), 'state_dict': f"model_4.bin", 'weight': 1}, # CV: ??
]


# In[ ]:


models = []
weights = []

for val in ENSEMBLES:
    model = val['model']
    model.to(config.device)
    model.load_state_dict(torch.load(val['state_dict']))
    model.eval()
    models.append(model)
    weights.append(val['weight'])


# In[ ]:


final_output = []


test_dataset = TweetDataset(
    tweet=df_test.text.values,
    sentiment=df_test.sentiment.values,
    selected_text=df_test.selected_text.values, 
    tweet_aug=[],
    selected_text_aug=[]
)

test_data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=config.VALID_BATCH_SIZE,
    num_workers=0
)

device = config.device


with torch.no_grad():
    tk0 = tqdm(test_data_loader, total=len(test_data_loader))
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

        outputs_start = []
        outputs_end = []
        
        for index, model in enumerate(models):
            output_start, output_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs_start.append(output_start * weights[index]) 
            outputs_end.append(output_end * weights[index]) 
            
        outputs_start = sum(outputs_start) / len(outputs_start) 
        outputs_end = sum(outputs_end) / len(outputs_end)
    
        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            _, output_sentence = utils.calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
                offsets=offsets[px]
            )
            
            final_output.append(output_sentence)


# In[ ]:


sample = pd.read_csv(f"{config.BASE_PATH}/sample_submission.csv")
sample.loc[:, 'selected_text'] = final_output

sample.to_csv("submission.csv", index=False)
sample.head(10)


# ## References:
# 
# * https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch
# * https://www.kaggle.com/shonenkov/nlp-albumentations

# # END NOTES
# 
# 
# **<span style="color:Red">Please upvote this kernel if you like it . It motivates me to produce more quality content :)**  
