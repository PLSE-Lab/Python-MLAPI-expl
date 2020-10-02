#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import copy
from transformers import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
import tokenizers


# # Define config

# In[ ]:


class Config:
    # config settings
    def __init__(self):
        # dataset setting
        self.max_seq_length = 192
        # dataloader settings
        self.val_batch_size = 256
        self.num_workers = 8


# In[ ]:


Config = Config()


# # Preprocess test data 

# In[ ]:


def preprocessing(text):

    text = text.replace("....", ". . . .")
    text = text.replace("...", ". . .")
    text = text.replace("..", ". .")
    text = text.replace("!!!!", "! ! ! !")
    text = text.replace("!!!", "! ! !")
    text = text.replace("!!", "! !")
    text = text.replace("????", "? ? ? ?")
    text = text.replace("???", "? ? ?")
    text = text.replace("??", "? ?")

    return text


def process_data(tweet, selected_text, old_selected_text, sentiment, tokenizer, model_type, max_len, augment=False):

    tweet_with_extra_space = preprocessing(copy.deepcopy(str(tweet).lower()))
    tweet = preprocessing(" " + " ".join(str(tweet).lower().split()))
    selected_text = preprocessing(" " + " ".join(str(selected_text).lower().split()))
    old_selected_text = " " + " ".join(str(old_selected_text).lower().split())

    # remove first " "
    len_st = len(selected_text) - 1
    idx0 = None
    idx1 = None

    # get char idx
    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
        if " " + tweet[ind: ind + len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    if idx0 is None and idx1 is None:
        print("--------------------------------------------- error cleaned selected----------------------------------")
        print("tweet:", tweet)
        print("selected_text:", selected_text)
        print("old_selected_text:", old_selected_text)
        print("--------------------------------------------- error cleaned selected----------------------------------")

        for ind in (i for i, e in enumerate(tweet) if e == old_selected_text[1]):
            if " " + tweet[ind: ind + len_st] == old_selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break
                
    # get char mask
    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1

    # get word offsets
    tweet_offsets_word_level = []
    tweet_offsets_token_level = []
    cursor = 0
    input_ids_orig = []

    for i, word in enumerate(tweet.split()):

        sub_words = tokenizer.tokenize(" " + word)
        encoded_word = tokenizer.convert_tokens_to_ids(sub_words)
        number_of_tokens = len(encoded_word)
        input_ids_orig += encoded_word

        start_offsets = cursor

        token_level_cursor = start_offsets

        for i in range(number_of_tokens):

            if (model_type == "bert-base-uncased") or (model_type == "bert-large-uncased")                     or (model_type == "bert-base-cased") or (model_type == "bert-large-cased"):

                # for bert tokenizer, replace "##" and add " " for first sub_word
                sub_word_len = len(sub_words[i].replace("##", ""))
                if i == 0:
                    sub_word_len += 1
            else:
                sub_word_len = len(sub_words[i])

            tweet_offsets_token_level.append((token_level_cursor, token_level_cursor + sub_word_len))
            cursor = token_level_cursor + sub_word_len
            token_level_cursor += sub_word_len

        end_offsets = cursor

        for i in range(number_of_tokens):
            tweet_offsets_word_level.append((start_offsets, end_offsets))

    # get word idx
    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets_token_level):

        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)

    if len(target_idx) == 0:
        print(tweet, selected_text)

    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    # print(tweet[tweet_offsets_token_level[targets_start][0]: tweet_offsets_token_level[targets_end][1]],
    #       "------------", selected_text)


    if model_type == "roberta-base" or model_type == "roberta-large" or model_type == "roberta-base-squad":

        sentiment_id = {
            'positive': 1313,
            'negative': 2430,
            'neutral': 7974
        }

        input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
        token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets_token_level = [(0, 0)] * 4 + tweet_offsets_token_level + [(0, 0)]
        tweet_offsets_word_level = [(0, 0)] * 4 + tweet_offsets_word_level + [(0, 0)]
        targets_start += 4
        targets_end += 4

    elif (model_type == "albert-base-v2") or (model_type == "albert-large-v2") or (model_type == "albert-xlarge-v2"):

        sentiment_id = {
            'positive': 2221,
            'negative': 3682,
            'neutral': 8387
        }

        input_ids = [2] + [sentiment_id[sentiment]] + [3] + input_ids_orig + [3]
        token_type_ids = [0, 0, 0] + [0] * (len(input_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets_token_level = [(0, 0)] * 3 + tweet_offsets_token_level + [(0, 0)]
        tweet_offsets_word_level = [(0, 0)] * 3 + tweet_offsets_word_level + [(0, 0)]
        targets_start += 3
        targets_end += 3

    elif (model_type == "xlnet-base-cased") or (model_type == "xlnet-large-cased"):

        sentiment_id = {
            'positive': 1654,
            'negative': 2981,
            'neutral': 9201
        }

        input_ids = [sentiment_id[sentiment]] + [4] + input_ids_orig + [3]
        token_type_ids = [0, 0] + [0] * (len(input_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets_token_level = [(0, 0)] * 2 + tweet_offsets_token_level + [(0, 0)]
        tweet_offsets_word_level = [(0, 0)] * 2 + tweet_offsets_word_level + [(0, 0)]
        targets_start += 2
        targets_end += 2

    elif (model_type == "bert-base-uncased") or (model_type == "bert-large-uncased"):

        sentiment_id = {
            'positive': 3893,
            'negative': 4997,
            'neutral': 8699
        }

        input_ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids_orig + [102]
        token_type_ids = [0, 0, 0] + [0] * (len(input_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets_token_level = [(0, 0)] * 3 + tweet_offsets_token_level + [(0, 0)]
        tweet_offsets_word_level = [(0, 0)] * 3 + tweet_offsets_word_level + [(0, 0)]
        targets_start += 3
        targets_end += 3

    elif (model_type == "bert-base-cased") or (model_type == "bert-large-cased"):

        sentiment_id = {
            'positive': 3112,
            'negative': 4366,
            'neutral': 8795
        }

        input_ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids_orig + [102]
        token_type_ids = [0, 0, 0] + [0] * (len(input_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets_token_level = [(0, 0)] * 3 + tweet_offsets_token_level + [(0, 0)]
        tweet_offsets_word_level = [(0, 0)] * 3 + tweet_offsets_word_level + [(0, 0)]
        targets_start += 3
        targets_end += 3

    else:
        raise NotImplementedError

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets_token_level = tweet_offsets_token_level + ([(0, 0)] * padding_length)
        tweet_offsets_word_level = tweet_offsets_word_level + ([(0, 0)] * padding_length)
    else:
        input_ids = input_ids[:max_len]
        mask = mask[:max_len]
        token_type_ids = token_type_ids[:max_len]
        tweet_offsets_token_level = tweet_offsets_token_level[:max_len]
        tweet_offsets_word_level = tweet_offsets_word_level[:max_len]


    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'orig_tweet': tweet,
        'orig_tweet_with_extra_space': tweet_with_extra_space,
        'orig_selected': old_selected_text,
        'sentiment': sentiment,
        'offsets_token_level': tweet_offsets_token_level,
        'offsets_word_level': tweet_offsets_word_level
    }


# In[ ]:


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text, old_selected_text, tokenizer, model_type, max_len):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.old_selected_text = old_selected_text
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_len = max_len
        
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data(
            self.tweet[item],
            self.selected_text[item],
            self.old_selected_text[item],
            self.sentiment[item],
            self.tokenizer,
            self.model_type,
            self.max_len,
        )

        return torch.tensor(data["ids"], dtype=torch.long),                torch.tensor(data["mask"], dtype=torch.long),                torch.tensor(data["token_type_ids"], dtype=torch.long),                torch.tensor(data["targets_start"], dtype=torch.long),                torch.tensor(data["targets_end"], dtype=torch.long),                data["orig_tweet"],                data["orig_tweet_with_extra_space"],                data["orig_selected"],                data["sentiment"],                torch.tensor(data["offsets_token_level"], dtype=torch.long),                torch.tensor(data["offsets_word_level"], dtype=torch.long)


# # Define dataloader

# In[ ]:


def get_test_loader(data_path="../input/tweet-sentiment-extraction/",
                    csv_name="test.csv",
                    max_seq_length=384,
                    model_type="bert-base-uncased",
                    batch_size=4,
                    num_workers=4):

    CURR_PATH = "../input/"
    csv_path = os.path.join(data_path, csv_name)
    df_test = pd.read_csv(csv_path)
    df_test.loc[:, "selected_text"] = df_test.text.values
    df_test.loc[:, "cleaned_selected_text"] = df_test.text.values
#     df_test = df_test[:6]

    if (model_type == "bert-base-uncased"):
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers-vocab/{}-vocab.txt".format(model_type)),
            lowercase=True,
        )
    elif (model_type == "bert-large-uncased"):
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers-vocab/{}-vocab.txt".format(model_type)),
            lowercase=True,
        )
    elif (model_type == "bert-base-cased"):
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers-vocab/{}-vocab.txt".format(model_type)),
            lowercase=True,
        )
    elif (model_type == "bert-large-cased"):
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers-vocab/{}-vocab.txt".format(model_type)),
            lowercase=True,
        )
    elif (model_type == "xlnet-base-cased") or (model_type == "xlnet-large-cased"):
        tokenizer = XLNetTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers-vocab/{}-spiece.model".format(model_type)),
            lowercase=True,
        )
    elif (model_type == "albert-base-v2") or (model_type == "albert-large-v2") or (model_type == "albert-xlarge-v2"):
        tokenizer = AlbertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(CURR_PATH, "transformers-vocab/{}-spiece.model".format(model_type)),
            lowercase=True,
        )
    elif model_type == "roberta-base":
        tokenizer = RobertaTokenizer(
            vocab_file=os.path.join(CURR_PATH, "transformers-vocab/{}-vocab.json".format(model_type)),
            merges_file=os.path.join(CURR_PATH, "transformers-vocab/{}-merges.txt".format(model_type)),
            lowercase=True,
        )
    elif model_type == "roberta-base-squad":
        tokenizer = RobertaTokenizer(
            vocab_file=os.path.join(CURR_PATH, "transformers-vocab/{}-vocab.json".format(model_type)),
            merges_file=os.path.join(CURR_PATH, "transformers-vocab/{}-merges.txt".format(model_type)),
            lowercase=True,
        )
    elif model_type == "roberta-large":
        tokenizer = RobertaTokenizer(
            vocab_file=os.path.join(CURR_PATH, "transformers-vocab/{}-vocab.json".format(model_type)),
            merges_file=os.path.join(CURR_PATH, "transformers-vocab/{}-merges.txt".format(model_type)),
            lowercase=True,
        )
    else:

        raise NotImplementedError

    ds_test = TweetDataset(
        tweet=df_test.text.values,
        sentiment=df_test.sentiment.values,
        selected_text=df_test.cleaned_selected_text.values,
        old_selected_text=df_test.selected_text.values,
        tokenizer=tokenizer,
        model_type=model_type,
        max_len=max_seq_length
    )
    # print(len(ds_test.tensors))
    loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return loader, tokenizer


# # Define model

# In[ ]:


############################################ Define Net Class
class TweetBert(nn.Module):
    def __init__(self, model_type="bert-large-uncased", hidden_layers=None):
        super(TweetBert, self).__init__()

        self.model_name = 'TweetBert'
        self.model_type = model_type

        if hidden_layers is None:
            hidden_layers = [-1]
        self.hidden_layers = hidden_layers

        if model_type == "bert-large-uncased":
            bert_model_config =             '../input/bertlargewholewordmaskingfinetunedsquad/bert-large-uncased-whole-word-masking-finetuned-squad-config.json'
            self.config = BertConfig.from_json_file(bert_model_config)
            self.config.output_hidden_states = True
            self.config.hidden_dropout_prob = 0.1
            self.bert = BertModel(self.config)   
        elif model_type == "bert-large-cased":
            bert_model_config =             '../input/bertlargewholewordmaskingfinetunedsquad/bert-large-cased-whole-word-masking-finetuned-squad-config.json'
            self.config = BertConfig.from_json_file(bert_model_config)
            self.config.output_hidden_states = True
            self.config.hidden_dropout_prob = 0.1
            self.bert = BertModel(self.config)   
        elif model_type == "bert-base-uncased":
            bert_model_config = '../input/pretrained-bert-models-for-pytorch/bert-base-uncased/bert_config.json'
            self.config = BertConfig.from_json_file(bert_model_config)
            self.config.output_hidden_states = True
            self.config.hidden_dropout_prob = 0.1
            self.bert = BertModel(self.config)   
        elif model_type == "bert-base-cased":
            bert_model_config = '../input/pretrained-bert-models-for-pytorch/bert-base-cased/bert_config.json'
            self.config = BertConfig.from_json_file(bert_model_config)
            self.config.output_hidden_states = True
            self.bert = BertModel(self.config)   
        elif model_type == "roberta-base":
            roberta_model_config = '../input/roberta-transformers-pytorch/roberta-base/config.json'
            self.config = RobertaConfig.from_json_file(roberta_model_config)
            self.config.output_hidden_states = True
            self.config.hidden_dropout_prob = 0.1
            model_path = os.path.join('../input/roberta-transformers-pytorch/roberta-base/pytorch_model.bin')
            self.bert = RobertaModel.from_pretrained(model_path, config=self.config)  
        elif model_type == "roberta-large":
            roberta_model_config = '../input/roberta-transformers-pytorch/roberta-large/config.json'
            self.config = RobertaConfig.from_json_file(roberta_model_config)
            self.config.output_hidden_states = True
            self.config.hidden_dropout_prob = 0.1
            model_path = os.path.join('../input/roberta-transformers-pytorch/roberta-large/pytorch_model.bin')
            self.bert = RobertaModel.from_pretrained(model_path, config=self.config)  
        elif model_type == "albert-large-v2":
            albert_model_config = '../input/pretrained-albert-pytorch/albert-large-v2/config.json'
            self.config = AlbertConfig.from_json_file(albert_model_config)
            self.config.output_hidden_states = True
            self.config.hidden_dropout_prob = 0.1
            model_path = os.path.join('../input/pretrained-albert-pytorch/albert-large-v2/pytorch_model.bin')
            self.bert = AlbertModel.from_pretrained(model_path, config=self.config)  
        elif model_type == "albert-base-v2":
            albert_model_config = '../input/pretrained-albert-pytorch/albert-base-v2/config.json'
            self.config = AlbertConfig.from_json_file(albert_model_config)
            self.config.output_hidden_states = True
            self.config.hidden_dropout_prob = 0.1
            model_path = os.path.join('../input/pretrained-albert-pytorch/albert-large-v2/pytorch_model.bin')
            self.bert = AlbertModel.from_pretrained(model_path, config=self.config) 
        elif model_type == "xlnet-base-cased":
            xlnet_model_config = '../input/xlnet-pretrained-models-pytorch/xlnet-base-cased-config.json'
            self.config = XLNetConfig.from_json_file(xlnet_model_config)
            self.config.output_hidden_states = True
            self.config.hidden_dropout_prob = 0.1
            model_path = os.path.join('../input/xlnet-pretrained-models-pytorch/' + model_type + '-pytorch_model.bin')
            self.bert = XLNetModel.from_pretrained(model_path, config=self.config)  
        else:
            raise NotImplementedError

        # hidden states fusion
        weights_init = torch.zeros(len(hidden_layers)).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)

        self.qa_start_end = nn.Linear(self.config.hidden_size, 2)

        def init_weights_linear(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.normal_(m.weight, std=0.02)
                torch.nn.init.normal_(m.bias, 0)

        self.qa_start_end.apply(init_weights_linear)

        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])

    def get_hidden_states(self, hidden_states):

        fuse_hidden = None
        # concat hidden
        for i in range(len(self.hidden_layers)):
            if i == 0:
                hidden_layer = self.hidden_layers[i]
                fuse_hidden = hidden_states[hidden_layer].unsqueeze(-1)
            else:
                hidden_layer = self.hidden_layers[i]
                hidden_state = hidden_states[hidden_layer].unsqueeze(-1)
                fuse_hidden = torch.cat([fuse_hidden, hidden_state], dim=-1)

        fuse_hidden = (torch.softmax(self.layer_weights, dim=0) * fuse_hidden).sum(-1)

        return fuse_hidden

    def get_logits_by_random_dropout(self, fuse_hidden, fc):

        logit = None
        h = fuse_hidden

        for j, dropout in enumerate(self.dropouts):

            if j == 0:
                logit = fc(dropout(h))
            else:
                logit += fc(dropout(h))

        return logit / len(self.dropouts)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
    ):

        if self.model_type == "roberta-base" or self.model_type == "roberta-base-squad"                 or self.model_type == "roberta-large":

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
            hidden_states = outputs[2]
        elif self.model_type == "xlnet-base-cased":

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            hidden_states = outputs[1]
        else:

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
            hidden_states = outputs[2]

        # bs, seq len, hidden size
        fuse_hidden = self.get_hidden_states(hidden_states)

        fuse_hidden_context = fuse_hidden
        if self.model_type == "xlnet-base-cased":
            hidden_classification = fuse_hidden[:, -1, :]
        else:
            hidden_classification = fuse_hidden[:, 0, :]

        # #################################################################### direct approach
        logits = self.get_logits_by_random_dropout(fuse_hidden_context, self.qa_start_end)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]

        return outputs  # start_logits, end_logits, (hidden_states), (attentions)


# In[ ]:


def load_check_point(model, checkpoint_path, skip_layers=[]):
    
    checkpoint_to_load = torch.load(checkpoint_path)
    model_state_dict = checkpoint_to_load['model']
    
    state_dict = model.state_dict()

    keys = list(state_dict.keys())

    for key in keys:
        if any(s in key for s in skip_layers):
            continue
        try:
            state_dict[key] = model_state_dict[key]
        except:
            print("Missing key:", key)

    model.load_state_dict(state_dict)
    
    return model


# # define logits function

# In[ ]:


def get_logits(model, checkpoint_path, folds, all_input_ids, all_attention_masks, all_token_type_ids=None):
    """
    Get start and end logits of a batch by ensemble of all folds
    Args:
        model: pytorch model instance
        checkpoint_path: string, path for checkpoints
        folds: list, all folds for ensemble
        all_input_ids: tensor, cuda, a batch of input_ids tensor
        all_attention_masks: tensor, cuda, a batch of attention_masks tensor
        all_token_type_ids: tensor, cuda, a batch of token_type_ids tensor
    Returns:
        start_logits: tensor, cpu
        end_logits: tensor, cpu
    """
    
    for model_idx, fold in enumerate(folds):
        
        checkpoint = os.path.join(checkpoint_path, "fold_{}.pth".format(fold))
        
        model = load_check_point(model, checkpoint, skip_layers=[])
        model.eval()
        
        outputs = model(input_ids=all_input_ids, attention_mask=all_attention_masks,
                                     token_type_ids=all_token_type_ids)
        if model_idx == 0:
            start_logits, end_logits = torch.softmax(outputs[0], dim=-1) / len(folds), torch.softmax(outputs[1], dim=-1) / len(folds)
#             start_logits, end_logits = outputs[0] / len(folds), outputs[1] / len(folds)
        else:
            start_logits += torch.softmax(outputs[0], dim=-1) / len(folds)
            end_logits += torch.softmax(outputs[1], dim=-1) / len(folds)
#             start_logits += outputs[0] / len(folds)
#             end_logits += outputs[1] / len(folds)
            
    return start_logits.detach().cpu(), end_logits.detach().cpu()


# In[ ]:


def get_word_level_logits(start_logits,
                          end_logits,
                          model_type,
                          tweet_offsets_word_level):

    tweet_offsets_word_level = np.array(tweet_offsets_word_level)

    if model_type == "roberta-base" or model_type == "roberta-large" or model_type == "roberta-base-squad":
        logit_offset = 4

    elif (model_type == "albert-base-v2") or (model_type == "albert-large-v2") or (
            model_type == "albert-xlarge-v2"):
        logit_offset = 3

    elif (model_type == "xlnet-base-cased") or (model_type == "xlnet-large-cased"):
        logit_offset = 2

    elif (model_type == "bert-base-uncased") or (model_type == "bert-large-uncased"):
        logit_offset = 3

    elif (model_type == "bert-base-cased") or (model_type == "bert-large-cased"):
        logit_offset = 3

    prev = tweet_offsets_word_level[logit_offset]
    word_level_bbx = []
    curr_bbx = []

    for i in range(len(tweet_offsets_word_level) - logit_offset - 1):

        curr = tweet_offsets_word_level[i + logit_offset]

        if curr[0] < prev[0] and curr[1] > prev[1]:
            break

        if curr[0] == prev[0] and curr[1] == prev[1]:
            curr_bbx.append(i)
        else:
            word_level_bbx.append(curr_bbx)
            curr_bbx = [i]

        prev = curr

    if len(word_level_bbx) == 0:
        word_level_bbx.append(curr_bbx)

    for i in range(len(word_level_bbx)):
        word_level_bbx[i].append(word_level_bbx[i][-1] + 1)

    start_logits_word_level = [np.max(start_logits[bbx[0]: bbx[-1]]) for bbx in word_level_bbx]
    end_logits_word_level = [np.max(end_logits[bbx[0]: bbx[-1]]) for bbx in word_level_bbx]

    return start_logits_word_level, end_logits_word_level, word_level_bbx


def get_token_level_idx(start_logits,
                        end_logits,
                        start_logits_word_level,
                        end_logits_word_level,
                        word_level_bbx):

    # get most possible word
    start_idx_word = np.argmax(start_logits_word_level)
    end_idx_word = np.argmax(end_logits_word_level)

    # get all token idx in selected word
    start_word_bbx = word_level_bbx[start_idx_word]
    end_word_bbx = word_level_bbx[end_idx_word]

    # find most possible token idx in selected word
    start_idx_in_word = np.argmax(start_logits[start_word_bbx[0]: start_word_bbx[-1]])
    end_idx_in_word = np.argmax(end_logits[end_word_bbx[0]: end_word_bbx[-1]])

    # find most possible token idx in whole sentence
    start_idx_token = start_word_bbx[start_idx_in_word]
    end_idx_token = end_word_bbx[end_idx_in_word]

#     return start_idx_token, end_idx_token
    return np.argmax(start_logits), np.argmax(end_logits)


def calculate_jaccard_score(
        original_tweet,
        selected_text,
        idx_start,
        idx_end,
        model_type,
        tweet_offsets,
        tokenizer,
        sentiment):
    
    if idx_end < idx_start:
        filtered_output = original_tweet

    else:
        input_ids_orig = tokenizer.encode(original_tweet).ids
        input_ids = input_ids_orig + [2]
        filtered_output = tokenizer.decode(input_ids[idx_start:idx_end+1])

    jac = 0
    return jac, filtered_output


# In[ ]:


def get_different_model_logits(model_type="roberta-base",                                hidden_layers=[-1, -2, -3, -4],                                checkpoint_path="../input/tweetrobertabase5fold42v8/",                                Config=None):
    """
    Get start and end logits of a batch by ensemble of all folds
    Args:
        model_type: string
        hidden_layers: list
        checkpoint_path: string
        Config: Config class instance
    Returns:
        start_logits_token_level: list of array
        end_logits_token_level: list of array
        start_logits_word_level: list of array
        end_logits_word_level: list of array
        word_level_bbx: list of list of list
        token_level_offsets: list of list of tuple
        tweet: list of string
    """
    
    if model_type == "roberta-base" or model_type == "roberta-large" or model_type == "roberta-base-squad":

        offsets = 4

    elif model_type == "albert-base-v2" or model_type == "albert-large-v2" or model_type == "albert-xlarge-v2":

        offsets = 3

    elif model_type == "xlnet-base-cased" or model_type == "xlnet-large-cased":

        offsets = 2

    elif model_type == "bert-base-uncased" or model_type == "bert-large-uncased" or model_type == "bert-base-cased" or model_type == "bert-large-cased":

        offsets = 3

    else:
        raise NotImplementedError

    loader, tokenizer = get_test_loader(data_path="../input/tweet-sentiment-extraction/",
                                        max_seq_length=Config.max_seq_length,
                                        model_type=model_type,
                                        batch_size=Config.val_batch_size,
                                        num_workers=Config.num_workers)

    model = TweetBert(model_type=model_type, hidden_layers=hidden_layers).cuda().eval()

    start_logits_token_level = []
    end_logits_token_level = []
    start_logits_word_level = []
    end_logits_word_level = []
    word_level_bbx = []
    token_level_offsets = []
    tweets = []
    tweets_with_extra_spaces = []
    sentiments = []
    
    # init cache
    torch.cuda.empty_cache()

    with torch.no_grad():

        for test_batch_i, (all_input_ids, all_attention_masks, all_token_type_ids, 
                               all_start_positions, all_end_positions, 
                               all_orig_tweet, all_orig_tweet_with_extra_space, 
                               all_orig_selected, all_sentiment, 
                               all_offsets_token_level, all_offsets_word_level) in enumerate(loader):

            if (test_batch_i % 5 == 0):
                print("Inferencing: ", test_batch_i, "of", len(loader))

            # set input to cuda mode
            all_input_ids = all_input_ids.cuda()
            all_attention_masks = all_attention_masks.cuda()
            all_token_type_ids = all_token_type_ids.cuda()

            start_logits, end_logits = get_logits(model,                                                   checkpoint_path,                                                   [0, 1, 2, 3, 4],                                                   all_input_ids,                                                   all_attention_masks,                                                   all_token_type_ids)

            start_logits = start_logits[:, offsets:]
            end_logits = end_logits[:, offsets:]
            
            start_logits = torch.softmax(start_logits, dim=-1)
            end_logits = torch.softmax(end_logits, dim=-1)
            
            def to_numpy(tensor):
                return tensor.numpy()

                # batch size, seq len

            start_logits = to_numpy(start_logits)
            end_logits = to_numpy((end_logits))
            
            for px, tweet in enumerate(all_orig_tweet):
                
                start_logits_word_level_sample, end_logits_word_level_sample, word_level_bbx_sample = get_word_level_logits(
                    start_logits[px],
                    end_logits[px],
                    model_type,
                    all_offsets_word_level[px])

                start_logits_token_level.append(start_logits[px])
                end_logits_token_level.append(end_logits[px])
                start_logits_word_level.append(start_logits_word_level_sample)
                end_logits_word_level.append(end_logits_word_level_sample)
                word_level_bbx.append(word_level_bbx_sample)
                token_level_offsets.append(all_offsets_token_level[px])
                tweets.append(tweet)
                tweets_with_extra_spaces.append(all_orig_tweet_with_extra_space[px])
                sentiments.append(all_sentiment[px])


    return start_logits_token_level, end_logits_token_level,            start_logits_word_level, end_logits_word_level,            word_level_bbx, token_level_offsets,            tweets, tweets_with_extra_spaces, sentiments, tokenizer


# # roberta-base

# In[ ]:


roberta_base_start_logits_token_level_42, roberta_base_end_logits_token_level_42, roberta_base_start_logits_word_level_42, roberta_base_end_logits_word_level_42, roberta_base_word_level_bbx, roberta_base_token_level_offsets, tweets, tweets_with_extra_spaces, sentiments, roberta_base_tokenizer = get_different_model_logits(model_type="roberta-base",                                                                        hidden_layers=[-1, -2, -3, -4],                                                                        checkpoint_path="../input/tweetrobertabasenewpipelinepreprocessing42/",                                                                       Config=Config)


roberta_base_start_logits_token_level_666, roberta_base_end_logits_token_level_666, roberta_base_start_logits_word_level_666, roberta_base_end_logits_word_level_666, _, _, _, _, _, _ = get_different_model_logits(model_type="roberta-base",                                                                        hidden_layers=[-1, -2, -3, -4],                                                                        checkpoint_path="../input/tweetrobertabasenewpipelinepreprocessing666/",                                                                       Config=Config)


roberta_base_start_logits_token_level_1234, roberta_base_end_logits_token_level_1234, roberta_base_start_logits_word_level_1234, roberta_base_end_logits_word_level_1234, _, _, _, _, _, _ = get_different_model_logits(model_type="roberta-base",                                                                        hidden_layers=[-1, -2, -3, -4],                                                                        checkpoint_path="../input/tweetrobertabasenewpipelinepreprocessing1234/",                                                                       Config=Config)


# In[ ]:


sample_size = len(roberta_base_start_logits_token_level_42)

roberta_base_start_logits_token_level = [(roberta_base_start_logits_token_level_42[i] + 
                                          roberta_base_start_logits_token_level_666[i] + 
                                          roberta_base_start_logits_token_level_1234[i]
                                         ) / 3 for i in range(sample_size)]

roberta_base_end_logits_token_level = [(roberta_base_end_logits_token_level_42[i] + 
                                        roberta_base_end_logits_token_level_666[i] +
                                        roberta_base_end_logits_token_level_1234[i]
                                       ) / 3 for i in range(sample_size)]

roberta_base_start_logits_word_level = [(np.array(roberta_base_start_logits_word_level_42[i]) + 
                                         np.array(roberta_base_start_logits_word_level_666[i]) + 
                                         np.array(roberta_base_start_logits_word_level_1234[i])
                                        ) / 3 for i in range(sample_size)]

roberta_base_end_logits_word_level = [(np.array(roberta_base_end_logits_word_level_42[i]) + 
                                       np.array(roberta_base_end_logits_word_level_666[i]) +
                                       np.array(roberta_base_end_logits_word_level_1234[i])
                                      ) / 3 for i in range(sample_size)]


# # albert-large

# In[ ]:


_, _, albert_large_start_logits_word_level, albert_large_end_logits_word_level, _, _, _, _, _, _ = get_different_model_logits(model_type="albert-large-v2",                                                                                                                           hidden_layers=[-1, -2, -3, -4],                                                                                                                           checkpoint_path="../input/tweetalbertlargenewpipelinepreprocessingv1/",                                                                                                                          Config=Config)


# # xlnet-base

# In[ ]:


_, _, xlnet_base_start_logits_word_level, xlnet_base_end_logits_word_level, _, _, _, _, _, _ = get_different_model_logits(model_type="xlnet-base-cased",                                                                                                                           hidden_layers=[-1, -2, -3, -4],                                                                                                                           checkpoint_path="../input/tweetxlnetbasenewpipelinepreprocessingv1/",                                                                                                                          Config=Config)


# # Ensemble

# In[ ]:


start_logits_word_level = []
end_logits_word_level = []

for i in range(len(roberta_base_start_logits_word_level)):
    
#     start_logits_word_level.append(np.array(roberta_base_start_logits_word_level[i]))
#     end_logits_word_level.append((np.array(roberta_base_end_logits_word_level[i])))
    
    start_logits_word_level.append((np.array(roberta_base_start_logits_word_level[i]) + np.array(albert_large_start_logits_word_level[i]) + np.array(xlnet_base_start_logits_word_level[i])) / 3)
    end_logits_word_level.append((np.array(roberta_base_end_logits_word_level[i]) + np.array(albert_large_end_logits_word_level[i]) + np.array(xlnet_base_end_logits_word_level[i])) / 3)


# In[ ]:


import tokenizers

# we based on roberta base token level offsets for final prediction
all_results = []
all_start_end = []
base_model_type = "roberta-base"

CURR_PATH = "../input/"
tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file=os.path.join(CURR_PATH, "transformers-vocab/{}-vocab.json".format(base_model_type)),
            merges_file=os.path.join(CURR_PATH, "transformers-vocab/{}-merges.txt".format(base_model_type)),
            lowercase=True,
            add_prefix_space=True
        )

for i in range(len(start_logits_word_level)):
    
    start_idx_token, end_idx_token = get_token_level_idx(roberta_base_start_logits_token_level[i],                                                          roberta_base_end_logits_token_level[i],                                                          start_logits_word_level[i],                                                          end_logits_word_level[i],                                                          roberta_base_word_level_bbx[i]
                                                         )
#     start_idx_token, end_idx_token = np.argmax(roberta_base_start_logits_token_level[i], axis=-1), np.argmax(roberta_base_end_logits_token_level[i], axis=-1)
    all_start_end.append((start_idx_token, end_idx_token))
    _, final_text = calculate_jaccard_score(
                original_tweet=tweets[i],
                selected_text=tweets[i],
                idx_start=start_idx_token,
                idx_end=end_idx_token,
                model_type=base_model_type,
                tweet_offsets=roberta_base_token_level_offsets[i],
                tokenizer=tokenizer,
                sentiment=sentiments[i]
            )
        
    all_results.append(final_text)


# # Generate csv

# In[ ]:


print(all_start_end[:30])


# In[ ]:


submission = pd.read_csv(os.path.join("/kaggle/input/tweet-sentiment-extraction", "sample_submission.csv"))
test = pd.read_csv(os.path.join("/kaggle/input/tweet-sentiment-extraction", "test.csv"))


# In[ ]:


for i in range(len(submission)):
    if test['sentiment'][i] == 'neutral' or len(test['text'][i].split()) < 4:  # neutral postprocessing
        submission.loc[i, 'selected_text'] = test['text'][i]
    else:
        submission.loc[i, 'selected_text'] = all_results[i]


# In[ ]:


submission.head(30)


# In[ ]:


submission["text"] = test["text"]


# In[ ]:


def reverse_preprocessing(text):

    text = text.replace(". . . .", "....")
    text = text.replace(". . .", "...")
    text = text.replace(". .", "..")
    text = text.replace("! ! ! !", "!!!!")
    text = text.replace("! ! !", "!!!")
    text = text.replace("! !", "!!")
    text = text.replace("? ? ? ?", "????")
    text = text.replace("? ? ?", "???")
    text = text.replace("? ?", "??")

    return text


def find_text_idx(text, selected_text):

    text_len = len(text)

    for start_idx in range(text_len):
        if text[start_idx] == selected_text[0]:
            for end_idx in range(start_idx+1, text_len+1):
                contained_text = "".join(text[start_idx: end_idx].split())
                # print("contained_text:", contained_text, "selected_text:", selected_text)
                if contained_text == "".join(selected_text.split()):
                    return start_idx, end_idx

    return None, None


def calculate_spaces(text, selected_text):

    selected_text = " ".join(selected_text.split())
    start_idx, end_idx = find_text_idx(text, selected_text)
    # print("text:", text[start_idx: end_idx], "prediction:", selected_text)

    if start_idx is None:
        start_idx = 0
        print("----------------- error no start idx find ------------------")
        print("text:", text, "prediction:", selected_text)
        print("----------------- error no start idx find ------------------")

    if end_idx is None:
        end_idx = len(text)
        print("----------------- error no end idx find ------------------")
        print("text:", text, "prediction:", selected_text)
        print("----------------- error no end idx find ------------------")

    x = text[:start_idx]
    try:
        if x[-1] == " ":
            x = x[:-1]
    except:
        pass

    l1 = len(x)
    l2 = len(" ".join(x.split()))
    return l1 - l2, start_idx, end_idx


def pp_v2(text, predicted):

    text = str(text).lower()
    predicted = predicted.lower()
    predicted = predicted.strip()

    if len(predicted) == 0:
        return predicted

    predicted = reverse_preprocessing(str(predicted))

    spaces, index_start, index_end = calculate_spaces(text, predicted)

    if spaces == 1:
        if len(text[max(0, index_start-1): index_end+1]) <= 0 or text[max(0, index_start-1): index_end+1][-1] != ".":
            return text[max(0, index_start - 1): index_end]
        else:
            return text[max(0, index_start-1): index_end+1]
    elif spaces == 2:
        return text[max(0, index_start-2): index_end]
    elif spaces == 3:
        return text[max(0, index_start-3): index_end-1]
    elif spaces == 4:
        return text[max(0, index_start-4): index_end-2]
    else:
        return predicted


# In[ ]:


submission["new_selected"] = submission.apply(lambda x: pp_v2(x.text, x.selected_text), axis=1)


# In[ ]:


submission.selected_text = submission["new_selected"]


# In[ ]:


submission[["textID","selected_text"]].to_csv("submission.csv", index=False)


# In[ ]:


submission.head(30)


# In[ ]:




