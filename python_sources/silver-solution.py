#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
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
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from IPython.display import FileLink
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import functional as F


# In[ ]:


get_ipython().system(' ls ../input')


# In[ ]:


MAX_LEN = 192
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 5
ROBERTA_PATH = "../input/roberta-base"

TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f"{ROBERTA_PATH}/vocab.json", 
    merges_file=f"{ROBERTA_PATH}/merges.txt", 
    lowercase=True,
    add_prefix_space=True
)


# In[ ]:


class RoBertaModel(transformers.BertPreTrainedModel):
    def __init__(self):
        model_config = transformers.RobertaConfig.from_pretrained(ROBERTA_PATH)
        model_config.output_hidden_states = True
        super(RoBertaModel, self).__init__(model_config)
        self.roberta = transformers.RobertaModel.from_pretrained(ROBERTA_PATH, config=model_config)
        self.drop_out = nn.Dropout(0.1)
        self.Cov1S = nn.Conv1d(768 * 2, 128 , kernel_size = 2 ,stride = 1 )
        self.Cov1E = nn.Conv1d(768 * 2, 128, kernel_size = 2 ,stride = 1 )
        self.Cov2S = nn.Conv1d(128 , 64 , kernel_size = 2 ,stride = 1)
        self.Cov2E = nn.Conv1d(128 , 64 , kernel_size = 2 ,stride = 1)
        self.lS = nn.Linear(64 , 1)
        self.lE = nn.Linear(64 , 1)
        torch.nn.init.normal_(self.lS.weight, std=0.02)
        torch.nn.init.normal_(self.lE.weight, std=0.02)

        
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.drop_out(out)
        out = out.permute(0,2,1)
        
        same_pad1 = torch.zeros(out.shape[0] , 768*2 , 1).cuda()
        same_pad2 = torch.zeros(out.shape[0] , 128 , 1).cuda()

        out1 = torch.cat((same_pad1 , out), dim = 2)
        out1 = self.Cov1S(out1)
        out1 = torch.cat((same_pad2 , out1), dim = 2)
        out1 = self.Cov2S(out1)
        out1 = F.leaky_relu(out1)
        out1 = out1.permute(0,2,1)
        start_logits = self.lS(out1).squeeze(-1)
        #print(start_logits.shape)

        out2 = torch.cat((same_pad1 , out), dim = 2)
        out2 = self.Cov1E(out2)
        out2 = torch.cat((same_pad2 , out2), dim = 2)
        out2 = self.Cov2E(out2)
        out2 = F.leaky_relu(out2)
        out2 = out2.permute(0,2,1)
        end_logits = self.lE(out2).squeeze(-1)
        #print(end_logits.shape)


        return start_logits, end_logits


# In[ ]:


def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
    tweet = " " + " ".join(str(tweet).split())
    selected_text = ' ' + " ".join(str(selected_text).split())

    len_st = len(selected_text) - 1
    idx0 = None
    idx1 = None

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
        if " " + tweet[ind: ind+len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st-1
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
    
    input_ids = [0] + input_ids_orig + [2] + [2] +  [sentiment_id[sentiment]] + [2]
    token_type_ids = [0] * (len(input_ids_orig) + 1) + [0, 0, 0, 0] 
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] + tweet_offsets + [(0, 0)]*4
    targets_start += 1
    targets_end += 1

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


# In[ ]:


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data(
            self.tweet[item], 
            self.selected_text[item], 
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )

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


def calculate_jaccard_score(
    original_tweet, 
    target_string, 
    sentiment_val, 
    best_idxs,
    offsets,
    verbose=False):
    
    if best_idxs[1] < best_idxs[0]:
        best_idxs[1] = best_idxs[0]
    
    filtered_output  = ""
    for ix in range(best_idxs[0], best_idxs[1] + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "

    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    if sentiment_val != "neutral" and verbose == True:
        if filtered_output.strip().lower() != target_string.strip().lower():
            print("********************************")
            print(f"Output= {filtered_output.strip()}")
            print(f"Target= {target_string.strip()}")
            print(f"Tweet= {original_tweet.strip()}")
            print("********************************")

    jac = 0
    return jac, filtered_output


# In[ ]:




device = torch.device("cuda")

modell1 = RoBertaModel()
modell1.to(device)
modell1.load_state_dict(torch.load("/kaggle/input/myroberta-fold/model_0.bin"))
modell1.eval()
'''
modell2 = RoBertaModel()
modell2.to(device)
modell2.load_state_dict(torch.load("/kaggle/input/myroberta-fold/model_1.bin"))
modell2.eval()

'''
modell3 = RoBertaModel()
modell3.to(device)
modell3.load_state_dict(torch.load("/kaggle/input/myroberta-fold/model_2.bin"))
modell3.eval()
'''
modell4 = RoBertaModel()
modell4.to(device)
modell4.load_state_dict(torch.load("/kaggle/input/myroberta-fold/model_3.bin"))
modell4.eval()
'''
modell5 = RoBertaModel()
modell5.to(device)
modell5.load_state_dict(torch.load("/kaggle/input/myroberta-fold/model_4.bin"))
modell5.eval()




device = torch.device("cuda")
modell6 = RoBertaModel()
modell6.to(device)
modell6.load_state_dict(torch.load("/kaggle/input/single2/modelbest_2.bin"))
modell6.eval()

device = torch.device("cuda")
modell7 = RoBertaModel()
modell7.to(device)
modell7.load_state_dict(torch.load("/kaggle/input/sinlge3/modelbest_3.bin"))
modell7.eval()


# In[ ]:




device = torch.device("cuda")
model1 = RoBertaModel()
model1.to(device)
model1.load_state_dict(torch.load("/kaggle/input/roberta-conv-8fold/trained_models/model_0.bin"))
model1.eval()

model2 = RoBertaModel()
model2.to(device)
model2.load_state_dict(torch.load("/kaggle/input/roberta-conv-8fold/trained_models/model_1.bin"))
model2.eval()

model3 = RoBertaModel()
model3.to(device)
model3.load_state_dict(torch.load("/kaggle/input/roberta-conv-8fold/trained_models/model_2.bin"))
model3.eval()

model4 = RoBertaModel()
model4.to(device)
model4.load_state_dict(torch.load("/kaggle/input/roberta-conv-8fold/trained_models/model_3.bin"))
model4.eval()

model5 = RoBertaModel()
model5.to(device)
model5.load_state_dict(torch.load("/kaggle/input/roberta-conv-8fold/trained_models/model_4.bin"))
model5.eval()

model6 = RoBertaModel()
model6.to(device)
model6.load_state_dict(torch.load("/kaggle/input/roberta-conv-8fold/trained_models/model_5.bin"))
model6.eval()
'''
model7 = RoBertaModel()
model7.to(device)
model7.load_state_dict(torch.load("/kaggle/input/roberta-conv-8fold/trained_models/model_6.bin"))
model7.eval()
'''
model8 = RoBertaModel()
model8.to(device)
model8.load_state_dict(torch.load("/kaggle/input/roberta-conv-8fold/trained_models/model_7.bin"))
model8.eval()


# In[ ]:


df_test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values


final_output = []
test_dataset = TweetDataset(
        tweet=df_test.text.values,
        sentiment=df_test.sentiment.values,
        selected_text=df_test.selected_text.values
    )

data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=VALID_BATCH_SIZE,
    num_workers=1
)


# In[ ]:


def get_best_start_end_idxs(_start_logits, _end_logits):
    best_logit = -1000
    best_idxs = None
    for start_idx, start_logit in enumerate(_start_logits):
        for end_idx, end_logit in enumerate(_end_logits[start_idx:]):
            logit_sum = (start_logit + end_logit).item()
            if logit_sum > best_logit:
                best_logit = logit_sum
                best_idxs = (start_idx, start_idx+end_idx)
    return best_idxs


# In[ ]:


# losses = utils.AverageMeter()
# tk0 = tqdm(data_loader, total=len(data_loader))
fin_output_start = []
fin_output_end = []
fin_padding_lens = []
fin_tweet_tokens = []
fin_orig_sentiment = []
fin_orig_tweet = []

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
    
    
    
    o11, o12 = modell1(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )
    '''
    o21, o22 = modell2(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )
    '''

    o31, o32 = modell3(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )
    '''
    o41, o42 = modell4(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )
    '''
    o51, o52 = modell5(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )
    
    o61, o62 = modell6(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )
    
    o71, o72 = modell7(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )
    
    
    
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

    outputs_start6, outputs_end6 = model6(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )
    '''
    outputs_start7, outputs_end7 = model7(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )
    '''
    outputs_start8, outputs_end8 = model8(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
        )
    outputs_start1 = torch.softmax(outputs_start1, dim=1).cpu().detach().numpy()
    outputs_end1 = torch.softmax( outputs_end1, dim=1).cpu().detach().numpy()
        
    outputs_start2 = torch.softmax(outputs_start2, dim=1).cpu().detach().numpy()
    outputs_end2 = torch.softmax( outputs_end2, dim=1).cpu().detach().numpy()
        
    outputs_start3 = torch.softmax(outputs_start3, dim=1).cpu().detach().numpy()
    outputs_end3 = torch.softmax( outputs_end3, dim=1).cpu().detach().numpy()
        
    outputs_start4 = torch.softmax(outputs_start4, dim=1).cpu().detach().numpy()
    outputs_end4 = torch.softmax( outputs_end4, dim=1).cpu().detach().numpy()
        
    outputs_start5 = torch.softmax(outputs_start5, dim=1).cpu().detach().numpy()
    outputs_end5 = torch.softmax( outputs_end5, dim=1).cpu().detach().numpy()
        
    outputs_start6 = torch.softmax(outputs_start6, dim=1).cpu().detach().numpy()
    outputs_end6 = torch.softmax( outputs_end6, dim=1).cpu().detach().numpy()
    '''
    outputs_start7 = torch.softmax(outputs_start7, dim=1).cpu().detach().numpy()
    outputs_end7 = torch.softmax( outputs_end7, dim=1).cpu().detach().numpy()
    '''
    outputs_start8 = torch.softmax(outputs_start8, dim=1).cpu().detach().numpy()
    outputs_end8 = torch.softmax( outputs_end8, dim=1).cpu().detach().numpy()
    
    o31=torch.softmax(o31, dim=1).cpu().detach().numpy()
    o32=torch.softmax(o32, dim=1).cpu().detach().numpy()
    '''
    o41=torch.softmax(o41, dim=1).cpu().detach().numpy()
    o42=torch.softmax(o42, dim=1).cpu().detach().numpy()
    
    o21=torch.softmax(o21, dim=1).cpu().detach().numpy()
    o22=torch.softmax(o22, dim=1).cpu().detach().numpy()
    '''
    
    o11=torch.softmax(o11, dim=1).cpu().detach().numpy()
    o12=torch.softmax(o12, dim=1).cpu().detach().numpy()
    

    
    o51=torch.softmax(o51, dim=1).cpu().detach().numpy()
    o52=torch.softmax(o52, dim=1).cpu().detach().numpy()
    
    o61=torch.softmax(o61, dim=1).cpu().detach().numpy()
    o62=torch.softmax(o62, dim=1).cpu().detach().numpy()
    
    o71=torch.softmax(o71, dim=1).cpu().detach().numpy()
    o72=torch.softmax(o72, dim=1).cpu().detach().numpy()
    
    #o1 = 0.1*o11 + 0.08*o21 + 0.18*o31 + 0.18*o41 + 0.18*o51 + 0.12*o61+0.08*o71
    #o2 = 0.1*o12 + 0.08*o22 + 0.18*o32 + 0.18*o42 + 0.18*o52 + 0.12*o62+0.08*o72
    
    outputs_start = (0.025*o31 + 0.1*o71 + 0.15*o51 + 0.12*o11 + 0.025*o61 +outputs_start1*0.15+ outputs_start2*0.08+ outputs_start3*0.15 + outputs_start4*0.1 + outputs_start5*0.15+ outputs_start6*0.05+ outputs_start8*0.025)
    outputs_end = (0.025*o32 + 0.1*o72 + 0.15*o52+ 0.12*o12 + 0.025*o62 +outputs_end1*0.15+ outputs_end2*0.08 + outputs_end3*0.15+ outputs_end4*0.1+ outputs_end5*0.05+ outputs_end6*0.15+ outputs_end8*0.025)    
    
    #outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
    #outputs_end = torch.softmax( outputs_end, dim=1).cpu().detach().numpy()
    
    for px, tweet in enumerate(orig_tweet):
        selected_tweet = orig_selected[px]
        tweet_sentiment = sentiment[px]
        _, output_sentence = calculate_jaccard_score(
            original_tweet=tweet,
            target_string=selected_tweet,
            sentiment_val=tweet_sentiment,
            best_idxs=get_best_start_end_idxs(outputs_start[px, :],outputs_end[px, :]),
            offsets=offsets[px]
        )
        final_output.append(output_sentence)


# In[ ]:



def get_indicies(text, find_text):
    indicies = []
    index = 0
    while index < len(text):
            index = text.find(find_text, index)
            if index == -1:
                break
            indicies.append(index)
            index += len(find_text) # +2 because len('ll') == 2
    return indicies       

def add_noise(tweet, pred, sentiment):
    pred_idx = tweet.find(pred)
    len_pred = len(pred)
    ds_idxs = get_indicies(tweet, '  ')
    ts_idxs = get_indicies(tweet, '   ')
    
    
    new_sl = pred
    # Double space and start space
    if ((sentiment != 'neutral') & (len(ds_idxs) > 0)):
        if (pred_idx > ds_idxs[0]) & (tweet[0] == ' ') & (len(pred) != len(tweet) & (pred_idx > 1)):
            new_sl = tweet[pred_idx-2:pred_idx+len_pred]
    # Triple space and starting space
    if ((sentiment != 'neutral') & (len(ts_idxs) > 0)):
        if (pred_idx > ts_idxs[0]) & (tweet[0] == ' ') & (len(pred) != len(tweet) & (pred_idx > 1)):
            new_sl = tweet[pred_idx-2-1:pred_idx+len_pred-1]
    # 2 double spaces
    if ((sentiment != 'neutral') & (len(ds_idxs) == 2)):
        if (pred_idx > ds_idxs[0]) & (pred_idx > ds_idxs[1]) & (len(pred) != len(tweet)):
            new_sl = tweet[pred_idx-2-1:pred_idx+len_pred-1]
    

            
    # Clean up neutral if first word starts with an underscore
    if ((sentiment == 'neutral') & (tweet[0] == '_') & (len(tweet.split())>1)):
        first_word_len = len(tweet.split()[0])
        new_sl = tweet[0+first_word_len+1:]
        
    
    return new_sl


# In[ ]:




# try again with lists
text_list = df_test.text.values.tolist()
pred_list = final_output
sentiment_list = df_test.sentiment.values.tolist()


# In[ ]:


new_preds = []

for i in range(0,len(text_list)):
    new_pred = add_noise(text_list[i], pred_list[i].strip(), sentiment_list[i])
    new_preds.append(new_pred)


# In[ ]:


sample = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
sample.loc[:, 'selected_text'] = new_preds


# In[ ]:




sample['selected_text'] = sample['selected_text'].apply(lambda x: x.replace('!!!!', '!') if len(x.split())==1 else x)
sample['selected_text'] = sample['selected_text'].apply(lambda x: x.replace('..', '.') if len(x.split())==1 else x)
sample['selected_text'] = sample['selected_text'].apply(lambda x: x.replace('...', '.') if len(x.split())==1 else x)

sample.to_csv("submission.csv", index=False)


# In[ ]:


sample.head(50)


# In[ ]:




