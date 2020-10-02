#!/usr/bin/env python
# coding: utf-8

# # QUEST BERT-base Pytorch minimalistic (training)
# 
# This is a PyTorch version of @akensert's brilliant TF kernel. Here I train all 5 folds, and have a separate inference kernel over [here]().
# 
# Feel free to comment if you have any questions!

# Versions:
# 
# V19: Fixed f-strings, trying linear scheduler with warmup of 0.05, commonly used for fine-tuning BERT. Also using AdamW, also commonly used for fine-tuning BERT.
# 
# V18: separated CV display into different code cell, because of output overflow/cutoff issues.
# 
# V17: Updated to train all 5-folds with GroupKFold, print all fold scores and CV score at end, cleaned up code so it's a little clearer.

# In[ ]:


import os

os.system('pip install ../input/sacremoses/sacremoses-master/ > /dev/null')
os.system('pip install ../input/transformers/transformers-master/ > /dev/null')


# # Imports

# In[ ]:


import transformers, sys, os, gc
import numpy as np, pandas as pd, math
import torch, random, os, multiprocessing, glob
import torch.nn.functional as F
import torch, time

from sklearn.model_selection import GroupKFold
# from ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold <--- if you want to use Neuron Engineer's CV
from scipy.stats import spearmanr
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer, BertModel, BertPreTrainedModel, BertForSequenceClassification, BertConfig,
    WEIGHTS_NAME, CONFIG_NAME, AdamW, get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup,
)

from tqdm import tqdm_notebook
print(transformers.__version__)


# In[ ]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 42
seed_everything(SEED)


# In[ ]:


train = pd.read_csv("../input/google-quest-challenge/train.csv",)
test = pd.read_csv("../input/google-quest-challenge/test.csv",)

target_cols = ['question_asker_intent_understanding', 'question_body_critical', 
               'question_conversational', 'question_expect_short_answer', 
               'question_fact_seeking', 'question_has_commonly_accepted_answer', 
               'question_interestingness_others', 'question_interestingness_self', 
               'question_multi_intent', 'question_not_really_a_question', 
               'question_opinion_seeking', 'question_type_choice',
               'question_type_compare', 'question_type_consequence',
               'question_type_definition', 'question_type_entity', 
               'question_type_instructions', 'question_type_procedure', 
               'question_type_reason_explanation', 'question_type_spelling', 
               'question_well_written', 'answer_helpful',
               'answer_level_of_information', 'answer_plausible', 
               'answer_relevance', 'answer_satisfaction', 
               'answer_type_instructions', 'answer_type_procedure', 
               'answer_type_reason_explanation', 'answer_well_written']


# # Helpers

# In[ ]:


# helpers

# From the Ref Kernel's
from math import floor, ceil

def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
        
    segments = []
    first_sep = True
    current_segment_id = 0
    
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False 
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def _trim_input(title, question, answer, max_sequence_length=512, t_max_len=30, q_max_len=239, a_max_len=239):
    
    #293+239+30 = 508 + 4 = 512
    
    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    
    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len+4) > max_sequence_length:
        
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len
      
        if a_max_len > a_len:
            a_new_len = a_len 
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len
            
            
        if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d"%(max_sequence_length, (t_new_len + a_new_len + q_new_len + 4)))
        
        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]
    
    return t, q, a

def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    
    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]

def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm_notebook(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(t, q, a, max_sequence_length)
        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(), 
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    ]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


# Check the below cell to see how to add colors to your output! (thanks to my teammate @ecdrid)

# In[ ]:


# Ref From Stack Overflow
# Make sure to end with bcolors.ENDC otherwise color is used in the rest of the terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# # PyTorch Dataset, training, validation, testing functions

# In[ ]:


# dataset and other utils..

class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, lengths, labels = None):
        
        self.inputs = inputs
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
        self.lengths = lengths

    def __getitem__(self, idx):
        
        input_ids       = self.inputs[0][idx]
        input_masks     = self.inputs[1][idx]
        input_segments  = self.inputs[2][idx]
        lengths         = self.lengths[idx]
        if self.labels is not None: # targets
            labels = self.labels[idx]
            return input_ids, input_masks, input_segments, labels, lengths
        return input_ids, input_masks, input_segments, lengths

    def __len__(self):
        return len(self.inputs[0])

def train_model(train_loader, optimizer, criterion, scheduler=None):
    
    model.train()
    avg_loss = 0.
    tk0 = tqdm_notebook(enumerate(train_loader))
    
    for idx, batch in tk0:
        
        input_ids, input_masks, input_segments, labels, _ = batch
        input_ids, input_masks, input_segments, labels = input_ids.to(device), input_masks.to(device), input_segments.to(device), labels.to(device)            
        
        output_train = model(input_ids = input_ids.long(),
                             labels = None,
                             attention_mask = input_masks,
                             token_type_ids = input_segments,
                            )
        logits = output_train[0] #output preds
        
        loss = criterion(logits, labels)
        loss.backward()
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        
        avg_loss += loss.item() / len(train_loader)
        del input_ids, input_masks, input_segments, labels

    torch.cuda.empty_cache()
    gc.collect()
    return avg_loss

def val_model(val_loader, val_shape, batch_size=8):

    avg_val_loss = 0.
    model.eval() # eval mode
    
    valid_preds = np.zeros((val_shape, 30))
    original = np.zeros((val_shape, 30))
    
    tk0 = tqdm_notebook(enumerate(val_loader))
    with torch.no_grad():
        
        for idx, batch in tk0:
            input_ids, input_masks, input_segments, labels, _ = batch
            input_ids, input_masks, input_segments, labels = input_ids.to(device), input_masks.to(device), input_segments.to(device), labels.to(device)            
            
            output_val = model(input_ids = input_ids.long(),
                             labels = None,
                             attention_mask = input_masks,
                             token_type_ids = input_segments,
                            )
            logits = output_val[0] #output preds
            
            avg_val_loss += criterion(logits, labels).item() / len(val_loader)
            valid_preds[idx*batch_size : (idx+1)*batch_size] = logits.detach().cpu().squeeze().numpy()
            original[idx*batch_size : (idx+1)*batch_size]    = labels.detach().cpu().squeeze().numpy()
        
        score = 0
        preds = torch.sigmoid(torch.tensor(valid_preds)).numpy()
        
        spear = []
        for i in range(preds.shape[1]):
            spear.append(np.nan_to_num(spearmanr(original[:, i], preds[:, i]).correlation))
        score = np.mean(spear)
    return avg_val_loss, score

def predict_result(model, test_loader, batch_size=32):
    
    test_preds = np.zeros((len(test), 30))
    
    model.eval();
    tk0 = tqdm_notebook(enumerate(test_loader))
    for idx, x_batch in tk0:
        with torch.no_grad():
            outputs = model(input_ids = x_batch[0].to(device), 
                            labels = None, 
                            attention_mask = x_batch[1].to(device),
                            token_type_ids = x_batch[2].to(device),
                           )
            predictions = outputs[0]
            test_preds[idx*batch_size : (idx+1)*batch_size] = predictions.detach().cpu().squeeze().numpy()

    output = torch.sigmoid(torch.tensor(test_preds)).numpy()        
    return output


# # BERT model configuration

# The below hidden code cell contains code to add a pooling layer to the model if you desire to do so, as according to @ecdrid's discussion post over [here](). 

# In[ ]:


# class BertForSequenceClassification(BertPreTrainedModel):
#     r"""
#         **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
#             Labels for computing the sequence classification/regression loss.
#             Indices should be in ``[0, ..., config.num_labels - 1]``.
#             If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
#             If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

#     Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
#         **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
#             Classification (or regression if config.num_labels==1) loss.
#         **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
#             Classification (or regression if config.num_labels==1) scores (before SoftMax).
#         **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
#             list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
#             of shape ``(batch_size, sequence_length, hidden_size)``:
#             Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#         **attentions**: (`optional`, returned when ``config.output_attentions=True``)
#             list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

#     Examples::

#         tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
#         input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
#         labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
#         outputs = model(input_ids, labels=labels)
#         loss, logits = outputs[:2]

#     """
#     def __init__(self, config):
#         super(BertForSequenceClassification, self).__init__(config)
#         self.num_labels = config.num_labels

#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

#         self.init_weights()

#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
#                 position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

#         outputs = self.bert(input_ids,
#                             attention_mask=attention_mask,
#                             token_type_ids=token_type_ids,
#                             position_ids=position_ids,
#                             head_mask=head_mask,
#                             inputs_embeds=inputs_embeds)

#         sequence_output = outputs[0]
#         pooled_output = torch.mean(sequence_output,axis=1) # Global Average Pooling (batch_size,hidden_size) 

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

#         if labels is not None:
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs

#         return outputs  # (loss), logits, (hidden_states), (attentions)


# Below is the BERT configuration code, which includes setting up the tokenizer class, and the configuration class. We are using a `bert-base-uncased` model, with 30 labels as associated with this task.

# In[ ]:


tokenizer = BertTokenizer.from_pretrained("../input/pretrained-bert-models-for-pytorch/bert-base-uncased-vocab.txt")
input_categories = list(train.columns[[1,2,5]]); input_categories

bert_model_config = '../input/pretrained-bert-models-for-pytorch/bert-base-uncased/bert_config.json'
bert_config = BertConfig.from_json_file(bert_model_config)
bert_config.num_labels = 30

bert_model = 'bert-base-uncased'
do_lower_case = 'uncased' in bert_model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_model_file = 'bert_pytorch.bin'

print(do_lower_case, bert_model, device, output_model_file)


# # Training and validation code

# Below, we define the folds, and go through the folds, train and validate the model. The final validation fold scores are printed out at the end and the models are saved for later inference.

# In[ ]:


from transformers import AdamW
NUM_FOLDS = 5  
BATCH_SIZE = 8
epochs = 4    # change this

# Define CV
kf = GroupKFold(n_splits = NUM_FOLDS)
scores = []

y_train = train[target_cols].values # dummy

print(bcolors.FAIL, f"For Every Fold, Train {epochs} Epochs", bcolors.ENDC)

for fold, (train_index, val_index) in enumerate(kf.split(X=train.question_body, groups=train.question_body)):
        
    print(bcolors.HEADER, "Current Fold:", fold, bcolors.ENDC)

    train_df, val_df = train.iloc[train_index], train.iloc[val_index]
    print("Train and Valid Shapes are", train_df.shape, val_df.shape)
    
    print(bcolors.HEADER, "Preparing train datasets....", bcolors.ENDC)
    
    inputs_train = compute_input_arays(train_df, input_categories, tokenizer, max_sequence_length=512)
    outputs_train = compute_output_arrays(train_df, columns = target_cols)
    outputs_train = torch.tensor(outputs_train, dtype=torch.float32)
    lengths_train = np.argmax(inputs_train[0] == 0, axis=1)
    lengths_train[lengths_train == 0] = inputs_train[0].shape[1]
    
    print(bcolors.HEADER, "Preparing Valid datasets....", bcolors.ENDC)
    
    inputs_valid = compute_input_arays(val_df, input_categories, tokenizer, max_sequence_length=512)
    outputs_valid = compute_output_arrays(val_df, columns = target_cols)
    outputs_valid = torch.tensor(outputs_valid, dtype=torch.float32)
    lengths_valid = np.argmax(inputs_valid[0] == 0, axis=1)
    lengths_valid[lengths_valid == 0] = inputs_valid[0].shape[1]
   
    
    print(bcolors.HEADER, "Preparing Dataloaders....", bcolors.ENDC)

    train_set    = QuestDataset(inputs=inputs_train, lengths=lengths_train, labels=outputs_train)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    valid_set    = QuestDataset(inputs=inputs_valid, lengths=lengths_valid, labels=outputs_valid)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    # Create model
    model = BertForSequenceClassification.from_pretrained('../input/pretrained-bert-models-for-pytorch/bert-base-uncased/', config=bert_config);
    model.zero_grad();
    model.to(device);
    torch.cuda.empty_cache()
    model.train();
    
    best_score      = -1.
    best_param_score = None
    best_param_epoch = None

    optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-7, correct_bias=False)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.05, num_training_steps= epochs*len(train_loader))
    #scheduler=None
    
    print("Training....")
    
    for epoch in tqdm_notebook(range(epochs)):
        
        torch.cuda.empty_cache()
        
        start_time   = time.time()
        avg_loss     = train_model(train_loader, optimizer, criterion, scheduler)
        avg_val_loss, score = val_model(valid_loader, val_shape=val_df.shape[0])
        elapsed_time = time.time() - start_time

        print(bcolors.WARNING, 'Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t score={:.6f} \t time={:.2f}s'.format(
            epoch + 1, epochs, avg_loss, avg_val_loss, score, elapsed_time),
        bcolors.ENDC
        )

        if best_score < score:
            best_score = score
            best_param_score = model.state_dict()
            best_epoch_score = epoch

    print(bcolors.WARNING, f'Best model came from Epoch {best_epoch_score} with score of {best_score}',bcolors.ENDC)
    model.load_state_dict(best_param_score)
    torch.save(best_param_score, 'best_param_score_{}.pt'.format(fold+1))
    scores.append(best_score)

    del train_df, val_df, model, optimizer, criterion, scheduler
    torch.cuda.empty_cache()
    del valid_loader, train_loader, valid_set, train_set
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()


# # Results:

# In[ ]:


print(bcolors.HEADER,'Individual Fold Scores:',bcolors.ENDC)
print(scores)
print(bcolors.HEADER,f'QUEST BERT-base {NUM_FOLDS}-fold CV score: ',np.mean(scores),bcolors.ENDC)

