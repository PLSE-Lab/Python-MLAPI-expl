#!/usr/bin/env python
# coding: utf-8

# ## Huggingface meets Fastai 
# 
# ![](https://huggingface.co/landing/assets/transformers-docs/huggingface_logo.svg)
# ![](https://docs.fast.ai/images/company_logo.png)

# This notebook shows training Bert model from `transformers` library  with `fastai` library interface. By doing so we get to use goodies like `lr_finder()`, `gradual_unfreezing`, `Callbacks`, `to_fp16()` and other customizations if necessary.
# 
# Since finetuning requires loading pretrained models from the internet this notebook can't be submitted directly but required models or output of this notebook can be saved as a Kaggle dataset for further submission.

# In[ ]:


from fastai.core import *
import transformers; transformers.__version__


# In[ ]:


KAGGLE_WORKING = Path("/kaggle/working")


# In[ ]:


path = Path("../input/tweet-sentiment-extraction/")
train_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')
train_df = train_df.dropna().reset_index(drop=True)


# In[ ]:


# max approx sequence length
(max(len(o.split()) for o in array(train_df['text'])), 
max(len(o.split()) for o in array(test_df['text'])))


# In[ ]:


train_df.sentiment.value_counts()


# ### SQUAD Q/A Data Prep
# 
# Here we are creating a SQUAD format dataset as we will leverage data prep utilities from `transformers` library. We could use SQUAD V1 or V2 for preparing data, but this dataset doesn't require `is_impossible` as it doesn't have any adversarial questions. Questions are coming from sentiment of the tweets; being either `positive` or `negative`. The idea has been taken from other kernels, so thanks!

# In[ ]:


# an example for SQUAD json data format
squad_sample = {
    "version": "v2.0",
    "data": [
        {
            "title": "Beyonc\u00e9",
            "paragraphs": [
                {
                    "qas": [
                        {
                            "question": "When did Beyonce start becoming popular?",
                            "id": "56be85543aeaaa14008c9063",
                            "answers": [
                                {
                                    "text": "in the late 1990s",
                                    "answer_start": 269
                                }
                            ],
                            "is_impossible": False
                        }
                    ],
                    "context": "Beyonc\u00e9 Giselle Knowles-Carter (/bi\u02d0\u02c8j\u0252nse\u026a/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyonc\u00e9's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles \"Crazy in Love\" and \"Baby Boy\"."
                }
            ]
        }
    ]
}


# In[ ]:


def get_answer_start(context, answer):
    len_a = len(answer)
    for i, _ in enumerate(context):
        if context[i:i+len_a] == answer: return i
    raise Exception("No overlapping segment found")


# In[ ]:


def generate_qas_dict(text_id, context, answer, question):
    qas_dict = {}
    qas_dict['question'] = question
    qas_dict['id'] = text_id
    qas_dict['is_impossible'] = False
    
    if answer is None: 
        qas_dict['answers'] = []
    else: 
        answer_start = get_answer_start(context, answer)
        qas_dict['answers'] = [{"text":answer, "answer_start":answer_start}]
    return qas_dict


# In[ ]:


def create_squad_from_df(df):
    data_dicts = []
    for _, row in df.iterrows():
        text_id = row['textID']
        context = row['text']
        answer =  row['selected_text'] if 'selected_text' in row else None
        question = row['sentiment']

        qas_dict = generate_qas_dict(text_id, context, answer, question)
        data_dict = {"paragraphs" : [{"qas" : [qas_dict], "context":context}]}
        data_dict['title'] = text_id
        data_dicts.append(data_dict)

    return {"version": "v2.0", "data": data_dicts}


# In[ ]:


# train_no_neutral_df = train_df[train_df.sentiment != 'neutral'].reset_index(drop=True)
# test_no_neutral_df = test_df[test_df.sentiment != 'neutral'].reset_index(drop=True)
# train_no_neutral_df.shape, test_no_neutral_df.shape


# ### Create KFold Validation
# 
# For training we will be using only positive and negative tweets, as neutral tweets have a score of `~0.97` when submitted as it is. Also, positive and negative tweets are balanced so I am here using a vanilla KFold cross validation approach. Trained models for all folds can be used for further ensembling if needed.

# In[ ]:


from sklearn.model_selection import KFold


# In[ ]:


# create 5 fold trn-val splits with only positive/negative tweets
os.makedirs("squad_data", exist_ok=True)
kfold = KFold(5, shuffle=True, random_state=42)
fold_idxs = list(kfold.split(train_df))

for i, (trn_idx, val_idx) in enumerate(fold_idxs):
    _trn_fold_df = train_df.iloc[trn_idx]
    _val_fold_df = train_df.iloc[val_idx]
    train_squad_data = create_squad_from_df(_trn_fold_df)
    valid_squad_data = create_squad_from_df(_val_fold_df)
    with open(f"squad_data/train_squad_data_{i}.json", "w") as f: f.write(json.dumps(train_squad_data))
    with open(f"squad_data/valid_squad_data_{i}.json", "w") as f: f.write(json.dumps(valid_squad_data))


# In[ ]:


# create for test 
test_squad_data =  create_squad_from_df(test_df)
with open("squad_data/test_squad_data.json", "w") as f: f.write(json.dumps(test_squad_data))


# ### Check SQUAD json

# In[ ]:


_train_dict = json.loads(open(KAGGLE_WORKING/'squad_data/train_squad_data_0.json').read())


# In[ ]:


sample_idx = np.random.choice(range(len(_train_dict['data'])))
_train_dict['data'][sample_idx]


# In[ ]:


textid = _train_dict['data'][sample_idx]['paragraphs'][0]['qas'][0]['id']


# In[ ]:


train_df[train_df.textID == textid]


# ### Data
# 
# Since we are training a BERT model we will use bert tokenizer for `bert-base-cased`. You can use `Auto*` classes from `transformers` library to load and train with any model available. For demonstration I am training with `foldnum=0`.

# In[ ]:


from fastai.text import *
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForQuestionAnswering
from transformers.data.processors.squad import (SquadResult, SquadV1Processor, SquadV2Processor,
                                                SquadExample, squad_convert_examples_to_features)


# In[ ]:


PRETRAINED_TYPE = 'roberta-base'
# PRETRAINED_TYPE = 'distilbert-base-uncased'


# In[ ]:


processor = SquadV2Processor()
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TYPE, do_lower_case=True)
os.makedirs(KAGGLE_WORKING/f'{PRETRAINED_TYPE}-tokenizer', exist_ok=True)
tokenizer.save_pretrained(KAGGLE_WORKING/f'{PRETRAINED_TYPE}-tokenizer')


# In[ ]:


max_seq_length = 128
max_query_length = 10

def get_dataset(examples, is_training):
    return squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        doc_stride=200,
        max_seq_length=max_seq_length,
        max_query_length=10,
        is_training=is_training,
        return_dataset="pt",
        threads=defaults.cpus,
    )


# ### Dataset

# In[ ]:


class SQUAD_Dataset(Dataset):
    def __init__(self, dataset_tensors, examples, features, is_training=True):
        self.dataset_tensors = dataset_tensors
        self.examples = examples
        self.features = features
        self.is_training = is_training
        
        
    def __getitem__(self, idx):
        'fastai requires (xb, yb) to return'
        'AutoModel handles loss computation in forward hence yb will be None'
        input_ids = self.dataset_tensors[0][idx]
        attention_mask = self.dataset_tensors[1][idx]
        token_type_ids = self.dataset_tensors[2][idx]
        xb = (input_ids, attention_mask, token_type_ids)
        if self.is_training: 
            start_positions = self.dataset_tensors[3][idx]
            end_positions = self.dataset_tensors[4][idx]
        yb = [start_positions, end_positions]
        return xb, yb
    
    def __len__(self): return len(self.dataset_tensors[0])


# In[ ]:


def get_fold_ds(foldnum):
    data_dir = "/kaggle/working/squad_data"
    train_filename = f"train_squad_data_{foldnum}.json"
    valid_filename = f"valid_squad_data_{foldnum}.json"
    test_filename = "test_squad_data.json"
    
    # tokenize
    train_examples = processor.get_train_examples(data_dir, train_filename)
    valid_examples = processor.get_train_examples(data_dir, valid_filename)
    test_examples = processor.get_dev_examples(data_dir, test_filename)

    # create tensor dataset
    train_features, train_dataset = get_dataset(train_examples, True)
    valid_features, valid_dataset = get_dataset(valid_examples, True)
    test_features, test_dataset = get_dataset(test_examples, False)
    
    # create pytorch dataset
    train_ds = SQUAD_Dataset(train_dataset.tensors, train_examples, train_features)
    valid_ds = SQUAD_Dataset(valid_dataset.tensors, valid_examples, valid_features)
    test_ds = SQUAD_Dataset(test_dataset.tensors, test_examples, test_features, False)
    
    return train_ds, valid_ds, test_ds    


# ### DataAugmentor & Dataset

# In[ ]:


#export
class TSEDataAugmentor():

    def __init__(self, tokenizer, input_ids, attention_mask, start_position, end_position, token_to_orig_map): 

        self.tokenizer = tokenizer 
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        
        # initial answer start and end positions
        self.ans_start_pos, self.ans_end_pos = start_position.item(), end_position.item()
                
        # initial context start and end positions
        self.token_to_orig_map = token_to_orig_map
        self.context_start_pos, self.context_end_pos = min(token_to_orig_map), max(token_to_orig_map)

        
    
    # left and right indexes excluding answer tokens and eos token
    @property
    def left_idxs(self): return np.arange(self.context_start_pos, self.ans_start_pos)
    
    @property
    def right_idxs(self): return np.arange(self.ans_end_pos+1, self.context_end_pos+1)
    
    @property
    def left_right_idxs(self): return np.concatenate([self.left_idxs, self.right_idxs])
    
    @property
    def rand_left_idx(self): return np.random.choice(self.left_idxs) if self.left_idxs.size > 0 else None
    
    @property
    def rand_right_idx(self): return np.random.choice(self.right_idxs) if self.right_idxs.size > 0 else None
        
    
    
    def right_truncate(self, right_idx):
        """
        Truncate context from random right index to beginning, answer pos doesn't change
        Note: token_type_ids NotImplemented
        """
        if not right_idx: raise Exception("Right index can't be None")
        
        # clone for debugging
        new_input_ids = self.input_ids.clone()
        nopad_input_ids = new_input_ids[self.attention_mask.bool()]
        
        # truncate from right idx to beginning - add eos_token_id to end
        truncated = torch.cat([nopad_input_ids[:right_idx+1], tensor([self.tokenizer.eos_token_id])])
        
        # pad new context until size are equal
        # replace original input context with new
        n_pad = len(nopad_input_ids) - len(truncated)
        new_context = F.pad(truncated, (0,n_pad), value=self.tokenizer.pad_token_id)
        new_input_ids[:self.context_end_pos+2] = new_context
        
        
        # find new attention mask, update new context end position (exclude eos token)
        # Note: context start doesn't change since we don't manipulate question
        new_attention_mask = tensor([1 if i != 1 else 0 for i in new_input_ids])
        new_context_end_pos = torch.where(new_attention_mask)[0][-1].item() - 1 
        self.context_end_pos = new_context_end_pos
        
        # update input_ids and attention_masks
        self.input_ids = new_input_ids
        self.attention_mask = new_attention_mask
        
        return self.input_ids, self.attention_mask, (tensor(self.ans_start_pos), tensor(self.ans_end_pos))

    def random_right_truncate(self):
        right_idx = self.rand_right_idx
        if right_idx: self.right_truncate(right_idx)
    
    
    def left_truncate(self, left_idx):
        """
        Truncate context from random left index to end, answer pos changes too
        Note: token_type_ids NotImplemented
        """
        
        if not left_idx: raise Exception("Left index can't be None")
        
        # clone for debugging
        new_input_ids = self.input_ids.clone()
        
        # pad new context until size are equal
        # replace original input context with new

        n_pad = len(new_input_ids[self.context_start_pos:]) - len(new_input_ids[left_idx:])
        
        new_context = F.pad(new_input_ids[left_idx:], (0,n_pad), value=self.tokenizer.pad_token_id)
        
        new_input_ids[self.context_start_pos:] = new_context
        
                
        # find new attention mask, update new context end position (exclude eos token)
        # Note: context start doesn't change since we don't manipulate question
        new_attention_mask = tensor([1 if i != 1 else 0 for i in new_input_ids])
        new_context_end_pos = torch.where(new_attention_mask)[0][-1].item() - 1
        self.context_end_pos = new_context_end_pos
        
        # find new answer start and end positions
        # update new answer start and end positions
        ans_shift = left_idx - self.context_start_pos
        self.ans_start_pos, self.ans_end_pos = self.ans_start_pos-ans_shift, self.ans_end_pos-ans_shift
        
        
        # update input_ids and attention_masks
        self.input_ids = new_input_ids
        self.attention_mask = new_attention_mask
        
        return self.input_ids, self.attention_mask, (tensor(self.ans_start_pos), tensor(self.ans_end_pos))
        
    def random_left_truncate(self):
        left_idx = self.rand_left_idx
        if left_idx: self.left_truncate(left_idx)
        
        
    def replace_with_mask(self, idxs_to_mask):
        """
        Replace given input ids with tokenizer.mask_token_id
        """
        # clone for debugging
        new_input_ids = self.input_ids.clone()
        new_input_ids[idxs_to_mask] = tensor([tokenizer.mask_token_id]*len(idxs_to_mask))
        self.input_ids = new_input_ids

        
    def random_replace_with_mask(self, mask_p=0.2):
        """
        mask_p: Proportion of tokens to replace with mask token id
        """
        idxs_to_mask = np.random.choice(self.left_right_idxs, int(len(self.left_right_idxs)*mask_p))
        if idxs_to_mask.size > 0: self.replace_with_mask(idxs_to_mask)
        
                


# In[ ]:


do_tfms = {}
do_tfms["random_left_truncate"] = {"p":0.3}
do_tfms["random_right_truncate"] = {"p":0.3}
do_tfms["random_replace_with_mask"] = {"p":0.3, "mask_p":0.2}
do_tfms


# In[ ]:


#export
class SQUAD_Dataset(Dataset):
    def __init__(self, tokenizer, dataset_tensors, examples, features, is_training=True, do_tfms=None):
        self.dataset_tensors = dataset_tensors
        self.examples = examples
        self.features = features
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.do_tfms = do_tfms
                
        
    def __getitem__(self, idx):
        'fastai requires (xb, yb) to return'
        
        input_ids = self.dataset_tensors[0][idx]
        attention_mask = self.dataset_tensors[1][idx]
        token_type_ids = self.dataset_tensors[2][idx]
        if self.is_training: 
            start_position = self.dataset_tensors[3][idx]
            end_position = self.dataset_tensors[4][idx]
            
            if self.do_tfms:
                token_to_orig_map = self.features[idx].token_to_orig_map
                
                augmentor = TSEDataAugmentor(self.tokenizer,
                                             input_ids,
                                             attention_mask,
                                             start_position, end_position,
                                             token_to_orig_map)

                if np.random.uniform() < self.do_tfms["random_left_truncate"]["p"]:
                    augmentor.random_left_truncate()
                if np.random.uniform() < self.do_tfms["random_right_truncate"]["p"]:
                    augmentor.random_right_truncate()
                if np.random.uniform() < self.do_tfms["random_replace_with_mask"]["p"]:
                    augmentor.random_replace_with_mask(self.do_tfms["random_replace_with_mask"]["mask_p"])

                input_ids = augmentor.input_ids
                attention_mask = augmentor.attention_mask
                start_position, end_position = tensor(augmentor.ans_start_pos), tensor(augmentor.ans_end_pos)
                
            
        xb = (input_ids, attention_mask, token_type_ids)
        if self.is_training: yb = (start_position, end_position)
        else: yb = 0
        
        return xb, yb
    
    def __len__(self): return len(self.dataset_tensors[0])


# In[ ]:


#export
def get_fold_ds(foldnum, tokenizer=tokenizer):
    data_dir = "/kaggle/working/squad_data"
    train_filename = f"train_squad_data_{foldnum}.json"
    valid_filename = f"valid_squad_data_{foldnum}.json"
    test_filename = "test_squad_data.json"
    
    # tokenize
    train_examples = processor.get_train_examples(data_dir, train_filename)
    valid_examples = processor.get_train_examples(data_dir, valid_filename)
    test_examples = processor.get_dev_examples(data_dir, test_filename)

    # features and tensors
    train_features, train_dataset = get_dataset(train_examples, True)
    valid_features, valid_dataset = get_dataset(valid_examples, True)
    test_features, test_dataset = get_dataset(test_examples, False)
    train_dataset_tensors = train_dataset.tensors
    valid_dataset_tensors = valid_dataset.tensors
    test_dataset_tensors = test_dataset.tensors
    
    # create pytorch dataset
    do_tfms = {}
    do_tfms["random_left_truncate"] = {"p":0.3}
    do_tfms["random_right_truncate"] = {"p":0.3}
    do_tfms["random_replace_with_mask"] = {"p":0.3, "mask_p":0.3}

    train_ds = SQUAD_Dataset(tokenizer, train_dataset_tensors, train_examples, train_features, True, do_tfms)
    valid_ds = SQUAD_Dataset(tokenizer, valid_dataset_tensors, valid_examples, valid_features, True)
    test_ds = SQUAD_Dataset(tokenizer, test_dataset_tensors, test_examples, test_features, False)
    
    return train_ds, valid_ds, test_ds    


# ### Model
# 
# Here we have `ModelWrapper` to make model from `transformers` to work with `fastai` 's `Learner` class. Also, loss is computed within the model, for this we will use a `DummyLoss` to work with `Learner.fit()`

# In[ ]:


from transformers import AutoModelForPreTraining, RobertaModel, BertModel


# In[ ]:


# MODEL_TYPE = 'distilbert'
MODEL_TYPE = 'roberta'


# In[ ]:


# train_ds, valid_ds, test_ds  = get_fold_ds(0)

# model = AutoModel.from_pretrained(PRETRAINED_TYPE)

# data = DataBunch.create(train_ds, valid_ds, test_ds, path=".", bs=32)

# xb,yb = data.one_batch()


# In[ ]:


class QAHead(Module): 
    def __init__(self, p=0.5):    
        self.d0 = nn.Dropout(p)
        self.l0 = nn.Linear(768, 2)
#         self.d1 = nn.Dropout(p)
#         self.l1 = nn.Linear(256, 2)        
    def forward(self, x):
        return self.l0(self.d0(x))
    
class TSEModel(Module):
    def __init__(self, model): 
        self.sequence_model = model
        self.head = QAHead()
        
    def forward(self, *xargs):
        inp = {}
        inp["input_ids"] = xargs[0]
        inp["attention_mask"] = xargs[1]
        inp["token_type_ids"] = xargs[2]
        if MODEL_TYPE in ["xlm", "roberta", "distilbert", "camembert"]: del inp["token_type_ids"]
    
        sequence_output, _ = self.sequence_model(**inp)
        start_logits, end_logits = self.head(sequence_output).split(1, dim=-1)
        return (start_logits.squeeze(-1), end_logits.squeeze(-1))


# In[ ]:


class CELoss(Module):
    "single backward by concatenating both start and logits with correct targets"
    def __init__(self): self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, inputs, start_targets, end_targets):
        start_logits, end_logits = inputs
        
        logits = torch.cat([start_logits, end_logits]).contiguous()
        
        targets = torch.cat([start_targets, end_targets]).contiguous()
        
        return self.loss_fn(logits, targets)


# In[ ]:


# tse_model = TSEModel(model)

# out = tse_model(*xb)

# loss_func = CELoss()

# loss = loss_func(out, *yb); loss


# Here we define parameter group split points for `gradual unfreezing`. Idea is coming from [ULMFIT paper](https://arxiv.org/pdf/1801.06146.pdf).

# In[ ]:


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# We will choose start and end indexes for predictions such that sum of logits is maximum while satisfying `start_idx <= end_idx`

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


# For each epoch we will calculate `JaccardScore` on validation set

# In[ ]:


def get_answer_by_char_offset(context_text, char_to_word_offset, start_idx, end_idx, token_to_orig_map):
    
    start_offset_id = token_to_orig_map[start_idx] 
    end_offset_id = token_to_orig_map[end_idx]
    
    
    return "".join([ct for ct, char_offs in zip(context_text, char_to_word_offset) if 
                                     (char_offs >= start_offset_id) & (char_offs <= end_offset_id)])


# In[ ]:


def model_split_func(m): 
    n = len(m.sequence_model.encoder.layer) - 5
    return (m.sequence_model.embeddings, m.sequence_model.encoder.layer[n], m.head)


# In[ ]:


def model_split_func(m): 
    "4 layer groups"
    n = len(m.sequence_model.encoder.layer)//2
    return (m.sequence_model.embeddings, m.sequence_model.encoder.layer[:n], m.sequence_model.encoder.layer[n:], m.head)


# In[ ]:


class JaccardScore(Callback):
    "Stores predictions and targets to perform calculations on epoch end."
    def __init__(self, valid_ds): 
        self.valid_ds = valid_ds
        self.token_to_orig_map = [o.token_to_orig_map for o in valid_ds.features]
        self.context_text = [o.context_text for o in valid_ds.examples]
        self.answer_text = [o.answer_text for o in valid_ds.examples]
        self.char_to_word_offset = [o.char_to_word_offset for o in valid_ds.examples]

        self.offset_shift = min(self.token_to_orig_map[0].keys())
        
        
    def on_epoch_begin(self, **kwargs):
        self.jaccard_scores = []  
        self.valid_ds_idx = 0
        
        
    def on_batch_end(self, last_input:Tensor, last_output:Tensor, last_target:Tensor, **kwargs):
        
#         import pdb;pdb.set_trace()
        
        input_ids = last_input[0]
        attention_masks = last_input[1].bool()
        token_type_ids = last_input[2].bool()

        start_logits, end_logits = last_output
        
        # mask select only context part
        for i in range(len(input_ids)):
            
            if MODEL_TYPE == "roberta": 
                
                _input_ids = input_ids[i].masked_select(attention_masks[i])
                _start_logits = start_logits[i].masked_select(attention_masks[i])[4:-1] # ignore first 4 (non context) and last special token:2
                _end_logits = end_logits[i].masked_select(attention_masks[i])[4:-1] # ignore first 4 (non context) and last special token:2
                start_idx, end_idx = get_best_start_end_idxs(_start_logits, _end_logits)
                start_idx, end_idx = start_idx + self.offset_shift, end_idx + self.offset_shift

#             else:
#                 _input_ids = input_ids[i].masked_select(token_type_ids[i])
#                 _start_logits = start_logits[i].masked_select(token_type_ids[i])
#                 _end_logits = end_logits[i].masked_select(token_type_ids[i])
#                 _offset_shift = sum(~token_type_ids[i][attention_masks[i]])
#                 start_idx, end_idx = get_best_start_end_idxs(_start_logits, _end_logits)
#                 start_idx, end_idx = start_idx + self.offset_shift, end_idx + self.offset_shift
            
            context_text = self.context_text[self.valid_ds_idx]
            char_to_word_offset = self.char_to_word_offset[self.valid_ds_idx]
            token_to_orig_map = self.token_to_orig_map[self.valid_ds_idx]
            
            _answer =  get_answer_by_char_offset(context_text, char_to_word_offset, start_idx, end_idx, token_to_orig_map)
            _answer_text = self.answer_text[self.valid_ds_idx]
            
            score = jaccard(_answer, _answer_text)
            self.jaccard_scores.append(score)

            self.valid_ds_idx += 1
            
    def on_epoch_end(self, last_metrics, **kwargs):        
        res = np.mean(self.jaccard_scores)
        return add_metrics(last_metrics, res)


# Here we are initialazing model, splitting parameter groups and putting model callback for mixed precision training. `bs=128` is a good choice for the GPU memory we have at hand.

# In[ ]:


from fastai.callbacks import *

def new_on_train_begin(self, **kwargs:Any)->None:
    "Initializes the best value."
    if not hasattr(self, 'best'):
        self.best = float('inf') if self.operator == np.less else -float('inf')

SaveModelCallback.on_train_begin = new_on_train_begin


# In[ ]:


model = AutoModel.from_pretrained(PRETRAINED_TYPE)
os.makedirs(f"{PRETRAINED_TYPE}-config", exist_ok=True)
model.config.save_pretrained(f"{PRETRAINED_TYPE}-config")
del model; gc.collect()


# TODO: 
# 
# - No wd to LayerNorm and Biases. 
# - Explore better optimizations methods for better jaccard score.

# In[ ]:


def run_fold(foldnum):
    n_epochs = 5
    wd = 0.002 # true weight decay
    
    # DATA
    train_ds, valid_ds, test_ds = get_fold_ds(foldnum, tokenizer)
    data = DataBunch.create(train_ds, valid_ds, test_ds, path=".", bs=64)
    
    # LEARNER
    model = AutoModel.from_pretrained(PRETRAINED_TYPE)
    tse_model = TSEModel(model)
    learner = Learner(data, tse_model, loss_func=CELoss(), metrics=[JaccardScore(valid_ds)], model_dir=f"models_fold_{foldnum}")
    learner.split(model_split_func)
    learner.to_fp16() 
    
    # CALLBACKS
    early_stop_cb = EarlyStoppingCallback(learner, monitor='jaccard_score',mode='max',patience=2)
    save_model_cb = SaveModelCallback(learner,every='improvement',monitor='jaccard_score',name=f'{MODEL_TYPE}-qa-finetune')
    csv_logger_cb = CSVLogger(learner, f"training_logs_{foldnum}", True)


    ### Train
    # We can find the maximun learning rate to start training using `lr_finder()`. `1e-2` seems like a good choice from the plot.

    lr = 1e-2
    
    # Last Param Group
    learner.freeze_to(3);
    learner.fit_one_cycle(1, lr, pct_start=0.4, div_factor=50,
                          wd=wd, callbacks=[early_stop_cb, save_model_cb, csv_logger_cb])

    # Last 2 Param Groups
    learner.freeze_to(2)
    learner.fit_one_cycle(n_epochs, slice(lr/100, lr/10), pct_start=0.4, div_factor=50, 
                          wd=wd, callbacks=[early_stop_cb, save_model_cb, csv_logger_cb])

    # All Param Groups
#     data = DataBunch.create(train_ds, valid_ds, test_ds, path=".", bs=64) # decrease bs to fit GPU MEM
#     learner.data = data
#     learner.to_fp16()
    learner.freeze_to(1) # exclude embeddings layer
    learner.fit_one_cycle(n_epochs, slice(lr/1000, lr/100), pct_start=0.4, div_factor=50, 
                          wd=wd, callbacks=[early_stop_cb, save_model_cb, csv_logger_cb])
    
    # don't save opt state
    learner.save(f'{MODEL_TYPE}-qa-finetune', with_opt=False)
    del learner; gc.collect()


# ### Run 5 folds

# In[ ]:


for foldnum in range(5): run_fold(foldnum)


# ### Predict

# In[ ]:


# learner.load('bert-large-cased-qa-step1')
# learner.model.eval();


# In[ ]:


# from tqdm import tqdm
# test_answers = []
# with torch.no_grad():
#     for xb,yb in tqdm(learner.data.test_dl):
#         output = learner.model(*xb)
#         input_ids = xb[0]
#         attention_masks = xb[1].bool()
#         token_type_ids = xb[2].bool()

#         start_logits, end_logits = output

#         batch_answers = []
#         # mask select only context part
#         for i in range(len(input_ids)):
#             _input_ids = input_ids[i].masked_select(token_type_ids[i])
#             _start_logits = start_logits[i].masked_select(token_type_ids[i])
#             _end_logits = end_logits[i].masked_select(token_type_ids[i])
#             best_idxs = get_best_start_end_idxs(_start_logits, _end_logits)
#             _answer = tokenizer.decode(_input_ids[best_idxs[0]:best_idxs[1]+1])
#             batch_answers.append(_answer)
            
#         test_answers += batch_answers


# ### Submit

# In[ ]:


# test_df['selected_text'] = test_answers

# # keep neutral as it is
# test_df['selected_text'] = test_df.apply(lambda o: o['text'] if o['sentiment'] == 'neutral' else o['selected_text'], 1)

# subdf = test_df[['textID', 'selected_text']]

# subdf.head()

# ## this shouldn't be necessary since evaluation code in Kaggle is fixed
# # def f(selected): return " ".join(set(selected.lower().split()))
# # subdf.selected_text = subdf.selected_text.map(f)

# subdf.to_csv("submission.csv", index=False)


# ### fin

# Hope this is helpful to someone :)
