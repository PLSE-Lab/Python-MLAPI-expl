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

for i, (trn_idx, val_idx) in enumerate(fold_idxs[:1]):
    _trn_fold_df = train_df.iloc[trn_idx]
    _val_fold_df = train_df.iloc[val_idx]
    train_squad_data = create_squad_from_df(_trn_fold_df)
    valid_squad_data = create_squad_from_df(_val_fold_df)
    with open(f"squad_data/train_squad_data_{i}.json", "w") as f: f.write(json.dumps(train_squad_data))
    with open(f"squad_data/valid_squad_data_{i}.json", "w") as f: f.write(json.dumps(valid_squad_data))


# In[ ]:


# # create for test 
test_squad_data =  create_squad_from_df(test_df)
with open("squad_data/test_squad_data.json", "w") as f: f.write(json.dumps(test_squad_data))


# ### Data
# 
# Since we are training a BERT model we will use bert tokenizer for `bert-base-cased`. You can use `Auto*` classes from `transformers` library to load and train with any model available. For demonstration I am training with `foldnum=0`.

# In[ ]:


from fastai.text import *
from transformers import (AutoTokenizer, AutoConfig, AutoModel, AutoModelForQuestionAnswering,
                         RobertaTokenizer)
from transformers.data.processors.squad import (SquadResult, SquadV1Processor, SquadV2Processor,
                                                SquadExample, squad_convert_examples_to_features)


# In[ ]:


PRETRAINED_TYPE = 'roberta-base'
# PRETRAINED_TYPE = 'distilbert-base-uncased'


# In[ ]:


PRETRAINED_TOK_PATH = Path("/kaggle/input/tse-fastai-squad-bert-pretrained/roberta-base-tokenizer/")


# In[ ]:


shutil.copytree(PRETRAINED_TOK_PATH, KAGGLE_WORKING/"tokenizer_dir")


# In[ ]:


PRETRAINED_TOK_PATH = KAGGLE_WORKING/"tokenizer_dir"


# In[ ]:


shutil.copyfile(str(PRETRAINED_TOK_PATH/"tokenizer_config.json"), str(PRETRAINED_TOK_PATH/"config.json"))


# In[ ]:


PRETRAINED_TOK_PATH.ls()


# In[ ]:


processor = SquadV2Processor()
tokenizer = RobertaTokenizer.from_pretrained(str(PRETRAINED_TOK_PATH))


# In[ ]:


max_seq_length = 192
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
        else:
            yb = 0
        return xb, yb
    
    def __len__(self): return len(self.dataset_tensors[0])


# In[ ]:


PRETRAINED_PATH = Path("/kaggle/input/tse-fastai-squad-bert-pretrained/")


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


# ### Model
# 
# Here we have `ModelWrapper` to make model from `transformers` to work with `fastai` 's `Learner` class. Also, loss is computed within the model, for this we will use a `DummyLoss` to work with `Learner.fit()`

# In[ ]:


from transformers import AutoModelForPreTraining, RobertaModel, BertModel


# In[ ]:


# MODEL_TYPE = 'distilbert'
MODEL_TYPE = 'roberta'


# In[ ]:


class QAHead(Module): 
    def __init__(self, p=0.5):    
        self.d0 = nn.Dropout(p)
        self.l0 = nn.Linear(768, 256)
        self.d1 = nn.Dropout(p)
        self.l1 = nn.Linear(256, 2)        
    def forward(self, x):
        return self.l1(self.d1(self.l0(self.d0(x))))
    
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


# Here we define parameter group split points for `gradual unfreezing`. Idea is coming from [ULMFIT paper](https://arxiv.org/pdf/1801.06146.pdf).

# In[ ]:


def model_split_func(m): 
    n = len(m.sequence_model.encoder.layer) - 5
    return (m.sequence_model.encoder.layer[n], m.head)


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


# In[ ]:


def get_best_start_end_idxs_v2(_start_logits, _end_logits):
    start_idx, end_idx = torch.argmax(_start_logits).item(), torch.argmax(_end_logits).item()
    if start_idx > end_idx: end_idx = start_idx
    return (start_idx, end_idx)


# For each epoch we will calculate `JaccardScore` on validation set

# In[ ]:


def get_answer_by_char_offset(context_text, char_to_word_offset, start_idx, end_idx, token_to_orig_map):
    
    start_offset_id = token_to_orig_map[start_idx] 
    end_offset_id = token_to_orig_map[end_idx]
    
    
    return "".join([ct for ct, char_offs in zip(context_text, char_to_word_offset) if 
                                     (char_offs >= start_offset_id) & (char_offs <= end_offset_id)])


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


# ### Predict

# In[ ]:


from tqdm import tqdm


# In[ ]:


train_ds, valid_ds, test_ds = get_fold_ds(0)
data = DataBunch.create(train_ds, valid_ds, test_ds, path=".", bs=128)


# In[ ]:


config = AutoConfig.from_pretrained(PRETRAINED_PATH/"roberta-base-config")
model = AutoModel.from_config(config)
tse_model = TSEModel(model)
learner = Learner(data, tse_model, loss_func=CELoss(), path=PRETRAINED_PATH, model_dir=f".")


# In[ ]:


token_to_orig_map = [o.token_to_orig_map for o in test_ds.features]
context_text = [o.context_text for o in test_ds.examples]
char_to_word_offset = [o.char_to_word_offset for o in test_ds.examples]
offset_shift = min(token_to_orig_map[0].keys())
test_ds_idx = 0 

MODEL_NAME = "roberta"
final_answers = []

with torch.no_grad():        
    for xb,yb in tqdm(learner.data.test_dl):
        model0 = learner.load(f'models_fold_0/{MODEL_NAME}-qa-finetune').model.eval()
        start_logits0, end_logits0 = to_cpu(model0(*xb))
        start_logits0, end_logits0 = start_logits0.float(), end_logits0.float()
        
        model1 = learner.load(f'models_fold_1/{MODEL_NAME}-qa-finetune').model.eval()
        start_logits1, end_logits1 = to_cpu(model1(*xb))
        start_logits1, end_logits1 = start_logits1.float(), end_logits1.float()
        
        model2 = learner.load(f'models_fold_2/{MODEL_NAME}-qa-finetune').model.eval()        
        start_logits2, end_logits2 = to_cpu(model2(*xb))
        start_logits2, end_logits2 = start_logits2.float(), end_logits2.float()
        
        model3 = learner.load(f'models_fold_3/{MODEL_NAME}-qa-finetune').model.eval()
        start_logits3, end_logits3 = to_cpu(model3(*xb))
        start_logits3, end_logits3 = start_logits3.float(), end_logits3.float()
        
        model4 = learner.load(f'models_fold_4/{MODEL_NAME}-qa-finetune').model.eval()
        start_logits4, end_logits4 = to_cpu(model4(*xb))
        start_logits4, end_logits4 = start_logits4.float(), end_logits4.float()
        
        
        input_ids = to_cpu(xb[0])
        attention_masks = to_cpu(xb[1].bool())
        token_type_ids = to_cpu(xb[2].bool())
        
        start_logits = (start_logits0 + start_logits1 + start_logits2 + start_logits3 + start_logits4) / 5
        end_logits = (end_logits0 + end_logits1 + end_logits2 + end_logits3 + end_logits4) / 5
        
        # mask select only context part
        for i in range(len(input_ids)):

            _input_ids = input_ids[i].masked_select(attention_masks[i])
            _start_logits = start_logits[i].masked_select(attention_masks[i])[4:-1] # ignore first 4 (non context) and last special token:2
            _end_logits = end_logits[i].masked_select(attention_masks[i])[4:-1] # ignore first 4 (non context) and last special token:2
            start_idx, end_idx = get_best_start_end_idxs(_start_logits, _end_logits)
            start_idx, end_idx = start_idx + offset_shift, end_idx + offset_shift

            _context_text = context_text[test_ds_idx]
            _char_to_word_offset = char_to_word_offset[test_ds_idx]
            _token_to_orig_map = token_to_orig_map[test_ds_idx]
            
            predicted_answer =  get_answer_by_char_offset(_context_text, _char_to_word_offset, start_idx, end_idx, _token_to_orig_map)
            final_answers.append(predicted_answer)
            
            test_ds_idx += 1


# ### Submit

# In[ ]:


test_df['selected_text'] = final_answers


# In[ ]:


# predict same text if word count < 3
test_df['selected_text'] = test_df.apply(lambda o: o['text'] if len(o['text']) < 3 else o['selected_text'], 1)


# In[ ]:


# keep neutral as it is or not?
test_df['selected_text'] = test_df.apply(lambda o: o['text'] if o['sentiment'] == 'neutral' else o['selected_text'], 1)


# In[ ]:


subdf = test_df[['textID', 'selected_text']]


# In[ ]:


subdf.head()


# In[ ]:


subdf


# In[ ]:


## this shouldn't be necessary since evaluation code in Kaggle is fixed
# def f(selected): return " ".join(set(selected.lower().split()))
# subdf.selected_text = subdf.selected_text.map(f)
subdf.to_csv("submission.csv", index=False)


# ### fin

# Hope this is helpful to someone :)
