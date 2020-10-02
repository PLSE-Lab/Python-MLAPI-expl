#!/usr/bin/env python
# coding: utf-8

# ## Super Fast Inference with TPUs for BERT
#  This kernel is based on the [excellent kernel](https://www.kaggle.com/shonenkov/tpu-inference-super-fast-xlmroberta) by @shonenkov. I have just adapted the code for my dataloaders and nn.module implementation so that it can be used for BERT (@abhishek).
#  Usually inference for bert-base-multilingual-uncased takes around 10 mins on GPU, but this one gets done in 2 mins. Have fun!

# In[ ]:


get_ipython().system('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py > /dev/null')
get_ipython().system('python pytorch-xla-env-setup.py --version 20200416 --apt-packages libomp5 libopenblas-dev > /dev/null')
get_ipython().system('pip install transformers > /dev/null')
get_ipython().system('pip install pandarallel > /dev/null')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from datetime import datetime

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from glob import glob
for path in glob(f'../input/*'):
    print(path)

from nltk import sent_tokenize
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=2, progress_bar=True)
# Any results you write to the current directory are saved as output.


# In[ ]:


import os
import torch
import pandas as pd
from scipy import stats
import numpy as np
import warnings
from tqdm import tqdm
from collections import OrderedDict, namedtuple
import torch.nn as nn
from torch.optim import lr_scheduler
import joblib
import time
import logging
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
import sys
from sklearn import metrics, model_selection
import re
warnings.filterwarnings("ignore")
import nltk
nltk.download('punkt')
import matplotlib.pyplot as plt


# In[ ]:


df_test = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv")
sample = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")


# In[ ]:


MAX_LEN = 192
BERT_PATH = "/kaggle/input/bert-base-multilingual-uncased/"
MODEL_PATH = "/kaggle/input/bert-base-multilingual-uncased/pytorch_model.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)
class BERTDataset:
    def __init__(self, comment_text, tokenizer, max_len):
        self.comment_text = comment_text
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.comment_text)
    
    def __getitem__(self, item):
        comment_text = str(self.comment_text[item])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }
    
class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768*2, 1)

    def forward(self, ids, mask, token_type_ids):
        o1 , o2 = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        mean_pooling = torch.mean(o1,1)
        max_pooling, _ = torch.max(o1,1)
        final = torch.cat((mean_pooling, max_pooling),1)
        final = self.bert_drop(final)
        output = self.out(final)
        return output


# In[ ]:


import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


class MultiTPUPredictor:
    
    def __init__(self, model, device):
        if not os.path.exists('node_submissions'):
            os.makedirs('node_submissions')

        self.model = model
        self.device = device

        xm.master_print(f'Model prepared. Device is {self.device}')


    def run_inference(self, test_loader, verbose=True, verbose_step=50):
        self.model.eval()
        result = {'id': [], 'toxic': []}
        t = time.time()
        with torch.no_grad():
            for bi, d in tqdm(enumerate(test_loader), total=len(test_loader)):
                ids = d["ids"]
                token_type_ids = d["token_type_ids"]
                mask = d["mask"]

                ids = ids.to(self.device, dtype=torch.long)
                inputs = token_type_ids.to(self.device, dtype=torch.long)
                attention_masks = mask.to(self.device, dtype=torch.long)

                outputs = self.model(
                    ids=ids,
                    mask=attention_masks,
                    token_type_ids=inputs
                )
                if verbose:
                    if bi % 50 == 0:
                        xm.master_print(f'Prediction Step {bi}, time: {(time.time() - t):.5f}')

                toxic = torch.sigmoid(outputs).cpu().detach().numpy()

                result['id'].extend(ids.cpu().detach().numpy())
                result['toxic'].extend(toxic)

        result = pd.DataFrame(result)
        node_count = len(glob('node_submissions/*.csv'))
        result.to_csv(f'node_submissions/submission_{node_count}_{datetime.utcnow().microsecond}.csv', index=False)


# In[ ]:


BERT_PATH="bert-base-multilingual-uncased"
mx = BERTBaseUncased()
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-multilingual-uncased", do_lower_case=True)
# mx.load_state_dict(torch.load("/content/drive/My Drive/TCC/Translated/TCC-Translated-Epoch-4-Fold-1-93.37.bin"))


# In[ ]:


test_dataset = BERTDataset(
        comment_text=df_test.content.values,
        tokenizer=TOKENIZER,
        max_len=MAX_LEN
    )


# In[ ]:


def _mp_fn(rank, flags):
    device = xm.xla_device()
    model = mx.to(device)

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        sampler=test_sampler,
        pin_memory=False,
        drop_last=False,
        num_workers=1
    )

    fitter = MultiTPUPredictor(model=model, device=device)
    fitter.run_inference(test_loader)


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nFLAGS={}\nxmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')")

