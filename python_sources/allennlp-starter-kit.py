#!/usr/bin/env python
# coding: utf-8

# # Simple baseline with AllenNlP
# 
# I haven't seen anyone try to use AllenNLP for a kaggle competition before, so I wrote this kernel to show how it could be done.  
# AllenNLP abstracts away most of the boilerplate code like training loops, loading pretrained embeddings, and keeping track of experiments which lets you write a lot less code. It also lets you change model architectures and hyperparameters by  creating new experiments entirely from configuration files instead of changing the code for each new experiment.

# ### Install AllenNLP from dataset

# In[ ]:


get_ipython().system('cp -r ../input/allennlp-packages/packages/packages ./')
get_ipython().system('pip install -r packages/requirements.txt --no-index --find-links packages')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

FOLD = 0

import os
import sys
import random
import glob
import gc
import logging
import requests
import re

from typing import Dict, Tuple, List
from collections import OrderedDict
from overrides import overrides
from time import sleep

import cv2
import numpy as np
import pandas as pd

import mlcrate as mlc

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim

import torchvision

import allennlp

from allennlp.common import Registrable, Params
from allennlp.common.util import START_SYMBOL, END_SYMBOL, JsonDict

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import ArrayField, TextField
from allennlp.data.iterators import BucketIterator, MultiprocessIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, CharacterTokenizer
from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper # MIGHT USE FOR ABSTRACTION

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.beam_search import BeamSearch

from allennlp.training.metrics import F1Measure, BLEU
from allennlp.training import Trainer

sys.path.insert(0, './math_handwriting_recognition')

logger = logging.getLogger()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# ## Load and split data

# In[ ]:


train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
sample_submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')


# In[ ]:


get_ipython().system('mkdir jigsaw')
get_ipython().system('touch jigsaw/__init__.py')

# Get a 5 fold cv
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = list(kfold.split(train))[0]
train_df, val_df = train.iloc[train_idx].reset_index(), train.iloc[val_idx].reset_index()
train_df.to_csv('train.csv')
val_df.to_csv('val.csv')


# ## Dataset reader

# In[ ]:


get_ipython().run_cell_magic('writefile', 'jigsaw/dataset.py', 'import os\nimport random\nfrom typing import Dict, Tuple, List\nfrom overrides import overrides\n\nimport cv2\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nimport torch\n\nimport spacy\n\nimport allennlp\n\nfrom allennlp.common.util import START_SYMBOL, END_SYMBOL, get_spacy_model\n\nfrom allennlp.data import DatasetReader, Instance\nfrom allennlp.data.fields import ArrayField, TextField, MetadataField, LabelField\nfrom allennlp.data.token_indexers import SingleIdTokenIndexer\nfrom allennlp.data.tokenizers import Token, Tokenizer, CharacterTokenizer, WordTokenizer\n\n@Tokenizer.register("simple")\nclass LatexTokenizer(Tokenizer):\n    def __init__(self) -> None:\n        super().__init__()\n\n    def _tokenize(self, text):\n        return [Token(token) for token in text.split()]\n\n    @overrides\n    def tokenize(self, text: str) -> List[Token]:\n        tokens = self._tokenize(text)\n\n        return tokens\n\n@DatasetReader.register(\'jigsaw\')\nclass JigsawDatasetReader(DatasetReader):\n    def __init__(self, root_path: str, tokenizer: Tokenizer, lazy: bool = True, subset: bool = False) -> None:\n        super().__init__(lazy)\n        \n        self.root_path = root_path\n        self.subset = subset\n        \n        self._tokenizer = tokenizer\n        self._token_indexer = {"tokens": SingleIdTokenIndexer()}\n\n    @overrides\n    def _read(self, file: str):\n        df = pd.read_csv(os.path.join(self.root_path, file))\n\n        if self.subset:\n            df = df.loc[:16]\n\n        for _, row in df.iterrows():\n            idx = row[\'id\']\n            comment_text = row[\'comment_text\']\n            \n            if \'target\' in df.columns:\n                target = int(row[\'target\'] > 0.5)\n                yield self.text_to_instance(idx, comment_text, target)\n            else:\n                yield self.text_to_instance(idx, comment_text)\n            \n    @overrides\n    def text_to_instance(self, idx: str, comment_text: str, target: float = None) -> Instance:\n        comment_text = self._tokenizer.tokenize(comment_text)\n        \n        fields = {}\n        fields[\'idx\'] = MetadataField({\'idx\': idx})\n        fields[\'comment_text\'] = TextField(comment_text, self._token_indexer)\n\n        if target is not None:\n            fields[\'target\'] = LabelField(target, skip_indexing=True)\n        \n        return Instance(fields)')


# ## Simple LSTM baseline model

# In[ ]:


get_ipython().run_cell_magic('writefile', 'jigsaw/model.py', 'import os\nimport random\nfrom typing import Dict, Tuple\nfrom overrides import overrides\n\nimport numpy as np\nimport pandas as pd\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nimport torchvision\n\nimport allennlp\n\nfrom allennlp.common import Registrable, Params\nfrom allennlp.common.util import START_SYMBOL, END_SYMBOL\n\nfrom allennlp.data.vocabulary import Vocabulary\n\nfrom allennlp.models import Model\n\nfrom allennlp.modules import FeedForward\nfrom allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder\nfrom allennlp.modules.token_embedders import Embedding\nfrom allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper\n\nfrom allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits\n\nfrom allennlp.nn.beam_search import BeamSearch\n\nfrom allennlp.training.metrics import F1Measure, BLEU, Auc, BooleanAccuracy\n\ndevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\n\n@Model.register(\'baseline\')\nclass Baseline(Model):\n   def __init__(self, embeddings: TextFieldEmbedder, encoder: Seq2VecEncoder, classifier: FeedForward, vocab: Vocabulary) -> None:\n       super().__init__(vocab)\n\n       self.embedding = embeddings\n       \n       self.encoder = encoder\n       self.classifier = classifier\n       \n       self.loss = nn.BCEWithLogitsLoss()\n       self.accuracy = BooleanAccuracy()\n       \n   @overrides\n   def forward(self, idx: Dict[str, torch.Tensor], comment_text: Dict[str, torch.Tensor], target: torch.Tensor = None) -> Dict[str, torch.Tensor]:\n       mask = get_text_field_mask(comment_text)\n\n       x = self.embedding(comment_text)\n       x = self.encoder(x, mask)\n       x = self.classifier(x).view(-1)\n       \n       logits = torch.sigmoid(x)\n               \n       out = {\'idx\': idx, \'pred\': logits}\n\n       if target is not None:\n           if not self.training:\n               self.accuracy((logits > 0.5).int(), target.int())\n\n           out[\'loss\'] = self.loss(x, target.float())\n\n       return out\n\n   @overrides\n   def get_metrics(self, reset: bool = False) -> Dict[str, float]:\n       if not self.training:\n           metrics = {\n               "accuracy": self.accuracy.get_metric(reset)\n           }\n       else:\n           metrics = {}\n       \n       return metrics')


# ## Predictor to get test predictions

# In[ ]:


get_ipython().run_cell_magic('writefile', 'jigsaw/predictor.py', 'import os\nimport random\nfrom typing import Dict, Tuple, List\nfrom overrides import overrides\nimport json\n\nimport numpy as np\nimport pandas as pd\n\nimport matplotlib.pyplot as plt\nimport skimage\nimport cv2\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nimport torchvision\n\nimport mlcrate as mlc\n\nimport allennlp\n\nfrom allennlp.common import Registrable, Params\nfrom allennlp.common.util import START_SYMBOL, END_SYMBOL, JsonDict, sanitize\n\nfrom allennlp.data import DatasetReader, Instance\nfrom allennlp.data.vocabulary import Vocabulary\n\nfrom allennlp.models import Model\n\nfrom allennlp.predictors.predictor import Predictor\n\nfrom allennlp.modules.token_embedders import Embedding\nfrom allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits\nfrom allennlp.nn.beam_search import BeamSearch\n\nfrom allennlp.training.metrics import F1Measure, BLEU\n\ndevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\n\n@Predictor.register(\'jigsaw\')\nclass JigsawPredictor(Predictor):\n    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:\n        super().__init__(model, dataset_reader)\n        \n    def dump_line(self, outputs: JsonDict) -> str:\n        pred = str(outputs[\'pred\'])\n\n        return f\'{pred}\\n\'')


# ## Config file to set up experiments without changing the code

# In[ ]:


get_ipython().run_cell_magic('writefile', 'config.json', '{\n    "dataset_reader": {\n        "type": "jigsaw",\n        "root_path": "./",\n        "lazy": true,\n        "subset": false,\n        "tokenizer": {\n            "type": "simple"\n        }\n    },\n    "train_data_path": "train.csv",\n    "validation_data_path": "val.csv",\n    "model": {\n        "type": "baseline",\n        "embeddings": {\n          "tokens": {\n            "type": "embedding",\n            "pretrained_file": "../input/quoratextemb/embeddings/glove.840B.300d/glove.840B.300d.txt",\n            "embedding_dim": 300,\n            "trainable": false\n          }\n        },\n        \'encoder\': {\n            \'type\': \'lstm\',\n            \'bidirectional\': false,\n            \'input_size\': 300,\n            \'hidden_size\': 64,\n            \'num_layers\': 1\n        },\n        \'classifier\': {\n            \'input_dim\': 64,\n            \'num_layers\': 1,\n            \'hidden_dims\': 1,\n            \'activations\': \'linear\' # sigmoid activation is applied separately\n        }\n    },\n    "iterator": {\n        "type": "bucket",\n        "sorting_keys":[["comment_text", "num_tokens"]],\n        "batch_size": 512\n    },\n    "trainer": {\n        "num_epochs": 4,\n        "cuda_device": 0,\n        "optimizer": {\n            "type": "adam",\n            "lr": 0.001\n        },\n        "grad_clipping": 5,\n        "learning_rate_scheduler": {\n            "type": "reduce_on_plateau",\n            "factor": 0.5,\n            "patience": 5\n        },\n        "num_serialized_models_to_keep": 1,\n        "summary_interval": 10,\n        "histogram_interval": 100,\n        "should_log_parameter_statistics": true,\n        "should_log_learning_rate": true\n    },\n    \'vocabulary\': {\n        \'max_vocab_size\': 100000,\n#         "directory_path": "./vocabulary"\n    }\n}')


# ## Train the model

# In[ ]:


get_ipython().system('allennlp train config.json -s ./logs --include-package jigsaw')
# !rm -rf logs/*


# ## Evaluate the model's performance on the train and val sets

# In[ ]:


get_ipython().system('allennlp evaluate --cuda-device 0 --include-package jigsaw ./logs/model.tar.gz train.csv')
get_ipython().system('allennlp evaluate --cuda-device 0 --include-package jigsaw ./logs/model.tar.gz val.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', "!allennlp predict --output-file ./train_preds.csv --batch-size 64 --cuda-device 0 --use-dataset-reader --predictor jigsaw --include-package jigsaw --silent ./logs/model.tar.gz train.csv\n# From https://superuser.com/questions/246837/how-do-i-add-text-to-the-beginning-of-a-file-in-bash\n!sed -i '1s/^/prediction\\n/' train_preds.csv\ntrain_preds = pd.read_csv('train_preds.csv')\ntrain_roc_auc_score = roc_auc_score(train_df.target.values > 0.5, train_preds.prediction.values)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "!allennlp predict --output-file ./val_preds.csv --batch-size 64 --cuda-device 0 --use-dataset-reader --predictor jigsaw --include-package jigsaw --silent ./logs/model.tar.gz val.csv\n# From https://superuser.com/questions/246837/how-do-i-add-text-to-the-beginning-of-a-file-in-bash\n!sed -i '1s/^/prediction\\n/' val_preds.csv\nval_preds = pd.read_csv('val_preds.csv')\nval_roc_auc_score = roc_auc_score(val_df.target.values > 0.5, val_preds.prediction.values)")


# In[ ]:


get_ipython().system('cat logs/metrics.json')


# In[ ]:


print(f'Train ROC-AUC: {round(train_roc_auc_score, 4)}')
print(f'Val ROC-AUC: {round(val_roc_auc_score, 4)}')


# ## Predict on the test set and save submission

# In[ ]:


get_ipython().run_cell_magic('time', '', "!allennlp predict --output-file ./test_preds.csv --batch-size 64 --cuda-device 0 --use-dataset-reader --predictor jigsaw --include-package jigsaw --silent ./logs/model.tar.gz ../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv\n# From https://superuser.com/questions/246837/how-do-i-add-text-to-the-beginning-of-a-file-in-bash\n!sed -i '1s/^/prediction\\n/' test_preds.csv\ntest_preds = pd.read_csv('test_preds.csv')\nsample_submission['prediction'] = test_preds['prediction'].values\nmlc.kaggle.save_sub(sample_submission, 'submission.csv')")


# In[ ]:


sample_submission.head()


# ## Delete unnecessary files to free up more space

# In[ ]:


get_ipython().system('rm -rf logs')
get_ipython().system('rm out.txt')
get_ipython().system('rm config.json')


# In[ ]:


get_ipython().system('rm -rf packages')


# In[ ]:







