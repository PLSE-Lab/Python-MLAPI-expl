#!/usr/bin/env python
# coding: utf-8

# I found out why this happens.
# 
# GPT2 was originally didn't have a dedicated padding token since it was trained on sequences of equal lengths. The maintainers of pytorch pretrained bert have gotten around this by letting you set special tokens with their own vocab indices.
# 
# This should fix the problem:
# 
#     # Add the <pad> token to the vocabulary
#     SPECIAL_TOKENS = ["<pad>"]
#     tokenizer.set_special_tokens(SPECIAL_TOKENS)
# 
#     # Set the number of special tokens in the model
#     model.set_num_special_tokens(len(SPECIAL_TOKENS))
# 
#     # Get the <pad> token's index
#     pad_idx = tokenizer.convert_tokens_to_ids(['<pad>'])[0]
#     
#     # Use keras's tokenizer to pad sequences with pad_idx
#     x = []
#     for i in tqdm(range(len(x_train))):
#         x.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x_train[i])[:MAX_LEN]))
#     
#     x_train = sequence.pad_sequences(x, maxlen=MAX_LEN, padding='post', value=pad_idx)
#     x_train = torch.tensor(x_train, dtype=torch.int32)
# 
# I also made a kernel where I preprocess the data and save it to disk [here](https://www.kaggle.com/bkkaggle/jigsaw-preprocessing-gpt2-1)
# 
# #### Resources
# - https://github.com/huggingface/pytorch-pretrained-BERT/issues/573
# - https://github.com/huggingface/pytorch-pretrained-BERT/issues/577
# - https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

# In[ ]:


get_ipython().run_cell_magic('time', '', '!cp -r ../input/jigsaw-pytorch-pretrained-bert/repository/huggingface-pytorch-pretrained-BERT-3fc63f1/ ./\n!pip install ./huggingface-pytorch-pretrained-BERT-3fc63f1/.')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Borrows a lot of code from https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version\nFOLD = 0\n\nimport os\nimport sys\nimport random\nimport glob\nimport gc\nimport requests\nimport pickle\nimport csv\n\nimport numpy as np\nimport pandas as pd\n\nimport mlcrate as mlc\n\nimport os\n\nfrom sklearn.model_selection import KFold\nfrom sklearn.metrics import roc_auc_score, roc_curve\nfrom sklearn.preprocessing import StandardScaler\n\nfrom tqdm._tqdm_notebook import tqdm_notebook as tqdm\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nfrom torch.utils import data\nfrom torch.nn.utils.rnn import pad_sequence\nimport torch.utils.checkpoint as checkpoint\n\nfrom keras.preprocessing import text, sequence\n\n# from apex import amp\n\nimport spacy\nfrom spacy.lang.en import English\n\nimport matplotlib.pyplot as plt\n\nfrom pytorch_pretrained_bert import BertTokenizer, GPT2Tokenizer\n\n# disable progress bars when submitting\ndef is_interactive():\n   return \'SHLVL\' not in os.environ\n\nif not is_interactive():\n    def nop(it, *a, **k):\n        return it\n\n    tqdm = nop\n\nSEED = 4242\n\ndef seed_everything(SEED=SEED):\n    random.seed(SEED)\n    os.environ[\'PYTHONHASHSEED\'] = str(SEED)\n    np.random.seed(SEED)\n    torch.manual_seed(SEED)\n    torch.cuda.manual_seed(SEED)\n    torch.backends.cudnn.deterministic = True\n\nseed_everything()\n\ndevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\n\ndef get_n_params(model):\n    pp=0\n    for p in list(model.parameters()):\n        nn=1\n        for s in list(p.size()):\n            nn = nn*s\n        pp += nn\n    return pp\n\n# from https://github.com/floydhub/save-and-resume\ndef save_checkpoint(state):\n    """Save checkpoint if a new best is achieved"""\n    print (" Saving checkpoint")\n\n    filename = f\'./checkpoint-{state["fold"]}.pt.tar\'\n    torch.save(state, filename)\n\ndef initialize(model, fold):\n    path = f\'./checkpoint-{fold}.pt.tar\'\n    \n    checkpoint = torch.load(path)\n    model.load_state_dict(checkpoint[\'model\'])\n\n    print(f\' Loaded checkpoint {path} | Trained for {checkpoint["epoch"] + 1} epochs\')\n    \n    return model')


# In[ ]:


WORKERS = 0

SPLITS = 5
MAX_LEN = 220
NUM_WORDS = 100000

BATCH_SIZE = 512


# In[ ]:


train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')


# In[ ]:


# train = train.loc[:1000]


# In[ ]:


x_train = train['comment_text'].values


# In[ ]:


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


# In[ ]:


x = []
for i in tqdm(range(len(x_train))):
    x.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x_train[i])[:MAX_LEN]))
    
x_train = sequence.pad_sequences(x, maxlen=MAX_LEN, padding='post')
x_train = torch.tensor(x_train, dtype=torch.int32)


# In[ ]:


with open('x_train_gpt.pkl', 'wb') as handle:
    pickle.dump(x_train, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


get_ipython().system('rm -rf huggingface-pytorch-pretrained-BERT-3fc63f1')

