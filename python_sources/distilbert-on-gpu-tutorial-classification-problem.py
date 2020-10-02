#!/usr/bin/env python
# coding: utf-8

# # Introduction :
# 
# ### This kernel is a leading one for [this kernel](https://www.kaggle.com/atulanandjha/distillbert-extensive-tutorial-starter-kernel), where i had implemented DISTILLBERT on CPU. Here, I am extending the same task on GPU, much faster and better accuracy.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Different versions of BERT and their performance standards:
# 
# ![bert-variants](https://miro.medium.com/max/1243/1*5PzGl1dNt_5jMH3_mm3PpA.png)

# #### ensuring python environment

# In[ ]:


import sys
print(sys.executable)


# ### to check GPU-configurations and current usage.

# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


# uncomment/Comment below line to install/skip-install hugging-face transformers

get_ipython().system('pip install transformers')


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb # pytorch-transformers by huggingface
import time
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#initiating Garbage Collector for GPU environment setup
import gc
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())
    except:
        pass


# #### cross-checking if GPU is used in this notebook

# In[ ]:


torch.cuda.is_available()


# ### loading dataset files

# In[ ]:


path = '../input/stanford-sentiment-treebank-v2-sst2/datasets/'

# to read via CSV files...
# df = pd.read_csv(path + 'csv-format/train.csv')

df = pd.read_csv(path + 'tsv-format/train.tsv', delimiter='\t')


# In[ ]:


df.shape


# ### Note : for now I am working on a sample of dataset, 2000 rows for now. although, we can loop across batches of equal sizes to cover whole dataset as well.
# 
# ### <span style="color:red;"> Consider Upvoting the kernel if you have come this far and you liked it.</span>

# In[ ]:


batch_1 = df[:2000]
batch_1['Ratings'].value_counts()


# In[ ]:


# https://www.kaggle.com/dansbecker/running-kaggle-kernels-with-a-gpu
USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    print('using device: cuda')
else:
    print('using device: cpu')


# In[ ]:


use_cuda = not False and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# ![BERT-versions-anime](http://jalammar.github.io/images/transformer-ber-ulmfit-elmo.png)

# ### Loading model class, weight matrices, and tokenizer classes from pre-trained DistillBERT model.

# In[ ]:


print(time.ctime())


model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights).to(device)

print(time.ctime())


# In[ ]:


tokenized = batch_1['Reviews'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))


# In[ ]:


tokenized.shape


# In[ ]:


print(time.ctime())

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

print(time.ctime())


# In[ ]:


np.array(padded).shape


# ### Applying attention mask to avoid Biasness.

# In[ ]:


attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape


# # note:::::::
# 
# 
# ### I get the error (CUDA out of memory) in below code cell with batch size > = 2500. Can some one suggest me any algo/procedure with a code snippet on how to overcome that problem??
# 
# ### Like, How can I make use of complete dataset : 6920 rows, instead of just 2000 (i used here).
# 
# Possible suggestion :  USe for loop and run the algo for the complete dataset...

# In[ ]:


# with GPU usage...


print(time.ctime())


if USE_GPU and torch.cuda.is_available():
    print('using GPU...')
    input_ids = torch.tensor(padded).to(device)  
    attention_mask = torch.tensor(attention_mask).to(device)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)# .to(device)
        
print(time.ctime())


# In[ ]:


# add .cpu to convert cuda tensor to numpy()

features = last_hidden_states[0][:,0,:].cpu().numpy()


# In[ ]:


labels = batch_1['Ratings']


# In[ ]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels)


# ![training-Bert](http://jalammar.github.io/images/bert-transfer-learning.png)
# 
# #### The two steps of how BERT is developed. You can download the model pre-trained in step 1 (trained on un-annotated data), and only worry about fine-tuning it for step 2.

# In[ ]:



# parameters = {'C': np.linspace(0.0001, 100, 20)}
# grid_search = GridSearchCV(LogisticRegression(), parameters)
# grid_search.fit(train_features, train_labels)

# print('best parameters: ', grid_search.best_params_)
# print('best scrores: ', grid_search.best_score_)


# In[ ]:


lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)


# In[ ]:


lr_clf.score(test_features, test_labels)


# In[ ]:


from sklearn.dummy import DummyClassifier
clf = DummyClassifier()

scores = cross_val_score(clf, train_features, train_labels)
print("Dummy classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ### Other tasks that can be performed on top of BERT Architecture.
# 
# ![bert-appplications](http://jalammar.github.io/images/openai-input%20transformations.png)

# ## References:
# 1. [Papers-with-code-DISTILLBERT](https://paperswithcode.com/paper/distilbert-a-distilled-version-of-bert)
# 2. [Jay-Alammar-Evolution-of-BERT-architectures](http://jalammar.github.io/illustrated-bert/)

# ## Shoutouts :
# 
# #### to these Kernels for providing useful code snippets to fully utilize GPU/CUDA for Pytorch.
# 
# 1. [Simple PyTorch with kaggle's GPU](https://www.kaggle.com/leighplt/simple-pytorch-with-kaggle-s-gpu) - @leighplt
# 
# 2. [Testing GPU-enabled Notebooks MNIST + Pytorch](https://www.kaggle.com/scottclowe/testing-gpu-enabled-notebooks-mnist-pytorch) - @scottclowe
# 
# 3. [CUDA Out of memory](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/91081) - @thawatt

# # <span style="color:red"># DO UPVOTE if you like this kernel. Feedbacks are always welcomed! </span>
# 
# # THANKS !
