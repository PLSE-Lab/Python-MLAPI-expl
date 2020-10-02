#!/usr/bin/env python
# coding: utf-8

# # Fine Tuning Transformer for Named Entity Recognition

# ### Introduction
# 
# In this tutorial we will be fine tuning a transformer model for the **Named Entity Recognition** problem. 
# This is one of the most common business problems where a given piece of text/sentence/document different entites need to be identified such as: Name, Location, Number, Entity etc.
# 
# #### Flow of the notebook
# 
# The notebook will be divided into seperate sections to provide a organized walk through for the process used. This process can be modified for individual use cases. The sections are:
# 
# 1. [Installing packages for preparing the system](#section00)
# 2. [Importing Python Libraries and preparing the environment](#section01)
# 3. [Importing and Pre-Processing the domain data](#section02)
# 4. [Preparing the Dataset and Dataloader](#section03)
# 5. [Creating the Neural Network for Fine Tuning](#section04)
# 6. [Fine Tuning the Model](#section05)
# 7. [Validating the Model Performance](#section06)
# 8. [Saving the model and artifacts for Inference in Future](#section07)
# 
# #### Technical Details
# 
# This script leverages on multiple tools designed by other teams. Details of the tools used below. Please ensure that these elements are present in your setup to successfully implement this script.
# 
#  - Data:
# 	- We are working from a dataset available on [Kaggle](https://www.kaggle.com/)
#     - This NER annotated dataset is available at the following [link](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus)
#     - We will be working with the file `ner.csv` from the dataset. 
#     - In the given file we will be looking at the following columns for the purpose of this fine tuning:
#         - `sentence_idx` : This is the identifier that the word in the row is part of the same sentence
#         - `word` : Word in the sentence
#         - `tag` : This is the identifier that is used to identify the entity in the dataset. 
#     - The various entites tagged in this dataset are as per below:
#         - geo = Geographical Entity
#         - org = Organization
#         - per = Person
#         - gpe = Geopolitical Entity
#         - tim = Time indicator
#         - art = Artifact
#         - eve = Event
#         - nat = Natural Phenomenon
# 
# 
#  - Language Model Used:
# 	 - We are using BERT for this project. Hugging face team has created a customized model for token classification, called **BertForTokenClassification**. We will be using it in our custommodel class for training. 
# 	 - [Blog-Post](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
#      - [Documentation for python](https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification)
# 
# 
#  - Hardware Requirements:
# 	 - Python 3.6 and above
# 	 - Pytorch, Transformers and All the stock Python ML Libraries
# 	 - TPU enabled setup. This can also be executed over GPU but the code base will need some changes. 
# 
# 
#  - Script Objective:
# 	 - The objective of this script is to fine tune **BertForTokenClassification**` to be able to identify the entites as per the given test dataset. The entities labled in the given dataset are as follows:

# <a id='section00'></a>
# ### Installing packages for preparing the system
# 
# We are installing 2 packages for the purposes of TPU execution and f1 metric score calculation respectively
# *You can skip this step if you already have these libraries installed in your environment*

# In[ ]:


get_ipython().system('curl -q https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py')
get_ipython().system('python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev')
get_ipython().system('pip -q install seqeval')


# <a id='section01'></a>
# ### Importing Python Libraries and preparing the environment
# 
# At this step we will be importing the libraries and modules needed to run our script. Libraries are:
# * Pandas
# * Pytorch
# * Pytorch Utils for Dataset and Dataloader
# * Transformers
# * BERT Model and Tokenizer
# 
# Followed by that we will preapre the device for TPU execeution. This configuration is needed if you want to leverage on onboard TPU. 

# In[ ]:


# Importing pytorch and the library for TPU execution

import torch
import torch_xla
import torch_xla.core.xla_model as xm


# In[ ]:


# Importing stock ml libraries

import numpy as np
import pandas as pd
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForTokenClassification, BertTokenizer, BertConfig, BertModel

# Preparing for TPU usage
dev = xm.xla_device()


# <a id='section02'></a>
# ### Importing and Pre-Processing the domain data
# 
# We will be working with the data and preparing for fine tuning purposes. 
# *Assuming that the `ner.csv` is already downloaded in your `data` folder*
# 
# * Import the file in a dataframe and give it the headers as per the documentation.
# * Cleaning the file to remove the unwanted columns.
# * We will create a class `SentenceGetter` that will pull the words from the columns and create them into sentences
# * Followed by that we will create some additional lists and dict to keep the data that will be used for future processing

# In[ ]:


df = pd.read_csv("../input/entity-annotated-corpus/ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)
dataset=df.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
       'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
       'prev-prev-word', 'prev-shape', 'prev-word','shape'],axis=1)
dataset.head()


# In[ ]:


# Creating a class to pull the words from the columns and create them into sentences

class SentenceGetter(object):
    
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w,p, t) for w,p, t in zip(s["word"].values.tolist(),
                                                       s['pos'].values.tolist(),
                                                        s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(dataset)


# In[ ]:


# Creating new lists and dicts that will be used at a later stage for reference and processing

tags_vals = list(set(dataset["tag"].values))
tag2idx = {t: i for i, t in enumerate(tags_vals)}
sentences = [' '.join([s[0] for s in sent]) for sent in getter.sentences]
labels = [[s[2] for s in sent] for sent in getter.sentences]
labels = [[tag2idx.get(l) for l in lab] for lab in labels]


# <a id='section03'></a>
# ### Preparing the Dataset and Dataloader
# 
# We will start with defining few key variables that will be used later during the training/fine tuning stage.
# Followed by creation of Dataset class - This defines how the text is pre-processed before sending it to the neural network. We will also define the Dataloader that will feed  the data in batches to the neural network for suitable training and processing. 
# Dataset and Dataloader are constructs of the PyTorch library for defining and controlling the data pre-processing and its passage to neural network. For further reading into Dataset and Dataloader read the [docs at PyTorch](https://pytorch.org/docs/stable/data.html)
# 
# #### *CustomDataset* Dataset Class
# - This class is defined to accept the `tokenizer`, `sentences` and `labels` as input and generate tokenized output and tags that is used by the BERT model for training. 
# - We are using the BERT tokenizer to tokenize the data in the `sentences` list for encoding. 
# - The tokenizer uses the `encode_plus` method to perform tokenization and generate the necessary outputs, namely: `ids`, `attention_mask`
# - To read further into the tokenizer, [refer to this document](https://huggingface.co/transformers/model_doc/bert.html#berttokenizer)
# - `tags` is the encoded entity from the annonated dataset. 
# - The *CustomDataset* class is used to create 2 datasets, for training and for validation.
# - *Training Dataset* is used to fine tune the model: **70% of the original data**
# - *Validation Dataset* is used to evaluate the performance of the model. The model has not seen this data during training. 
# 
# #### Dataloader
# - Dataloader is used to for creating training and validation dataloader that load data to the neural network in a defined manner. This is needed because all the data from the dataset cannot be loaded to the memory at once, hence the amount of dataloaded to the memory and then passed to the neural network needs to be controlled.
# - This control is achieved using the parameters such as `batch_size` and `max_len`.
# - Training and Validation dataloaders are used in the training and validation part of the flow respectively

# In[ ]:


# Defining some key variables that will be used later on in the training

MAX_LEN = 200
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[ ]:


class CustomDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels, max_len):
        self.len = len(sentences)
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        label = self.labels[index]
        label.extend([4]*200)
        label=label[:200]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'tags': torch.tensor(label, dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len


# In[ ]:


# Creating the dataset and dataloader for the neural network

train_percent = 0.8
train_size = int(train_percent*len(sentences))
# train_dataset=df.sample(frac=train_size,random_state=200).reset_index(drop=True)
# test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_sentences = sentences[0:train_size]
train_labels = labels[0:train_size]

test_sentences = sentences[train_size:]
test_labels = labels[train_size:]

print("FULL Dataset: {}".format(len(sentences)))
print("TRAIN Dataset: {}".format(len(train_sentences)))
print("TEST Dataset: {}".format(len(test_sentences)))

training_set = CustomDataset(tokenizer, train_sentences, train_labels, MAX_LEN)
testing_set = CustomDataset(tokenizer, test_sentences, test_labels, MAX_LEN)


# In[ ]:


train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


# <a id='section04'></a>
# ### Creating the Neural Network for Fine Tuning
# 
# #### Neural Network
#  - We will be creating a neural network with the `BERTClass`. 
#  - This network will have the `BertForTokenClassification` model. 
#  - The data will be fed to the `BertForTokenClassification` as defined in the dataset. 
#  - Final layer outputs is what will be used to calcuate the loss and to determine the accuracy of models prediction. 
#  - We will initiate an instance of the network called `model`. This instance will be used for training and then to save the final trained model for future inference. 
#  
# #### Loss Function and Optimizer
#  - `Optimizer` is defined in the next cell.
#  - We do not define any `Loss function` since the specified model already outputs `Loss` for a given input. 
#  - `Optimizer` is used to update the weights of the neural network to improve its performance.
#  
# #### Further Reading
# - You can refer to my [Pytorch Tutorials](https://github.com/abhimishra91/pytorch-tutorials) to get an intuition of Loss Function and Optimizer.
# - [Pytorch Documentation for Loss Function](https://pytorch.org/docs/stable/nn.html#loss-functions)
# - [Pytorch Documentation for Optimizer](https://pytorch.org/docs/stable/optim.html)
# - Refer to the links provided on the top of the notebook to read more about `BertForTokenClassification`. 

# In[ ]:


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=18)
        # self.l2 = torch.nn.Dropout(0.3)
        # self.l3 = torch.nn.Linear(768, 200)
    
    def forward(self, ids, mask, labels):
        output_1= self.l1(ids, mask, labels = labels)
        # output_2 = self.l2(output_1[0])
        # output = self.l3(output_2)
        return output_1


# In[ ]:


model = BERTClass()
model.to(dev)


# In[ ]:


optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


# <a id='section05'></a>
# ### Fine Tuning the Model
# 
# After all the effort of loading and preparing the data and datasets, creating the model and defining its loss and optimizer. This is probably the easier steps in the process. 
# 
# Here we define a training function that trains the model on the training dataset created above, specified number of times (EPOCH), An epoch defines how many times the complete data will be passed through the network. 
# 
# Following events happen in this function to fine tune the neural network:
# - The dataloader passes data to the model based on the batch size. 
# - Subsequent output from the model and the actual category are compared to calculate the loss. 
# - Loss value is used to optimize the weights of the neurons in the network.
# - After every 500 steps the loss value is printed in the console.
# 
# As you can see just in 1 epoch by the final step the model was working with a miniscule loss of 0.08503091335296631 i.e. the output is extremely close to the actual output.

# In[ ]:


def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(dev, dtype = torch.long)
        mask = data['mask'].to(dev, dtype = torch.long)
        targets = data['tags'].to(dev, dtype = torch.long)

        loss = model(ids, mask, labels = targets)[0]

        # optimizer.zero_grad()
        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(optimizer)
        xm.mark_step() 


# In[ ]:


for epoch in range(5):
    train(epoch)


# <a id='section06'></a>
# ### Validating the Model
# 
# During the validation stage we pass the unseen data(Testing Dataset) to the model. This step determines how good the model performs on the unseen data. 
# 
# This unseen data is the 30% of `ner.csv` which was seperated during the Dataset creation stage. 
# During the validation stage the weights of the model are not updated. Only the final output is compared to the actual value. This comparison is then used to calcuate the accuracy of the model. 
# 
# The metric used for measuring the performance of model for these problem statements is called F1 score. We will create a helper function for helping us with f1 score calcuation and also import a library for the same. `seqeval`

# In[ ]:


from seqeval.metrics import f1_score

def flat_accuracy(preds, labels):
    flat_preds = np.argmax(preds, axis=2).flatten()
    flat_labels = labels.flatten()
    return np.sum(flat_preds == flat_labels)/len(flat_labels)


# In[ ]:


def valid(model, testing_loader):
    model.eval()
    eval_loss = 0; eval_accuracy = 0
    n_correct = 0; n_wrong = 0; total = 0
    predictions , true_labels = [], []
    nb_eval_steps, nb_eval_examples = 0, 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(dev, dtype = torch.long)
            mask = data['mask'].to(dev, dtype = torch.long)
            targets = data['tags'].to(dev, dtype = torch.long)

            output = model(ids, mask, labels=targets)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            label_ids = targets.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)
            accuracy = flat_accuracy(logits, label_ids)
            eval_loss += loss.mean().item()
            eval_accuracy += accuracy
            nb_eval_examples += ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
        valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


# In[ ]:


valid(model, testing_loader)

