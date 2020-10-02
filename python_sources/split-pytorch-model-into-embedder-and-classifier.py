#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import transformers
from transformers import (
    BertPreTrainedModel,
    BertConfig,
    RobertaConfig,
    BertModel,
    RobertaModel
)


# ## Models as two distinct parts
# 
# In this competiton NLP transformer-based models like BERT consist of two parts:
# 
# 1. an embedding layer that embeds a batch of integer vectors into a batch of dense vectors
# 2. a classification head that classifies a batch of dense vectors into a batch of logits
# 
# Embedding layers such as BERT, RoBERTa, XLNet, etc. are essentially static: there are no hyperparameters to tune. 
# Instead, a model can be improved by tweaking the classification head by:
# 
# * selecting which of the 12 transformer hidden layer outputs to use and in which way (sum, average, concatenate, etc.)
# * choosing the hyperparameters of the classification layer (How many layers? Which dropout?)
# 
# I was wondering if there's a way to define many combination of embedders and transformers without making copies of the entire model's code.
# 
# 
# ### Why bother?
# 
# The advantages of this [separation of concerns](https://en.wikipedia.org/wiki/Separation_of_concerns) are obvious:
# 
# * bugs become less likely and the code more readable
# * we can use the same classifier across multiple embedders
# 
# ### Combining models in PyTorch
# 
# I found a [post on the PyTorch forums](https://discuss.pytorch.org/t/combining-trained-models-in-pytorch/28383) that gave me inspiration on how to accomplish this.
# In the code below I show how a PyTorch model can be constructed as a combination of an embedder and a classifier. This works because BertModel is a PyTorch torch.nn.Module sub-class.

# ## 1. Models and their config
# 
# I use a BERT and RoBERTa model and config from public datasets on Kaggle.

# In[ ]:


BERT_PATH = '../input/bert-base-uncased'
ROBERTA_PATH = '../input/roberta-base'


# In[ ]:


bert_config = BertConfig.from_pretrained(
    f'{BERT_PATH}/config.json'
)
bert_config.output_hidden_states = True
roberta_config = RobertaConfig.from_pretrained(
    f'{ROBERTA_PATH}/config.json',
)
roberta_config.output_hidden_states = True


# ## 2. Two embedders

# Here we define two embedders that return all 12 transformer hidden layer outputs. 
# Therefore it's important to set the `output_hidden_states` to True in the model's config.
# I also defined a parameter `freeze_weights` to freeze the weights of the embedder during training.
# This way, we can quickly experiment with different classifiers without retraining the embedder's parameters. 

# In[ ]:


class BERTEmbedder(BertPreTrainedModel):
    def __init__(self, config=bert_config, freeze_weights=False):
        super().__init__(config=bert_config)
        self.bert = BertModel.from_pretrained(
            BERT_PATH,
            config=bert_config
        )
        if freeze_weights is True:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, ids, mask, token_type_ids):
        _, _, hidden_states = self.bert(
            ids,
            mask,
            token_type_ids
        )
        return hidden_states


class RoBERTaEmbedder(BertPreTrainedModel):
    def __init__(self, config=roberta_config, freeze_weights=False):
        super().__init__(config=roberta_config)
        self.roberta = RobertaModel.from_pretrained(
            ROBERTA_PATH,
            config=roberta_config
        )
        if freeze_weights is True:
            for param in self.roberta.parameters():
                param.requires_grad = False

    def forward(self, ids, mask, token_type_ids):
        _, _, hidden_states = self.roberta(
            ids,
            mask,
            token_type_ids
        )
        return hidden_states
    


# ## 3. Three classes of classifiers

# Below I defined three classifiers that accept either 2 or 3 of the transformer hidden layer outputs.
# They calculate the logits for the sequences start and end token index.

# In[ ]:


class ClassifierConcatLastTwo(nn.Module):
    """ concatenate output of last two hidden layers """
    def __init__(self, hidden_size=768):
        super().__init__()
        self.drop_out = nn.Dropout(0.1)
        self.linear = nn.Linear(hidden_size * 2, 2)
        torch.nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, hidden_states):
        out = torch.cat((hidden_states[-1], hidden_states[-2]), dim=-1)
        out = self.drop_out(out)
        logits = self.linear(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits


class ClassifierConcatLastThree(nn.Module):
    """ concatenate output of last three hidden layers """
    def __init__(self, hidden_size=768):
        super().__init__()
        self.drop_out = nn.Dropout(0.1)
        self.linear = nn.Linear(hidden_size * 3, 2)
        torch.nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, hidden_states):
        out = torch.cat(
            (hidden_states[-1], hidden_states[-2], hidden_states[-3]),
            dim=-1
        )
        out = self.drop_out(out)
        logits = self.linear(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits


class ClassifierAverageLastThree(nn.Module):
    """ average output of last three hidden layers """
    def __init__(self, hidden_size=768):
        super().__init__()
        self.drop_out = nn.Dropout(0.1)
        self.linear = nn.Linear(hidden_size, 2)
        torch.nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, hidden_states):
        out = torch.stack(
            (hidden_states[-1], hidden_states[-2], hidden_states[-3])
        )
        out = self.drop_out(out)
        logits = self.linear(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits


# ## 4. Class to combine an embedder with a classifier

# The class below combines an embedder with a classifier by combining their forward methods into one forward method.

# In[ ]:


class CombinedModel(nn.Module):
    def __init__(self, embedder, classifier):
        super().__init__()
        self.embedder = embedder
        self.classifier = classifier

    def forward(self, ids, mask, token_type_ids):
        hidden_states = self.embedder(ids, mask, token_type_ids)
        logits = self.classifier(hidden_states)
        start_logits = logits[0]
        end_logits = logits[1]
        return start_logits, end_logit


# To make experimentation easy, I put the various classes into Python dicts which is an idea I took from Abhishek.

# In[ ]:


EMBEDDER_DISPATCHER = {
    'bert': BERTEmbedder(),
    'roberta': RoBERTaEmbedder()
}

CLASSIFIER_DISPATCHER = {
    'concat_last_two': ClassifierConcatLastTwo(),
    'concat_last_three': ClassifierConcatLastThree(),
    'average_last_three': ClassifierAverageLastThree()
}


# ## 5. Examples

# Below I show some examples of how to use the dicts to create various model combinations.

# ### BERT embedder with classifier that concatenates last two hidden layer ouputs

# In[ ]:


embedder = EMBEDDER_DISPATCHER['bert']
classifier = CLASSIFIER_DISPATCHER['concat_last_two']
model = CombinedModel(embedder, classifier)
print(model)


# ### RoBERTa embedder with classifier that concatenates last two hidden layer ouputs

# In[ ]:


embedder = EMBEDDER_DISPATCHER['roberta']
classifier = CLASSIFIER_DISPATCHER['concat_last_two']
model = CombinedModel(embedder, classifier)
print(model)


# I hope you enjoyed this little PyTorch trick!
