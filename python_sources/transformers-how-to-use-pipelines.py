#!/usr/bin/env python
# coding: utf-8

# ## How can I leverage State-of-the-Art Natural Language Models with only one line of code ?

# Newly introduced in transformers v2.3.0, **pipelines** provides a high-level, easy to use,
# API for doing inference over a variety of downstream-tasks, including: 
# 
# - Sentence Classification (Sentiment Analysis): Indicate if the overall sentence is either positive or negative. _(Binary Classification task or Logitic Regression task)_
# - Token Classification (Named Entity Recognition, Part-of-Speech tagging): For each sub-entities _(**tokens**)_ in the input, assign them a label _(Classification task)_.
# - Question-Answering: Provided a tuple (question, context) the model should find the span of text in **content** answering the **question**.
# - Mask-Filling: Suggests possible word(s) to fill the masked input with respect to the provided **context**.
# - Feature Extraction: Maps the input to a higher, multi-dimensional space learned from the data.
# 
# Pipelines encapsulate the overall process of every NLP process:
#  
#  1. Tokenization: Split the initial input into multiple sub-entities with ... properties (i.e. tokens).
#  2. Inference: Maps every tokens into a more meaningful representation. 
#  3. Decoding: Use the above representation to generate and/or extract the final output for the underlying task.
# 
# The overall API is exposed to the end-user through the `pipeline()` method with the following 
# structure:
# 
# ```python
# from transformers import pipeline
# 
# # Using default model and tokenizer for the task
# pipeline("<task-name>")
# 
# # Using a user-specified model
# pipeline("<task-name>", model="<model_name>")
# 
# # Using custom model/tokenizer as str
# pipeline('<task-name>', model='<model name>', tokenizer='<tokenizer_name>')
# ```

# In[ ]:


get_ipython().system('pip install transformers')


# In[ ]:


from __future__ import print_function
import ipywidgets as widgets
from transformers import pipeline


# ## 1. Sentence Classification - Sentiment Analysis

# In[ ]:


nlp_sentence_classif = pipeline('sentiment-analysis')
nlp_sentence_classif('Such a nice weather outside !')


# ## 2. Token Classification - Named Entity Recognition

# In[ ]:


nlp_token_class = pipeline('ner')
nlp_token_class('Hugging Face is a French company based in New-York.')


# ## 3. Question Answering

# In[ ]:


nlp_qa = pipeline('question-answering')
nlp_qa(context='Hugging Face is a French company based in New-York.', question='Where is based Hugging Face ?')


# ## 4. Text Generation - Mask Filling

# In[ ]:


nlp_fill = pipeline('fill-mask')
nlp_fill('Hugging Face is a French company based in <mask>')


# ## 5. Projection - Features Extraction 

# In[ ]:


import numpy as np
nlp_features = pipeline('feature-extraction')
output = nlp_features('Hugging Face is a French company based in Paris')
np.array(output).shape   # (Samples, Tokens, Vector Size)


# Alright ! Now you have a nice picture of what is possible through transformers' pipelines, and there is more
# to come in future releases. 
# 
# In the meantime, you can try the different pipelines with your own inputs

# In[ ]:


task = widgets.Dropdown(
    options=['sentiment-analysis', 'ner', 'fill_mask'],
    value='ner',
    description='Task:',
    disabled=False
)

input = widgets.Text(
    value='',
    placeholder='Enter something',
    description='Your input:',
    disabled=False
)

def forward(_):
    if len(input.value) > 0: 
        if task.value == 'ner':
            output = nlp_token_class(input.value)
        elif task.value == 'sentiment-analysis':
            output = nlp_sentence_classif(input.value)
        else:
            if input.value.find('<mask>') == -1:
                output = nlp_fill(input.value + ' <mask>')
            else:
                output = nlp_fill(input.value)                
        print(output)

input.on_submit(forward)
display(task, input)


# In[ ]:


context = widgets.Textarea(
    value='Einstein is famous for the general theory of relativity',
    placeholder='Enter something',
    description='Context:',
    disabled=False
)

query = widgets.Text(
    value='Why is Einstein famous for ?',
    placeholder='Enter something',
    description='Question:',
    disabled=False
)

def forward(_):
    if len(context.value) > 0 and len(query.value) > 0: 
        output = nlp_qa(question=query.value, context=context.value)            
        print(output)

query.on_submit(forward)
display(context, query)

