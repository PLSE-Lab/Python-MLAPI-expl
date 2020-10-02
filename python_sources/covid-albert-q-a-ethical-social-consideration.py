#!/usr/bin/env python
# coding: utf-8

# This work uses the [**ALBERT**](https://arxiv.org/abs/1909.11942) model (**a light [BERT](https://en.wikipedia.org/wiki/BERT_(language_model) model**) to perform **question answering** tasks on CORD-19 dataset.
# 
# **Pros**: Currently, ALBERT outperforms most other BERT variants, including BERT itself, on the popular Q&A benchmark [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/). As of April 9, 2020, it is ranked 9th in all Q&A models and its accuracy is only marginally lower than the top ones.
# **Cons**: The ALBERT model was not specifically trained on bio- or medical-related database, so output is still sometimes inaccurate. Fine-tuning it on SQuAD-like bio database could help.
# 
# ALBERT is relatively new (developed around Sep 2019), so it is difficult to find pretrained model specifically fine-tuned to Q&A, unlike BERT model. So I have went ahead and fine-tuned it with SQuAD 2.0 myself, and I simply attached the pretrained model as a Kaggle dataset. This notebook will use the pretrained model. Both pretraining and inference are done thanks to Huggingface's [Transformers](https://github.com/huggingface/transformers) module.

# For brevity, this notebook is focused on the scientific task. For detailed walkthrough of the code, plase refer to the [notebook](https://www.kaggle.com/joljol/covid-19-albert-transformer-for-q-a-on-cord-19) from which this notebook was originally forked.

# In[ ]:


import torch 
device = 'cuda' if torch.cuda.is_available() else 'cpu' # GPU recommended

# Loading custom pre-trained ALBERT model already fine-tuned to SQuAD 2.0
import transformers
from transformers import AlbertTokenizer, AlbertForQuestionAnswering
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForQuestionAnswering.from_pretrained(
    '/kaggle/input' \
    '/nlp-albert-models-fine-tuned-for-squad-20'\
    '/albert-base-v2-tuned-for-squad-2.0').to(device)

# Loading the CORD-19 dataset and pre-processing
import pandas as pd
data = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv',
                   keep_default_na=False)
data = data[data['abstract']!='']        .reset_index(drop=True) # Remove rows with no abstracts


# Code for interfencing the ALBERT model.

# In[ ]:


import numpy as np
from tensorflow.keras.utils import Progbar

def inference_ALBERT(question):
    
    spans, scores, token_ids = [], [], []
    
    # Iterating over all CORD-19 articles and perform model inference
    progress_bar = Progbar(len(data))
    for i in range(len(data)):
        if i % 500 == 0:
            progress_bar.update(i)
        text = data['abstract'][i]
        input_ids = tokenizer.encode(question, text)
        
        # We have token limit of 512, so truncate if needed
        if len(input_ids) > 512:
            input_ids, token_type_ids =                 input_ids[:511] + [3], token_type_ids[:512]
                # [3] is the SEP token
        
        token_type_ids = [0 if i <= input_ids.index(3) 
                          else 1 for i in range(len(input_ids))]

        # Preparing the tensors for feeding into model
        input_ids_tensor = torch.tensor([input_ids]).to(device)
        token_type_ids_tensor = torch.tensor([token_type_ids]).to(device)
        
        # Performing model inference
        start_scores, end_scores =             model(input_ids_tensor, 
                  token_type_ids=token_type_ids_tensor)
        
        # Releasing GPU memory by moving each tensor back to CPU
        # If GPU is not used, this step is uncessary but won't give error
        input_ids_tensor, token_type_ids_tensor, start_scores, end_scores =             tuple(map(lambda x: x.to('cpu').detach().numpy(), 
                     (input_ids_tensor, token_type_ids_tensor, \
                      start_scores, end_scores)))
        # Let me know if there's an easier way to do this, as I mostly work
        # with tensorflow and I'm not very familiar with Keras

        # Appending results to the corresponding lists
        # Spans are the indices of the start and end of the answer
        spans.append( [start_scores.argmax(), end_scores.argmax()+1] )
        # Scores are the "confidence" level in the start and end
        scores.append( [start_scores.max(), end_scores.max()] )
        token_ids.append( input_ids )

    spans = np.array(spans, dtype='int')
    scores = np.array(scores)
    
    return spans, scores, token_ids


# Code for formatting and displaying results.

# In[ ]:


# Define a helper function to directly convert token IDs to string
convert_to_str = lambda token_ids:     tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(token_ids))

from IPython.display import display, HTML

def display_results(spans, scores, token_ids, first_n_entries=15,
                    max_disp_len=100):
    
    display(HTML(
        'Model output (<text style=color:red>red font</text> '\
        'highlights the answer predicted by ALBERT NLP model)'\
        ))
    
    # We first sort the results based on the confidence in either the 
    # start or end index of the answer, whichever is smaller
    min_scores = scores.min(axis=1) 
    sorted_idx = (-min_scores).argsort() # Descending order
    
    counter = 0    
    for idx in sorted_idx:
        
        # Stop if first_n_entries papers have been displayed
        if counter >= first_n_entries:
            break
        
        # If the span is empty, the model prdicts no answer exists 
        # from the article. In rare cases, the end is smaller than
        # the start. Both will be skipped
        if spans[idx,0] == 0 or spans[idx,1] == 0 or             spans[idx,1]<=spans[idx,0]:
            continue

        # Obtaining the start and end token indices of answer
        start, end = spans[idx, :]

        abstract = data['abstract'][idx]
        abstract_highlight = convert_to_str(token_ids[idx][start:end])
        
        # If we cannot fully convert tokens to original text,
        # we then use the detokenized text (lower cased)
        # Otherwise it would be best to have the original text,
        # because there's lots of formatting especially in bio articles
        start = abstract.lower().find(abstract_highlight)
        if start == -1:
            abstract = convert_to_str(token_ids[idx]
                                      [token_ids[idx].index(3)+1:])
            start = abstract.find(abstract_highlight)
            end = start + len(abstract_highlight)
            abstract = abstract[:-5] # to remove the [SEP] token in the end
        else:
            end = start + len(abstract_highlight)
            abstract_highlight = abstract[start:end]
        abstract_before_highlight, abstract_after_highlight =             abstract[: start],             abstract[end : ]
    
        # Putting information in HTML format
        html_str = f'<b>({counter+1}) {data["title"][idx]}</b><br>' +                    f'Confidence: {scores[idx].min():.2f} | ' +                    f'<i>{data["journal"][idx]}</i> | ' +                    f'{data["publish_time"][idx]} | ' +                    f'<a href={data["url"][idx]}>{data["doi"][idx]}</a>' +                    '<p style=line-height:1.1><font size=2>' +                    abstract_before_highlight +                    '<text style=color:red>%s</text>'%abstract_highlight +                    abstract_after_highlight + '</font></p>'
        
        display(HTML(html_str))
        
        counter += 1

# Combining the inference function and the display function into one
def inference_ALBERT_and_display_results(question, 
                                         first_n_entries=15,
                                         max_disp_len=100):
    
    spans, scores, token_ids = inference_ALBERT(question)
    display_results(spans, scores, token_ids, 
                    first_n_entries, max_disp_len)


# # **Subtask 1**
# 
# Original subtask description:
# 
# Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019

# In[ ]:


inference_ALBERT_and_display_results(
    'Articulation and translation of existing ethical '
    'principles and standards to salient issues in COVID-2019 ')


# # **Subtask 2**
# 
# Original subtask description:
# 
# Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight

# In[ ]:


inference_ALBERT_and_display_results(
    'Embedding ethics across all thematic areas, '
    'engage with novel ethical issues that arise and '
    'coordinate to minimize duplication of oversight ')


# # **Subtask 3**
# 
# Original subtask description:
# 
# Efforts to support sustained education, access, and capacity building in the area of ethics

# In[ ]:


inference_ALBERT_and_display_results(
    'Support for sustained education, access, and '
    'capacity building in the area of ethics ')


# # **Subtask 4**
# 
# Original subtask description:
# 
# Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.

# In[ ]:


inference_ALBERT_and_display_results(
    'Establishment of a team at WHO that will be '
    'integrated within multidisciplinary research and '
    'operational platforms and that will connect with '
    'existing and expanded global networks of social '
    'sciences')


# # **Subtask 5**
# 
# Original subtask description:
# 
# Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)

# In[ ]:


inference_ALBERT_and_display_results(
    'Development qualitative assessment frameworks '
    'to systematically collect information related to '
    'local barriers and enablers for the uptake and '
    'adherence to public health measures for prevention '
    'and control, including the rapid identification '
    'of the secondary impacts of these measures '
    '(e.g. use of surgical masks, modification of health '
    'seeking behaviors for SRH, school closures) ')


# # **Subtask 6**
# 
# Original subtask description:
# 
# Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.

# In[ ]:


inference_ALBERT_and_display_results(
    'Identification of how the burden of responding '
    'to the outbreak and implementing public health '
    'measures affects the physical and psychological '
    'health of those providing care for Covid-19 '
    'patients and identify the immediate needs that '
    'must be addressed ')


# # **Subtask 7**
# 
# Original subtask description:
# 
# Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media.

# In[ ]:


inference_ALBERT_and_display_results(
    'Identification the underlying drivers of fear, '
    'anxiety and stigma that fuel misinformation and 
    'rumor, particularly through social media.')

