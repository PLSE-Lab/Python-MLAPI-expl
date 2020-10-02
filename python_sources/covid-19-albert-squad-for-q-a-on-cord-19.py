#!/usr/bin/env python
# coding: utf-8

# This work uses the [**ALBERT**](https://arxiv.org/abs/1909.11942) model (**a light [BERT](https://en.wikipedia.org/wiki/BERT_(language_model) model**) to perform **question answering** tasks on CORD-19 dataset.
# 
# Currently, ALBERT outperforms most other BERT variants, including BERT itself, on the popular Q&A benchmark [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/). As of April 9, 2020, it is ranked 9th in all Q&A models and its accuracy is only marginally lower than the top ones. Therefore, it is definitely worth the time to apply ALBERT to this urgent task.
# 
# ALBERT is relatively new (developed around Sep 2019), so it is difficult to find pretrained model specifically fine-tuned to Q&A, unlike BERT model. So I have went ahead and fine-tuned it with SQuAD 2.0 myself, and I simply attached the pretrained model as a Kaggle dataset. This notebook will use the pretrained model. Both pretraining and inference are done thanks to Huggingface's [Transformers](https://github.com/huggingface/transformers) module.

# ALBERT is a bit computationally heavy. We could use some methods to reduce ALBERT calculations, e.g. do simple search then apply ALBERT only to the search results. But I found ALBERT could capture interesting relationships that traditional search engines couldn't. So I decided to apply ALBERT to all ~40k articles in the dataset. Overall, it takes around 15 minutes for a full iteration using the "base" ALBERT model, which is still managable. We could also use other methods, perhaps caching, to reduce ALBERT run time. But for simplicity and time, I sticked to the pure ALBERT approach.
# 
# This notebook is for walking through the code. I will submit individual notebooks to the 9 tasks very soon.
# 
# It is highly recommended to use GPU to run this notebook.

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


# Next, we will define the function where we inference the pre-trained ALBERT model over all CORD-19 papers.

# In[ ]:


from tqdm.notebook import tqdm
import numpy as np

def inference_ALBERT(question):
    
    spans, scores, token_ids = [], [], []
    
    # Iterating over all CORD-19 articles and perform model inference
    for i in tqdm(range(len(data))): # tqdm is for showing the process bar
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


# Next, we will define the function where we display the results in an organized way.

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


# Finally, we test the display results function. 
# 
# This notebook is for walking through the code. I will submit individual notebooks to the 9 tasks very soon.

# In[ ]:


# Combining the inference function and the display function into one
def inference_ALBERT_and_display_results(question, 
                                         first_n_entries=15,
                                         max_disp_len=100):
    
    spans, scores, token_ids = inference_ALBERT(question)
    display_results(spans, scores, token_ids, 
                    first_n_entries, max_disp_len)

# Testing the functions
inference_ALBERT_and_display_results(
    'What is the host range for the coronavirus pathogen?')
print('Done!')

