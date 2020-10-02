#!/usr/bin/env python
# coding: utf-8

# # CORD-19 QA Transformer Model
# 
# This notebook builds a CORD-19 extractive QA model based on the BERT-Small model. The model is designed to help extract structured information out of CORD-19 articles. 
# 
# The following sections go through detailed steps in building these models, background information on each step and a link to pre-trained models on Hugging Face's website.

# # Install libraries and download build scripts

# In[ ]:


# Install latest transformers library
get_ipython().system('pip install transformers --upgrade')

# Language modeling
get_ipython().system('wget -P /tmp https://raw.githubusercontent.com/huggingface/transformers/master/examples/language-modeling/run_language_modeling.py')

# SQuAD 2.0
get_ipython().system('wget -P /tmp https://raw.githubusercontent.com/huggingface/transformers/master/examples/question-answering/run_squad.py')
get_ipython().system('wget -P /tmp https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json')
get_ipython().system('wget -P /tmp https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json')

# Remove wandb
get_ipython().system('pip uninstall -y wandb')


# # Model Discussion
# 
# [BERT-Small](https://huggingface.co/google/bert_uncased_L-6_H-512_A-8) was selected as the root model after testing various models ([BERT-Base](https://huggingface.co/bert-base-uncased), [ALBERT-base v2](https://huggingface.co/albert-base-v2), [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased), [BioBERT](https://huggingface.co/monologg/biobert_v1.1_pubmed), [DistilBERT](https://huggingface.co/distilbert-base-uncased), [BERT-Tiny](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) and [BERT-Mini](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4)).
# 
# The CORD-19 QA extractive model is primarily designed to execute subqueries against large lists of search results, therefore it needs to be performant. BERT-Base, ALBERT-base v2 and SciBERT all roughly had the same execution time. BERT-Tiny and BERT-Mini were much faster than BERT-Small but just couldn't reach a reasonable level of answering accuracy. DistilBERT was slower than BERT-Small.
# 
# The second part of model selection is whether to start with a general English language model or a medical/scientific model. My assumption going in was that models trained on medical data would be more accurate and perform better. But what I found was that BERT-Small performed better as I built out the CORD-19 QA dataset. My hypothesis behind this is that given my limited medical background, the way I constructed the questions to ask of the data was better suited to a general language model.
# 
# Others may find different results and it's easy to modify/test. Simply clone this notebook and change a single line in the code below to substitute an alternate model.

# # bert-small-cord19
# 
# The file [cord19.txt](https://www.kaggle.com/davidmezzetti/cord19-qa?select=cord19.txt) is a partial export of sentences from the CORD-19 dataset, representing the best articles, ones with detected study designs. 
# 
# A pretrained model is available on Hugging Face's website: [bert-small-cord19](https://huggingface.co/NeuML/bert-small-cord19)

# In[ ]:


get_ipython().system('python /tmp/run_language_modeling.py     --model_type bert     --model_name_or_path google/bert_uncased_L-6_H-512_A-8     --do_train     --mlm     --line_by_line     --block_size 512     --train_data_file ../input/cord19-qa/cord19.txt     --per_gpu_train_batch_size 4     --learning_rate 3e-5     --num_train_epochs 3.0     --output_dir bert-small-cord19     --save_steps 0     --overwrite_output_dir')


# # bert-small-cord19-squad2
# 
# The next step takes the fine-tuned language model and trains it on SQuAD 2.0. SQuAD 2.0 is a better fit than 1.1 as it handled abstaining from answering a question, which is important for this dataset.
# 
# A pretrained model is available on Hugging Face's website: [bert-small-cord19-squad2](https://huggingface.co/NeuML/bert-small-cord19-squad2)

# In[ ]:


get_ipython().system('python /tmp/run_squad.py     --model_type bert     --model_name_or_path bert-small-cord19     --do_train     --do_eval     --do_lower_case     --version_2_with_negative     --train_file /tmp/train-v2.0.json     --predict_file /tmp/dev-v2.0.json     --per_gpu_train_batch_size 8     --learning_rate 3e-5     --num_train_epochs 3.0     --max_seq_length 384     --doc_stride 128     --output_dir bert-small-cord19-squad2     --save_steps 0     --threads 2     --overwrite_cache     --overwrite_output_dir')


# # bert-small-cord19qa
# 
# The last step is taking the fine-tuned SQuAD 2.0 model and further fine-tuning it on [700+ CORD-19 specific QA pairs](https://www.kaggle.com/davidmezzetti/cord19-qa).
# 
# A pretrained model is available on Hugging Face's website: [bert-small-cord19qa](https://huggingface.co/NeuML/bert-small-cord19qa)

# In[ ]:


get_ipython().system('python /tmp/run_squad.py     --model_type bert     --model_name_or_path bert-small-cord19-squad2     --do_train     --do_lower_case     --version_2_with_negative     --train_file ../input/cord19-qa/cord19-qa.json     --per_gpu_train_batch_size 8     --learning_rate 5e-5     --num_train_epochs 10.0     --max_seq_length 384     --doc_stride 128     --output_dir bert-small-cord19qa     --save_steps 0     --threads 2     --overwrite_cache     --overwrite_output_dir')


# In[ ]:


# Remove large training cache files from output
get_ipython().system('rm -rf cached_*')


# # Testing the model
# Test the model with the following handful of question/context/answer groups

# In[ ]:


import csv
import string
import sys

from transformers.pipelines import pipeline

# Create NLP pipeline
nlp = pipeline("question-answering", model="bert-small-cord19qa", tokenizer="bert-small-cord19qa")

# Init questions/contexts/answers
questions = ["What containment method?",
             "What weather factor?",
             "What is the incubation period range?",
             "What is model prediction?",
             "What is cancer risk number?"]

contexts = ["With contact tracing, the proportion q of individuals exposed to the virus is quarantined.",
            "Higher temperatures and higher RH (38 C, and >95% RH) have been found to reduce virus viability.",
            "The average incubation period is 5-6 days, ranging from 1-14 days 6 .",
            "Therefore, if person-to-person transmission persists from February, we predict the epidemic peak would occur in June.",
            "cancer was associated with an increased risk for severe events (odds ratio, 5.34; 95% confidence interval [CI], 1.80 to 16.18; P = .0026),"]

answers = ["contact tracing",
           "Higher temperatures and higher RH",
           "5-6 days",
           "epidemic peak would occur in June",
           "odds ratio, 5.34"]

# Show results
for x, result in enumerate(nlp(question=questions, context=contexts)):
    # Remove leading/trailing punctuation
    result["answer"] = result["answer"].strip(string.punctuation)
    print(result, answers[x])

