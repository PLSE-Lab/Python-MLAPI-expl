#!/usr/bin/env python
# coding: utf-8

# # Generate Text with OpenAI's GPT-2 Language Model (117M version)
# 
# ## For more information, refer to OpenAI's original blog post:
# https://blog.openai.com/better-language-models
# 
# ## Details:
# OpenAI only released their smaller 117M parameter model but even this small model performs suprisingly well. You can generate conditional samples from a given sentence or generate unconditional samples. 
# 
# ### Tuning parameters for optimal predictions
# The model starts repeating itself more often when given short prompts, but changing the temperature from the default of 0.7 can give you better results. Increasing the temperature forces the model to make more novel predictions, but often causes the model to go off topic. Decreasing the temperature keeps the model from going off topic, but causes the model to repeat itself more often. 
# 
# ## Options:
# ```
# --text : sentence to begin with.
# --quiet : not print all of the extraneous stuff like the "================"
# --nsamples : number of sample sampled in batch when multinomial function use
# --unconditional : If true, unconditional generation.
# --batch_size : number of batch size
# --length : sentence length (< number of context)
# --temperature: the thermodynamic temperature in distribution (default 0.7)
# --top_k : Returns the top k largest elements of the given input tensor along a given dimension. (default 40)
# ```
# 
# ## Code:
#  https://github.com/graykode/gpt-2-Pytorch

# In[ ]:


# From https://github.com/graykode/gpt-2-Pytorch
import os
get_ipython().system('git clone https://github.com/graykode/gpt-2-Pytorch.git')
os.chdir('./gpt-2-Pytorch')
get_ipython().system('curl --output gpt2-pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin')
get_ipython().system('pip install -r requirements.txt')


# # Generating conditional samples

# In[ ]:


get_ipython().system('python main.py --text "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."')


# In[ ]:


os.chdir('../')
get_ipython().system('rm -rf gpt-2-Pytorch')

