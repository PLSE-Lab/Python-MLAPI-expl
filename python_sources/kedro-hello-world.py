#!/usr/bin/env python
# coding: utf-8

# **[Kedro](https://kedro.readthedocs.io/en/latest/index.html) Hello World**
# 
# **Goal:**
# A public kernel for those like me who want to learn and start using a better development and production workflow in machine learning. And for those who do not always have access to the latest packages or best practice in their day jobs.
# 
# **Background:**
# I'd met QuantumBlack at one of their Meetups earlier this year and have been waiting for them to release kedro. I've decided to give it a go and make it work on Kaggle!
# 
# The example below is lifted from their 'Hello World' example and adapted to work on Kaggle. 

# In[ ]:


get_ipython().system('kedro info  # requires Kaggle Settings -> Packages Install kedro')
get_ipython().system('pwd')


# In[ ]:


# requires Kaggle Settings -> Internet On
# used to display a tree of directory structure created by kedro
get_ipython().system('apt-get -qq install tree')


# In[ ]:


# Kaggle does not support interactive shell, so creating structure using yaml config instead
import yaml

yaml_dict = dict(output_dir='.',
    project_name='Getting Started',
    repo_name='getting-started',
    python_package='getting_started',
    include_example=True
    )

with open('kedro_new_config.yml', 'w') as outfile:
    yaml.dump(yaml_dict, outfile, default_flow_style=False)

get_ipython().system('cat kedro_new_config.yml')


# In[ ]:


get_ipython().system('kedro new --config kedro_new_config.yml')


# In[ ]:


get_ipython().run_line_magic('cd', './getting-started')
get_ipython().system('tree')


# In[ ]:


get_ipython().system('kedro test')


# In[ ]:


get_ipython().system('kedro run')


# Looks like it worked!
# 
# Next kernel will attempt QuantumBlack's Spaceflights tutorial
