#!/usr/bin/env python
# coding: utf-8

# # Pre-trained LDA Topic Modelling For News Website Topics
# 
# A pre-trained model and Python library for LDA topic modelling. This can be re-trained, however, has been trained by default for identifying topics for online data (Specifically news articles). [Github link](https://github.com/user1342/Pre-Trained-News-Topic_Modelling)
# 
# ## Installation
# Install the library with  ```pip``` and the GitHub link.
# ```
# python -m pip install git+https://github.com/user1342/Topic-Modelling-For-Online-Data.git
# ```
# 
# ## Usage
# 
# ```python
# from topic_modelling.topic_modelling import topic_modelling
# 
# modeller = topic_modelling()
# 
# print(modeller.identify_topic("hello world"))
# print(modeller.get_topics())
# ```
# 
# ### Re-training The Model
# The number of topics, passes, and dataset (seperated by ```\n``` with one line per article) can be defined. Topics defaults to 4, passes to 15, and the dataset defaults to a list of 22545 news articles located in the ```data``` directory of the library called ```dataset.txt```. Once retrained the new model files will be created in the libraries ```models``` folder and will be used for later use.
# 
# ```python
# modeller = topic_modelling()
# modeller.re_train(number_of_topics=4, number_of_passes=70, dataset="new_data.txt")
# ```
# 
# ### Visualising Groups
# When using a Jupyter Notebook you can use the ```get_lda_display``` function to return the LDA data to visualise the model.
# 
# ```python
# modeller = topic_modelling()
# pyLDAvis.display(modeller.get_lda_display())
# ```
# ## Groups
# The model is pre-trained for 4 'news-related' groups. These group descriptions are detailed below:
# 
# | Group ID | Group Description | Example Keywords                                   |
# |----------|-------------------|----------------------------------------------------|
# | 0        | People            | Player, People, Family, League                     |
# | 1        | Government        | President, Government, Company, National, Minister |
# | 2        | Technology        | Google, Apple, Phone, Technology, Microsoft        |
# | 3        | Residential       | Bedroom, Garden, House, Water                      |
# 
# ## Dataset
# This dataset has been gathered from the global most used news websites (written in English), where they're most recent pages have been identified as news (using:  [Website-Category-Identification-Tool](https://github.com/user1342/Website-Category-Identification-Tool)) then these articles have been added to the dataset.
# 
# - Kaggle Dataset and Kernel: www.kaggle.com/dataset/061b55bb510ebb7c484ef2c9ed5f5ddc474239d5952d9a75e7cc587923bec7df
# 

# In[ ]:


# When using the Kaggle consol the package download wouldn't persist. 
# OS.system being used instead.
import os
os.system("pip install git+https://github.com/user1342/Topic-Modelling-For-Online-Data.git")


# In[ ]:


from topic_modelling.topic_modelling import topic_modelling

# Return a list of the current topics and their keywords
modeller = topic_modelling()
print(modeller.get_topics())


# In[ ]:


# Visualise the topics
import pyLDAvis.gensim
pyLDAvis.display(modeller.get_lda_display())


# In[ ]:


# Identify the topic of a given news article
print(modeller.identify_topic("With the introduction of the General Data Protection Regulation (GDPR), the EU is enacting a set of mandatory regulations for businesses that go into effect soon, on May 25, 2018. Organisations found in non-compliance could face hefty penalties of up to 20 million euros, or 4 percent of worldwide annual turnover, whichever is higher."))


# In[ ]:


# Retrain the model. By default it uses a pre-provided list of 22545 news articles. However this can be changed. You can also specify the amount of passes and groups. 
modeller.re_train(number_of_topics=3, number_of_passes=15)
# Display the new model
pyLDAvis.display(modeller.get_lda_display())

