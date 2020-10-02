#!/usr/bin/env python
# coding: utf-8

# # Assignment 5: IMDB review sample Classification with fast.ai Deep Learning Framework

# **Question 1: (10 points)**
# 
# Export the sample data set you created in [Classwork 3](https://colab.research.google.com/drive/1be7ksupqRkdjU1fZUAS37F5GiLCkFzkR) with pickle library, name it `imdb-sample.pickle`, and upload it to Google Colab. Then run the following codes.

# In[ ]:


import os
print(os.listdir("../input"))
import pickle
#"../input/imdb-sample.pickle"


# In[ ]:


from fastai.text import *
path = Path('.')
with open('../input/imdb-sample.pickle', 'rb') as f:
    train, valid = pickle.load(f)


# In[ ]:


valid.tail()


# You should see the output like this:
# 
# ![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-03-26-10-11-10-085256.png)

# **Question 2: (10 points)**
# 
# Create a TextLMDataBunch instance called `data_lm`, load your train and valid Dataframe into it, and run the following code.

# In[ ]:


# Your code here:
data_lm = TextLMDataBunch.from_df(path=path, train_df=train , valid_df=valid, text_cols=0, label_cols=1 )


# In[ ]:


data_lm.show_batch()


# You should see the output like this:
# 
# ![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-03-26-10-14-25-459713.png)

# **Question 3: (10 points)**
# 
# Create a `language_model_learner` named `learn`, use `data_lm` as input data, `AWD_LSTM` as architecture, and choose 0.5 as Dropout rate. Draw the result of learning rate finder.

# In[ ]:


# Your code here:
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn.lr_find()
learn.recorder.plot(skip_end=5)


# You should see the output like this:
# 
# ![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-03-26-10-17-47-701082.png)

# **Question 4: (20 points)**
# 
# Fit one cycle with your language learner (`learn`), unfreeze it and fit another 3 cycles. Save the language learner's encoder as `ft_enc`.

# In[ ]:


# Your code here:
learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
learn.save('fit_head')
learn.load('fit_head');
learn.unfreeze()
learn.fit_one_cycle(3, 1e-3, moms=(0.8,0.7))
learn.save('fine_tuned')
learn.load('fine_tuned')
learn.save_encoder('ft_enc')


# You should see the output like this:
# 
# ![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-03-26-10-20-09-953060.png)

# **Question 5: (10 points)**
# 
# Create a TextClasDataBunch instance called `data_clas`, load your train and valid Dataframe into it, use the vocab from `data_lm.train_ds.vocab`, set batch size as 32, and run the following code.

# In[ ]:


# Your code here:
data_clas = TextClasDataBunch.from_df(path, train_df=train, valid_df=valid, text_cols=0, label_cols=1)


# In[ ]:


data_clas.show_batch()


# You should see the output like this:
# 
# ![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-03-26-10-24-53-893234.png)

# **Question 6: (10 points)**
# 
# Create a `text_classifier_learner` named `learn`, use `data_clas` as input data, `AWD_LSTM` as architecture, and choose 0.5 as Dropout rate. Note to compare the result with Scikit-learn and textblob later, you need to make sure Precision and Recall are in the metrics list. Load the encoder  (`ft_enc`) you saved just now into `learn`. Draw the result of learning rate finder.

# In[20]:


# Your code here:
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc')
learn.lr_find()
learn.recorder.plot(skip_end=5)


# In[28]:


learn = text_classifier_learner(data_clas,AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc')
f1_label1 = Precision()
f1_label0 = Recall()
learn.metrics=[accuracy, f1_label1,f1_label0]


# You should see the output like this:
# 
# ![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-03-26-10-27-55-239469.png)

# **Question 7: (20 points)**
# 
# Fit one cycle with your text classifier learner (`learn`). Unfreeze the last two layers, and fit 3 cycles. Then unfreeze it totally, and fit another 2 cycles. Show the training result.

# In[29]:


# Your code here:
learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
learn.save('fit_head')
learn.load('fit_head');


# In[33]:


learn.unfreeze()
learn.fit_one_cycle(3, 1e-3, moms=(0.8,0.7))
learn.save('fine_tuned')
learn.load('fine_tuned')
learn.save_encoder('result')
learn.unfreeze()
learn.fit_one_cycle(2, 1e-3, moms=(0.8,0.7))
learn.save('fine_tuned')
learn.load('fine_tuned')
learn.save_encoder('result')


# You should see the output like this:
# 
# ![](https://github.com/wshuyi/github_pub_img/raw/master/assets/2019-03-26-10-31-32-631875.png)

# **Question 8: (10 points)**
# 
# Comparing the result with those from textblob, scikit-learn in Classwork 3, what is your finding? How about comparing with the result from Self-study 7 (the whole IMDB dataset)? Write down your answer and comments.

# *Your answer here:*
# 
# 

# **The result from textblob on classification in the test dataset shown the prediction result of each sentiment is:**
# 
# 0: precision 0.91  recall 0.43 f1-score  0.59  support 500                               
# 1: precision 0.63  recall 0.96 f1-score  0.76   support 500
#        
# **The result from Scikit-Learn on classification in the test dataset shown the prediction result of each sentiment is:**
# 
# 0: precision 0.79  recall 0.85 f1-score  0.82  support 500                               
# 1: precision 0.84  recall 0.78 f1-score  0.81  support 500
#           
# **The result from Self-Study 7 with the whole dataset, which does not provide precision and recall:**
# 
# * epoch	train_loss	valid_loss	accuracy	time
# *  0	0.246402	0.166181	0.936880	07:58
# *  1	0.254337	0.163558	0.938560	07:04
# Note: Part of the processing in Self-Study 7 was incomplete due to timeout in Kaggle after maxing out on the time allowed. https://www.kaggle.com/ladybirdcj/fast-ai-v3-lesson-3-imdb-ckj
# 
# **The result from this assignment is:**
# 
# * epoch	train_loss	valid_loss	accuracy	precision	recall	time
#   *0	0.261334	0.327046	0.851000	0.922892	0.936000	00:27
#   *1	0.235252	0.271704	0.887000	0.875728	0.872000	00:27
# 
# According to Scikit_Learn documentation on the classification report, the precision is the ratio of true positives to the sum of true and false positives. Recall is the ability of a classifier to find all positive instances. The F1 score is a weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. As a rule of thumb, the weighted average of F1 should be used to compare classifier models, not global accuracy. Reference:https://www.scikit-yb.org/en/latest/api/classifier/classification_report.html
# 
# Since we are missing part of the metrics from Self-Study 7, they are not exact metrics to compare between all four models. However, we can look at the overall scores and distribution of those scores to get a sense of the performance of each model. 
# 
# The largest dataset from Self-Study 7 is producing the highest accuracy rate (0.94 for 0 and 0.94 for 2). This is apparent due to the large dataset that is used to develop the training and validation step. For this assignment, we are using the parititon data for the test and valid dataset from the original file. Therefore, the dataset is much smaller and is producing similar result to Scikit-Learn f1-score to fastai accuracy with a +/- 5% range. However, the precision and recall of fastai is higher to both sci-kit learn and textblob. 
# 
# Textblob result has the same dataset of 500 positive and 500 negative sentiment but performance varied a lot between precision and recall. Therefore, it is the least reliable model to use out of the four methods.  

# 

# In[ ]:




