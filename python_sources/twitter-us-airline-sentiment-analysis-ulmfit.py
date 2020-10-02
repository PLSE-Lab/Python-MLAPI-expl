#!/usr/bin/env python
# coding: utf-8

# # Fellowship.AI challenge 2020 - Twitter US Airline Sentiment Analysis with ULMFiT
# 

# Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task. It has been extremely useful for computer vision tasks. However, in their paper, [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf), Jeremy Howard and Sebastian Ruder proposed **Universal Language Model Fine-tuning (ULMFiT)**, a transfer learning method that can be used for most NLP tasks. They open-sourced the model, source, and methods for fine tuning a language model for a given dataset. The model has been pretrained on Wikitext-103 corpus consisting of 28,595 preprocessed Wikipedia articles and 103 million words.

# In this notebook, we shall be creating an ULMFiT model that can correctly classify the sentiment of tweets from from the [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) dataset. The data consists of 14640 labeled tweets about customer experiences/opinions regarding six major US airlines. The tweets are classified as negative, neutral and positive, and we will train a model to classify them as such.

# Let's start by importing the necessary libraries. We're using fast.ai as it contains an implementation of AWD-LSTM [(Merity et al., 2017a)](https://arxiv.org/pdf/1708.02182.pdf) architecure according to ULMFiT paper. It was found to result in lower error rates across the board.

# In[ ]:


from fastai.text import *

import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report


# In[ ]:


path = '../input/twitter-airline-sentiment/Tweets.csv'
df = pd.read_csv(path, index_col = 'tweet_id')
#df[['airline_sentiment','airline','text']]
df.head()


# ## Exploratory Analysis
# Let's examine our dataset.

# In[ ]:


pd.Series(df['airline_sentiment']).value_counts().plot(kind = "bar" , title = "Class Distribution", )


# In[ ]:


pd.crosstab(index = df["airline"] ,columns = df["airline_sentiment"] ).plot(kind = "bar", stacked=True, title="Sentiment Distribution by Airline", )


# In[ ]:


pd.Series(df["negativereason"]).value_counts().plot(kind = "bar" , title = "Reasons for Negative Reviews", )


# In[ ]:


pd.crosstab(index = df["airline"] ,columns = df["negativereason"] ).plot(kind = "bar", title = "Negative Reasons by Airline", stacked = True, figsize=(10,10))


# ## Model training
# 
# The only two relevant features for the task at hand are the actual tweet, and the given sentiment for it. Let's create a new dataframe comprised of said features.

# In[ ]:


df_final = df[['airline_sentiment', 'text']]
df_final.head()


# We'll split the dataset 90-10, and use the method ```from_df``` of the ```TextLMDataBunch``` module (to get the data ready for a language model).

# In[ ]:


train, val = train_test_split(df_final, test_size=0.1)
data_lm = TextLMDataBunch.from_df(path="/output/kaggle/working", train_df = train, valid_df = val)


# Let's see the modified data. The xx words, might look weird, but they are simply special FastAI tokens that represent various features of a sentence, such as start, unknown words, etc.

# In[ ]:


data_lm.show_batch()


# Next, we load up the pre-trained AWD-LSTM language model learner and feed it our data. 

# In[ ]:


learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5, model_dir='/output/kaggle/working')


# 
# We will use the fastai learning rate finder to determine how to set our learning rate. Learning rate is used to dictate the speed of the model in training, so it is important to choose a suitable value so we don't spend days training. Fast.ai explores ```lr``` from ```1e-07``` to ```10``` over 100 iterations by default.
# 
# ```recorder.plot()``` allows us to observe the value for which the minimum loss is observed. I set the ```suggestion``` parameter as ```True``` in earlier versions of the notebook, but I eventually started to observe the training gave better results when the ```lr``` was less than the suggestion by a factor of 10.

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# We will train our model for 4 epochs at a learning rate of ```1e-2``` and a momentum range between 0.8 and 0.7. Momentum controls the training process w.r.t learning rate. These values have been obtained by manual trial and error, as well as looking at the documentation and tutorial on the fast.ai website.

# In[ ]:


learn.fit_one_cycle(4, 1e-2, moms=(0.8, 0.7))
learn.recorder.plot()


# We will then unfreeze the langauge model and train it for another 4 epochs at a lower learning rate of 1e-3.

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(4, 1e-3, moms=(0.8,0.7))


# Let us save the encoding of the model to be used later for classification. 

# In[ ]:


learn.save_encoder('fine_tuned_enc')


# In[ ]:


#Testing the LM so far

learn.predict('Terrible customer service @UnitedAirlines!', n_words=10)


# Now that we are done training our language model, let's build the classifier. We will need to create an additional test split here for analyzing model accuracy later on.

# In[ ]:


train_valid, test = train_test_split(df_final, test_size=0.1)
train, val = train_test_split(train_valid, test_size=0.1)


# In[ ]:


#creating classifer data bunch with vocab from the LM

data_clas = TextClasDataBunch.from_df(path,train,val,test, vocab=data_lm.train_ds.vocab, 
                                      text_cols='text', label_cols='airline_sentiment', bs=32)


# In[ ]:


data_clas.show_batch()


# Finally we will create a text classifier learner. We'll load in the 'fine_tuned_enc' encoder that we had saved earlier.

# In[ ]:


learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, model_dir='/output/kaggle/working')


# In[ ]:


learn.load_encoder('fine_tuned_enc')
learn.freeze()


# In[ ]:


#finding lr for training process

learn.lr_find()
learn.recorder.plot()


# Instead of fine-tuning all the layers at once, the paper suggests gradually unfreezing the layers from the last layer. One point to note is that deeper layers require more training, so we will have to adjust the learning rates accordingly in each step.

# In[ ]:


learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.freeze_to(-2)
learn.fit_one_cycle(3, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))


# In[ ]:


learn.freeze_to(-3)
learn.fit_one_cycle(2, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))


# Now that everything is trained, it's time to see if our model actually works.

# In[ ]:


pred_class = learn.predict("Terrible Flight!")
print(f"Predicted Sentiment: {list(pred_class)[0]}")


# Looks like it's predicting the sentiment for an input string correctly.

# Let's create a new column in the ```test``` dataframe called ```pred_sentiment```, which will allow us to compare the actual sentiment of the tweet and the predicted.

# In[ ]:


test['pred_sentiment'] = test['text'].apply(lambda row: str(learn.predict(row)[0]))


# In[ ]:


learn.save('classifier')


# ## Model Evaluation

# In[ ]:


test.head()


# ```TextClassificationInterpretation``` provides an interpretation of classification based on input sensitivity for AWD-LSTM models. The ```show_intrinsic_attention function``` calculates the intrinsic attention of the input, which allows us to observe the importance of each word in the input sentence. The darker the shade, more the importance of the word.

# In[ ]:


text_ci = TextClassificationInterpretation.from_learner(learn)
test_text = "@UnitedAirlines I am extremely disappointed."
text_ci.show_intrinsic_attention(test_text,cmap=cm.Blues)


# Observing the confusion matrix, we can see that the most common wrong prediction is neutral tweets being classified as negative.

# In[ ]:


text_ci.plot_confusion_matrix()


# In[ ]:


text_ci.show_top_losses(5)


# Overall, the model achieved 83% accuracy, which is pretty good, considering we didn't have to perform any tokenization or excessive model training.

# In[ ]:


accuracy_score(test['pred_sentiment'], test['airline_sentiment'])


# For classification tasks, the terms true positives, true negatives, false positives, and false negatives compare the results of the classifier under test with trusted external judgments. The terms positive and negative refer to the classifier's prediction and the terms true and false refer to whether that prediction corresponds to the external judgment.
# 
# $Precision$ = $\frac{True Positive}{True Positive + False Positive}$
# 
# $Recall$ = $\frac{True Positive}{True Positive + False Negative}$
# 
# The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
# 
# And then $F1$ = 2 $\times$ $\frac{precision \times recall} {precision + recall}$
# 
# 

# In[ ]:


f1_score(test['airline_sentiment'],test['pred_sentiment'],average = 'macro')


# Let's take a look at the final classification report.

# In[ ]:


print(classification_report( test['airline_sentiment'], test['pred_sentiment']))


# ## Possible improvements
# 
# * Increase amount of time spent training
# * The dataset contains 9178 negative samples, 3099 neutral samples and 2363 positive samples. This imbalance makes sense from the point of view of the dataset, as people are more likely to tweet about a negative experience than a positive one. However, augmenting the neutral and postive samples could improve the overall model accuracy. One way to it is to generate tweets trained just on the positive tweets using a Transformer model. Synonym Replacement is also a viable option, in which words with the highest intrinsic attention value are swapped out for their synonyms in a copy.
