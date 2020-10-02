#!/usr/bin/env python
# coding: utf-8

# 

# # Error analysis between ELMo and BERT prediction

# This notebooks aims to give a look out error analysis from two famous NLP machine learning models : ELMo and BERT. 
# 
# A combinaison of their predictions is also looked out and then submitted for the competition.

# ## Acknowledgments
# 
# This kernel uses the following kernels, their great explanations and inspirations:
# 
# - [NLP EDA bag-of-words tfidf GloVe BERT (vbmokin)](https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert)
# - [Basic EDA Cleaning and GloVe (shahules)](https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove)
# - [Disaster NLP Keras BERT using TfHub (xhulu)](https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub)

# ## Imports

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string
import re
from nltk.tokenize import word_tokenize
import json
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#Utility function to handle the two classes' probability outputs of ELMo's model. 
def return_low(x):
    high = x["1"] - 1
    if(x["0"] > x["1"]): return (1-x["0"])
    else : return x["1"]


# In[ ]:


#Loading predictions from ELMo and BERT models.
#Loading training and test datasets.
import pandas as pd

test = pd.read_csv("../input/predictions-datasets/test.csv")
train = pd.read_csv("../input/predictions-datasets/train.csv")


# ## Cleaning

# In[ ]:


def clean_tweets(tweet):
    """Removes links and non-ASCII characters"""
    
    tweet = ''.join([x for x in tweet if x in string.printable])
    
    # Removing URLs
    tweet = re.sub(r"http\S+", "", tweet)
    
    return tweet

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punctuations(text):
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    
    for p in punctuations:
        text = text.replace(p, f' {p} ')

    text = text.replace('...', ' ... ')
    
    if '...' not in text:
        text = text.replace('..', ' ... ')
    
    return text

def convert_abbrev(word):
    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word

def convert_abbrev_in_text(text):
    tokens = word_tokenize(text)
    tokens = [convert_abbrev(word) for word in tokens]
    text = ' '.join(tokens)
    return text


# Here we handle errors found in the dataset and the cleaning of the sentences stopwords and unwanted characters.

# In[ ]:


ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]
train.at[train['id'].isin(ids_with_target_error),'target'] = 0
train[train['id'].isin(ids_with_target_error)]

train = train.drop(train[train["text"].duplicated()].index)

with open('../input/abbreviation/abbreviation.json') as json_file:
    abbreviations = json.load(json_file)

train["text"] = train["text"].apply(lambda x: clean_tweets(x))
test["text"] = test["text"].apply(lambda x: clean_tweets(x))

train["text"] = train["text"].apply(lambda x: remove_emoji(x))
test["text"] = test["text"].apply(lambda x: remove_emoji(x))

train["text"] = train["text"].apply(lambda x: remove_punctuations(x))
test["text"] = test["text"].apply(lambda x: remove_punctuations(x))

train["text"] = train["text"].apply(lambda x: convert_abbrev_in_text(x))
test["text"] = test["text"].apply(lambda x: convert_abbrev_in_text(x))


# ## Models
# 
# 

# Note that the Model section won't be able to work here as ELMo only works tf version 1.15 and the BERT model is designed with tf version 2.1.
# You can jump directly to the "Result Analysis" section after reading the Models section.

# ### ELMo
# 
# First model is a prediction from ELMo's [module]('https://tfhub.dev/google/elmo/3') (non fine-tuned).

# ```python
# !pip uninstall tensorflow
# !pip install tensorflow==1.15
# ```

# ```python
# import tensorflow as tf
# import tensorflow_hub as hub
# 
# embed = hub.Module('https://tfhub.dev/google/elmo/3', trainable=True)
# 
# embeddings = embed(
#     np.array(train["text"]),
#     signature="default",
#     as_dict=True)["default"]
# ```

# ```python
# from sklearn.model_selection import train_test_split
# 
# #Need to clean "train[label]" for actual labels
# xtrain, xvalid, ytrain, yvalid = train_test_split(emb_train, 
#                                                   train["target"],  
#                                                   random_state=42, 
#                                                   test_size=0.15)
# ```

# ```python
# from sklearn.neural_network import MLPClassifier
# from itertools import product
# 
# 
# mlp_clf = MLPClassifier(50, learning_rate = "constant", max_iter=100, random_state = 42)
# mlp_clf.fit(xtrain, ytrain)
# 
# elmo_result_train = mlp_clf.predict(ytrain)
# elmo_f1_sco_train = f1_score(elmo_result_train, train["target"].values, average='weighted')
# 
# full_predict_mlp = mlp_clf.predict_proba(emb_train)
# ```

# ### BERT

# ```python
# import tensorflow_hub as hub
# 
# module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
# bert_layer = hub.KerasLayer(module_url, trainable=True)
# ```

# ```python
# def bert_encode(texts, tokenizer, max_len=512):
#     all_tokens = []
#     all_masks = []
#     all_segments = []
#     
#     for text in texts:
#         text = tokenizer.tokenize(text)
#             
#         text = text[:max_len-2]
#         input_sequence = ["[CLS]"] + text + ["[SEP]"]
#         pad_len = max_len - len(input_sequence)
#         
#         tokens = tokenizer.convert_tokens_to_ids(input_sequence)
#         tokens += [0] * pad_len
#         pad_masks = [1] * len(input_sequence) + [0] * pad_len
#         segment_ids = [0] * max_len
#         
#         all_tokens.append(tokens)
#         all_masks.append(pad_masks)
#         all_segments.append(segment_ids)
#     
#     return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
#     return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
# 
# def build_model(bert_layer, max_len=512):
#     input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
#     input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
#     segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
# 
#     _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
#     clf_output = sequence_output[:, 0, :]
#     
#     if Dropout_num == 0:
#         # Without Dropout
#         out = Dense(1, activation='sigmoid')(clf_output)
#     else:
#         # With Dropout(Dropout_num), Dropout_num > 0
#         x = Dropout(Dropout_num)(clf_output)
#         out = Dense(1, activation='sigmoid')(x)
# 
#     model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
#     model.compile(Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
#     
#     return model
# ```

# ```python
# vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
# do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
# tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
# ```

# ```python
# train_input = bert_encode(train.text.values, tokenizer, max_len=160)
# test_input = bert_encode(test.text.values, tokenizer, max_len=160)
# train_labels = train.target.values
# ```

# ```python
# model = build_model(bert_layer, max_len=160)
# model.summary()
# ```

# ```python
# train_history = model.fit(
#     train_input, train_labels,
#     validation_split=0.2,
#     epochs=10,
#     batch_size=16
# )
# ```

# ## Result Analysis

# A first cleaning of ELMo's prediction is needed because the model has a MLP layer on top for predictions which outputs through [proba_predict()](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier.predict_proba) (which outputs a probability of the sample for each class of the model). 

# In[ ]:


ELMo_full_proba = pd.read_csv("../input/predictions-datasets/ELMo_full_proba.csv")[["0", "1"]]
elmo_full = ELMo_full_proba.apply(return_low, axis = 1)
elmo_full_around = elmo_full.apply(lambda x: np.int(np.around(x)))

bert_full = pd.read_csv("../input/predictions-datasets/newTrainPredict.csv")["0"]
bert_full_around = bert_full.apply(lambda x: np.around(x))

bert_test = pd.read_csv("../input/predictions-datasets/newTestPredict.csv")["0"]
bert_test_around = bert_test.apply(lambda x: np.around(x))


# Analysis of ELMo's and BERT's predictions to divide values into confusion matrix's subsets (*True Positive, False Positives, False Negatives, True Negatives*)
# > [Confusion Matrix Wiki](https://en.wikipedia.org/wiki/Confusion_matrix)
# 
# Here the focus is kept on **False** values to focus on errors made from the models.
# 
# 

# In[ ]:


Predict = bert_full_around
Predict.index = train.index
bert_evaluationOnes = (Predict.loc[Predict == 1] == train.loc[Predict == 1]["target"])
bert_evaluationZeros = (Predict.loc[Predict == 0] == train.loc[Predict == 0]["target"])

tot_false = (bert_evaluationOnes[bert_evaluationOnes == False]).shape[0] + (bert_evaluationZeros[bert_evaluationZeros == False]).shape[0]
bert_acc = 1-tot_false/train.index.shape[0]
print("accuracy is ", bert_acc)

FP_bert = train.loc[bert_evaluationOnes[bert_evaluationOnes == False].index]
FN_bert = train.loc[bert_evaluationZeros[bert_evaluationZeros == False].index]
print(np.shape(FP_bert), np.shape(FN_bert))


# In[ ]:


Predict = elmo_full_around
Predict.index = train.index
elmo_evaluationOnes = (Predict.loc[Predict == 1] == train.loc[Predict == 1]["target"])
elmo_evaluationZeros = (Predict.loc[Predict == 0] == train.loc[Predict == 0]["target"])

tot_false = (elmo_evaluationOnes[elmo_evaluationOnes == False]).shape[0] + (elmo_evaluationZeros[elmo_evaluationZeros == False]).shape[0]
elmo_acc = 1-tot_false/train.index.shape[0]
print("accuracy is ", elmo_acc)

FP_elmo = train.loc[elmo_evaluationOnes[elmo_evaluationOnes == False].index]
FN_elmo = train.loc[elmo_evaluationZeros[elmo_evaluationZeros == False].index]
print(np.shape(FP_elmo), np.shape(FN_elmo))


# One trick analysed here is to average the predictions' probabilities of both models. This allows for the most "**confident**" model to take over. But why would that be something we want as the BERT model being more performant, would be a intrinsically "confident". While that is right for **True** values, it is not exactly the same for **False** values. 
# 
# "*Confidence*" is simply "How close to a 100% is the model when predicting a value". A model that has a precision of 98%, but an average confidence of 70% gives a different impression than a confidence of 90%.
# 
# The more a model is confident when it is right, usually the better. Inversly, the less confident the model usually is in its mistakes, the better. It shows an "hesitation" and the possibility of a doubt. 

# In[ ]:


combined_pd = pd.DataFrame({'bert': bert_full, 'elmo':elmo_full})
combined_pd = combined_pd.apply(lambda x: np.average(x), axis = 1)
combined_around_pd = combined_pd.apply(lambda x: np.int(np.around(x)))

Predict = combined_around_pd
Predict.index = train.index
combined_evaluationOnes = (Predict.loc[Predict == 1] == train.loc[Predict == 1]["target"])
combined_evaluationZeros = (Predict.loc[Predict == 0] == train.loc[Predict == 0]["target"])

tot_false = (combined_evaluationOnes[combined_evaluationOnes == False]).shape[0] + (combined_evaluationZeros[combined_evaluationZeros == False]).shape[0]

FP_comb = train.loc[combined_evaluationOnes[combined_evaluationOnes == False].index]
FN_comb = train.loc[combined_evaluationZeros[combined_evaluationZeros == False].index]

comb_acc = 1-tot_false/train.index.shape[0]
print("accuracy is ", comb_acc)
print(np.shape(FP_comb), np.shape(FN_comb))


# We now can show the 10 most mistakenly labeled **keywords** by both model and their averaged prediction. 

# In[ ]:


fig = plt.figure(figsize = (30,10))

ax = fig.add_subplot(131)
ax.axis([0, 10, 0, 18])
plt.xticks(size = 20)
plt.yticks(size = 15)
plt.title('FP_elmo')
FP_elmo["keyword"].value_counts().head(10).plot.bar(ax = ax)

ax = fig.add_subplot(132)
ax.axis([0, 10, 0, 18])
plt.xticks(size = 20)
plt.yticks(size = 15)
plt.title('FP_bert')
FP_bert["keyword"].value_counts().head(10).plot.bar(ax = ax)

ax = fig.add_subplot(133)
ax.axis([0, 10, 0, 18])
plt.xticks(size = 20)
plt.yticks(size = 15)
plt.title('FP_comb')
FP_comb["keyword"].value_counts().head(10).plot.bar(ax = ax)


# First remark is that the two models have very distinct errors. Although BERT has better accuracy, it seems to get more *stuck* on certain formulations. Let's look at the keyword "**windstorm**" and see why it could label it wrong. 

# In[ ]:


#Don't hesitate to test different keywords such as [detonate, pandemonium, tsunami, ...]
keyword = "windstorm"

for idx in FP_bert.index:  
    if(FP_bert["keyword"][idx] == keyword):
        print(FP_bert["id"][idx], FP_bert["keyword"][idx], " : ")
        print(FP_bert["text"][idx])
        print('--'*20)


# We rapidly see here that BERT gets stuck on the formulation of "**Windstorm Insurer**" and labels those examples as actual threats. It probably learns to see an association with threats and cataclysm from other tweets as there are 16 positively labeled examples with the keyword "**windstorm**".

# In[ ]:


train[train["keyword"] == "windstorm"]["target"].value_counts()


# let's do the same with ELMo's model and the keyword "**trapped**". 

# In[ ]:


#Don't hesitate to test different keywords such as [eyewitnessed, destroyed, demolition, ...]
keyword = "trapped"

for idx in FP_elmo.index:  
    if(FP_elmo["keyword"][idx] == keyword):
        print(FP_elmo["id"][idx], FP_elmo["keyword"][idx], " : ")
        print(FP_elmo["text"][idx])
        print('--'*20)
        
print(train[train["keyword"] == keyword]["target"].value_counts())


# Although the idea of a cataclysm is present inside this repeating set of tweets, ELMo fails to grab the context of the Hollywood movie, that is clearly stated, and that doesn't represent a treat. 

# Let's compare ELMo's and the combined's predictions on the classification of tweets with the keyword "**windstorm**"

# In[ ]:


keyword = "windstorm"

for idx in FP_elmo.index:  
    if(FP_elmo["keyword"][idx] == keyword):
        print(FP_elmo["id"][idx], FP_elmo["keyword"][idx], " : ")
        print(FP_elmo["text"][idx])
        print('--'*20)


# In[ ]:


keyword = "windstorm"

for idx in FP_comb.index:  
    if(FP_comb["keyword"][idx] == keyword):
        print(FP_comb["id"][idx], FP_comb["keyword"][idx], " : ")
        print(FP_comb["text"][idx])
        print('--'*20)


# ELMo only has only 6 predictions in the False Positive group for the keyword "**windstorm**". Although the misconception of "windstorm insurer" is still present, it is way less frequent. This has a lot of chance to involve the architecture of ELMo that is not built on transformers but on LSTM and that is specialized on single sentence comprehension.
# We see that it has more confidence in certain predictions than BERT allowing the combined prediction to reduce the number of mislabelled "**windstorm**" to 13. 

# In[ ]:


fig = plt.figure(figsize = (30,10))

ax = fig.add_subplot(131)
ax.axis([0, 10, 0, 12])
plt.title('FN_elmo')
plt.xticks(size = 20)
plt.yticks(size = 15)
FN_elmo["keyword"].value_counts().head(10).plot.bar(ax = ax)

ax = fig.add_subplot(132)
ax.axis([0, 10, 0, 12])
plt.xticks(size = 20)
plt.yticks(size = 15)
plt.title('FN_bert')
FN_bert["keyword"].value_counts().head(10).plot.bar(ax = ax)

ax = fig.add_subplot(133)
ax.axis([0, 10, 0, 12])
plt.xticks(size = 20)
plt.yticks(size = 15)
plt.title('FN_comb')
FN_comb["keyword"].value_counts().head(10).plot.bar(ax = ax)


# The same effect is found with the False Negatives groups where the combined predictions the model to get less *stuck* on some keywords.

# ## Confidence

# In[ ]:


def confidenceCalc(x):
    if x<0.5 : 
        return (0.5-x)*2
    else : 
        return (x-0.5)*2


# Calculation of the average confidence in the predictions of each model. 
# For rightfully labeled inputs, we want to see the confidence as high as possible (*True confidence*). 
# > This would show that, when it's right, it is confident it is right.
# 
# For wrongfully labeled inputs, we want to see the confidence as low as possible (*False confidence*).
# > This would show that, when it's wrong, it was closer to doubt.
# 
# 

# In[ ]:


Predict = elmo_full_around
Predict.index = train.index
elmo_full.index = train.index
TrueValues = elmo_full[train["target"] == Predict]
FalseValues = elmo_full[train["target"] != Predict]
meanTrueConf = TrueValues.apply(confidenceCalc).mean()
meanFalseConf = FalseValues.apply(confidenceCalc).mean()

print("ELMo's accuracy :", elmo_acc)
print("True confidence :", meanTrueConf, "False confidence :", meanFalseConf)
print("Value diff :", meanTrueConf - meanFalseConf)


# In[ ]:


Predict = bert_full_around
Predict.index = train.index
bert_full.index = train.index
TrueValues = bert_full[train["target"] == Predict]
FalseValues = bert_full[train["target"] != Predict]
meanTrueConf = TrueValues.apply(confidenceCalc).mean()
meanFalseConf = FalseValues.apply(confidenceCalc).mean()

print("BERT's accuracy :", bert_acc)
print("True confidence :", meanTrueConf, "False confidence :", meanFalseConf)
print("Value diff :", meanTrueConf - meanFalseConf)


# In[ ]:


Predict = combined_around_pd
Predict.index = train.index
combined_pd.index = train.index
TrueValues = combined_pd[train["target"] == Predict]
FalseValues = combined_pd[train["target"] != Predict]
meanTrueConf = TrueValues.apply(confidenceCalc).mean()
meanFalseConf = FalseValues.apply(confidenceCalc).mean()

print("combined's accuracy :", comb_acc)
print("True confidence :", meanTrueConf, "False confidence :", meanFalseConf)
print("Value diff :", meanTrueConf - meanFalseConf)


# We have an interesting effect here !
# Although the True confidence of the combined prediction isn't averaged between both model's True Confidence, there is a large drop of the False confidence.
# This "drop" is simply a consequence of the avering of both predictions. If for an example, ELMo predicted a `0.05` (confidence of **90**%) and BERT predicted a `0.80` (confidence of **60**%), the average will be `0.425` (confidence **15**%). If the boundary is set at `0.5`, than the final predicted class will be 0 (because **0.425 < 0.5**) but the confidence will be much lower than any of the other two models. 
# 
# That means a handling of errors that is in a much better direction in the combined probabilities model, at the cost of loosing a bit of confidence in the correctly labeled predictions.
# 
# In the design of programs to analyse tweets, the handling of errors is an important specificity that can have a lot of influence on the machine learning model. Having a model to have a very low percentage of False Negative might be a requirement when dealing with strong consequences. 

# ## Submissions

# A first submission from the BERT model was made with a test accuracy of **0.82822** which is not that suprising being really close to the scores from other BERT models in some notebooks on Kaggle and from the notebook this model was extracted from. 

# The combined prediction model performs an accuracy on the test set of **0.82719** which is really close to BERT's model performance. This is pretty much in line with the accuracy difference on the validation dataset used in training between both models (although the score is a bit higher than expected). 

# In[ ]:


submission_combined = pd.read_csv("../input/submission/submission.csv")


# In[ ]:


submission_combined.to_csv('/kaggle/working/submission.csv', index=False)


# ## Ending note

# Thank you for reading ! 
# Hope this little analysis might inspire some of you in improving your models or your comprehension ! Good luck with the competition !! 
