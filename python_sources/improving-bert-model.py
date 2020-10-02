#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
import os
import re
import json
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig, TFRobertaModel, RobertaTokenizerFast, RobertaTokenizer, BertTokenizerFast 
print(tf.__version__) 


# In[ ]:


data = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")


# In[ ]:


data = data[data.sentiment != "neutral"]


# In[ ]:


data.head()


# In[ ]:


max_len = 384
configuration = BertConfig()  # default paramters and configuration for BERT

# model_to_use = 'bert-base-uncased'

# # # Save the slow pretrained tokenizer
# #slow_tokenizer = BertTokenizer.from_pretrained("../input/bert-base-uncased/")
# tokenizer = BertTokenizer.from_pretrained(model_to_use)
# save_path = "bert_base_uncased/"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# tokenizer.save_pretrained(save_path)

# # Load the fast tokenizer from saved file
# #tokenizer = BertWordPieceTokenizer("../input/bert-base-uncased/vocab.txt", lowercase=True)
# tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=False)


# In[ ]:


rtokenizer = BertTokenizerFast.from_pretrained("../input/spanbert-pt/spanbert-base-cased/")


# In[ ]:


# s = tokenizer.encode(data.text.values[1])
# print(data.text.values[1])
# print(s)
# print(s.ids)
# print(s.tokens)
# print(s.offsets)


# In[ ]:


#rtokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')


# In[ ]:


s = rtokenizer.encode_plus(data.text.values[1], return_offsets_mapping=True)
print(data.text.values[1])
print(s)
print(s.offset_mapping)
print(s.input_ids)


# In[ ]:


data


# In[ ]:


class TweetExample:
    def __init__(self, sentiment, text, start_char_idx, selected_text, all_answers):
        self.sentiment = sentiment
        self.text = text
        self.start_char_idx = start_char_idx
        self.selected_text = selected_text
        self.all_answers = all_answers
        self.skip = False

    def preprocess(self):
        text = self.text
        sentiment = self.sentiment
        selected_text = self.selected_text
        start_char_idx = self.start_char_idx

        # Clean context, answer and question
        sentiment = " ".join(str(sentiment).split())
        text = " ".join(str(text).split())
        answer = " ".join(str(selected_text).split())

        # Find end character index of answer in context
        end_char_idx = start_char_idx + len(answer)
        if end_char_idx >= len(text):
            self.skip = True
            return

        # Mark the character indexes in context that are in answer
        is_char_in_ans = [0] * len(text)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1

        # Tokenize context
        tokenized_context = rtokenizer.encode_plus(text, return_offsets_mapping=True, max_length = max_len)

        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_context.offset_mapping):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)

        if len(ans_token_idx) == 0:
            self.skip = True
            return

        # Find start and end token index for tokens from answer
        start_token_idx = ans_token_idx[0]
        end_token_idx = ans_token_idx[-1]

        # Tokenize question
        #tokenized_question = tokenizer.encode(question)
        tokenized_question = rtokenizer.encode_plus(sentiment, return_offsets_mapping=True, max_length = max_len)

        # Create inputs
        input_ids = tokenized_context.input_ids + tokenized_question.input_ids[1:]
        token_type_ids = [0] * len(tokenized_context.input_ids) + [1] * len(
            tokenized_question.input_ids[1:]
        )
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = max_len - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx
        self.context_token_to_char = tokenized_context.offset_mapping


# In[ ]:


# for idx, value in enumerate(data.text): ## remove hyperlinks
#     words = str(value).split()
#     words = [x for x in words if not x.startswith("http")]
#     data["text"][idx] = " ".join(words)
#     #print(data["text"][idx])
    
# for idx, value in enumerate(test.text):
#     words = str(value).split()
#     words = [x for x in words if not x.startswith("http")]
#     test["text"][idx] = " ".join(words)
    


# In[ ]:


data = data[pd.notnull(data.selected_text)]


# In[ ]:


data


# In[ ]:


test


# In[ ]:


print(len(data.text), len(data.textID))


# In[ ]:


rtokenizer.encode(test.sentiment[0])


# In[ ]:


def create_tweet_examples(data):
    squad_examples = []
    all_answers = data.selected_text.values
    count = 0
    for index, row in data.iterrows():
        sentiment = row['sentiment']
        text = row['text']
        selected_text = row['selected_text']
        #print(sentiment, text, selected_text)
        try:
            start_char_idx = row.text.index(row.selected_text.split()[0])
            squad_eg = TweetExample(
                sentiment, text, start_char_idx, selected_text, all_answers
            )
            squad_eg.preprocess()
            squad_examples.append(squad_eg)
        except:
            count += 1
    print("Couldn't find ",count,"tokens")
    return squad_examples


# In[ ]:


class InputTweetExample:
    def __init__(self, sentiment, text):
        self.sentiment = sentiment
        self.text = text
        self.skip = False

    def preprocess(self):
        text = self.text
        sentiment = self.sentiment

        # Clean context, answer and question
        sentiment = " ".join(str(sentiment).split())
        text = " ".join(str(text).split())

        # Tokenize context
        tokenized_context = rtokenizer.encode_plus(text, return_offsets_mapping=True, max_length = max_len)

        # Tokenize question
        #tokenized_question = tokenizer.encode(question)
        tokenized_question = rtokenizer.encode_plus(sentiment, return_offsets_mapping=True, max_length = max_len)

        # Create inputs
        input_ids = tokenized_context.input_ids + tokenized_question.input_ids[1:]
        token_type_ids = [0] * len(tokenized_context.input_ids) + [1] * len(
            tokenized_question.input_ids[1:]
        )
        attention_mask = [1] * len(input_ids)

        # Pad and create attention masks.
        # Skip if truncation is needed
        padding_length = max_len - len(input_ids)
        if padding_length > 0:  # pad
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:  # skip
            self.skip = True
            return

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.context_token_to_char = tokenized_context.offset_mapping
        self.start_token_idx = 0
        self.end_token_idx = 0


# In[ ]:


def create_tweet_examples_test(data):
    squad_examples = []
    all_answers = None
    count = 0
    for index, row in data.iterrows():
        sentiment = row['sentiment']
        text = row['text']
        selected_text = None
        #print(sentiment, text, selected_text)
        try:
            start_char_idx = 0
            squad_eg = InputTweetExample(
                sentiment, text
            )
            squad_eg.preprocess()
            squad_examples.append(squad_eg)
        except:
            count += 1
    print("Couldn't find ",count,"tokens")
    return squad_examples


# In[ ]:


def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"],
    ]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y


# In[ ]:


from sklearn.model_selection import train_test_split

train, validation = train_test_split(data, test_size = 0.1)


# In[ ]:


train_squad_examples = create_tweet_examples(train)
x_train, y_train = create_inputs_targets(train_squad_examples)
print(f"{len(train_squad_examples)} training points created.")

eval_squad_examples = create_tweet_examples(validation)
x_eval, y_eval = create_inputs_targets(eval_squad_examples)
print(f"{len(eval_squad_examples)} evaluation points created.")


# In[ ]:


test_squad_examples = create_tweet_examples_test(test)
x_test, y_test = create_inputs_targets(test_squad_examples)
print(f"{len(test_squad_examples)} test points created.")


# "y_test" is nothing, but now we've changed the test data to the format that the model takes in.

# In[ ]:


def create_model():
    ## BERT encoder
    #encoder = TFBertModel.from_pretrained("../input/bert-base-uncased/", from_pt = True)
#     encoder = TFBertModel.from_pretrained(model_to_use)

    PATH = '../input/spanbert-pt/spanbert-base-cased/'
    #encoder = TFRobertaModel.from_pretrained('../input/roberta-base', from_pt = True)

    #config = RobertaConfig.from_pretrained(PATH+'config.json')
    encoder = TFBertModel.from_pretrained(PATH, from_pt = True)
    
## QA Model
    input_ids = layers.Input(shape=(max_len ,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(
         input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
    )[0]
    

    start_logits = layers.Dropout(0.3)(embedding)
    start_logits = layers.Conv1D(128,2,padding='same')(start_logits)
    start_logits = layers.LeakyReLU()(start_logits)
    start_logits = layers.Conv1D(64,2,padding='same')(start_logits)
    start_logits = layers.Dense(1, name="start_logit")(start_logits)
    start_logits = layers.Flatten()(start_logits)

    end_logits = layers.Dropout(0.3)(embedding)
    end_logits = layers.Conv1D(128,2,padding='same')(end_logits)
    end_logits = layers.LeakyReLU()(end_logits)
    end_logits = layers.Conv1D(64,2,padding='same')(end_logits)
    end_logits = layers.Dense(1, name="end_logit")(end_logits)
    end_logits = layers.Flatten()(end_logits)

    start_probs = layers.Activation(keras.activations.softmax)(start_logits)
    end_probs = layers.Activation(keras.activations.softmax)(end_logits)

    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[start_probs, end_probs],
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(lr=5e-5)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    return model


# In[ ]:


use_tpu = False
if use_tpu:
    # Create distribution strategy
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    # Create model
    with strategy.scope():
        model = create_model()
else:
#     strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
#     with strategy.scope():
    model = create_model()

model.summary()


# In[ ]:


def normalize_text(text):
    text = text.lower()

    # Remove punctuations
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)

    # Remove articles
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = re.sub(regex, " ", text)

    # Remove extra white space
    text = " ".join(text.split())
    return text


class ExactMatch(keras.callbacks.Callback):
    """
    Each `SquadExample` object contains the character level offsets for each token
    in its input paragraph. We use them to get back the span of text corresponding
    to the tokens between our predicted start and end tokens.
    All the ground-truth answers are also present in each `SquadExample` object.
    We calculate the percentage of data points where the span of text obtained
    from model predictions matches one of the ground-truth answers.
    """

    def __init__(self, x_eval, y_eval):
        self.x_eval = x_eval
        self.y_eval = y_eval

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = self.model.predict(self.x_eval)
        count = 0
        eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip == False]
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = eval_examples_no_skip[idx]
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = squad_eg.text[pred_char_start:pred_char_end]
            else:
                pred_ans = squad_eg.text[pred_char_start:]

            normalized_pred_ans = normalize_text(pred_ans)
            normalized_true_ans = [normalize_text(_) for _ in squad_eg.all_answers]
            if normalized_pred_ans in normalized_true_ans:
                count += 1
        acc = count / len(self.y_eval[0])
        print(f"\nepoch={epoch+1}, exact match score={acc:.2f}")


# In[ ]:


batch_size = 8
div = len(x_train[0]) - (len(x_train[0]) % batch_size)
x = list(np.array(x_train)[:,:div]) ## inputs must be divisible by batch size 
y = list(np.array(y_train)[:,:div])


# In[ ]:


exact_match_callback = ExactMatch(x_eval, y_eval)
model.fit(
    list(np.array(x)),
    list(np.array(y)),
    epochs=1,  # For demonstration, 3 epochs are recommended
    verbose=2,
    batch_size=8,
    callbacks=[exact_match_callback],
)


# In[ ]:


np.array(x_test).shape


# In[ ]:


pred = model.predict(x_test)
np.array(pred).shape


# In[ ]:


pred


# In[ ]:


test['prediction'] = np.zeros(len(test.text.values))


# In[ ]:


pred_start, pred_end = pred
count = 0
#test_examples_no_skip = [_ for _ in test_squad_examples if _.skip == False]
pred_text = []
for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
    squad_eg = test_squad_examples[idx]
    if (squad_eg.skip == True):
        print(idx)
        pred_text.append("")
    else: 
        offsets = squad_eg.context_token_to_char
        start = np.argmax(start)
        end = np.argmax(end)
        if start >= len(offsets):
            continue
        pred_char_start = offsets[start][0]
        if end < len(offsets):
            pred_char_end = offsets[end][1]
            pred_ans = squad_eg.text[pred_char_start:pred_char_end + 1]
        else:
            pred_ans = squad_eg.text[pred_char_start:]

        normalized_pred_ans = normalize_text(pred_ans)
        #print(normalized_pred_ans)
        pred_text.append(normalized_pred_ans)
        test['prediction'][idx] = normalized_pred_ans


# In[ ]:


pred_text


# In[ ]:


test[-50:]


# In[ ]:


test.prediction[test.prediction == 0] = test.text[test.prediction == 0]
test.prediction[test.sentiment == "neutral"] = test.text[test.sentiment=="neutral"]
test.prediction[test.prediction ==''] = test.text[test.prediction =='']

import string
def clean_text(text):
    words = str(text).split()
    words = [x for x in words if not x.startswith("http")]
    words = " ".join(words)
    return words

test['prediction'] = test['prediction'].apply(clean_text)


# In[ ]:


test[:50]


# In[ ]:


evaluation = test.textID.copy().to_frame()
evaluation['selected_text'] = test['prediction']
evaluation


# In[ ]:


evaluation.to_csv("submission.csv", index=False)


# In[ ]:




