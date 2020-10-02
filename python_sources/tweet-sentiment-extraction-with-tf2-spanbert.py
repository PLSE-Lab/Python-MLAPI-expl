#!/usr/bin/env python
# coding: utf-8

# Here, we're going to adapt this example from the Keras website (https://keras.io/examples/nlp/text_extraction_with_bert/) to apply it to the Tweet Sentiment Extraction competition. We're going to take the text of a tweet and its sentiment, and return the part of the text that embodies its sentiment (positive, negative, or neutral).

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
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertTokenizer, TFBertModel, BertConfig, BertTokenizerFast 
print(tf.__version__)


# In[ ]:


data = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")


# ## Data Preprocessing

# I've noticed that for the neutral tweets, most of the time the "selected text" is just the whole tweet. So, I'm going to go by that assumption, and only predict the selected text for the positive or negative tweets.

# In[ ]:


data = data[data.sentiment != "neutral"]
data = data[pd.notnull(data.selected_text)]


# In[ ]:


data.head()


# We're going to use SpanBERT, a version of BERT specifically trained for predicting spans, as our transformer.
# 
# For more information about SpanBERT: https://github.com/facebookresearch/SpanBERT
# 
# For more information about BERT: https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
# 
# To learn about transformers in general: https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04
# 
# Feel free to explore different transformers- there are many great options!

# The first step is to tokenize the text, which turns it into an array of numbers.

# In[ ]:


max_len = 384

tokenizer = BertTokenizerFast.from_pretrained("../input/spanbert-pt/spanbert-base-cased/")


# The example from the Keras website (https://keras.io/examples/nlp/text_extraction_with_bert/ is for the SQUAD dataset, one of the most famous text extraction datasets. In that dataset, there is a "context" (e.g. a paragraph), a "question", and an "answer" (which is a subset of the context). In this case, we can treat the tweet as the context, the sentiment as the question, and the selected text of the tweet as the answer.

# Below, we create an object for the input data. This tokenizes the text and selected text, finds the indices of the selected text in the text, and prepares the data for input to BERT.

# In[ ]:


class TrainTweetExample:
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

        # Clean text, sentiment, and answer
        sentiment = " ".join(str(sentiment).split())
        text = " ".join(str(text).split())
        answer = " ".join(str(selected_text).split())

        # Find end character index of answer in text
        end_char_idx = start_char_idx + len(answer)
        if end_char_idx >= len(text):
            self.skip = True
            return

        # Mark the character indexes in text that are in answer
        is_char_in_ans = [0] * len(text)
        for idx in range(start_char_idx, end_char_idx):
            is_char_in_ans[idx] = 1

        # Tokenize text
        tokenized_text = tokenizer.encode_plus(text, return_offsets_mapping=True, max_length = max_len)

        # Find tokens that were created from answer characters
        ans_token_idx = []
        for idx, (start, end) in enumerate(tokenized_text.offset_mapping):
            if sum(is_char_in_ans[start:end]) > 0:
                ans_token_idx.append(idx)

        if len(ans_token_idx) == 0:
            self.skip = True
            return

        # Find start and end token index for tokens from answer
        start_token_idx = ans_token_idx[0]
        end_token_idx = ans_token_idx[-1]

        # Tokenize sentiment
        tokenized_sentiment = tokenizer.encode_plus(sentiment, return_offsets_mapping=True, max_length = max_len)

        # Create inputs
        input_ids = tokenized_text.input_ids + tokenized_sentiment.input_ids[1:]
        token_type_ids = [0] * len(tokenized_text.input_ids) + [1] * len(
            tokenized_sentiment.input_ids[1:]
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
        self.context_token_to_char = tokenized_text.offset_mapping


# In[ ]:


def create_tweet_examples(data):
    tweet_examples = []
    all_answers = data.selected_text.values
    count = 0
    for index, row in data.iterrows():
        sentiment = row['sentiment']
        text = row['text']
        selected_text = row['selected_text']

        try: ## turn the data into TrainTweetExample objects
            start_char_idx = row.text.index(row.selected_text.split()[0])
            tweet_eg = TrainTweetExample(
                sentiment, text, start_char_idx, selected_text, all_answers
            )
            tweet_eg.preprocess()
            tweet_examples.append(tweet_eg)
        except: ## keep track of the number that can't be tokenized/processed
            count += 1 
    print("Couldn't process",count,"points")
    return tweet_examples


# We do the same thing for the testing data, which only has the text and sentiment of the tweet.

# In[ ]:


class TestTweetExample:
    def __init__(self, sentiment, text):
        self.sentiment = sentiment
        self.text = text
        self.skip = False

    def preprocess(self):
        text = self.text
        sentiment = self.sentiment

        # Clean text, answer and sentiment
        sentiment = " ".join(str(sentiment).split())
        text = " ".join(str(text).split())

        # Tokenize text
        tokenized_text = tokenizer.encode_plus(text, return_offsets_mapping=True, max_length = max_len)

        # Tokenize sentiment
        tokenized_sentiment = tokenizer.encode_plus(sentiment, return_offsets_mapping=True, max_length = max_len)

        # Create inputs
        input_ids = tokenized_text.input_ids + tokenized_sentiment.input_ids[1:]
        token_type_ids = [0] * len(tokenized_text.input_ids) + [1] * len(
            tokenized_sentiment.input_ids[1:]
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
        self.text_token_to_char = tokenized_text.offset_mapping
        self.start_token_idx = 0
        self.end_token_idx = 0


# In[ ]:


def create_tweet_examples_test(data):
    tweet_examples = []
    all_answers = None
    count = 0
    for index, row in data.iterrows():
        sentiment = row['sentiment']
        text = row['text']
        selected_text = None

        try: ## turn the data into TestTweetExample objects
            start_char_idx = 0
            tweet_eg = TestTweetExample(
                sentiment, text
            )
            tweet_eg.preprocess()
            tweet_examples.append(tweet_eg)
        except: ## keep track of the number that can't be tokenized/processed
            count += 1
    print("Couldn't process",count,"points")
    return tweet_examples


# In[ ]:


def create_inputs_targets(tweet_examples):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in tweet_examples:
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


# Now, we can split the data into training and validation sets, and apply the preprocessing to the training and testing sets. We want to keep track of the number of data points successfully processed, and if there are any that caused errors.

# In[ ]:


from sklearn.model_selection import train_test_split

train, validation = train_test_split(data, test_size = 0.1)


# In[ ]:


train_tweet_examples = create_tweet_examples(train)
x_train, y_train = create_inputs_targets(train_tweet_examples)
print(f"{len(train_tweet_examples)} training points created.")

eval_tweet_examples = create_tweet_examples(validation)
x_eval, y_eval = create_inputs_targets(eval_tweet_examples)
print(f"{len(eval_tweet_examples)} evaluation points created.")


# In[ ]:


test_tweet_examples = create_tweet_examples_test(test)
x_test, _ = create_inputs_targets(test_tweet_examples)
print(f"{len(test_tweet_examples)} test points created.")


# ## Creating & Training Model

# Now, we can create the model that uses the SpanBERT transformer, and puts other layers on top of it. Shout-out to this notebook (https://www.kaggle.com/al0kharba/tensorflow-roberta-0-712) for helping me put together the layers.

# In[ ]:


def create_model():

    PATH = '../input/spanbert-pt/spanbert-base-cased/'
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


# The competition was restricted to CPU/GPU, but I found that TPUs greatly sped up training time. To use a TPU, just select it in the menu and change the "use_tpu" value below to True.

# In[ ]:


use_tpu = True
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
    model = create_model()

model.summary()


# Below is how we're going to evaluate the results of the model.

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
    Each `TweetExample` object contains the character level offsets for each token
    in its input text. We use them to get back the span of text corresponding
    to the tokens between our predicted start and end tokens.
    All the ground-truth answers are also present in each `TweetExample` object.
    We calculate the percentage of data points where the span of text obtained
    from model predictions matches one of the ground-truth answers.
    """

    def __init__(self, x_eval, y_eval):
        self.x_eval = x_eval
        self.y_eval = y_eval

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = self.model.predict(self.x_eval)
        count = 0
        eval_examples_no_skip = [_ for _ in eval_tweet_examples if _.skip == False]
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            tweet_eg = eval_examples_no_skip[idx]
            offsets = tweet_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = tweet_eg.text[pred_char_start:pred_char_end]
            else:
                pred_ans = tweet_eg.text[pred_char_start:]

            normalized_pred_ans = normalize_text(pred_ans)
            normalized_true_ans = [normalize_text(_) for _ in tweet_eg.all_answers]
            if normalized_pred_ans in normalized_true_ans:
                count += 1
        acc = count / len(self.y_eval[0])
        print(f"\nepoch={epoch+1}, exact match score={acc:.2f}")


# Now, we train the model. I'm only using one epoch, but feel free to update and explore different training setups.

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
    epochs=1,
    verbose=1,
    batch_size=batch_size,
    callbacks=[exact_match_callback],
)


# ## Generating Predictions

# Let's use the model to predict the selected text for the test set.

# In[ ]:


pred = model.predict(x_test)
np.array(pred).shape


# In[ ]:


test['prediction'] = np.zeros(len(test.text.values))


# In[ ]:


pred_start, pred_end = pred
count = 0

pred_text = []
for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
    tweet_eg = test_tweet_examples[idx]
    if (tweet_eg.skip == True):
        pred_text.append("")
    else: 
        offsets = tweet_eg.text_token_to_char
        start = np.argmax(start)
        end = np.argmax(end)
        if start >= len(offsets):
            continue
        pred_char_start = offsets[start][0]
        if end < len(offsets):
            pred_char_end = offsets[end][1]
            pred_ans = tweet_eg.text[pred_char_start:pred_char_end + 1]
        else:
            pred_ans = tweet_eg.text[pred_char_start:]

        normalized_pred_ans = normalize_text(pred_ans)
        #print(normalized_pred_ans)
        pred_text.append(normalized_pred_ans)
        test['prediction'][idx] = normalized_pred_ans


# Now, we can go back and fill in any gaps, like neutral tweets, and remove hyperlinks from the final results.

# In[ ]:


test.prediction[test.prediction == 0] = test.text[test.prediction == 0]
test.prediction[test.sentiment == "neutral"] = test.text[test.sentiment=="neutral"]
test.prediction[test.prediction ==''] = test.text[test.prediction =='']

def clean_text(text):
    words = str(text).split()
    words = [x for x in words if not x.startswith("http")]
    words = " ".join(words)
    return words

test['prediction'] = test['prediction'].apply(clean_text)


# In[ ]:


test[:50]


# Now, we can just write our submission file!

# In[ ]:


evaluation = test.textID.copy().to_frame()
evaluation['selected_text'] = test['prediction']
evaluation


# In[ ]:


evaluation.to_csv("submission.csv", index=False)


# In[ ]:




