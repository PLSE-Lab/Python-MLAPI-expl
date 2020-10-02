#!/usr/bin/env python
# coding: utf-8

# ## Importing Necessary Libraries

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
import transformers #huggingface transformers library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# ## Configuration for TPUs
# 
# The given news classifier is trained on using BERT model. Since, BERT is a very large model, it requires gpu's and tpu's to train quickly. Here, I have used TPU's provided by Kaggle to train this model.

# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# ## Preprocessing

# In[ ]:


df = pd.read_json('/kaggle/input/news-category-dataset/News_Category_Dataset_v2.json', lines = True)
df.head()


# In[ ]:


# WORLDPOST and THE WORLDPOST were given as two separate categories in the dataset. Here I change the category THE WORLDPOST to WORLDPOST 
df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)


# In[ ]:


print(f"The dataset contains { df.category.nunique() } unique categories")


# In[ ]:


#label encoding the categories. After this each category would be mapped to an integer.
encoder = LabelEncoder()
df['categoryEncoded'] = encoder.fit_transform(df['category'])


# In[ ]:


#since I am using bert-large-uncased as the model, I am converting each of the news headlines and descriptions into lower case.
df['headline'] = df['headline'].apply(lambda headline: str(headline).lower())
df['short_description'] = df['short_description'].apply(lambda descr: str(descr).lower())


# In[ ]:


#calculating the length of headlines and descriptions
df['descr_len'] = df['short_description'].apply(lambda x: len(str(x).split()))
df['headline_len'] = df['headline'].apply(lambda x: len(str(x).split()))


# In[ ]:


df.describe()


# As we can see, that there are a lot of samples that have a description length of 0, however almost all articles have a headline. Going with the intuition that the headline is often more descriptive of the category of the news, as well as to provide more text data to the model, I add news headlines to the short description and modify the description of the news samples.

# In[ ]:


df['short_description'] =df['headline']+df['short_description']


# ## Tokenizing
# 
# * I use the HuggingFace tokenizer for bert to tokenize the given news descriptions. After splitting up the dataset into train and test, we encode both the training data and the testing data. 
# * One more thing to notice is that we truncate the last part of the sentence when encoding it, thus adding the headlines to the front of the description means that our model will always have the headline of the news sample.

# In[ ]:


def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])


# In[ ]:


#bert large uncased pretrained tokenizer
tokenizer = transformers.BertTokenizer.from_pretrained('bert-large-uncased')


# In[ ]:


X_train,X_test ,y_train,y_test = train_test_split(df['short_description'], df['categoryEncoded'], random_state = 2020, test_size = 0.3)


# In[ ]:


#tokenizing the news descriptions and converting the categories into one hot vectors using tf.keras.utils.to_categorical
Xtrain_encoded = regular_encode(X_train.astype('str'), tokenizer, maxlen=80)
ytrain_encoded = tf.keras.utils.to_categorical(y_train, num_classes=40,dtype = 'int32')
Xtest_encoded = regular_encode(X_test.astype('str'), tokenizer, maxlen=80)
ytest_encoded = tf.keras.utils.to_categorical(y_test, num_classes=40,dtype = 'int32')


# ## Building the model

# In[ ]:


def build_model(transformer, loss='categorical_crossentropy', max_len=512):
    input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    #adding dropout layer
    x = tf.keras.layers.Dropout(0.3)(cls_token)
    #using a dense layer of 40 neurons as the number of unique categories is 40. 
    out = tf.keras.layers.Dense(40, activation='softmax')(x)
    model = tf.keras.Model(inputs=input_word_ids, outputs=out)
    #using categorical crossentropy as the loss as it is a multi-class classification problem
    model.compile(tf.keras.optimizers.Adam(lr=3e-5), loss=loss, metrics=['accuracy'])
    return model


# In[ ]:


#building the model on tpu
with strategy.scope():
    transformer_layer = transformers.TFAutoModel.from_pretrained('bert-large-uncased')
    model = build_model(transformer_layer, max_len=80)
model.summary()


# ## Training

# In[ ]:


#creating the training and testing dataset.
BATCH_SIZE = 32*strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE 
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((Xtrain_encoded, ytrain_encoded))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)
test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(Xtest_encoded)
    .batch(BATCH_SIZE)
)


# In[ ]:


#training for 10 epochs
n_steps = Xtrain_encoded.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    epochs=10
)


# ## Evaluation

# In[ ]:


#making predictions
preds = model.predict(test_dataset,verbose = 1)
#converting the one hot vector output to a linear numpy array.
pred_classes = np.argmax(preds, axis = 1)


# In[ ]:


#extracting the classes from the label encoder
encoded_classes = encoder.classes_
#mapping the encoded output to actual categories
predicted_category = [encoded_classes[x] for x in pred_classes]
true_category = [encoded_classes[x] for x in y_test]


# In[ ]:


result_df = pd.DataFrame({'description':X_test,'true_category':true_category, 'predicted_category':predicted_category})
result_df.head()


# In[ ]:


print(f"Accuracy is {sklearn.metrics.accuracy_score(result_df['true_category'], result_df['predicted_category'])}")


# In[ ]:


result_df.to_csv('testPredictions.csv', index = False)


# ## Confusion Matrix

# In[ ]:


result_df[result_df['true_category']!=result_df['predicted_category']]


# In[ ]:


confusion_mat = confusion_matrix(y_true = true_category, y_pred = predicted_category, labels=list(encoded_classes))


# In[ ]:


df_cm = pd.DataFrame(confusion_mat, index = list(encoded_classes),columns = list(encoded_classes))
plt.rcParams['figure.figsize'] = (20,20)
sns.heatmap(df_cm)

