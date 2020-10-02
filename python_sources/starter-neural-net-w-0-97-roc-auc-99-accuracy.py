#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import keras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.manifold import TSNE
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Dense, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

df = pd.read_csv("/kaggle/input/bertembedded-spam-messages/spam_encoded.csv")
df.head()


# In[ ]:


plt.subplots(figsize=(10,8))
ax = sns.countplot(x="spam", data=df)


# In[ ]:


df["spam"].value_counts()


# We can see that there is the same amout of class imbalance as in the original dataset: **13.41%** of messages are spam
# 
# # Basic feature engineering
# 
# Adding just some counts based on the original messages (number of words, number of characters, etc.)

# In[ ]:


# Count the number of words in the message
df["num_words"] = df["original_message"].apply(lambda s: len(re.findall(r'\w+', s)))

# Get the length of the text message
df["message_len"] = df["original_message"].apply(len)

# Count the number of uppercased characters
df["num_uppercase_chars"] = df["original_message"].apply(lambda s: sum(1 for c in s if c.isupper())) 

# Count the numbe rof uppercased words
df["num_uppercase_words"] = df["original_message"].apply(lambda s: len(re.findall(r"\b[A-Z][A-Z]+\b", s)))

# Check if the message contains the word "free" or "win"
df["contains_free_or_win"] = df["original_message"].apply(lambda s: int("free" in s.lower() or "win" in s.lower()))


# # Standardizing features
# 
# For better performance in algorithms that optimize a loss-function it is best if every feature is in the same scale. To this end lets standardize all features except the binary features and the original text column

# In[ ]:


# Initialize StandardScaler
scaler = preprocessing.StandardScaler()

# Dont standerdize binary columns and the text column
feats_to_scale = df.drop(["spam", "original_message", "contains_free_or_win"], axis=1)

# Create a new dataframe with the standardized features
scaled_features = pd.DataFrame(scaler.fit_transform(feats_to_scale))
scaled_features.rename(
    {768: "num_words", 769:"message_len", 770: "num_uppercase_chars", 771: "num_uppercase_words"},
    axis=1,
    inplace=True
)

# Update the dataset with the new standerdized features
scaled_df = df.copy()
scaled_df.update(scaled_features)


# # Train-Test split
# 
# We'll leave a held out test set and perform our analysis on the training set

# In[ ]:


X = scaled_df.drop(["spam", "original_message"], axis=1)
y = scaled_df["spam"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


# # Exploratory data analysis
# 
# Here the idea is to get a sense of the data.
# 
# First we can see the manifold distribution of the data using T-SNE, where we can see how the embeddings encode the messages

# In[ ]:


X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X_train)


# Creating additional dataframes for plotting

# In[ ]:


# creating the dataframe for plotting
def creat_plotting_data(data, labels=y_train, rename=False):
    """Creates a dataframe from the given data, used for plotting"""
    
    df = pd.DataFrame(data)
    df["spam"] = labels.to_numpy()
    
    if rename:
        df.rename({0:"v1", 1:"v2"}, axis=1, inplace=True)
        
    return df

# creating the dataframes for plotting
plotting_data_embedded = creat_plotting_data(X_embedded, rename=True)


# # T-SNE visualization of SMS messages
# 
# We can see how the messages are distributed in manifold space. Since we have the BERT embedings for every message it should produce interesting results

# In[ ]:


plt.figure(figsize=(16, 10))
ax = sns.scatterplot(x="v1", y="v2", hue="spam", data=plotting_data_embedded)
ax.set(title = "Spam messages are generally closer together due to the BERT embeddings")
plt.show()


# We clearly see that the BERT embeddings together with the additional engineered features we get a training set where spam messages are quite close together with some outliers amongst ham messages.
# 
# # Do spam messages have more words than ham messages?

# In[ ]:


_,ax = plt.subplots(figsize=(16,10))
sns.kdeplot(df.loc[df.spam == 0, "num_words"], shade=True, label="Ham", clip=(0, 35)) # removing observations with message length above 35 because there is an outlier
sns.kdeplot(df.loc[df.spam == 1, "num_words"], shade=True, label="Spam")
ax.set(xlabel = "Number of words", ylabel = "Density",title = "Spam messages have more words than ham messages")
plt.show()


# This plot shows us that yes **spam** messages, generally, have **more words** than ham messages. 
# 
# **Spam** messages seem to be concentrated around 30 words while **ham** messages are concentrated around 10 words
# 
# # Is there any pattern in the length of the messages?

# In[ ]:


_,ax = plt.subplots(figsize=(16,10))
sns.kdeplot(df.loc[df.spam == 0, "message_len"], shade=True, label="Ham", clip=(0, 250)) # removing observations with message length above 250 because there is an outlier
sns.kdeplot(df.loc[df.spam == 1, "message_len"], shade=True, label="Spam")
ax.set(xlabel = "Message length", ylabel = "Density",title = "Spam messages are longer than ham messages, concentrated on 150 characters")
plt.show()


# This plots shows two things:
# 
# - Spam messages are, generally, longer than ham messages (which is to be expected since they have more words)
# - There seems to be a common value of around 150 characters for spam messages
# 
# # Do spam messages have more uppercased words?

# In[ ]:


_,ax = plt.subplots(figsize=(16,10))
sns.kdeplot(df.loc[df.spam == 0, "num_uppercase_words"], shade=True, label="Ham", clip=(0, 250)) # removing observations with message length above 250 because there is an outlier
sns.kdeplot(df.loc[df.spam == 1, "num_uppercase_words"], shade=True, label="Spam")
ax.set(xlabel = "Number of uppercased words", ylabel = "Density",title = "Number of uppercased words don't seem to have any patter with spam")
plt.show()


# So the number of uppercased words does not seem to have any particular pattern with Spam
# 
# ## Is having the word "free" or "win"  more associated with spam messages?

# In[ ]:


plt.subplots(figsize=(10,8))

# Get the proportion of the genders grouped by the attrition status
grouped_data = df.groupby("spam")["contains_free_or_win"].value_counts(normalize = True).rename("Percentage of group").reset_index()
print(grouped_data)

# Plot the result
ax = sns.barplot(x="spam", y="Percentage of group", hue="contains_free_or_win", data=grouped_data)


# - **36.95% of spam** messages contain the word "free" or "win"
# - Only **2.69% of ham** messages contain the word "free" or "win"

# In[ ]:


plt.subplots(figsize=(16,10))
ax = sns.scatterplot(x="message_len", y="num_uppercase_chars", hue="spam", data=df)
ax.set(
    xlabel="Number of characters in the message",
    ylabel="Number of uppercase characters in the message",
    title="Spam messages are clustered together. There is a strong linear pattern for ham messages with high number of uppercase characters")
plt.show()


# - **Spam** messages are really clustered together in terms of message length with some spam messages having more uppercase characters than others
# 
# - There is a strong linear pattern for **ham** messages with higher number of uppercased characters
# 
# # Simple neural net
# 
# To demonstrate the efficiency of the embeddings we'll create a simple neural network
# 
# ## Defining the architecture of the network
# 
# Lets go for a 1000-256-256-128-10-1 layered network, not particular reason for it.
# 
# We'll add dropout to help prevent overfitting and batchnorm for better convergence

# In[ ]:


model = Sequential()

model.add(Dense(1000, input_shape=(773,), activation="relu"))
model.add(BatchNormalization(axis=-1))

model.add(Dense(256, activation="relu"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

model.add(Dense(256, activation="relu"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.5))

model.add(Dense(128, activation="relu"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

model.add(Dense(10, activation="relu"))
model.add(BatchNormalization(axis=-1))
model.add(Dropout(0.25))

model.add(Dense(1, activation="sigmoid"))


# ## Compiling the network
# 
# We'll use a RMSProp optimizer with 0.001 learning rate but we'll reduce this learning rate by half if the validation accuracy has not increased in the last 3 epochs (we should not be using accuracy since its a skewed dataset but it will do for now)

# In[ ]:


# defining the learning rate, the number of epochs and the batch size
INIT_LR = 0.001
NUM_EPOCHS = 30
BS = 64
opt = RMSprop(lr = INIT_LR)

# This is just a necessary step to compile the model, we don't actually need it because we're not using the old model
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

# Reduce the learning rate by half if validation accuracy has not increased in the last 3 epochs
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

fitted_network = model.fit(X_train, y_train, validation_split=0.2, batch_size=BS, epochs=NUM_EPOCHS, callbacks=[learning_rate_reduction])


# # Run predictions on the test set
# 
# Our network achieved pretty good results in the training and validation data, let's see if that is the case for the test set as well

# In[ ]:


# predict results
preds = np.round(model.predict(X_test)).flatten()

# Plot confusion matrix
plt.figure(figsize=(10,4))
heatmap = sns.heatmap(data = pd.DataFrame(confusion_matrix(y_test, preds)), annot = True, fmt = "d", cmap=sns.color_palette("Reds", 50))
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
plt.ylabel('Ground Truth')
plt.xlabel('Prediction')
plt.show()

# Print accuracy, ROC and classification report for the test-set
print(f"""Accuray: {round(accuracy_score(y_test, preds), 5) * 100}%
ROC-AUC: {round(roc_auc_score(y_test, preds), 5)}""")
print(classification_report(y_test, preds))


# # Results
# 
# It is clear that our network with the BERT embeddings and the engineered features achieved great results
# 
# - Retrieved 95% of all spam messages
# - Of all those messages the network said were spam, it was right 97% of the times
# - Only 5 ham messages were missclassified
# 
# For comparison in my [first analysis](https://www.kaggle.com/mrlucasfischer/bert-the-spam-detector-that-uses-just-10-words) I created a baseline random forest (with no hyperparameter tunning) that achieved 0.8806 ROC-AUC
# 
# Hope this helps get started using this dataset, and I hope this motivates with the usefulness of the BERT embeddings
# 
# Cheers! :)
