#!/usr/bin/env python
# coding: utf-8

# # Sentiment analysis with keras
# ## Data description: A csv with reviews of restaurants from Tripadvisor. 
# ## There are a number of problems with the data, for example the review score is expressed as "4 of 5 bubbles" instead of just "4"

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


raw = pd.read_csv("../input/londonbased-restaurants-reviews-on-tripadvisor/tripadvisor_co_uk-travel_restaurant_reviews_sample.csv")


# ## Let's have a quick check at the data

# In[ ]:


raw.head()


# ## There are a lot of columns there, this is a very comprehensive dataset, but in order to keep things simple, lets just use the review and the score

# In[ ]:


cleaned = raw[["review_text", "rating"]]


# In[ ]:


cleaned.head()


# In[ ]:


set(cleaned["rating"])


# ## The first thing is to convert the "N of 5 bubbles" into just "N" we will use the apply function for that, notice that some rows do not seem to follow that, we have some funny values such as "April 2015 or "September 2015" so our function will have to take care of that

# In[ ]:


def clean(raw_data):
    as_string = str(raw_data["rating"])
    try:
        return int(as_string[0]) # Our number of bubbles is simply the first character
    except:
        # Some values cannot be converted... in which case set to -1 and we will remove them later
        return -1

cleaned["y"] = cleaned.apply(clean, axis=1)


# In[ ]:


cleaned.head()


# ## Lets take a closer look at the text

# In[ ]:


print(cleaned["review_text"][0], "\n")
print(cleaned["review_text"][1], "\n")
print(cleaned["review_text"][2])


# ## As we can see, we are going to need to clean our data (I know... booooring) but this is an important step, if you feed crappy data to a model you will get... well a crappy model. So I know we all want to jump into defining the neural network model, but this has to be done first :)
# 
# ## We will need to 
# 
# 1. Make everything lower case.
# 2. Get rid of pretty much everything that is not a a letter.

# In[ ]:


import re
def clean_text(df):
    text = str(df["review_text"])
    text = text.lower()
    text = re.sub("[^a-z\s]", "", text)
    return text

cleaned["review_text"] = cleaned.apply(clean_text, axis=1)


# In[ ]:


print(cleaned["review_text"][0], "\n")
print(cleaned["review_text"][1], "\n")
print(cleaned["review_text"][2], "\n")


# ## Things look more clear now... as we want to keep this simple, lets just rename our columns to X and y

# In[ ]:


del cleaned["rating"]
cleaned.columns = ["X", "y"]


# In[ ]:


cleaned.head()


# ## Remember that we used a score = -1 to those columns we could not convert, it is time to drop them now

# In[ ]:


cleaned = cleaned[cleaned["y"] > 0]
cleaned.head()


# ## Next, It would be interesting to know how big are our reviews... ultimately we will need to feed to our neural network a fixed text size so we might as well understand the data a big more

# In[ ]:


cleaned["review_size"] = cleaned["X"].str.count(" ")
cleaned["review_size"].describe()


# ## This is relevant information, we seem to have reviews with very few words... lets plot it

# In[ ]:


cleaned["review_size"].plot(title="Number of words per review", figsize=(20, 8))


# ## In the chart above we can see that some reviews have very few words... lets get rid of any review with less than 50 words.

# In[ ]:


cleaned = cleaned[cleaned["review_size"] >= 50]
cleaned["review_size"].describe()


# ## We got rid of quite a lot of reviews, now lets analyze the scores
# 
# ## And lets do a super quick look at our data

# In[ ]:


del cleaned["review_size"]
cleaned.describe()


# ## Here comes the first "surprise" it seems that people are very generous when giving out reviews, the mean is over 4 out of 5, lets plot them to get a better idea

# In[ ]:


scores = cleaned.groupby(["y"]).agg("count")


# In[ ]:


scores.plot(kind='bar', title="Review by number of stars", figsize=(10, 7))


# ## As we mentioned before, we have an interesting distribution, most of people label restaurants with 4 or 5 stars... it is something to take into account, we are dealing with a n asymmetric distribution
# 
# ## Now we are going to prepare the text, lets get our hands dirty

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# ## We are going to tokenize the whole corpus, this means to assign a numbe (id) to each word, so that we will have a list of ints instead of a list of words.

# In[ ]:


# Lets tokenize the works, we will only keep the 20.000 most common words
VOCAB_SIZE=20000
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(cleaned["X"])
sequences = tokenizer.texts_to_sequences(cleaned["X"])
print("Number of sequences", len(sequences))


# In[ ]:


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
print("Index for 'great' is ", word_index["great"])


# ## The next thing is to convert the text to sequences of the same size. Remeber that ultimately a neural network works with matrices, the number of rows is of course the number of reviews, what about the number of columns? in our case we will define a variable ```SEQUENCE_SIZE``` to determine what is that value.
# 
# ## The reviews with less than ```SEQUENCE_SIZE``` words will simply add zeros to the end, and the reviews that are too big will be truncated

# In[ ]:


SEQUENCE_SIZE=120
data = pad_sequences(sequences, maxlen=SEQUENCE_SIZE)
print("Our padded data has a shape of ", data.shape)


# In[ ]:


labels = cleaned["y"]
# Normalize the labels to values between 0 and 1, this can be done by simply dividing by 5, we will later need to multiply our predictions by 5
labels = labels / 5


# In[ ]:


print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# ## Time to do train and test data split, we will use 70% of data for training

# In[ ]:


validation_split = int(data.shape[0] * 0.7)
X_train = data[0: validation_split]
y_train = labels[0: validation_split]

X_test = data[validation_split:]
y_test = labels[validation_split:]


# In[ ]:


print("X_train", X_train.shape, "y_train", y_train.shape)
print("X_test", X_test.shape, "y_test", y_test.shape)


# ## Lets play now: We are going to build a simple network with just an embedding layer, an LSTM (Long Short Term Memory) layer and the output one, notice that this is a REGRESSION problem, not a CLASSIFICATION one, we are not interested on determining whether the review is a 1 or 2 or 3... 5, in our case the output will be a number, for example 3.687, the metric used to determine how good we are going will be the mean square distance
# 
# ## It is important to understand why this will be treated as a regression problem: Imagine we have a given review and its label is 4 stars, now lets imagine we have two different models
# 
# 1.** model A predicts a label of 1.**
# 2. ** model B predicts a label of 5.**
# 
# ## Which model is better? our intuition obviously goes for model B and in this case this is correct, if we measure the error as the absolute difference between our prediction and the actual prediction, model A has an error of 3 (4-1) and model B has an error of 1 (4-1).
# 
# ## What occurs if we treat this as a classification problem? Well, in that case the error would depend on the cross entropy, which will penalize equally models A and B, however if we use the mean average error or the mean squared error, we will have more accurate predictions.
# 
# ## Let's quickly review the formula for mean average error: 
# 
# ![ ](https://wikimedia.org/api/rest_v1/media/math/render/svg/3ef87b78a9af65e308cf4aa9acf6f203efbdeded)
# 
# ## Now do not get scared, the SUM symbol represents simply a loop, Y represents the actual value and X represents each of our predictions, N is just the number of samples
# 

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, TimeDistributed, Embedding, Dropout
EMBEDDING_DIM = 100
model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=SEQUENCE_SIZE))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dropout(0.3))
model.add(Dense(1, activation="linear"))


# In[ ]:


from keras.optimizers import RMSprop

rmsprop = RMSprop(lr=0.0001)

model.compile(loss='mae', optimizer=rmsprop)
for layer in model.layers:
    print("Layer", layer.name, "Is trainable? ==>", layer.trainable)
model.summary()


# ### Aaaand, time to train now

# In[ ]:


history = model.fit(X_train, y_train, epochs=8, batch_size=32, validation_split=0.3)


# ### After training, it is always fundamental to understand how things went, I have learned (the hard way) to always plot the learning curves, so lets do that

# In[ ]:


model_history = pd.DataFrame.from_dict(history.history)
model_history


# In[ ]:


model_history[["loss", "val_loss"]].plot(title="Learning curve, loss", figsize=(17, 6))


# In[ ]:


model.evaluate(X_test, y_test)


# ## Lets make one prediction with our model

# In[ ]:


five_star_review = """My wife and I were here for dinner last night, what an amazing experience. Entree were crab and pork, both amazing dishes full of flavor. I understand what others have said in their reviews re: portion size, however; the skill involved in making these dishes deserves somebody to savour it and not rush it. Mains were the wagyu, and the lamb. Despite both being meat dishes, I couldn't say which one I liked more despite trying both as they were so different. Full of flavours, expertly cooked. The wagyu was tender and the flavours amazing, the lamb was perfect and all the accompaniments were expertly suited. Dessert was a chocolate mousse, you only need a little bit because it's so rich. The whole experience was amazing. A special mention to Ashe who had the most amazing/funny/professional table manner when serving, he's a true asset to the establishment. Unfortunately cash flow has been tight this month for good reasons, but it meant we weren't able to tip after receiving the bill. Despite this, we hope to return again and will make up for our evening last night on our next visit."""
three_star_review = """We had heard good things about this place, but it has let us down twice. The first time it was busy and we gave up waiting for service after about 20 minutes. So, yesterday we tried again with 6 friends who were visiting the 'Mountains.' Sadly, the waiter seemed unaware of almost anything on the menu, got four orders wrong and failed to even place one order at all! He then went home and another server had to deal with the mistakes. Four of us were disappointed with the food and the other four thought it was just okay. The location is great but this cafe needs a real shake-up."""
one_star_review = """The location was spectacular, the fireplace created a beautiful ambiance on a winters day. However the food was a lot to be desired. My friend had been previously with her husband and enjoyed the meals, however what we got was disgusting! We ordered sausages from the local butcher with polenta....well Coles cheap beef sausages are a lot tastier and moist, they looked like they boiled them and put them on a plate, absolutely no grill marks, the polenta was so dry I gagged. My other friend ordered their homemade pie that was stuck in a microwave oven to be heated and ended with soggy pastry. Sorry but definitely need someone new in the kitchen. The staff, well the first lady that took us to our table was lovely, but the second person we ordered our coffees with was so upset that we were going to delay her from leaving work on time. Definitely won't be returning or recomending!"""

texts = [five_star_review, three_star_review, one_star_review]

text_data = pd.DataFrame(data=texts, columns=["review_text"])
text_data["review_text"] = text_data.apply(clean_text, axis=1)
sequences_test = tokenizer.texts_to_sequences(text_data["review_text"])
to_predict = pad_sequences(sequences_test, maxlen=SEQUENCE_SIZE)

output = model.predict(to_predict, verbose=False)
print("Predicted", output*5)


# ## No too bad, our system is actually learning and our validation and training lines are similar, now the question is, can we do better?
# 
# ## And of course the answer is YES!! 
# 
# ## Instead of training the embeddings ourselves, lets use a trained result already, fortunately someone spent a lot of time training the whole wikipedia and providing embeddings of either 50, 100 or 200 dimensions, in our case we will use the 100 dimensions one

# In[ ]:


print(os.listdir("../input"))
print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join("../input/glove-global-vectors-for-word-representation", 'glove.6B.100d.txt'))
count = 0
for line in f:
    count += 1
    if count % 100000 == 0:
        print(count, "vectors so far...")
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))


# In[ ]:


print("Word 'great' has a vector like this \n", embeddings_index["great"])


# In[ ]:


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print("Size of the embedding matrix is ", embedding_matrix.shape)


# In[ ]:


embedding_layer = Embedding(len(word_index) + 1,  # Input dim
                            EMBEDDING_DIM,  # Output dim
                            weights=[embedding_matrix],
                            input_length=SEQUENCE_SIZE,  
                            trainable=False)

# This layer will take an input of 
#           BatchSize x SEQUENCE_SIZE
# And the output will be
#           None, SEQUENCE_LENGTH, 100


# In[ ]:


from keras.layers import Flatten, TimeDistributed
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.3))
model.add(LSTM(128))
model.add(Dropout(0.3))
model.add((Dense(1, activation='linear')))


# In[ ]:


rmsprop = RMSprop(lr=0.001)

model.compile(loss='mae', optimizer=rmsprop)

for layer in model.layers:
    print("Layer", layer.name, "Is trainable? ==>", layer.trainable)
model.summary()


# In[ ]:


history = model.fit(X_train, y_train, epochs=8, batch_size=32, validation_split=0.3)


# In[ ]:


model_history = pd.DataFrame.from_dict(history.history)
model_history


# In[ ]:


model_history[["loss", "val_loss"]][model_history["loss"]< 3].plot(title="Learning curve, loss", figsize=(17, 6))


# In[ ]:


output = model.predict(to_predict, verbose=False)
print("Predicted", output*5)


# In[ ]:


model.evaluate(X_test, y_test)


# In[ ]:




