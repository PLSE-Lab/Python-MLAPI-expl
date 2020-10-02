#!/usr/bin/env python
# coding: utf-8

# # Predicting Recommendation from Review Text : LSA vs Autoencoder vs NN

# ## Table of Content <a id="toc"></a>
# * [Global Variables](#gv)
# * [1. Data Preprocessing](#data_preprocessing)
#     * [1.1 Importing Data and Separating Data of Our Interest](#1.1)
#     * [1.2 Creating Preprocessing Function and Applying it on Our Data](#1.2)
#     * [1.3 Creating TF-IDF Matrix](#1.3)
# * [2. Apply SVD to TF-IDF Matrix](#apply_svd)
#     * [2.1 Create Term and Document Representation](#2.1)
#     * [2.2 Visulize Those Representation](#2.2)
# * [3 Information Retreival Using LSA](#ir_lsa)
# * [4. Create Model to Predict Recommendation](#4)
# * [5. Train Autoencoder on TF-IDF Matrix](#5)
# * [6. Information Retrieval Using Autoencoder](#6)
# * [7. Predict Recommendation using Encoding of Autoencoder](#7)
# * [8. Use simple NN to predict Recommendation](#8)

# In[ ]:


# Global Variables 
K = 2 # number of components
query = 'nice good price'


# ##  1. Data Preprocessing <a id="data_preprocessing"></a>

# ### 1.1 Importing Data and Separating Data of Our Interest <a id="1.1"></a>

# In[ ]:


import pandas as pd
import numpy as np

# Data filename
dataset_filename = "../input/Womens Clothing E-Commerce Reviews.csv"

# Loading dataset
data = pd.read_csv(dataset_filename, index_col=0)


# We are reducing the size of our dataset to decrease the running time of code
list_of_clothing_id = data['Clothing ID'].value_counts()[:10].index
y1 = [x == 0 for x in data['Recommended IND']]
y2 = [x in list_of_clothing_id for x in data['Clothing ID']]
y3 = [a or b for a,b in zip(y1,y2)]

datax = data.loc[y3, :]


# Delete missing observations for variables that we will be working with
for x in ["Recommended IND","Review Text"]:
    datax = datax[datax[x].notnull()]

# Keeping only those features that we will explore
datax = datax[["Recommended IND","Review Text"]]

# Resetting the index
datax.index = pd.Series(list(range(datax.shape[0])))
    
print('Shape : ',datax.shape)
datax.head()


# ### 1.2 Creating Preprocessing Function and Applying it on Our Data <a id="1.2"></a>

# In[ ]:


from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

wordnet_lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'[a-z]+')
stop_words = set(stopwords.words('english'))

def preprocess(document):
    document = document.lower() # Convert to lowercase
    words = tokenizer.tokenize(document) # Tokenize
    words = [w for w in words if not w in stop_words] # Removing stopwords
    # Lemmatizing
    for pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]:
        words = [wordnet_lemmatizer.lemmatize(x, pos) for x in words]
    return " ".join(words)


# In[ ]:


datax['Processed Review'] = datax['Review Text'].apply(preprocess)

datax.head()


# ### 1.3 Creating TF-IDF Matrix <a id="1.3"></a>

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
TF_IDF_matrix = vectorizer.fit_transform(datax['Processed Review'])
TF_IDF_matrix = TF_IDF_matrix.T

print('Vocabulary Size : ', len(vectorizer.get_feature_names()))
print('Shape of Matrix : ', TF_IDF_matrix.shape)


# ## 2. Apply SVD to TF-IDF Matrix <a id="apply_svd"></a>

# ### 2.1 Create Term and Document Representation  <a id="2.1"></a>

# In[ ]:


# import numpy as np

# # Applying SVD
# U, s, VT = np.linalg.svd(TF_IDF_matrix.toarray()) # .T is used to take transpose and .toarray() is used to convert sparse matrix to normal matrix

# TF_IDF_matrix_reduced = np.dot(U[:,:K], np.dot(np.diag(s[:K]), VT[:K, :]))

# # Getting document and term representation
# terms_rep = np.dot(U[:,:K], np.diag(s[:K])) # M X K matrix where M = Vocabulary Size and N = Number of documents
# docs_rep = np.dot(np.diag(s[:K]), VT[:K, :]).T # N x K matrix 


# In[ ]:


import numpy as np
from scipy.sparse.linalg import svds

# Applying SVD
U, s, VT = svds(TF_IDF_matrix) # .T is used to take transpose and .toarray() is used to convert sparse matrix to normal matrix

TF_IDF_matrix_reduced = np.dot(U[:,:K], np.dot(np.diag(s[:K]), VT[:K, :]))

# Getting document and term representation
terms_rep = np.dot(U[:,:K], np.diag(s[:K])) # M X K matrix where M = Vocabulary Size and N = Number of documents
docs_rep = np.dot(np.diag(s[:K]), VT[:K, :]).T # N x K matrix 


# ### 2.2 Visulize Those Representation <a id="2.2"></a>

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(docs_rep[:,0], docs_rep[:,1], c=datax['Recommended IND'])
plt.title("Document Representation")
plt.show()


# In[ ]:


plt.scatter(terms_rep[:,0], terms_rep[:,1])
plt.title("Term Representation")
plt.show()


# ## 3 Information Retreival Using LSA <a id="ir_lsa"></a>

# In[ ]:


# This is a function to generate query_rep

def lsa_query_rep(query):
    query_rep = [vectorizer.vocabulary_[x] for x in preprocess(query).split()]
    query_rep = np.mean(terms_rep[query_rep],axis=0)
    return query_rep


# In[ ]:


from scipy.spatial.distance import cosine

query_rep = lsa_query_rep(query)

query_doc_cos_dist = [cosine(query_rep, doc_rep) for doc_rep in docs_rep]
query_doc_sort_index = np.argsort(np.array(query_doc_cos_dist))

print_count = 0
for rank, sort_index in enumerate(query_doc_sort_index):
    print ('Rank : ', rank, ' Consine : ', 1 - query_doc_cos_dist[sort_index],' Review : ', datax['Review Text'][sort_index])
    if print_count == 4 :
        break
    else:
        print_count += 1


# ## 4. Create Model to Predict Recommendation <a id="4"></a>

# In[ ]:


from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

def create_logistic_model(X,y):
    
    # Splitting data for training and validation
    x_train, x_test, y_train, y_test = train_test_split(pd.DataFrame(X),y,test_size=0.1, random_state=1)
    
    # Getting the input dimension
    input_dim = X.shape[1]
    
    # this is our input placeholder
    input_doc = Input(shape=(input_dim,))
    # This is dense layer
    dense_layer = Dense(1, activation='sigmoid')(input_doc)
    # Our final model
    model = Model(input_doc, dense_layer)
    
    # Compiling model
    model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])
    
    
    # Training model
    history = model.fit(x_train, y_train,
                epochs=5,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, y_test),
                verbose=0)
    # Printing Accuracy
    print('Accuracy on Training Data : ', history.history['acc'][-1])
    print('Accuracy on Validation Data : ', history.history['val_acc'][-1])
    
    # Returning model
    return model, history


# In[ ]:


model_using_lsa, history = create_logistic_model(docs_rep, datax['Recommended IND'])


# In[ ]:


datax['Recommended IND'].value_counts() / datax['Recommended IND'].shape[0]


# In[ ]:


print(np.sum(model_using_lsa.predict(docs_rep) > .5))
print(docs_rep.shape[0])


# ## 5. IR using Autoencoder with TF-IDF matrix <a id="5"></a>

# In[ ]:


from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model

encoding_dim = 2 # Size of encoding
input_dim = TF_IDF_matrix.shape[0] # Size of docs

# Splitting Data for training and validation
df = pd.DataFrame(TF_IDF_matrix.T.toarray())
x_train, x_val, y_train, y_val = train_test_split(df, df[0], test_size=0.1, random_state=1)

# Encoder and Decoder
# this is our input placeholder
input_docs = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='tanh')(input_docs)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='relu')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_docs, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_docs, encoded)

# create a placeholder for an encoded (2-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(loss='mean_squared_error', optimizer='sgd')

history = autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=100,
                shuffle=True,
                verbose=0,
                validation_data=(x_val, x_val))

# encode and decode some data points
print('Original Data : ', x_val[:5])
encoded_datapoints = encoder.predict(x_val[:5])
print('Encodings : ', encoded_datapoints)
decoded_datapoints = decoder.predict(encoded_datapoints)
print('Reconstructed Data : ', decoded_datapoints)


# In[ ]:


docs_rep_autoencoder = encoder.predict(TF_IDF_matrix.T)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(docs_rep_autoencoder[:,0], docs_rep_autoencoder[:,1], c=datax['Recommended IND'])
plt.title("Document Representation")
plt.show()


# ## 6 Information Retreival Using Autoencoder <a id="6"></a>

# In[ ]:


# This is a function to generate query_rep
def autoencoder_query_rep(query):
    query_rep = vectorizer.transform([query])
    query_rep = encoder.predict(query_rep)
    return query_rep


# In[ ]:


from scipy.spatial.distance import cosine

query_rep = autoencoder_query_rep(query)

query_doc_cos_dist = [cosine(query_rep, doc_rep) for doc_rep in docs_rep]
query_doc_sort_index = np.argsort(np.array(query_doc_cos_dist))

print_count = 0
for rank, sort_index in enumerate(query_doc_sort_index):
    print ('Rank : ', rank, ' Consine : ', 1 - query_doc_cos_dist[sort_index],' Review : ', datax['Review Text'][sort_index])
    if print_count == 4 :
        break
    else:
        print_count += 1


# ## 7. Predict Recommendation using Encoding of Autoencoder <a id="7"></a>

# In[ ]:


model_using_autoencoder, history = create_logistic_model(docs_rep_autoencoder, datax['Recommended IND'])


# In[ ]:


datax['Recommended IND'].value_counts() / datax['Recommended IND'].shape[0]


# In[ ]:


print(np.sum(model_using_lsa.predict(docs_rep) > .5))
print(docs_rep.shape[0])


# ## 8. Use simple NN to predict Recommendation <a id="8"></a>

# In[ ]:


from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model

input_dim = TF_IDF_matrix.shape[0] # Size of docs

# Splitting Data for training and validation
df = pd.DataFrame(TF_IDF_matrix.T.toarray())
x_train, x_val, y_train, y_val = train_test_split(df, datax['Recommended IND'], test_size=0.1, random_state=1)

# this is our input placeholder
input_docs = Input(shape=(input_dim,))
layer1 = Dense(100, activation='relu')(input_docs)
layer2 = Dense(10, activation='relu')(layer1)
layer3 = Dense(2, activation='relu')(layer2)
layer4 = Dense(1, activation='sigmoid')(layer3)

# Get encoding
encoder = Model(input_docs, layer3)

# Final Model
model = Model(input_docs, layer4)

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(x_train, y_train,
                epochs=125,
                batch_size=100,
                shuffle=True,
                verbose=0,
                validation_data=(x_val, y_val))

# Printing Accuracy
print('Accuracy on Training Data : ', history.history['acc'][-1])
print('Accuracy on Validation Data : ', history.history['val_acc'][-1])


# In[ ]:


import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# # Displaying all plot of encodings and saving these for report

# In[ ]:


docs_rep_nn = encoder.predict(TF_IDF_matrix.T)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(docs_rep_nn[:,0], docs_rep_nn[:,1], c=datax['Recommended IND'])
plt.show()

plt.savefig('doc_rep_plot_nn.png')


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(docs_rep[:,0], docs_rep[:,1], c=datax['Recommended IND'])
plt.show()

plt.savefig('doc_rep_plot_lsa.png')


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.scatter(docs_rep_autoencoder[:,0], docs_rep_autoencoder[:,1], c=datax['Recommended IND'])
plt.show()

plt.savefig('doc_rep_plot_ae.png')


# In[ ]:




