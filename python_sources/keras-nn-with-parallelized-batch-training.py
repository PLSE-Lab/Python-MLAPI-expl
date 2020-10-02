#!/usr/bin/env python
# coding: utf-8

# I start with [Alex Papiu's](https://www.kaggle.com/apapiu/ridge-script) script where he preprocesses the data into a ```scipy.sparse``` matrix and train a neural network. Keras does not like ```scipy.sparse``` matrices and converting the entire training set to a matrix will lead to computer memory issues; so the model is trained in batches: 32 samples at a time, and these few samples can be converted to matrices and fed into the network. 
# 
# This requieres a batch generator, which I pieced together from this [stack overflow question](https://stackoverflow.com/questions/41538692/using-sparse-matrices-with-keras-and-tensorflow) and I set up an iterator to make it threadsafe for parallelization. ~~Kaggle allows the use of 32 cores which speeds up the training~~. Seems like kaggle only allows four cores.  
# 
# I have been tuning the network and it seems like a smaller network with longer epochs yields better results. Currently I have a two hidden layers with 25  and 10 nodes. This is quite small but, with the input layer considered, this network still yields approximately 1.5M parameters!
# 
# Give it a try and let me know what you think. There are still plenty of things on can try:
# * Add a validation set for early stopping. 
# * Tune `batch_size`, `samples_per_epoch`, and nodes in hidden layers.
# * Add dropout.
# * Add L1 and/or L2 regularization.
#    
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import scipy
import gc
import time
import threading
import multiprocessing

from sklearn import metrics
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

from keras import regularizers 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# the following functions allow for a parallelized batch generator
class threadsafe_iter(object):
    """
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def batch_generator(X_data, y_data, batch_size):
    
    #index = np.random.permutation(X_data.shape[0])    
    #X_data = X_data[index]
    #y_data = y_data[index]
    
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    #idx = 1
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].todense()
        y_batch = y_data[index_batch]
        counter += 1
        yield np.array(X_batch),y_batch
        #print("")
        #print(X_batch.shape)
        #print("")
        #print('generator yielded a batch %d' % idx)
        #idx += 1
        if (counter > number_of_batches):
            counter=0
            
            
@threadsafe_generator
def batch_generator_x(X_data,batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(X_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].todense()
        counter += 1
        yield np.array(X_batch)
        if (counter > number_of_batches):
            counter=0

np.random.seed(1337)  # for reproducibility
start_time = time.time()

NUM_BRANDS = 2500
NAME_MIN_DF = 10
MAX_FEAT_DESCP = 50000

print("Reading in Data")

df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_test = pd.read_csv('../input/test.tsv', sep='\t')

df = pd.concat([df_train, df_test], 0)
nrow_train = df_train.shape[0]
y_train = np.log1p(df_train["price"])

del df_train
gc.collect()

#print(df.memory_usage(deep = True))

df["category_name"] = df["category_name"].fillna("Other").astype("category")
df["brand_name"] = df["brand_name"].fillna("unknown")

pop_brands = df["brand_name"].value_counts().index[:NUM_BRANDS]
df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "Other"

df["item_description"] = df["item_description"].fillna("None")
df["item_condition_id"] = df["item_condition_id"].astype("category")
df["brand_name"] = df["brand_name"].astype("category")

#print(df.memory_usage(deep = True))

print("Encodings...")
count = CountVectorizer(min_df=NAME_MIN_DF)
X_name = count.fit_transform(df["name"])

print("Category Encoders...")
unique_categories = pd.Series("/".join(df["category_name"].unique().astype("str")).split("/")).unique()
count_category = CountVectorizer()
X_category = count_category.fit_transform(df["category_name"])

print("Descp encoders...")
count_descp = TfidfVectorizer(max_features = MAX_FEAT_DESCP, 
                              ngram_range = (1,3),
                              stop_words = "english")
X_descp = count_descp.fit_transform(df["item_description"])

print("Brand encoders...")
vect_brand = LabelBinarizer(sparse_output=True)
X_brand = vect_brand.fit_transform(df["brand_name"])

print("Dummy Encoders..")
X_dummies = scipy.sparse.csr_matrix(pd.get_dummies(df[[
    "item_condition_id", "shipping"]], sparse = True).values)

X = scipy.sparse.hstack((X_dummies, 
                         X_descp,
                         X_brand,
                         X_category,
                         X_name)).tocsr()

#print([X_dummies.shape, X_category.shape, X_name.shape, X_descp.shape, X_brand.shape])

X_train = X[:nrow_train]

tpoint1 = time.time()
print("Time for Preprocessing: {}".format(hms_string(tpoint1-start_time)))

print("Train dimensions:{}".format(X_train.shape))




print("Fitting Model")

n_workers = multiprocessing.cpu_count() #it's 32 on kaggle, 4 on my personal machine
batch_size = 32
            
model = Sequential()
model.add(Dense(25,
                input_dim=X_train.shape[1],
                kernel_initializer='normal',
                activation='relu',
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01),
               ))
model.add(Dense(10, kernel_initializer='normal', activation='relu'))
#model.add(Dropout(0.01))
model.add(Dense(1, kernel_initializer='normal'))
#monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit_generator(generator=batch_generator(X_train, y_train, batch_size),
                    workers=n_workers, 
                    steps_per_epoch=1024, #samples_per_epoch=1024,
                    max_queue_size=128,
                    epochs=37, 
                    verbose=1,
                    use_multiprocessing=False
                   )

tpoint2 = time.time()
print("Time for Training: {}".format(hms_string(tpoint2-tpoint1)))


X_test = X[nrow_train:]
print("Test dimensions:{}".format(X_test.shape))

s = np.floor(X_test.shape[0]/batch_size)+1
print(s)

# unfortunately, predicting is tougher to parallelize...
pred = model.predict_generator(generator=batch_generator_x(X_test, batch_size),
                               steps=s,
                               #workers=n_workers, 
                               #max_queue_size=32, 
                               use_multiprocessing=False,
                               verbose=1
                              )

tpoint3 = time.time()
print("Predict dimensions:{}".format(pred.shape))
print("Time for Predicting: {}".format(hms_string(tpoint3-tpoint2)))



df_test["price"] = np.expm1(pred)
df_test[["test_id", "price"]].to_csv("submission_NN.csv", index = False)

elapsed_time = time.time() - start_time
print("Total Time: {}".format(hms_string(elapsed_time)))




