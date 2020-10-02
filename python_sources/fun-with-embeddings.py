#!/usr/bin/env python
# coding: utf-8

# # Fun with Embeddings
# In this kernel, I show a solution with Embeddings of all categorical features. This approach was popularized by the 3rd place solution of the Rossmann Competition (see python scripts on [github](https://github.com/entron/entity-embedding-rossmann)) and this kernel is inspired by their scripts. Further information about this approach including guidelines about the number of embeddings are given in the [fast.ai](http://fast.ai) course. A nice summary of the approach is given in this [fast.ai blogpost](http://www.fast.ai/2018/04/29/categorical-embeddings/).
# The beauty of the embedding approach is that not much feature engineering is needed. In theory, the fitted embeddings could be analyzed and interpreted. However, this is not done in this version.
# I use the Keras API in this example. Would be interesting to try pytorch with fast.ai-wrapper.

# In[ ]:


import numpy as np
import pandas as pd
import os


# ## Read in the training data set
# The training data set has the following columns: date, store, item, sales. Aim is to predict the sales for each date, store and item. It is important to learn how the data set is ordered (do this by plyaing around with the nrows parameter): It goes through all dates for store=1 and item=1 (1826 entries), then it will go through all dates for store=2 and item=1 etc.
# This becomes relevant when defining the validation set: If we want the last N days as validation set (because we see it as a time series problem), then we cannot simply take the last N entries of the table.

# In[ ]:


nrows = 10

dat = pd.read_csv("../input/train.csv")
dat.head(nrows)


# ## Expand date features
# The following function takes the input data frame and produces the feature matrix. The data is converted to datetime format and year, month, day, weekday are extracted. Also (in the case of training data) the target column ("sales") and in the case of training data the "id" column are split from the feature data. Also, I subtract the first year of the training set (2013) from the year column so that all values in the feature matrix are in the same order of magnitude. Playing around with the model showed that this actually makes a difference...

# In[ ]:


def expand_table(dat):
    dn = dat['date'].map(lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='ignore'))
    X = pd.DataFrame({'year': dn.dt.year-2013, 'month': dn.dt.month, 'day': dn.dt.day, 
                       'weekday': dn.dt.weekday,
                       'store': dat.store, 'item': dat.item,
                      }, columns = ['year', 'month', 'day', 'weekday', 'store', 'item'],
                    )
    
    X = np.array(X)
    if 'sales' in list(dat):
        y = np.array(dat['sales'])
    if 'id' in list(dat):
        y = np.array(dat['id'])

    print(X.shape, y.shape)
    
    return X, y


# In[ ]:


X, y = expand_table(dat)


# ## Read in test set
# I already read in the test set here, since we need the whole data set for one-hot encoding in one of the next steps. Since the data set is quite small, there is no memory issue. Actually, we need the one-hot encoding only for the RandomForest Benchmark, so for a pure Embedding approach, we could also skip the one-hot encoding and read in the test set later.

# In[ ]:


dat_test = pd.read_csv("../input/test.csv")
dat_test.head()


# In[ ]:


X_test, id_test = expand_table(dat_test)


# ## Determine number of unique entries for embedding
# Here, I determine the number of unique entries in each column (the number of categories). This is needed to set the embedding sizes. (Also it can be used to double-check the one-hot encoding result)
# For the embedding sizes, fast.ai suggest to take about half of the original number of the categories (+1) but sometimes it is worth playing around with this. (I am also still testing if there is room for improvement here.)

# In[ ]:


print("determine number of unique entries for embedding...")
X_tot = np.vstack((X, X_test))
print("# of unique years: ", len(np.unique(X_tot[:,0])))
print("# of unique months: ", len(np.unique(X_tot[:,1])))
print("# of unique days: ", len(np.unique(X_tot[:,2])))
print("# of unique weekdays: ", len(np.unique(X_tot[:,3])))
print("# of unique stores: ", len(np.unique(X_tot[:,4])))
print("# of unique items: ", len(np.unique(X_tot[:,5])))


# ## One-hot encoding
# The following cell allows one-hot encoding of the feature matrices. This is recommended for the Mean and Random Forest Benchmark. For the Embedding approach, however, we don't need it!

# In[ ]:


import sklearn.preprocessing as prepro

input_one_hot = False

if input_one_hot:
    print("Using one-hot encoding...")
    enc = prepro.OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')
    enc.fit(np.vstack((X[:,1:], X_test[:,1:])))
    X = np.hstack((X[...,[0]],enc.transform(X[:,1:])))
    X_test = np.hstack((X_test[...,[0]],enc.transform(X_test[:,1:])))

print(X.shape)
print(X_test.shape)


# ## Validation set
# Choose whether you want a random split or take the last xx% of the data as validation set. Beware: If we treat it as a time series and we want the last 10% of the data as validation set, we mean the last 10% in time. However, this is not how the data is sorted... So what I do here is take the data of the last 6 month (July - December 2017) as validation set.

# In[ ]:


from sklearn.model_selection import train_test_split

random_split = True
train_ratio = 0.9

if random_split:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1-train_ratio), random_state=0, shuffle = True)
else:
    train_size = int(train_ratio*len(X))
    X_train = X[(X[:,0]==4)&(X[:,1]>6)]
    y_train = y[(X[:,0]==4)&(X[:,1]>6)]
    X_val = X[(X[:,0]!=4)|(X[:,1]<=6)]
    y_val = y[(X[:,0]!=4)|(X[:,1]<=6)]
    
print("training: ", X_train.shape, y_train.shape)
print("validation: ", X_val.shape, y_val.shape)
print("test: ", X_test.shape)


# ## Speeding it up for quick test...
# Choose whether you want to run the test with a sub sample only to check the workflow. For the final test, you want to use all the data of course.

# In[ ]:


sub_sample = False
sub_sample_size = 50000

def subsample(X, y, n=sub_sample_size):
    ind = np.random.randint(X.shape[0], size=n)
    return X[ind,:], y[ind]

if sub_sample:
    X_train, y_train = subsample(X_train, y_train, 50000)
    print("subsample of training set: ", X_train.shape, y_train.shape)


# ## Modelling
# Here I define a class Model that has the method "evaluate". This method makes predictions and returns the SMAPE, which is the evalutation metric used in this competition.

# In[ ]:


class Model(object):
    
    def evaluate(self, X, y):
        preds = self.prediction(X)
        ind = np.where((y!=0)|(preds!=0))
        return 100*np.sum(np.abs(preds[ind] - y[ind])/((np.abs(preds[ind]) + np.abs(y[ind]))/2))/len(y)


# ### Benchmark Model: Mean
# This is a simple benchmark. The model returns the mean of the training set for every input row.

# In[ ]:


class BenchmarkMean(Model):
    
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.pred = np.mean(y_train)
        print("Result on validation data: ", self.evaluate(X_val, y_val))
        
    def prediction(self, X):
        return np.full(X.shape[0], self.pred)


# In[ ]:


benchmark_mean = False

if benchmark_mean:
    res = BenchmarkMean(X_train, y_train, X_val, y_val)
    pred = res.prediction(X_test)
    pd.Series(pred).describe()


# ### Benchmark: Random Forest
# Another benchmark: A simple random forest. For the random forest, a one-hot-encoding is recommended. In this kernel, the RandomForest only serves as a benchmark. To give better predictions with the Random Forest, more time series features should be employed.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor        
        
class RandomForest(Model):
    
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self.clf = RandomForestRegressor(n_estimators=200, verbose=True, max_depth=35, min_samples_split=2)
        self.clf.fit(X_train, y_train)
        print("Result on validation data: ", self.evaluate(X_val, y_val))
    
    def prediction(self, X):
        return self.clf.predict(X)


# In[ ]:


random_forest = False

if random_forest:
    res = RandomForest(X_train, y_train, X_val, y_val)
    pred = res.prediction(X_test)


# ### The main model: Neural Network with Embeddings
# The following model is the main model of this kernel: A Neural Network with Embeddings. I embed all features except for year . This is for the following reason: 1. the sales increase from year to year, in that sense there is a simple linear trend. 2. the test set is from a different year that the training set. If we were to embed the year, the year category of the test set has not been seen while training, so basically all information in the year would be lost...
# I am still experimenting with Dropout and Early Stopping, till now I don't see any clear advantage.
# I run the model fit 5 times and then take the average. It might be worth testing a bit more with different epoch numbers, learning rates, etc.

# In[ ]:


from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Activation, Reshape, Dropout
from keras.layers import Concatenate
from keras.layers.embeddings import Embedding
from keras import optimizers, regularizers
from keras.callbacks import EarlyStopping
import keras.backend as K

# helper function for NN input
def split_features(X):
    X_list = []
    for i in range(6):
        X_list.append(X[:,i])
    
    return X_list

# custom loss function
def smape(x, y):
    return 100.*K.mean(2*K.abs(x-y)/(K.abs(x)+K.abs(y)))


class NNwithEmbeddings(Model):
    
    def __init__(self, X_train, y_train, X_val, y_val):
        super().__init__()
        self._build_model()
        self.fit(X_train, y_train, X_val, y_val)
        
    def preprocess(self, X):
        X_list = split_features(X)
        return X_list
        
    def _build_model(self):
        ## year is a continuous feature
        inp_year = Input(shape=(1,), name="year")
        
        ## all other features are categorical and need embedding
        inp_month = Input(shape=(1,))
        out_month = Embedding(12+1, 7, name='month_embedding')(inp_month)
        out_month = Reshape(target_shape=(7,))(out_month)
        
        inp_day = Input(shape=(1,))
        out_day = Embedding(31+1, 16, name='day_embedding')(inp_day)
        out_day = Reshape(target_shape=(16,))(out_day)
        
        inp_weekday = Input(shape=(1,))
        out_weekday = Embedding(7+1, 4, name='weekday_embedding')(inp_weekday)
        out_weekday = Reshape(target_shape=(4,))(out_weekday)
        
        inp_stores = Input(shape=(1,))
        out_stores = Embedding(10+1, 6, name='stores_embedding')(inp_stores)
        out_stores = Reshape(target_shape=(6,))(out_stores)
        
        inp_items = Input(shape=(1,))
        out_items = Embedding(50+1, 26, name='items_embedding')(inp_items)
        out_items = Reshape(target_shape=(26,))(out_items)
        
        
        inp_model = [inp_year, inp_month, inp_day, inp_weekday, inp_stores, inp_items]
        out_embeddings = [inp_year, out_month, out_day, out_weekday, out_stores, out_items]
        
        out_model = Concatenate()(out_embeddings)
        out_model = Dense(100)(out_model)
        out_model = Activation('relu')(out_model)
        #out_model = Dropout(0.3)(out_model)
        out_model = Dense(10)(out_model)
        out_model = Activation('relu')(out_model)
        #out_model = Dropout(0.3)(out_model)
        out_model = Dense(1)(out_model)
        
        self.model = KerasModel(inputs=inp_model, outputs=out_model)
        
        self.model.compile(optimizer='Adam', loss=smape)
        
    
    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(self.preprocess(X_train), y_train,
                       validation_data=(self.preprocess(X_val), y_val),
                       epochs=10, batch_size=128,
                       #callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
                      )
        print("Result on validation data: ", self.evaluate(X_val, y_val))
        
    def prediction(self, X):
        return self.model.predict(self.preprocess(X)).flatten()
        


# In[ ]:


embedding_nn = True

if embedding_nn:
    res = []
    for i in range(5):
        res.append(NNwithEmbeddings(X_train, y_train, X_val, y_val))
    ps = np.array([r.prediction(X_test) for r in res])
    pred = ps.mean(axis=0)


# ## Submission
# The following lines produce an output file for submission.

# In[ ]:


subm = pd.DataFrame({'id': id_test, 'sales': pred})
subm.head()


# In[ ]:


subm.to_csv('subm_embedding.csv', index=False)

