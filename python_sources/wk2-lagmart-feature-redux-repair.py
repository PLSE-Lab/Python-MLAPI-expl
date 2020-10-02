#!/usr/bin/env python
# coding: utf-8

# # Wk 2. Lagmart, Rik's take on Enea's brilliant insight
# So the competition is over. We had a very brief summary of the winner's approach. Enea Caccia (group 8) scored a loss (MAE) of 2144 on the private leaderboard. Our attempts kept lagging around 13000. The surprising bit was that only 5 teams (out of 11) did better than the 2650 hurdle.  The rest of us scored 12950 or worse.
# 
# My old model simply tried to enhance the simplistic Baseline Jannes Klaas had provided. That was the simplest one layer linear regressor possible. I tried to improve it by making the NN more complicated; more layers, more units and activation functions. My feature engineering was a convoluted attempt to compute distance from today to the nearest holiday.
# 
# No matter how I tried, the model kept giving me predictions that were all the same number for 120000 different inputs. It was time for a radical change, but I did not know what to radically change. Until Enea's debriefing yesterday...
# 
# ## Enea's humble median
# The winning entry had not even used a neural net. Enea computed the median from grouped samples:
# * median per Store per Dept per Month (of year) per IsHoliday (boolean field in the data)
# 
# Here is [his kernel](https://www.kaggle.com/eneacaccia/week2-challenge-simple-stats). Here is my cleaned up, completed [version](https://www.kaggle.com/rikdedeken/week2-challenge-simple-stats). Both score a Loss = 2144.
# 
# 
# ## My own variation: ensemble of means
# So precompute median Weekly_Sales heh? Bene. I can do that. I can even precompute a handfull. Throw away every other bit of input? Easy. Than find a suitable weighted average of the precomputed numbers? Hey! That sounds like an optimization job for a neural net....
# 
# That's what I do in this notebook. And it still did not work. **Until**.... I threw away my activation functions and most of the layers.
# 
# My model now looks something like this:
# 
# 11 input units -> 1 output unit
# 
# Without any activation function.
# 
# ## Data and features
# I precomputed the following medians:
# * StoreMean (an average of all Sales for a particular store, over the entire rage of dates)
# * DeptMean (an average of all Sales for a particular department ID, across the whole Walmart chain, for all dates)
# * Store_DeptMean (an average of all Sales for this dep in this store, for all dates)
# * MonthMean (an average for all Sales this month of the year, across the Walmart chain)
# * Store_MonthMean (an average for this month, this store, all dates)
# * Dept_MonthMean (an average for this month, this department ID, across the chain, for al dates)
# * Store_Dept_MonthMean (an average for this month, this Dept, in this Store)
# * IsHolidayMean (one average from all samples that have IsHoliday=true)
# 

# The input consists of:
# 
# 1. IsHoliday 
# 2. Month
# 3. Week
# 4. MonthMean
# 5. Store_MonthMean
# 6. Dept_MonthMean
# 7. Store_DeptMean
# 8. IsHolidayMean
# 9. StoreMean
# 10. DeptMean
# 11. Store_Dept_MonthMean
# 
# None of these were scaled, standardized, normalized or anything. I fed them raw into the NN.
#        

# ## my best score so far?
# 
# 
# 
# 
# During training I validated on 20% of training data. The Loss got as low as:
# ```
# Loss by Mean Absolute Error
# train:       1720.68063151
# validation:  1721.73183077
# ```
# After submitting these predictions to Kaggle, I scored ```2018.75718```
# 
# My latest submissions:
# ```
# .       train/val  -> kaggle
# feb 21: 1650/1683  -> 2025 (one layer NN: in>out)
# feb 21: 1656/1661  -> 2043 (two layer NN: in>2>out)
# ```

# ## Possible improvements
# * normalize inputs
# * bigger NN
# * use actual .mean() instead of .median()
# * fewer inputs
# * more inputs
# * train more/longer
# * final training round with 100% training data (who needs validation that late in the race anyway?)
# * reintroduce optimization layers (batchnorm, dropout), tell the model to regularize (still trying to find out where I went wrong in the first place!)
# 
# ## Possible code improvements
# * Make computations of grouped means easier on the CPU. Use the groupby() method in combination with merge() or join().
# * Prettier graphs 8-)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.layers import Dense, Activation, BatchNormalization, Dropout
import keras.models
from keras import regularizers
from keras import optimizers

import re, datetime, time
from sklearn.model_selection import train_test_split


# In[ ]:


print(datetime.datetime.now())


# ## load data from CSV and organize

# In[ ]:


# load hard work on features from csv files
get_ipython().system(' find ../input ')


# In[ ]:


# @rik: the only reason you indexed Date is to produce a date_plot
# with blanks in it at all the right places

get_ipython().run_line_magic('xdel', 'train')
get_ipython().run_line_magic('xdel', 'test')
get_ipython().run_line_magic('xdel', 'train_f')
get_ipython().run_line_magic('xdel', 'test_f')
get_ipython().run_line_magic('xdel', 'df_b')
get_ipython().run_line_magic('xdel', 'df_f')
get_ipython().run_line_magic('xdel', 'df')
get_ipython().run_line_magic('xdel', 'pristine')

# choose: precomputed features, or bare Walmart data, or both

# precomputed features:
# train_f = pd.read_csv('../input/bletchley-wk2-featured-walmart/train_features_redux2.csv')
# test_f = pd.read_csv('../input/bletchley-wk2-featured-walmart/test_features_redux2.csv')
# df_f = pd.concat([train_f,test_f],axis=0) # Join train and test
# df_f.drop(['IsHoliday', 'Weekly_Sales'], inplace=True, axis=1)

# bare Walmart:
train = pd.read_csv('../input/course-material-walmart-challenge/train.csv')
test = pd.read_csv('../input/course-material-walmart-challenge/test.csv')

# train = pd.read_csv('../input/train.csv')
# test = pd.read_csv('../input/test.csv')

df = pd.read_csv('../input/bletchley-wk2-featured-walmart/df_features_redux2.csv')

# df = pd.concat([train,test],axis=0) # Join train and test
# df_b = pd.concat([train,test],axis=0) # Join train and test
# df = pd.concat([df_b,df_f],axis=1) 

len_train = len(train)
len_test = len(test)

pristine = df.copy()
# pristine['Date'] = pd.to_datetime(df['Date'])
#### DO NOT SORT concatenated df: this will make the cat irreversible!!!!!!!!!!!!!!!!!!!!!!!


# In[ ]:


df.columns


# ## new features: categories from date

# In[ ]:


def add_datepart(df, fldname, parts=[], drop=True):
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.
    """

    # also available on Kaggle: import fastai
    # This is my adaptation
    # Acknowledgment: Fast.ai ML library https://github.com/fastai/fastai/blob/master/fastai/structured.py
    # Jeremy Howard c.s.

    if len(parts) == 0:
        parts = [
            'Year', 
            'Month', 
            'Week', 
            'Dayofyear',
            'Dayofweek', 
            'Day', 
            'Is_month_end', 
            'Is_month_start', 
            'Is_quarter_end', 
            'Is_quarter_start', 
            'Is_year_end', 
            'Is_year_start',
            'Elapsed',
                ]

    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)

    # remove the word "date" from the end of the fldname
    targ_pre = re.sub('[Dd]ate$', '', fldname) 
    for n in parts:
        target = targ_pre+n
        if not target in df.columns:
            df[target] = getattr(fld.dt,n.lower())
            if n == 'Elapsed':
                df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9

    if drop: df.drop(fldname, axis=1, inplace=True)


# In[ ]:


# This clever function adds numerical and boolean fields for every(?) conceivable attribute of a Date object as new colums to df
add_datepart(df, 'Date', parts=['Year', 'Month', 'Week'], drop=False)


# ## new features: boolean (dummy) categories from categories

# In[ ]:


# create dummy features/columns for categorical data

catcats = [
#     "Year",
#     "Month",
#     "Type",
#     "Store",
#     "Dept",
]

for c in catcats:
    if c in df.columns:
        dummies = pd.get_dummies(df[c], prefix=c, prefix_sep='')
        df = pd.concat([df, dummies], axis=1)


# # new feature: mean weekly sales
# 
# per Store, per Dept, per Month, per IsHoliday, we establish the mean Weekly_Sales (overall mean is ca 10445). 
# Let's introduce these means into a column of their own. 
# This idea was heavily inspired by Enea's winning entry in the in-class competition at Bletchley Bootcamp, 19 feb 2018.
# 

# In[ ]:


def add_feature_mean_y(df, feature, y):
    # create new feature in column of df: fill with mean of y for every category of feature
    # the new column gets a name: name of given feature + "Mean"
    cname = "".join([feature, "Mean"])
    if cname in df.columns: return(False)
    
    df[cname] = df[y].mean() # create column, fill with not so random value
    
    # iterate over all unique categories in given feature
    # for combined features (e.g. Store_Dept_Month) this takes a LOT of compute time (say 1 hour)
    for k in df[feature].unique():
        # in rows of cat k, to new column , assign of all rows of cat k the mean(y)
#         mean = train.loc[train[feature] == k, [y]].median().values[0]
        mean = df.loc[df[feature] == k, [y]].median().values[0]   ## FIXME if shit happens 201802211020
        df.loc[df[feature] == k, [cname]] = mean
    return(cname)


# In[ ]:


def concat_features(df, features):
    # create new categorical feature named after a concatenation of the individual feature names
    # fill with concatenated values of those feature
    # example: features named   Given + Family = Given_Family
    #          feature values   joan  + jet    = joan_jet
    # separator is underscore _ for both names and values
    # any number of features (>1) can be concatenated
    # df['Store_Dept'] = df[['Store', 'Dept']].astype(str).apply(lambda x: '_'.join(x), axis=1)
    cname = '_'.join(features)
    if cname in df.columns: return(False)

    df[cname] = df[features].astype(str).apply(lambda x: '_'.join(x), axis=1)
    return(cname)


# ## create new features: combined categories

# In[ ]:


concat_features(df, ['Store', 'Month'])


# In[ ]:


concat_features(df, ['Dept', 'Month'])


# In[ ]:


concat_features(df, ['Store', 'Dept'])


# In[ ]:


concat_features(df, ['Store_Dept', 'Month'])


# In[ ]:


concat_features(df, ['Year', 'Month'])


# ## create features: mean_y for categories

# In[ ]:


# we make this split in between feature engineering, because we want to extract mean values from the y column
# and we do not want the zero ys from the test set to drag the mean down
# train = df.iloc[:len_train]


# In[ ]:


add_feature_mean_y(df, 'Store', 'Weekly_Sales')


# In[ ]:


add_feature_mean_y(df, 'Dept', 'Weekly_Sales')


# In[ ]:


add_feature_mean_y(df, 'Store_Dept', 'Weekly_Sales')


# In[ ]:


add_feature_mean_y(df, 'Month', 'Weekly_Sales')


# In[ ]:


add_feature_mean_y(df, 'Store_Month', 'Weekly_Sales')


# In[ ]:


add_feature_mean_y(df, 'Dept_Month', 'Weekly_Sales')


# In[ ]:


add_feature_mean_y(df, 'Store_Dept_Month', 'Weekly_Sales')


# In[ ]:


add_feature_mean_y(df, 'IsHoliday', 'Weekly_Sales')


# In[ ]:


add_feature_mean_y(df, 'Year_Month', 'Weekly_Sales')


# ## nice graphics

# In[ ]:


# plots sales and holidays in time for specific Store/Dept
def plot_store_dept(plotme):
    wsm = plotme.Weekly_Sales.median()
    fig, ax = plt.subplots(figsize=(13,5))
    ax.set_ylabel("Weekly_Sales")
    ax.bar(x=plotme.index, height=plotme["Weekly_Sales"], color='g', width=7, label="known sales")
    ax.axhline(y=wsm, c='g', label="Store_DeptMean")
    ax.plot_date(x=plotme.index, y=(plotme["IsHoliday"] * wsm), fmt='*m', ms=16)
    ax.plot_date(x=plotme.index, y=plotme['Dept_MonthMean'], fmt='.k', ms=3)
    ax.plot_date(x=plotme.index, y=plotme['Store_MonthMean'], fmt='.y')
    ax.plot_date(x=plotme.index, y=plotme['Store_Dept_MonthMean'], fmt='+c')
    ax.legend()
    plt.title(f"Store: {store}\nDept: {dept}")

    plt.show()


# In[ ]:


store=1
dept=1


# In[ ]:


store += 1
where = (df["Store"] == store) & (df["Dept"] == dept) 
select = ['Date', 'Weekly_Sales', 'IsHoliday', 'Store_MonthMean', 'Dept_MonthMean', 'Store_Dept_MonthMean']
plotme = df.loc[where, select]
plotme.set_index('Date', inplace=True)
plot_store_dept(plotme)


# ## last minute cleanup
# mostly categorical noise and features that we do not want our model exposed to, also fill Nulls (Nans) with zeroes, but most of all save save save!

# In[ ]:


df.fillna(0, inplace=True) # fill NAs with zeroes


# In[ ]:


# df.to_csv('df_features_redux2.csv', index=False)


# In[ ]:


drop_these = [
    'Date',
    'Store',
    'Dept',
    'Type',
    'IsHoliday',

    'CPI',
    'Fuel_Price',
    'MarkDown1',
    'MarkDown2',
    'MarkDown3',
    'MarkDown4',
    'MarkDown5',
    'Size',
    'Temperature',
    'Unemployment',

    'Week',
    'Month',
    'Year',
    'Year_Month',
    'Store_Month',
    'Dept_Month',
    'Store_Dept',
    'Store_Dept_Month',

#     'MonthMean',
#     'Store_MonthMean',
#     'Store_Dept_MonthMean',
#    'IsHolidayMean',
#     'StoreMean',
#     'DeptMean',
        ]

for dropje in drop_these:
    if dropje in df.columns:
        df.drop(dropje, axis=1, inplace=True)


# In[ ]:


# final sanity check
df[:12]


# In[ ]:


for c in df.columns.sort_values():
    print(f"\'{c}\',")


# # split df into train + test and (maybe) val

# In[ ]:


train = df.iloc[:len_train]
X_labels = train.drop('Weekly_Sales',axis=1).columns.astype(str)
X = train.drop('Weekly_Sales',axis=1).values
y = train['Weekly_Sales'].values

test = df.iloc[len_train:]
test = test.drop('Weekly_Sales',axis=1) # We should remove the nonsense values from test


# In[ ]:


# save hard work on features in new csv files
# train.to_csv('train_features_redux2.csv',index=False)
# test.to_csv('test_features_redux2.csv',index=False)
# ! ls -l *csv


# In[ ]:


# Split training set into sets for training and validation
# or, alternatively: order model.fit() to take its own splits

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
X=X_train
y=y_train
print(X.shape)
print(X_val.shape)
m_val,n_val = X_val.shape
m,n = X.shape


# ## Tweak your hyper parameters here

# In[ ]:


lamb = .01   # regularization rate
drop = 0.10  # dropout rate


# In[ ]:


#, kernel_regularizer = regularizers.l2(lamb)
# model.add(BatchNormalization())
# model.add(Activation('tanh'))
# model.add(Dropout(rate=drop))

get_ipython().run_line_magic('xdel', 'model')
model = keras.models.Sequential()

model.add(Dense(units=n, input_dim=n))
model.add(Dense(units=n))
model.add(Dense(units=3))
# model.add(BatchNormalization())

model.add(Dense(units=1))

model.compile(optimizer=optimizers.adam(), loss='mae')


# In[ ]:


model.summary()


# ## alpha, batchsize, epochs

# In[ ]:


# these hyper parameters can be tweaked after compiling the model
# this is useful for retraining an existing model under different params
alpha = .01    # learning rate
bs = min(2**14, m//2)  # batch size: maxed at half size testset
epochs = 400  # number of epochs per training round
model.optimizer.lr = alpha


# In[ ]:


# evaluate model on training and validation sets
print("Loss by Mean Absolute Error")

print("train:      ", model.evaluate(x=X,     y=y,     verbose=0, batch_size=bs))
print("validation: ", model.evaluate(x=X_val, y=y_val, verbose=0, batch_size=bs))


# # start training

# In[ ]:


timestart = datetime.datetime.now()


# In[ ]:


df.head()


# In[ ]:


print("Learning Rate: %f, BatchSize: %i "% (alpha, bs))
# first we train one epoch to get the ugly random result out of my pretty graphs
model.fit(X,y, epochs=1, verbose=1, batch_size=bs)

# history = model.fit(X,y,batch_size=bs,epochs=epochs, validation_split=0.1, verbose =1)
history = model.fit(X,y, batch_size=bs, epochs=epochs, validation_data=(X_val, y_val), verbose=0)

loss     = int(model.evaluate(x=X,     y=y,     verbose=0, batch_size=bs))
loss_val = int(model.evaluate(x=X_val, y=y_val, verbose=0, batch_size=bs))


# In[ ]:


loss


# In[ ]:


fname = f"model_feature_redux_loss_{loss_val}.h5"
model.save(fname, include_optimizer=False)


# In[ ]:


timedone = datetime.datetime.now()
runtime = timedone - timestart
print(runtime)


# # stop training

# In[ ]:


plt.plot(history.history['loss'])
plt.title('model loss (mae)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# In[ ]:


# evaluate model on training and validation sets
print("Loss by Mean Absolute Error")

print("train:      ", model.evaluate(x=X,     y=y,     verbose=0, batch_size=bs))
print("validation: ", model.evaluate(x=X_val, y=y_val, verbose=0, batch_size=bs))


# ## train some more, without validation data

# In[ ]:


# reload train from df
train = df.iloc[:len_train]
X_labels = train.drop('Weekly_Sales',axis=1).columns.astype(str)
X = train.drop('Weekly_Sales',axis=1).values
y = train['Weekly_Sales'].values

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
# X=X_train
# y=y_train
print(X.shape)
# print(X_val.shape)
# m_val,n_val = X_val.shape
m,n = X.shape


# In[ ]:


# these hyper parameters can be tweaked after compiling the model
# this is useful for retraining an existing model under different params
alpha = .001    # learning rate
bs = min(2**14, m//2)  # batch size: maxed at half size testset
epochs = 300  # number of epochs per training round
model.optimizer.lr = alpha


# In[ ]:


print("Learning Rate: %f, BatchSize: %i "% (alpha, bs))
# history = model.fit(X,y,batch_size=bs,epochs=epochs, validation_split=0.1, verbose =1)
history = model.fit(X,y, batch_size=bs, epochs=epochs, verbose=0)
loss     = int(model.evaluate(x=X,     y=y,     verbose=0))


# In[ ]:


fname = f"model_feature_redux_loss_{loss}.h5"
model.save(fname, include_optimizer=False)
get_ipython().system(' ls -l *h5')


# In[ ]:


# evaluate model on training and validation sets
print("Loss by Mean Absolute Error")

print("train:      ", loss)


# In[ ]:


timedone = datetime.datetime.now()
runtime = timedone - timestart
print(runtime)


# # stop 2nd training

# In[ ]:


plt.plot(history.history['loss'])
plt.title('model loss (mae)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# ## predict for submission

# In[ ]:


X_test = test.values
y_pred = model.predict(X_test)  # ,batch_size=bs
testfile = pd.read_csv('../input/course-material-walmart-challenge/test.csv')
submission = pd.DataFrame({'id':testfile['Store'].map(str) + '_' + testfile['Dept'].map(str) + '_' + testfile['Date'].map(str),
                          'Weekly_Sales':y_pred.flatten()})
fname = f"submission_features_redux2_L{loss}.csv"
submission.to_csv(fname, index=False)


# In[ ]:


get_ipython().system(' ls -l')
get_ipython().system(' wc -l submission*.csv')


# In[ ]:


fig,ax = plt.subplots(figsize=(14,5))
ax.scatter(range(len(y_pred)), y_pred, alpha=.05, c='m')
plt.show()


# In[ ]:


inputlayer=True
for layer in model.layers:
    if type(layer) == keras.layers.core.Dense:
        weights = layer.get_weights()
        fig,ax = plt.subplots( figsize=(5,5))
        layer_img = np.vstack((weights[1], weights[0]))
        plt.imshow(layer_img, cmap='PiYG')
        
        if inputlayer:
            labels = X_labels.insert(0, 'Bias')
            ax.set_yticklabels(labels)
            ax.set_yticks(range(len(labels)))
            inputlayer=False
        else:
            labels = np.array(range( 1, len(weights[0])+1 ))
            labels = np.hstack((['Bias'], labels))
            ax.set_yticklabels(labels)
            ax.set_yticks(range(len(labels)))
        plt.colorbar()
        plt.title(layer.name)
        plt.show()


# In[ ]:


for layer in model.layers:
    weights = layer.get_weights()
    print (weights)


# In[ ]:


predicted = testfile.copy()
predicted['Weekly_Prediction'] = y_pred
predicted = pd.concat([predicted,pristine],axis=0) # Join predicted and pristine
# predicted = pd.concat([predicted,df],axis=0) # Join predicted and pristine
predicted['Date'] = predicted['Date'].astype('datetime64[ns]')
predicted.set_index(['Date'], inplace=True)


# In[ ]:


# plots sales and predicted sales in time for specific Store/Dept
def plot_store_dept_pred(plotme):
    wpm = int(plotme.Weekly_Prediction.mean())
    wsm = int(plotme.Weekly_Sales.mean())
    fig, ax = plt.subplots(figsize=(13,4))
    ax.set_ylabel("Sales")
    ax.bar(x=plotme.index, height=plotme["Weekly_Sales"],width=7, color='g', label="known sales")
    ax.axhline(y=wsm, c='g', label=f"known sales mean {wsm}")
    ax.bar(x=plotme.index, height=plotme["Weekly_Prediction"],width=5, 
           label="predicted sales", color='g', alpha=.5)
    ax.axhline(y=wpm, c='c', label=f"predicted sales mean {wpm}")
    ax.plot_date(x=plotme.index, y=(plotme["IsHoliday"] * wsm), fmt='*m')
    ax.legend()
    plt.title(f"Store: {store}\nDept: {dept}")

    plt.show()


# In[ ]:


store=9
dept=30


# In[ ]:


# store += 1
dept  += 1

where = (predicted['Store'] == store) & (predicted['Dept'] == dept)
select = ['Date', 'Weekly_Prediction', 'Weekly_Sales', 'IsHoliday']
plotme = predicted.loc[where, select]
plot_store_dept_pred(plotme)
# print(plotme.Weekly_Prediction.mean(),plotme.Weekly_Sales.mean())
# print(plotme.Weekly_Prediction.count(),plotme.Weekly_Sales.count())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




