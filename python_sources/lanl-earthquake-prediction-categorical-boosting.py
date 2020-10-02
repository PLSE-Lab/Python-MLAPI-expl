#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler # for data scaling
from sklearn.model_selection import GridSearchCV # hyperparameter optimization
from catboost import CatBoostRegressor, Pool #catagorical gradient boosting
from sklearn.svm import NuSVR, SVR

import os
IS_LOCAL = False
if(IS_LOCAL):
    PATH="../input/LANL/"
else:
    PATH="../input/"
os.listdir(PATH)
# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv(PATH+'train.csv', nrows = 6000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})


# In[ ]:


train.head(10)


# In[ ]:


#visualize the dataset
train_ad_sample_df = train['acoustic_data'].values[::100]
train_ttf_sample_df = train['time_to_failure'].values[::100]

def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title = "acoustic data + ttf"):
    fig, ax1 = plt.subplots(figsize = (12, 8))
    plt.title(title)
    plt.plot(train_ad_sample_df, color = 'r')
    ax1.set_ylabel('acoustic data', color='r')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(train_ttf_sample_df, color ='b')
    ax2.set_ylabel('time to failure', color = 'b')
    plt.legend(['time to faliure'], loc = (0.01, 0.9))
    plt.grid(True)
    
plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)
del train_ad_sample_df
del train_ttf_sample_df


# In[ ]:


train = pd.read_csv(PATH+'train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
rows = 150000
segments = int(np.floor(train.shape[0] / rows))
print("Number of segments: ", segments)


# In[ ]:


features = ['mean','max','variance','min', 'stdev', 'quantile(0.01)', 'quantile(0.05)', 'quantile(0.95)', 'quantile(0.99)']

X = pd.DataFrame(index=range(segments), dtype=np.float64, columns=features)
Y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

for segment in tqdm(range(segments)):
    seg = train.iloc[segment*rows:segment*rows+rows]
    
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    
    Y.loc[segment, 'time_to_failure'] = y
    X.loc[segment, 'mean'] = x.mean()
    X.loc[segment, 'stdev'] = x.std()
    X.loc[segment, 'variance'] = np.var(x)
    X.loc[segment, 'max'] = x.max()
    X.loc[segment, 'min'] = x.min()
#     X.loc[segment, 'kur'] = x.kurtosis()
#     X.loc[segment, 'skew'] = x.skew()
    X.loc[segment, 'quantile(0.01)'] = np.quantile(x, 0.01)
    X.loc[segment, 'quantile(0.05)'] = np.quantile(x, 0.05)
    X.loc[segment, 'quantile(0.95)'] = np.quantile(x, 0.95)
    X.loc[segment, 'quantile(0.99)'] = np.quantile(x, 0.99)
    
    #FFT transform values -
    """
    from: 'https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction' kernel
     FFT is useful for sequence data feature extraction
     other than FFT there is Wavelet transform, which can be used to extract low level features, 
     Wavelet transform is though a lil bit complex in terms of computation
    """
    
    z = np.fft.fft(x)
    realFFT = np.real(z)
    imagFFT = np.imag(z)
    X.loc[segment, 'A0'] = abs(z[0])
    X.loc[segment, 'Rmean'] = realFFT.mean()
    X.loc[segment, 'Rstd'] = realFFT.std()
    X.loc[segment, 'Rmax'] = realFFT.max()
    X.loc[segment, 'Rmin'] = realFFT.min()
    X.loc[segment, 'Imean'] = imagFFT.mean()
    X.loc[segment, 'Istd'] = imagFFT.std()
    X.loc[segment, 'Imax'] = imagFFT.max()
    X.loc[segment, 'Imin'] = imagFFT.min()
    
X.describe().T


# In[ ]:


X.head()


# In[ ]:


# Scaling the data
scaler = StandardScaler()
scaler.fit(X)
scaled_X = pd.DataFrame(scaler.transform(X), columns = X.columns)


# In[ ]:


scaled_X.head(5)


# In[ ]:


# process the test data
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns = X.columns, dtype = np.float64, index = submission.index)
X_test.describe()


# In[ ]:


submission.shape, X_test.index.shape


# We'll used index of submission file to get the segment id and do all the operations.

# In[ ]:


# process the test data
for i, seg_id in enumerate(tqdm(X_test.index)):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    
    x = pd.Series(seg['acoustic_data'].values)
    z = np.fft.fft(x)
    realFFT = np.real(z)
    imagFFT = np.imag(z)
    
    X_test.loc[seg_id, 'mean'] = x.mean()
    X_test.loc[seg_id, 'stdev'] = x.std()
    X_test.loc[seg_id, 'variance'] = np.var(x)
    X_test.loc[seg_id, 'max'] = x.max()
    X_test.loc[seg_id, 'min'] = x.min()
    X_test.loc[seg_id, 'quantile(0.01)'] = np.quantile(x, 0.01)
    X_test.loc[seg_id, 'quantile(0.05)'] = np.quantile(x, 0.05)
    X_test.loc[seg_id, 'quantile(0.95)'] = np.quantile(x, 0.95)
    X_test.loc[seg_id, 'quantile(0.99)'] = np.quantile(x, 0.99)
    X_test.loc[seg_id, 'A0'] = abs(z[0])
    X_test.loc[seg_id, 'Rmean'] = realFFT.mean()
    X_test.loc[seg_id, 'Rstd'] = realFFT.std()
    X_test.loc[seg_id, 'Rmax'] = realFFT.max()
    X_test.loc[seg_id, 'Rmin'] = realFFT.min()
    X_test.loc[seg_id, 'Imean'] = imagFFT.mean()
    X_test.loc[seg_id, 'Istd'] = imagFFT.std()
    X_test.loc[seg_id, 'Imax'] = imagFFT.max()
    X_test.loc[seg_id, 'Imin'] = imagFFT.min()


# In[ ]:


# build a model
X_test.shape


# In[ ]:


# Scaling the test data
scaled_test_x = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
scaled_test_x.shape


# In[ ]:


scaled_test_x.tail()


#     this is a supervised learning problem but we need more features
#     feature engineeering
#     we'll use statistical feature learning here in which statistical features such as:
#     mean, variance, standard daviation are used to build new features
#         def gen_features(X):
#             strain = []
#             strain.append(X.mean())
#             strain.append(X.std())
#             strain.append(X.min())
#             strain.append(X.max())
#             strain.append(X.kurtosis()) #tailed data feature
#             strain.append(X.skew()) #skewness
#             strain.append(np.quantile(X,0.01))
#             strain.append(np.quantile(X,0.05))
#             strain.append(np.quantile(X,0.95))
#             strain.append(np.quantile(X,0.99)) #sample distributions of same probabilities
#             return pd.Series(strain)
#             
#     train = pd.read_csv(PATH+'train.csv', iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
#     X_train = pd.DataFrame()
#     y_train = pd.Series()
#     for df in train:
#         ch = gen_features(df['acoustic_data'])
#         X_train = X_train.append(ch, ignore_index=True)
#         y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))

# ### implement CatBoost Model
#     using gradient boosting based on catagorical features
#     catagorical features can't be related to each other, essentially
#     there are other boosting models available such as XGboost
#     """
#     gradient Boosting:
#     Step - 1 - computing gradient of loss fucntion we want to optimize for each input object
#     step - 2 - learning the decision tree which predicts gradients of the loss function
# 
#     ELI5 Time
#     step - 1 - first model data with simple models and analyze data for errors
#     step - 2 - errors signify data points that are difficult to fit by a simple model
#     step - 3 - in later models, we particularly focus on those hard to fit data to get them right
#     step - 4 - lastly, we combine all the predictors by giving some weighs to each predictor.
#     """
# 
# # MODEL - 1 CatBoost

# In[ ]:


train_pool = Pool(X, Y)
m = CatBoostRegressor(iterations = 10000, loss_function = 'MAE', boosting_type = 'Ordered')
m.fit(X, Y, silent = True)
m.best_score_


# In[ ]:


#predictions
predictions = np.zeros(len(scaled_test_x))
predictions += m.predict(scaled_test_x)


# In[ ]:


submission['time_to_failure'] = predictions


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv')


# ## MODEL 2 - SVM

# In[ ]:


parameters = [{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
               'C': [0.1, 0.2, 0.5, 1, 1.5, 2]}]
reg1 = GridSearchCV(SVR(kernel='rbf', tol=0.01), parameters, cv=5, scoring='neg_mean_absolute_error')
reg1.fit(scaled_X, Y.values.flatten())


# In[ ]:


predictions = reg1.predict(scaled_test_x)
print(predictions.shape)


# In[ ]:


submission['time_to_failure'] = predictions
submission.head()


# In[ ]:


submission.to_csv('submissionSVM.csv')


# In[ ]:





# In[ ]:




