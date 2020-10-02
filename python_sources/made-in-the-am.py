#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#basic imports in this section
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import gc
import os
print(os.listdir("../input"))
import time

# Any results you write to the current directory are saved as output.


# In[ ]:


#ignoring any future warnings in the code
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# **Loading the Two Sigma environment.**

# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()


# Two datasets, namely, market dataset and news dataset will be loaded in market and news dataframes respectively

# In[ ]:


#get_training_data is an inbiult kaggle method that returns datasets.
(market, news) = env.get_training_data()


# **Simple garbage collection**

# In[ ]:


gc.collect()


# > **Data Description**

# In[ ]:


#number of rows in the market dataset
market.shape[0]


# In[ ]:


#number of features in the market dataset
market.shape[1]


# In[ ]:


#datatype of each feature in the market dataset 
market.dtypes


# In[ ]:


#number of rows in the news dataset
news.shape[0]


# In[ ]:


#number of features in the news dataset
news.shape[1]


# In[ ]:


#datatype of each feature in the market dataset 
news.dtypes


# **Preprocessing starts from here!**

# In[ ]:


#this method contains code for basic preprocessing of both market and news dataset
'''
this method
1. converts time feature from market dataset whose datatype is datetime into date datatype
2. converts time feature from news dataset whose datatype is datetime into float datatype considering only the hours of the entire datetime datatype
3. converts sourceTimeStamp feature from news dataset whose datatype is datetime into float datatype considering only the hours of the entire datetime datatype
4. converts firstCreated feature from news dataset whose datatype is datetime into date datatype
5. converts universe and volume feartures from float datatype to integer datatype
6. sorts the time columns in both the datasets
7. places 0 in place of null values, which was proven damaging, hence, dropped the idea
8. the eval function used will convert string column into a dictionary containing all the assetCodes
    news['assetCodesLen'] = news['assetCodes'].map(lambda x: len(eval(x))) -> this line takes the length of the column of dictionary that was just created from the cloumn of strings using eval function and figures out how many assetCodes are there in news dataset.
    news['assetCodes'] = news['assetCodes'].map(lambda x: list(eval(x))[0]) -> this line converts dictionary into a list. It takes the only first value from the list and attaches it to the assetCodes column.
9. basic rounding of float datatype
10. returns both, the market and news dataset.
'''
def preprocess_data(market, news):
    market.time = market.time.dt.date
    news.time = news.time.dt.hour
    news.sourceTimestamp = news.sourceTimestamp.dt.hour
    news.firstCreated = news.firstCreated.dt.date
    news['assetCodesLen'] = news['assetCodes'].map(lambda x: len(eval(x)))
    news['assetCodes'] = news['assetCodes'].map(lambda x: list(eval(x))[0])
    market['volume'] = market.volume.astype(int)
    market['universe'] = market.universe.astype(int)
    #this is just an experiment
    market = market.sort_values('time')
    news = news.sort_values('time')
    #if accuracy decreases then change null values to raw values from that same row
    #just uncomment the next four lines
    '''columns = ['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
    columns_raw = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10']
    for i in range(len(column_raw)):
        market[column_market[i]] = market[column_market[i]].fillna(market[column_raw[i]])'''
    
    
    market.round({'close': 2, 'open' : 2, 'returnsClosePrevRaw1' : 4, 'returnsOpenPrevRaw1' : 4, 'returnsClosePrevRaw10' : 4, 
                  'returnsOpenPrevRaw10' : 4, 'returnsOpenNextMktres10' : 4})
    news.round({'relevance' : 3, 'sentimentNegative' : 3, 'sentimentNeutral' : 3, 'sentimentPositive' : 3})
    market_num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                    'returnsOpenPrevMktres10']
    news_num_cols = ['urgency', 'takeSequence', 'bodySize', 'companyCount', 'sentenceCount', 'wordCount', 
                     'firstMentionSentence', 'sentimentClass', 'sentimentNegative', 'sentimentNeutral', 'sentimentPositive', 
                     'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H', 'noveltyCount3D', 'noveltyCount5D', 
                     'noveltyCount7D', 'volumeCounts12H', 'volumeCounts24H', 'volumeCounts3D', 
                     'volumeCounts5D', 'volumeCounts7D']
    #this helps reducing the accuracy so we will replace nan values with mean in next phase and comment the next two lines
    #market[market_num_cols] = market[market_num_cols].fillna(0)
    #news[news_num_cols] = news[news_num_cols].fillna(0)
    return market, news;
[market, news] = preprocess_data(market, news)


# changing nan values by mean alues of the feature

# In[ ]:


#this method changes all the nan values in every column by the mean of that respective column.
#many classifiers cannot handle nan values and hence we have to compromise with the data(either drop row, which is loss of some important data or drop a column, which is also loss of important data, but less damaging than dropping a row)
#there were only 6 columns remaining after dropping the columns containing nan values.
#hence we decided to write this method.
#the plus point of y\this method is that we can now apply any classifier to the dataset and predict results.
#the most logical way is to replace nan values by the mean of the column.
#this way the mean of the column will not be affected and the further process won't get hurt.
def handling_nan_values(df):
    for i in df.columns:
        print(i, df[i].dtype)
        
        if df[i].dtype == "object":
            df[i] = df[i].fillna("other")
        elif (df[i].dtype == "int64" or df[i].dtype == "float64" or df[i].dtype == "int16" or df[i].dtype == "int8" or df[i].dtype == "int32" or df[i].dtype == "float32"):
            df[i] = df[i].fillna(df[i].mean())
            print(i, df[i].mean())
        else:
            pass
    return df


# There's no need perform one hot encoding on string columns because tree-like classifiers can easily handle those.

# 
# **Next is the function to remove outliers.**

# In[ ]:


#adjusting outliers
#rather than removing outliers, we decided to adjust them near to the farthest point after removing that outlier, or in more technical words, clipping the outliers.
#outliers are the noise in the dataset.
#they are corrupted data and affects prediction powers of the model severely in a bad way.
#hence it only makes sense to control these outliers in the dataset by decreasing or increasing its values to a specific value.
#here, in this method we are clipping all rows that contains outliers to a specified value(0.05 and  0.95).
#an outlier is a data point which has high variance and its value is far away from the mean/ average value.
def remove_outliers(data_frame, column_list, low=0.05, high=0.95):
    temp_frame = data_frame
    for column in column_list:
        this_column = data_frame[column]
        quant_df = this_column.quantile([low,high])
        low_limit = quant_df[low]
        high_limit = quant_df[high]
        temp_frame[column] = data_frame[column].clip(lower=low_limit, upper=high_limit)
    return temp_frame


# **outliers for news data**

# In[ ]:


#the outliers in the columns of news dataset, mentioned in columns_outliers are handled using the remove_outliers method
#only the outliers in numerical columns are handled as we are planning to remove categorical columns
columns_outlier = ['takeSequence', 'bodySize', 'sentenceCount', 'wordCount', 'sentimentWordCount', 'firstMentionSentence',
                   'noveltyCount12H','noveltyCount24H', 'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 
                   'volumeCounts12H', 'volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D']
news = remove_outliers(news, columns_outlier)


# **outliers for market data**

# the difference between open and close values can not be too much. The threshold in following code can be changed as per convenience. The lower threshold is 0.5 and upper threshold is 1.5. Every row not in the range[0.5, 1.5] will be dropped. Because those are noise data points.

# In[ ]:


#the outliers in market dataset are removed as per the logic mentioned above
market['close_open_ratio'] = np.abs(market['close']/market['open'])
threshold = 0.5
market = market.loc[market['close_open_ratio'] < 1.5]
market = market.loc[market['close_open_ratio'] > 0.5]
market = market.drop(columns=['close_open_ratio'])


# If returns exceed by 50% or fall by 50% of their value then the data is noisy data

# In[ ]:


columns =['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10', 'returnsOpenPrevMktres10', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsOpenNextMktres10']
for column in columns:
    market = market.loc[market[column]>=-2]
    market = market.loc[market[column]<=2]


# removing data having unknown asset name

# In[ ]:


#market = market[~market['assetCode'].isin(['PGN.N','EBRYY.OB'])]
market = market[~market['assetName'].isin(['Unknown'])]


# In[ ]:


market


# In[ ]:


news


# **Visualizing both datasets**

# In[ ]:


import matplotlib.pyplot as plt
plot = (market.nunique()).sort_values(ascending=False)
plot.plot(kind="bar", figsize = (20,10), fontsize = 15)
plt.xlabel("Columns", fontsize = 15)
plt.ylabel("Unique Values", fontsize = 15)
print('Done!')


# In[ ]:


market.nunique()


# In[ ]:


import matplotlib.pyplot as plt
plot = (news.nunique()).sort_values(ascending=False)
plot.plot(kind="bar", figsize = (20,10), fontsize = 15)
plt.xlabel("Columns", fontsize = 15)
plt.ylabel("Unique Values", fontsize = 15)
print('Done!')


# In[ ]:


news.nunique()


# **merging the dataset in next section**

# In[ ]:


#we will merge dataset based on assetCodes and not assetName because there are unknown values in the assetName column and also we handled the assetCodes column very nicely in data preprocessing
#we will make a left join on time and assetCode of market data and a righ join on firstCreated and assetCodes. Because in preprocessing part we converted time and firstCreated into date format and assetCode and assetCodes is explained above.
def dataset_merge(market, news):
    grouping_cols = ['firstCreated', 'assetCodes']
    news = news.groupby(grouping_cols, as_index=False).mean() 
    market = pd.merge(market, news, how='left', left_on= ['time', 'assetCode'], right_on= ['firstCreated', 'assetCodes'])
    return market
market = dataset_merge(market, news)


# **Correlation among data** : Pearson Correlation

# In[ ]:


pearson_dataframe = market.corr(method= 'pearson') 


# In[ ]:


pearson_dataframe


# **Inferences based on the correlation matrix:**
# The pairs having >0.7 correlation are asfollows:
# 1. (open, close)
# 2. (returnsClosePrevMktres1, returnsClosePrevRaw1)
# 3. (returnsOpenPrevRaw1, returnsOpenPrevMktres1)
# 4. (returnsOpenPrevRaw10, returnsClosePrevRaw10)
# 5. (returnsOpenPrevMktres10, returnsOpenPrevRaw10)
# 6. (returnsOpenPrevMktres10, returnsClosePrevMktres10)
# 7. (time_y, source_time_stamp)
# 8. (body_size, sentenceCount)
# 9. (body_size, wordCount)
# 10. (sentenceCount, wordCount)
# 11. (sentimentClass, sentimentPositive) and 
# 12. every columns of valueCounts

# **Covariance among data**

# In[ ]:


cov_dataframe = market.cov()


# In[ ]:


cov_dataframe


# In[ ]:


#plotting the correlation
import matplotlib.pyplot as plt
plt.matshow(pearson_dataframe)
plt.title("pearson correlation chart\n")


# In[ ]:


# defining only those columns that are important to the dataset
# we deduced this result based on prior preprocessing
reduced_cols = [c for c in market if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 'firstCreated', 'headline', 'headlineTag', 'provider', 'bodySize', 'wordCount', 'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'time_y', 'universe','sourceTimestamp']]


# In[ ]:


#we will make target variable in this section
target = market.returnsOpenNextMktres10 >= 0
target = target.values
op = market.returnsOpenNextMktres10.values


# We already filled nan values by the mean.... so no need of the next block anymore

# market_num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10', 'urgency', 'takeSequence', 'companyCount', 'marketCommentary', 'sentenceCount', 'firstMentionSentence', 'relevance', 'sentimentClass', 'sentimentNegative', 'sentimentNeutral', 'sentimentPositive', 'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H', 'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H', 'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D', 'volumeCounts7D']
# market[market_num_cols] = market[market_num_cols].fillna(0)

# In[ ]:


reduced_cols


# In[ ]:


market = market[reduced_cols]


# In[ ]:


market


# In[ ]:


#after making a left join on the dataset, there will be some nan values in the combined dataset.
#hence we will call handling_nan_values method again
market = handling_nan_values(market)


# In[ ]:


# we will scale the merged dataset using standard scaler.
# there are other approaches using min-max scaler, etc. 
# but we chose standard scaler because it handles categorical data which min-max fails to handle.
# the range of standard scaler is typically [-2, 4] while the range of min-max scaler is [0, 1] which will not give us important data to classify a stock as negative
from sklearn.preprocessing import StandardScaler

data_scaler = StandardScaler()
market[reduced_cols] = data_scaler.fit_transform(market[reduced_cols])


# The accuracy is decreasing when PCA is applied with less n_components, hence we will keep large values in n_components such as 20, 25, etc because the dataset is such that the model will require more data than a threshold to predict correctly

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=27)
market = pca.fit_transform(market)  
#market = pca.transform(market)


# In[ ]:


market.shape[1]


# In[ ]:


op


# In[ ]:


#spliting the dataset using test_train_split
#our next task is to split using time series as the dataset contains time related data
from sklearn import *
X_train, X_test, target_train, target_test, op_train, op_test = model_selection.train_test_split(market, target, op, test_size=0.2, random_state=5)


# In[ ]:


X_train


# #fitting the dataset using XGBoost classifier
# from xgboost import XGBClassifier
# t = time.time()
# xgb = XGBClassifier(n_jobs=4, n_estimators=200, max_depth=8, eta=0.1)
# xgb.fit(X_train, target_train)
# print(f'Done, time = {time.time() - t}')

# #printing the confusion matrix and accuracy using built in library
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# t = time.time()
# target_test, prediction = target_test, xgb.predict(X_test)
# print(classification_report(target_test, prediction))
# print("Detailed confusion matrix:")
# print(confusion_matrix(target_test, prediction))
# print(prediction)
# accuracy_score(target_test, prediction)
# print(f'Done, time = {time.time() - t}')

# **CatBoost next**

# In[ ]:


#fitting the dataset using CatBoost classifier
from catboost import CatBoostClassifier
t = time.time()
cat = CatBoostClassifier(thread_count=4, n_estimators=200, max_depth=8, eta=0.1, loss_function='Logloss' , verbose=10)
cat.fit(X_train, target_train)
print(f'Done, time = {time.time() - t}')


# In[ ]:


#printing the confusion matrix and accuracy using built in library
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
t = time.time()
target_test, prediction = target_test, cat.predict(X_test)
print(classification_report(target_test, prediction))
print("Detailed confusion matrix:")
print(confusion_matrix(target_test, prediction))
print(prediction)
print(accuracy_score(target_test, prediction))
print(f'Done, time = {time.time() - t}')


# In[ ]:


from sklearn.metrics import mean_squared_error
mean_squared_error(target_test, prediction)


# **ROC curve**

# In[ ]:


ycat_pred_proba = cat.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(target_test,  ycat_pred_proba)
auc = metrics.roc_auc_score(target_test, ycat_pred_proba)
plt.figure()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.plot(fpr, tpr, color='blue', lw=2, label="roc curve" % auc)
plt.legend(loc="lower right")
plt.show()


# In[ ]:


confidence_test = cat.predict_proba(X_test)[:,1]*2 -1
print(accuracy_score(confidence_test>0,target_test))
plt.hist(confidence_test, bins='auto')
plt.title("Cat predicted confidence")
plt.show()


# In[ ]:


# distribution of confidence that will be used as submission
plt.hist(confidence_test, bins='auto', label='Prediciton')
plt.hist(op_test, bins='auto',alpha=0.8, label='True data')
plt.title("predicted confidence")
plt.legend(loc='best')
plt.xlim(-1,1)
plt.show()

