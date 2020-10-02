#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# In this kernel, I will be mainly looking at pre- and post-processing methods. I believe that the modeling aspects of this challenge have been exhaustively documented by other users, but any model (especially a trading model) requires adequate data preperation and postprocessing to perform well. I will first go over some data selection issues, including a brief discussion on sample selection and train/validation splitting. I will then train a model and introduce some analytics to better examine how and why the model performs in a certain way. Finally, I will discuss some alternate validation methods that may be used to identify model false positives.
# 
# Much of the code in the first half of this kernel was adapted from other users' contributions on Kaggle. Additionally, as a disclaimer, my functions were not exhaustively tested and a few of them may be quite buggy.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
(marketdf, newsdf) = env.get_training_data()


# ## Data Preprocessing

# In[ ]:


marketdf.describe()


# In[ ]:


#Truncate data, reduce to just investing universe
uppernext = marketdf['returnsOpenNextMktres10'].quantile(q=0.9999)
lowernext = marketdf['returnsOpenNextMktres10'].quantile(q=0.0001)
upperprev = marketdf['returnsOpenPrevMktres10'].quantile(q=0.9999)
lowerprev = marketdf['returnsOpenPrevMktres10'].quantile(q=0.0001)
marketdf = marketdf[marketdf['returnsOpenNextMktres10']<uppernext]
marketdf = marketdf[marketdf['returnsOpenNextMktres10']>lowernext]
marketdf = marketdf[marketdf['returnsOpenPrevMktres10']<upperprev]
marketdf = marketdf[marketdf['returnsOpenPrevMktres10']>lowerprev]
#marketdf = marketdf[marketdf['universe']==1] ##Difference seems non-existant.
marketdf.describe()


# In this notebook, I will avoid doing much exploration of the features as this has already been poured over by other competition participants. Instead, I'll look at justifying a few filters. Firstly, I elect to remove the top/bottom 0.01% of samples based on the `returnsOpenNextMktres10` and `returnsOpenPrevMktres10` features, just to filter significant outliers that are likely data entry errors. This is justified as the `returns` features are all in percentage terms, and removing outliers along this axis reduces the standard deviation of most of the returns series to values in the <10% range. See the descriptive output above.
# 
# Next, a relevant question is whether we should limit the data universe to just the investable universe or not. In theory, it would be best to only use the same sample of securities in training data as will be used for actual prediction. However, over 40% of the sample would be excluded in this case, which could affect the accuracy of our model.
# 
# It is reasonable to expect most of the return features to have roughly similar distributions both in and out of the investable universe, as most of these features are related to returns and will be roughly symmetrical with mean = 0. The crucial piece of difference, though, is that volume for the investable universe is nearly an order of magnitutde larger on average than the volume outside of the universe. This is because the assets in the investible universe are only larger-cap US equities. Differences in volume may have a significant impact on the sensitivity of the securities to news articles, and could make the model less robust. As a result, I believe it is prudent to use only the data in the investable universe, even though this may make the model less accurate in training and validation. See the output below:

# In[ ]:


# Justify universe == 1
from scipy.stats import t, f

np.sum(marketdf['universe']) # 2422390 samples in the investable universe
np.mean(marketdf['universe']) # 59.5% of the sample is in the investable universe

n0 = np.sum(marketdf['universe'])
n1 = len(marketdf['universe']) - n0

means = marketdf.groupby(['universe'], sort=False).aggregate(np.mean)
stdev = marketdf.groupby(['universe'], sort=False).aggregate(np.std)
var = marketdf.groupby(['universe'], sort=False).aggregate(np.var)

#F-test for equal variance
f.cdf(stdev.values[0]/stdev.values[1],n0,n1) #strongly reject null of equal variance

#t tests for equal mean
t.cdf(means.values[0]-means.values[1],n0+n1,loc=0,scale=np.sqrt((n0*var.values[0] + n1*var.values[1])/(n0+n1))) #does not reject null of equal means


# As shown, we actually fail to reject the null of different means across all features, but there is a strong rejection of equal variance. In my opinion, I think this creates a sufficient case to consider leaving the securities outside our investing universe out of the model, but overall there probably will not be a significant difference with or without that part of the sample.
# 
# Moving on the data merging, we can now combine the the news dataframe with the market dataframe. I also create a few new features here, including the intraday return `bartrend`, the intraday average, and price-volume (total number of dollars exchanged, note it is unclear to me if volume is already stated in this way or if the feature is just based on securities exchanged). 
# 
# Additionally, I choose to exclude data from before 2012 for qualitative reasons. It seems likely that the 2008-9 crisis and subsequent recovery may cause the sample to behave qualitatively different. 
# 
# News features are also added, including `position` which gives a number representing how early in the article the subject company is mentioned, and `coverage`, which measures the proportion of sentiment words relative to the size of the article.
# 
# I previously experimented with using other conditional features, e.g. importance conditional on the news being positive, but found that decision tree models were generally unimpacted.
# 
# Finally, several news features are excluded as previous tests have found very low feature importance. This is a pervasive problem in this competition, and many participants elected not to use the news data at all. In keeping with the spirit of the competition, though, I choose to retain some of the slightly more useful features.

# In[ ]:


print('preparing data...')
def prepare_data(marketdf, newsdf):
    # a bit of feature engineering
    marketdf['time'] = marketdf.time.dt.strftime("%Y%m%d").astype(int)
    marketdf['bartrend'] = marketdf['close'] / marketdf['open']
    marketdf['average'] = (marketdf['close'] + marketdf['open'])/2
    marketdf['pricevolume'] = marketdf['volume'] * marketdf['close']
    
    newsdf['time'] = newsdf.time.dt.strftime("%Y%m%d").astype(int)
    newsdf['assetCode'] = newsdf['assetCodes'].map(lambda x: list(eval(x))[0])
    newsdf['position'] = newsdf['firstMentionSentence'] / newsdf['sentenceCount']
    newsdf['coverage'] = newsdf['sentimentWordCount'] / newsdf['wordCount']

    # filter pre-2012 data
    marketdf = marketdf.loc[marketdf['time'] > 20120000]
    
    # get rid of extra junk from news data
    droplist = ['sourceTimestamp','firstCreated','sourceId','headline','takeSequence','provider','firstMentionSentence',
                'sentenceCount','bodySize','headlineTag','marketCommentary','subjects','audiences','sentimentClass',
                'assetName', 'assetCodes','urgency','wordCount','sentimentWordCount']
    newsdf.drop(droplist, axis=1, inplace=True)
    marketdf.drop(['assetName', 'volume'], axis=1, inplace=True)
    
    # combine multiple news reports for same assets on same day
    newsgp = newsdf.groupby(['time','assetCode'], sort=False).aggregate(np.mean).reset_index()
    
    # join news reports to market data, note many assets will have many days without news data
    return pd.merge(marketdf, newsgp, how='left', on=['time', 'assetCode'], copy=False) #, right_on=['time', 'assetCodes'])

cdf = prepare_data(marketdf, newsdf)    
del marketdf, newsdf  # save the precious memory


# ## Model Prep & Tuning
# 
# For data splitting, I drop the assetcode and time features (to avoid potential model bias by picking a successful security over and over). Additionally, I use a time based split with an 11-day embargo period, as some of the lag features are 10-days in length, to avoid any data leaking.
# 
# Finally, I also store the original target variable amounts before classifying them, as these will be used later to compute train and validation scores.

# In[ ]:


print('building training set...')
targetcols = ['returnsOpenNextMktres10']
traincols = [col for col in cdf.columns if col not in ['time', 'assetCode', 'universe'] + targetcols]

##Experiment: Try cutting last x features
traincols = [col for col in cdf.columns if col not in ['time', 'assetCode', 'universe', 'relevance', 'noveltyCount24H', 'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'noveltyCount12H'] + targetcols]

dates = cdf['time'].unique()
train = range(len(dates))[:int(0.85*len(dates))]
val = range(len(dates))[int(0.85*len(dates)+11):] ## add 11 day gap to avoid data leakage

# we be classifyin - also store actual amounts
realreturn = cdf[targetcols[0]]
valreal = realreturn.loc[cdf['time'].isin(dates[val])].values
cdf[targetcols[0]] = (cdf[targetcols[0]] > 0).astype(int)
targetreal = realreturn.loc[cdf['time'].isin(dates[train])].values


# In[ ]:


# train data
Xt = cdf[traincols].fillna(0).loc[cdf['time'].isin(dates[train])].values
Yt = cdf[targetcols].fillna(0).loc[cdf['time'].isin(dates[train])].values

# validation data
Xv = cdf[traincols].fillna(0).loc[cdf['time'].isin(dates[val])].values
Yv = cdf[targetcols].fillna(0).loc[cdf['time'].isin(dates[val])].values

print(Xt.shape, Xv.shape)


# I use a slightly tuned LightGBM model for this competition. I experimented with using ensemble models as well, but failed to see any significant change despite a far longer train time. LightGBM checks the two boxes of being generally robust, and being very fast to train allowing us to conduct repeat experiments easily.
# 
# The parameter values have been tuned based on other user kernals and my own experiments.

# In[ ]:


import lightgbm as lgb
print ('Training lightgbm')


# In[ ]:


params = {"objective" : "binary",
          "metric" : "binary_error",
          "num_leaves" : 120, #previously 240, originally 60
          "max_depth": -1,
          "learning_rate" : 0.0005,
          "max_bin" : 200, #new
          "bagging_fraction" : 0.2,  # subsample
          "feature_fraction" : 0.8,  # colsample_bytree
          "bagging_freq" : 10,        # subsample_freq
          "bagging_seed" : 23,
          "verbosity" : -1 }


# In[ ]:


lgtrain, lgval = lgb.Dataset(Xt, Yt[:,0]), lgb.Dataset(Xv, Yv[:,0])
lgbmodel = lgb.train(params, lgtrain, 4000, valid_sets=[lgtrain, lgval], early_stopping_rounds=200, verbose_eval=100)


# In[ ]:


from matplotlib import pyplot as plt
ax = lgb.plot_importance(lgbmodel, max_num_features=20)
plt.show()

importance = np.argsort(lgbmodel.feature_importance())[::-1][:20]
j = 1
for i in importance:
    print(j, ": Feature #", i, traincols[i])
    j += 1


# Finally, we can look at the feature importance of the model. The lack of predictive power among the news sentiment features is a persistent problem in this challenge, and at this stage it may be appropriate to go back and collect different data or engineer different features (note: we should always do additional feature engineering based on feature importance, rather than validation performance, to prevent a look ahead bias). Obviously for this competition additional data collection is not a possibility, and I will continue using news sentiment features out of spirit of the competition. I must stress here that many of the top kernels in this competition mentioned that they only used price history data for their features. This is not egregious from a data science perspective, but using only price history does lack theoretical foundation, see Fama (1970) "Efficient Capital Markets: A Review of Theory and Empirical Work". Put simply, price history is easily collectible public data, and any advantage found using price history should theoretically be obtainable by many other traders as well, which is likely to eliminate any profitable opportunities. 

# # Experimentation with post-processing techinques
# 
# In this section, I'll introduce a few post-processing methods to scale our predictions. Bet sizing matters as much as the accuracy of the prediction, and we ideally desire some sizing method that maximizes our correct predictions and minimizes our incorrect predictions. 
# 
# Below, I'll examine the distributional properties of our predictions based on the LGB model trained above, as well as using a few different scaling methods including platt scaling, an inclusion tree, and ranked sizing, to show that bet sizing can considerably impact returns and score.

# In[ ]:


## Generate training data for examination:
lgbm_train = lgbmodel.predict(Xt)
lgbm_val = lgbmodel.predict(Xv)


# In[ ]:


from matplotlib import pyplot as plt

## Confusion matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(Yt, lgbm_train>0.5))

## Histogram of predictions vs actual directions
plt.hist(lgbm_train*(2*Yt[:,0]-1), range=(-1, 1), bins=20,
             histtype="step", lw=2)
plt.xlabel("Correct Predictions (Confidence Weighted)")
plt.ylabel("Count")
plt.title("Confidence-Weighted Prediction Accuracy")
plt.show()


## Correlation of predictions to actual returns
plt.hist(lgbm_train*targetreal, range=(-.1, .1), bins=20,
             histtype="step", lw=2)
plt.xlabel("Realized Returns")
plt.ylabel("Count")
plt.title("Classifier Returns")
plt.show()

## calibration curve
#  Shows the relative over/under prediction of a class based on output probability
#  Shallow calibration curve indicates that proportion of positives predicted is very sensitive to change in probability threshold (plot 1)
#  This is due high concentration of predicted probabilities around 0.5 (plot 2)
from sklearn.calibration import calibration_curve

plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

fop, mpv = calibration_curve(Yt, lgbm_train, n_bins=10, normalize=True)
ax1.plot([0, 1], [0, 1], linestyle='--')
ax1.plot(mpv, fop, marker='.')
ax1.set_ylabel("Fraction of positives")
ax2.hist(lgbm_train, range=(0, 1), bins=20,
             histtype="step", lw=2)
ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")

print("Calibration Curve & Probability Histogram:")
plt.show()


# As we can see from the above, the default probabilities generally are more accurate for more extreme probabilities, which should not be surprising. The return distribution (position size * returns) is slightly positive but generally resembles a normal distribution. This is not necessarily a good sign for our model, because it shows that the majority of predictions give essentially zero returns, and there are almost as many incorrect predictions resulting in high losses as there are high return correct predictions.
# 
# Both of these effects can be explained by examining the histogram of predicted values along with the calibration curve: almost all the probability predictions are very close to 0.5, and the calibration curve indicates that the model is somewhat poorly calibrated for class prediction.
# 
# Although having probabilities close to 0.5 may be irrelevant with some models, in this competition the confidence values are used for bet sizing, which requires some additional attention. Bet sizing is the idea that we wish to maximize the size of our correct predictions, especially when those predictions are of high returns, and minimize the size of the incorrect positions. Any trading system that cannot do this is bound to fail by attrition alone, even with a high degree of classifier accuracy.
# 
# To begin, I'll make a function that computes the competition evaluation score (roughly the Sharpe Ratio), which we can then use to assess various bet sizing techniques.

# In[ ]:


""" 
Model scoring function:
    inputs:
    pred - vector of predictions / bet sizes
    actual - real return of assets
    assets - list of asset/dates to include
    
    output:
    score based on competition evaluation metric

""" 
def score_model(pred,actual,assets):
    score = 0
    #ret = pred*actual
    returns = pd.DataFrame(data=np.array(pred*actual),columns=['returns'])
    returns['assetCode'] = assets
    x = pd.DataFrame(data=returns.groupby(['assetCode'], sort=False).aggregate(np.sum).values, columns=['x'])
    #xm = returns.groupby(['assetCode'], sort=False).aggregate(np.mean).values#.reset_index()
    #xsd = returns.groupby(['assetCode'], sort=False).aggregate(np.std).values#.reset_index()
    score = np.mean(x)/np.std(x)
    return score

#demonstration
#recall: train, val variables store index of dates for train/validation sets.
#targetreal/valreal are train, val real returns
trainList = cdf['assetCode'].loc[cdf['time'].isin(dates[train])]
valList = cdf['assetCode'].loc[cdf['time'].isin(dates[val])]
valList = valList.reset_index()['assetCode']
score_model(lgbm_train,targetreal,trainList) #test with the default probabilities.


# As shown above, the score with the model probabilities is very poor, further showing the need to create better sizing functions.

# ## Bet Sizing and Scaling
# 
# Below, I'll compose three different sizing functions:
#    1. Linear scaling: demean and stretch the confidence values so that most bets are sized near +-1 (the limit for this competition, leverage isn't allowed). This was very popular in this competition because its easy to implement and has seen some boost in performance.
#    2. Ranked scaling: ranks all probabilities from highest to lowest, then demeans and scales to +-1 (e.g. highest probability gets a value of 1, next highest gets a value close to 1, lowest probability gets a value of -1, etc.) 
#    3. Platt scaling: uses a scaled sigmoid function, the A parameter can be tuned on the training data.

# In[ ]:


"""
Bet sizing functions:
    - linear_scale: simply expands the predictions and truncates at +-1, popular in the competition
    - ranked_scale: ranks all predictions by probability, then scales with a linear stretch
    - platt_scale: uses a platt scaling algorithm with a linear stretch
"""

from scipy.stats import rankdata

def linear_scale(df):
    mean, std = np.mean(df), np.std(df)
    df = (df - mean)/ (std * 8)
    return np.clip(df,-1,1)

def ranked_scale(df):
    ranks = rankdata(df)
    ranks = ranks - np.mean(ranks)
    ranks = ranks/np.max(ranks)
    return np.clip(ranks,-1,1)

def platt_scale(df, A):
    df = (df-np.mean(df))
    probs = 1 / (1 + np.exp(A*df)) #sigmoid
    probs = (probs*2)-1
    return np.clip(probs,-1,1)


# The out of sample process for this model would be to determine up/down probabilities and then scale the probabilities in to bet sizes for a given day. To emulate this, the validation set will also be fit on a day by day basis. For linear and platt scaling this won't make a difference as the bet sizes are determined independently of other predictions, but ranked scaling specifically could be impacted.

# In[ ]:


# Fit validation measure for each method:
time = cdf['time'][cdf['time'].isin(dates[val])].values
date = np.unique(time)
preds_linear = np.zeros(len(valreal))
preds_ranked = np.zeros(len(valreal))
preds_platt = np.zeros(len(valreal))

for t in date:
    idx = (time == t)
    preds_linear[idx] = linear_scale(lgbm_val[idx])
    preds_ranked[idx] = ranked_scale(lgbm_val[idx])
    preds_platt[idx] = platt_scale(lgbm_val[idx],-900)


# In[ ]:


# Linear Scaling
plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

fop, mpv = calibration_curve(Yv, preds_linear, n_bins=10, normalize=True)
ax1.plot([0, 1], [0, 1], linestyle='--')
ax1.plot(mpv, fop, marker='.')
ax1.set_ylabel("Fraction of positives")
ax2.hist((preds_linear+1)/2, range=(0, 1), bins=20,
             histtype="step", lw=2)
ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")

print("Calibration Curve & Probability Histogram (Linear Scaling):")
plt.show()
print("Score: ", score_model(preds_linear,valreal,valList).values)

# Ranked Scaling
plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))
fop, mpv = calibration_curve(Yv, preds_ranked, n_bins=10, normalize=True)
ax1.plot([0, 1], [0, 1], linestyle='--')
ax1.plot(mpv, fop, marker='.')
ax1.set_ylabel("Fraction of positives")
ax2.hist((preds_ranked+1)/2, range=(0, 1), bins=20,
             histtype="step", lw=2)
ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")

print("Calibration Curve & Probability Histogram (Ranked Scaling):")
plt.show()
print("Score: ", score_model(preds_ranked,valreal,valList).values)


# Platt Scaling
plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))
fop, mpv = calibration_curve(Yv, preds_platt, n_bins=10, normalize=True)
ax1.plot([0, 1], [0, 1], linestyle='--')
ax1.plot(mpv, fop, marker='.')
ax1.set_ylabel("Fraction of positives")
ax2.hist((preds_platt+1)/2, range=(0, 1), bins=20,
             histtype="step", lw=2)
ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")

print("Calibration Curve & Probability Histogram (Platt Scaling):")
plt.show()
print("Score: ", score_model(preds_platt,valreal,valList).values)




# In the code block above, we score each ranking system and examine the prediction values and calibration curve. This gives some qualitative information on the pattern of predictions. For instance, the ranked sizing will assign the bets distributed uniformly, where as linear scaling tends to concentrate bets to a small size, and Platt scaling adopts mainly extreme bet sizes. Overall, Platt scaling appears to perform the best. The return distribution achieved by using each of these scalers is shown below. As expected, the didstributions are more "Gaussian" shaped for scalers that give bet sizes along the entire [-1,1] range (Platt and Ranked), and these scalers tend to be more successful at achieving positing outliers.

# In[ ]:


## Correlation of predictions to actual returns
plt.subplot(2,2,1)
plt.hist(preds_linear*valreal, range=(-.1, .1), bins=20,
             histtype="step", lw=2)
plt.ylabel("Count")
plt.title("Linear-Scaled Returns")


plt.subplot(2,2,2)
plt.hist(preds_platt*valreal, range=(-.1, .1), bins=20,
             histtype="step", lw=2)
plt.title("Platt-Scaled Returns")

plt.subplot(2,2,3)
plt.hist(preds_ranked*valreal, range=(-.1, .1), bins=20,
             histtype="step", lw=2)
plt.xlabel("Realized Returns")
plt.ylabel("Count")
plt.title("Rank-Scaled Returns")

plt.subplot(2,2,4)
plt.hist(((lgbm_val*2)-1)*valreal, range=(-.1, .1), bins=20,
             histtype="step", lw=2)
plt.xlabel("Realized Returns")
plt.title("Un-Scaled Returns")

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()


# ## Backtest Performance
# 
# In this section, I will show some equity curve computations. The equity curve essentially represents how a portfolio based on the model would perform. This can be more descriptive than using only a single score, for instance by examining for outliers, or stability of the returns.

# In[ ]:


"""
Equity Curve Computation

"""

def equity_curve(pred,actual,initial_capital,times):
    returns = pd.DataFrame(data=np.array(pred*actual),columns=['returns'])
    returns['time'] = times
    #returns['weight'] = np.abs(pred)
    #returns['weight'] = pred
    returns['returns_pos'] = (pred > 0)*returns['returns']
    returns['returns_neg'] = -1*(pred < 0)*returns['returns']
    returns['weight_pos'] = (pred > 0)*pred
    returns['weight_neg'] = -1*(pred < 0)*pred
    returns['bets_pos'] = 1*(pred > 0)
    returns['bets_neg'] = 1*(pred < 0)
    x = pd.DataFrame(data=returns.groupby(['time'], sort=False).aggregate(np.sum).values, columns=['x','x_pos','x_neg','weight_pos','weight_neg','bets_pos','bets_neg'])
    #weight = pd.DataFrame(data=returns.groupby(['time'], sort=False).aggregate(np.sum).values, columns=['weight'])
    #equity = initial_capital*np.ones(len(dates))
    equity = pd.DataFrame(data=np.ones(len(x)),columns=['totalReturns'])
    equity['time'] = np.unique(times)
    #equity['dailyReturnsLong'] = x['x_pos']/x['weight_pos']
    #equity['dailyReturnsShort'] = x['x_neg']/x['weight_neg']
    equity['dailyReturnsLong'] = x['x_pos']/x['bets_pos']
    equity['dailyReturnsShort'] = x['x_neg']/x['bets_neg']
    equity['dailyReturns'] = equity['dailyReturnsLong'] + equity['dailyReturnsShort'] + 1
    equity['totalReturns'] = np.cumprod(equity['dailyReturns'].values)
    
    
    return equity


# In[ ]:


from datetime import datetime
time_date = pd.DataFrame(date)[0].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))


# In[ ]:


test = equity_curve(preds_platt,valreal,100,time)
test2 = equity_curve(preds_ranked,valreal,100,time)
test3 = equity_curve(preds_linear,valreal,100,time)
test4 = equity_curve(lgbm_val*2-1,valreal,100,time)


# In[ ]:


plt.plot(time_date, test['totalReturns'], lw=2, label='Platt')
plt.plot(time_date, test2['totalReturns'], lw=2, label='Ranked')
plt.plot(time_date, test3['totalReturns'], lw=2, label='Linear')
plt.plot(time_date, test4['totalReturns'], lw=2, label='Unscaled')
plt.legend(loc='upper left')
plt.show()


# Overall, despite the differences in bet sizes, we see fairly comparable performance in terms of the equity curve shape, although both Platt and Ranked scaling methods produce positive returns while the default and linear scales give disappointing results. Notably, the performance of the equity curve is defined through very significant leaps up or down. In particular, the max and minimum daily returns are around 3 standard deviations above the mean. This should motivate the need for a more sophisticated bet sizing method which tries to achieve maximum accuracy for the prediction of significant swings, while sacrificing some accuracy when movements are small.

# # Additional Validation
# 
# As noted above, the overall score of the model seems to be highly dependent on the correct prediction of a few impactful days. In this final section, I will look at using a more robust cross validation method to ensure the stability of the model. Combinatorial Purged Cross Validation (CPCV) uses N partitions of the dataset in to combinations with K partitions retained for validation. Due to the combinatorial approach, the validation sets can then be used to reconstruct more than one backtesting path. The advantage of this is that we can repeatedly test a model on the same data without incurring any sort of look ahead bias, which may reduce false positives. The model also features data purging to prevent data leaking due to the lag features.
# 
# The implementation is worked out based on De Prado's *Advances in Financial Machine Learning* (2018). This particular function isn't totally tested, though, so I elected to just use the last 500,000 samples in the dataset to avoid any crashing. 

# In[ ]:


"""
Combinatorial Purged Cross Validation (CPCV):
Functions:
nCr: computes "n choose r"

cv_purge: adds sample embargo to the training set to prevent data leak
    - data: training data
    - val_ind:  index of the validation set
    - purge: integer for embargo range
    
CPCV: implements combinatorial cross validation, with data embargoes 
      to prevent leaks
    - data: full set of data (excluding target variable)
    - Y: target variable
    - N_PARTITIONS: number of splits for the data
    - k_val: number of splits to be retained for validation with each model fit
    - model: which model object to train (not implemented)
    - model_args: parameters for the model
 **NOTE**: This function has trouble with very large datasets.   

"""
from itertools import combinations

import math

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def cv_purge(data, val_ind, purge):
    purged = data
    val_up = val_ind+purge
    val_down = val_ind-purge
    val_down = val_down[val_down > 0]
    purged = purged.drop(index = purged.loc[purged.index.isin(val_up)].index, axis=0)
    purged = purged.drop(index = purged.loc[purged.index.isin(val_down)].index, axis=0)
    
    return purged
    

def CPCV(data, Y, N_PARTITIONS, k_val, model, model_args):
    SIZE = int(len(data)/N_PARTITIONS)
    combos = list(combinations(range(0,N_PARTITIONS),k_val))
    lc = len(combos)
    index = range(len(data))
    preds = pd.DataFrame(index=range(len(data)),columns=range(lc))
    
    for i in range(lc):
        current = combos[i]
        train_ind = []
        train = pd.DataFrame([])
        val = data
        for j in current:
            train_ind = index[(j*SIZE):((j+1)*SIZE)]
            train = train.append(data.iloc[train_ind])
            val = val.drop(train_ind, axis=0)
            
        #purge data
        train = cv_purge(train, val.index, 10)
        y_train = Y[Y.index.isin(train.index)]
        y_val = Y[np.invert(Y.index.isin(train.index))]
        
        
        #fit data
        print("Fitting model #", i)
        lgtrain, lgval = lgb.Dataset(train, y_train.values[:,0]), lgb.Dataset(val, y_val.values[:,0])
        lgbmodel = lgb.train(model_args, lgtrain, 1000, valid_sets=[lgtrain], early_stopping_rounds=100, verbose_eval=100)

        preds.loc[val.index,i] = lgbmodel.predict(val) #PH 
    
    N_PATHS = int(k_val / N_PARTITIONS * nCr(N_PARTITIONS, k_val)) 
    paths = pd.DataFrame(index=range(len(data)),columns=range(N_PATHS)) 

    for i in range(lc):
        for j in range(N_PARTITIONS):
            ind = index[(j*SIZE):((j+1)*SIZE)]
            if not preds.loc[ind,i].dropna().empty:
                for k in range(N_PATHS):
                    if paths.loc[ind,k].dropna().empty:
                        paths.loc[ind,k] = preds.loc[ind,i]
                        break

    
    return paths
    
    


# In[ ]:


## Prep data
X = cdf[traincols].fillna(0)
Y = cdf[targetcols].fillna(0)
idx0 = 1598481
idx1 = 2098481

X = X[idx0:idx1].reset_index().drop('index',axis=1)
Y = Y[idx0:idx1].reset_index().drop('index',axis=1)


# In[ ]:


cv_preds = CPCV(X,Y,6,2,0,params) #uses the same parameters as earlier model


# In[ ]:


cv_return = realreturn.iloc[idx0:idx1].reset_index()['returnsOpenNextMktres10']
cv_list = cdf['assetCode'].iloc[idx0:idx1]
cv_list = cv_list.reset_index()['assetCode']


# In[ ]:


for t in range(cv_preds.shape[1]):
    print("Model #", t, " score:", score_model(ranked_scale(cv_preds[t]),cv_return,cv_list).values)


# In[ ]:


# Fit validation measure for each method:
time = cdf['time'].iloc[idx0:idx1].values
date = np.unique(time)
preds0 = np.zeros(len(cv_preds))
preds1= np.zeros(len(cv_preds))
preds2 = np.zeros(len(cv_preds))
preds3 = np.zeros(len(cv_preds))
preds4 = np.zeros(len(cv_preds))

for t in date:
    idx = (time == t)
    preds0[idx] = ranked_scale(cv_preds[0].iloc[idx])
    preds1[idx] = ranked_scale(cv_preds[1].iloc[idx])
    preds2[idx] = ranked_scale(cv_preds[2].iloc[idx])
    preds3[idx] = ranked_scale(cv_preds[3].iloc[idx])
    preds4[idx] = ranked_scale(cv_preds[4].iloc[idx])


# In[ ]:


test5 = equity_curve(preds0,cv_return,100,cdf['time'].iloc[idx0:idx1].values)
test6 = equity_curve(preds1,cv_return,100,cdf['time'].iloc[idx0:idx1].values)
test7 = equity_curve(preds2,cv_return,100,cdf['time'].iloc[idx0:idx1].values)
test8 = equity_curve(preds3,cv_return,100,cdf['time'].iloc[idx0:idx1].values)
test9 = equity_curve(preds4,cv_return,100,cdf['time'].iloc[idx0:idx1].values)


# In[ ]:


from datetime import datetime
time_date = test5['time'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))


# In[ ]:


plt.plot(time_date, test5['totalReturns'], lw=2)
plt.plot(time_date, test6['totalReturns'], lw=2)
plt.plot(time_date, test7['totalReturns'], lw=2)
plt.plot(time_date, test8['totalReturns'], lw=2)
plt.xlabel('period')
plt.ylabel('cumulative returns')
plt.show()


# So as we can see, by using a somewhat more robust validation method, we actually can get return spreads around 5-10% in a 14 month period. This indicates that the models may be unstable, and should demonstrate the possibility of introducing false positives.

# # Conclusion
# 
# Overall, this competition had a very exciting premise. It is difficult to produce news sentiment data, and the applications to the stock market are quite clear.
# 
# In practice, there was essentially a 0% correlation between the news sentiment features and the market price history features, feature importance corroborated that features based on news sentiment were essentially useless, and many of the top scorers report that they did not use the news sentiment data at all. So instead, we essentially have a pool of people competing to find a strategy that goes against the weak form of the efficient market hypothesis.
# 
# Nearly 3000 individuals or teams made submissions in this competition, if everyone submitted strategies at random then we'd still see around 150 of them outperforming at a statistically significant (5%) level, regardless of how robust to overfitting LightGBM may be. What's worse, is that the top 10 participants alone made over 500 submissions, and then picked presumably their best submission for final assessment.
# 
# Thankfully, this is just a competition and not legitimate academic research, but it still (in my opinion) reflects an epistemology problem in the field - overemphasis on accuracy and validation scores at all costs, and ignoring methodology to reduce false positives.
# 
# I hope that my discussion of using CPCV, and examination of the model performance may inspire more rigorous investigation in to the source and stability of the performance of the features in others' models. I think areas of future research are additional feature engineering and testing the robustness of these features, as well as looking in to meta-labeling methods that may produce better bet sizing.

# In[ ]:




