#!/usr/bin/env python
# coding: utf-8

# # Comparing XGBoost and Catboost
# 
# This is a simple comparison of how both models do on only the market data. I wanted to compare CatBoost with XGBoost since a lot of people have been using XGBoost and CatBoost is supposedly faster, and also is enabled to handle categorical data. As you'll see we do our own label encoding for XGBoost, but CatBoost handles this very conveniently for us.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from kaggle.competitions import twosigmanews


# In[ ]:


env = twosigmanews.make_env()
(market_train, _) = env.get_training_data()


# # Data Preparation
# 
# We're going to define a function to make our analysis easier and that we can call when we want to process the data one day at a time for our submission. We'll keep all the columns except time and we'll add an extra column for our label encoded assets. We'll then delete the assetCode column when we use the XGBoost model, or delete the label encoded column when we use the CatBoost model since it can handle the strings directly.

# In[ ]:


lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())} #the function that gets a label for our assetcodes

def prep_data(market_data):
    # add asset code representation as int (as in previous kernels)
    market_data['assetCodeT'] = market_data['assetCode'].map(lbl) #assigns an integer label to each asset code using the function above
    market_col = ['assetCode','assetCodeT', 'volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 
                        'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 
                        'returnsOpenPrevMktres10']
    
    X = market_data[market_col].fillna(0).values #fillna values with zero
    if "returnsOpenNextMktres10" in list(market_data.columns):#if training data we need these to train
        up = (market_data.returnsOpenNextMktres10 >= 0).values
        r = market_data.returnsOpenNextMktres10.values
        universe = market_data.universe
        day = market_data.time.dt.date
        assert X.shape[0] == up.shape[0] == r.shape[0] == universe.shape[0] == day.shape[0]
    else:#data for prediction only we won't have or need these
        up = []
        r = []
        universe = []
        day = []
    return X, up, r, universe, day


# In[ ]:


X, up, r, universe, day = prep_data(market_train) #prepare the data


# In[ ]:


# r, u and d are used to calculate the scoring metric on test
X_train, X_test, up_train, up_test, _, r_test, _, u_test, _, d_test = train_test_split(X, up, r, universe, day, test_size=0.25, random_state=99)


# # Fit
# 
# Let's fit both models and then see how they do. Before fitting each I'll get rid of the duplicate column, either the assetCode or the label encoding.

# In[ ]:


xgb_market = XGBClassifier(n_jobs=4, n_estimators=200, max_depth=8, eta=0.1)

XGB_X_train = np.delete(X_train,0,1) #get rid of assetcodes in both train and test data
XGB_X_test = np.delete(X_test,0,1)

t = time.time()
print('Fitting Up')
xgb_market.fit(XGB_X_train,up_train)
print(f'XGB Done, time = {time.time() - t}s')


# In[ ]:


cat_up = CatBoostClassifier(thread_count=4, n_estimators=200, max_depth=8, eta=0.1, loss_function='Logloss' , verbose=10)

cat_X_train = np.delete(X_train,1,1) #get rid of encoded labels since this is duplicate information, CatBoost will handle the assetCodes directly
cat_X_test = np.delete(X_test,1,1)

t = time.time()
print('Fitting Up')
cat_features=[0] #this is just the column(s) in our data set that has categorical data and then we pass it to the model fitting function below
cat_up.fit(cat_X_train, up_train, cat_features) 
print(f'cat Done, time = {time.time() - t}')


# So we can see the CatBoost model has a much faster fitting with similar parameters set which should be a relief to anyone fitting the XGBoost model over and over again like I'd been doing. It took 26 minutes for the XGBoost model to fit for me, and only 5.5 for the CatBoost model. Let's see how the models prediction's compare.

# # Evaluation of Test
# 
# Here we'll compare plots of the predicted confidence for both models, and then score each using the criteria of the competition.

# In[ ]:


confidence_test = xgb_market.predict_proba(XGB_X_test)[:,1]*2 -1

print(accuracy_score(confidence_test>0,up_test))
plt.hist(confidence_test, bins='auto')
plt.title("XGB predicted confidence")
plt.show()


# In[ ]:


catconfidence_test = cat_up.predict_proba(cat_X_test)[:,1]*2 -1

print(accuracy_score(catconfidence_test>0,up_test))
plt.hist(catconfidence_test, bins='auto')
plt.title("CatBoost predicted confidence")
plt.show()


# It's interesting to note the different shape of the distributions, CatBoost predicts a much more normal distribution, while XGBoost seems to be a little skewed. I think we should expect a roughly normal distribution with mean just barely larger than zero. The CatBoost plot agrees with my intuition more but I do not have any good explanation as to why they differ. Finally, scoring both:

# In[ ]:


# calculation of actual metric that is used to calculate final score
r_test = r_test.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = confidence_test * r_test * u_test
data = {'day' : d_test, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_test = mean / std
print(f'XGBoost score: {score_test}')


# In[ ]:


r_test = r_test.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_icat = catconfidence_test * r_test * u_test
data2 = {'day' : d_test, 'x_t_icat' : x_t_icat}
df2 = pd.DataFrame(data2)
x_tcat = df2.groupby('day').sum().values.flatten()
mean = np.mean(x_tcat)
std = np.std(x_tcat)
score_testcat = mean / std
print(f'CatBoost score: {score_testcat}')


# This is really shockingly good. I'm not sure what exactly happened here, it may just be an outlier from our dataset, or maybe we are overfitting somehow. The leaderboard submission scores a bit better than XGBoost I think, but definitely not as well as the above result would imply. Finally let's use our CatBoost model to make predictions we can submit and then we'll look at feature importance.

# # Prediction

# In[ ]:


days = env.get_prediction_days()


# In[ ]:


n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
predicted_confidences = np.array([])
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    print(n_days,end=' ')
    
    t = time.time()
    # discard assets that are not scored
    market_obs_df = market_obs_df[market_obs_df.assetCode.isin(predictions_template_df.assetCode)]
    X_market_obs = prep_data(market_obs_df)[0]
    X_market_obs  = np.delete(X_market_obs ,1,1) #drop the label encoded column again
    prep_time += time.time() - t
    
    t = time.time()
    market_prediction = cat_up.predict_proba(X_market_obs)[:,1]*2 -1
    predicted_confidences = np.concatenate((predicted_confidences, market_prediction))
    prediction_time += time.time() -t
    
    t = time.time()
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':market_prediction})
    # insert predictions to template
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t

env.write_submission_file()
total = prep_time + prediction_time + packaging_time
print(f'Preparing Data: {prep_time:.2f}s')
print(f'Making Predictions: {prediction_time:.2f}s')
print(f'Packing: {packaging_time:.2f}s')
print(f'Total: {total:.2f}s')


# In[ ]:


# distribution of confidence as a sanity check: they should be distributed as above
plt.hist(predicted_confidences, bins='auto')
plt.title("CatBoost predicted confidence")
plt.show()


# Again we get a pretty good normal distribution which is a good consistency test.

# # Feature importances
# 
# Last we can compare the importance of each feature in the two models and see how they compare.

# In[ ]:


market_col = ['assetCodeT', 'volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 
                        'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 
                        'returnsOpenPrevMktres10']
plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.bar(range(len(xgb_market.feature_importances_)), xgb_market.feature_importances_)
plt.title("XGB Feature Importance")
plt.xticks(range(len(xgb_market.feature_importances_)), market_col, rotation='vertical');


# In[ ]:


plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.bar(range(len(cat_up.get_feature_importance(prettified=False))), cat_up.get_feature_importance(prettified=False))
plt.title("Cat Feature Importance")
plt.xticks(range(len(cat_up.get_feature_importance(prettified=False))), market_col, rotation='vertical');


# This is really striking because it looks like assetCodes are way more important in the CatBoost model. (Note: assetCodeT in the CatBoost plot is the real assetCode, not the encoded label, I just didn't change the bin label) It's very likely this is due to the nature of however CatBoost is encoding the labels,  you can read about what it's doing here: https://tech.yandex.com/catboost/doc/dg/concepts/algorithm-main-stages_cat-to-numberic-docpage/  To me at least it's a  bit unclear how much of a good thing this is. Should whether an asset price goes up or down depend on who/what it is? It strikes me sort of like saying "well Netflix stock always goes up." 
# 
# We also see Volume is the 3rd most important predictor in XGBoost and the worst in CatBoost which is a bit strange. You'd think if it were good for one it should be good for both although it's unclear to me why volume would be a good predictor of a stock increasing or decreasing. I could see it being a good predictor of the confidence e.g. if a stock is being sold off, a high volume might imply that it is really being sold heavily and will end the day down signficantly.
# 
# Close price is reasonably close in both models, 4th in XGB and 3rd in Cat.
# 
# We also see both models agree that returnsOpenPrevRaw10 and returnsOpenPrevMktres10 are fairly important. But why the return from ten days ago and not the return from yesterday? Does it have something to do with the fact that we are predicting ten days in the future? That would seem odd.
# 
# Overall I wouldn't try to pull too much out of these plots. I think the disagreement between the two may indicate just how noisy the data is and how difficult it is to really find patterns in.
# 
# I hope this was helpful and if you have any questions or think you can answer any of mine please comment! Cheers!
