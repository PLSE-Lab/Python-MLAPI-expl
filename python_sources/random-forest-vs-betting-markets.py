"""
This is a slightly modified Anthony Goldbloom's script.
The analysis in the original is good. Prediction and evaluation, 
not so much - waaaaay too optimistic.

Changes:
* shuffle test (otherwise there is leakage indeed)
* validation split by time
* using random forest (haven't had XGBoost handy)

Return? Always negative. No, wait. Got 0.01% once.
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier as RF

market_file = "../input/markets.csv"
runners_file = "../input/runners.csv"
odds_file = "../input/odds.csv"

df_market = pd.read_csv( market_file )
df_runners = pd.read_csv( runners_file, 
	dtype = {'barrier': np.int16,'handicap_weight': np.float16})
df_odds = pd.read_csv( odds_file )

#

##merge the runners and markets data frames
df_runners_and_market = pd.merge( df_runners, df_market, left_on='market_id', right_on='id', how='outer')
df_runners_and_market.index = df_runners_and_market['id_x'] 

numeric_features = ['position','market_id','barrier','handicap_weight']
categorical_features = ['rider_id']

#convert to factors
for feature in categorical_features:
    df_runners_and_market[feature] = df_runners_and_market[feature].astype(str)
    df_runners_and_market[feature] = df_runners_and_market[feature].replace('nan','0') #have to do this because of a weird random forest bug

    df_features = df_runners_and_market[numeric_features]

for feature in categorical_features:
    encoded_features = pd.get_dummies(df_runners_and_market[feature])
    encoded_features.columns = feature + encoded_features.columns
    df_features = pd.merge(df_features,encoded_features,left_index=True,right_index=True,how='inner') 

#turn the target variable into a binary feature: did or did not win
df_features['win'] = False
df_features.loc[df_features['position'] == 1,'win'] = True

#del df_runners_and_market, encoded_features, df_features['position']

split_at = int( len( df_features ) * 0.7 )

df_train = df_features.iloc[:split_at]
df_test = df_features.iloc[split_at:]

#

rf = RF( n_estimators = 100, verbose = 2, n_jobs = -1 )

rf.fit(df_train.drop(df_train[['win','position','market_id']],axis=1)
, df_train['win'])

predictions = rf.predict_proba(df_test.drop(df_test[['win','position','market_id']],axis=1))[:,0]
df_test['predictions'] = predictions
df_test = df_test[['predictions','win','market_id']]
#del df_train


df_odds = df_odds[df_odds['runner_id'].isin(df_test.index)]

#I take the mean odds for the horse rather than the odds 1 hour before or 10 mins before. You may want to revisit this.
average_win_odds = df_odds.groupby(['runner_id'])['odds_one_win'].mean()

#delete when odds are 0 because there is no market for this horse
average_win_odds[average_win_odds == 0] = np.nan
df_test['odds'] = average_win_odds
df_test = df_test.dropna(subset=['odds'])

#given that I predict multiple winners, there's leakage if I don't shuffle the test set (winning horse appears first and I put money on the first horse I predict to win)
df_test = df_test.iloc[np.random.permutation(len(df_test))]

#select the horse I picked as most likely to win
df_profit = df_test.loc[df_test.groupby("market_id")["predictions"].idxmax()]
df_profit
investment = 0
payout = 0
for index, row in df_profit.iterrows():
    investment +=1
    
    if (row['win']):
        payout += row['odds']

investment_return = (payout - investment)/investment
print( "\nreturn: {:.2%}".format( investment_return ))
