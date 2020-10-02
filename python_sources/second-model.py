import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, auc

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print('data read.....!')
SEED=1234

y = train['QuoteConversion_Flag']
train.drop(train.columns[2], axis=1, inplace=True)

train_length = len(train)

# Now we combine test and train for preprocessing
complete_data = train.append(test)
print(complete_data.shape)

# Things to try as extract day and month part from the quotes date, create bins for them
# Check the unique values for all the columns

def pre_processing(data):
    # Extract month and date part
    data['Date'] = pd.to_datetime(pd.Series(data['Original_Quote_Date']))
    # Now drop this column from the data frame
    data = data.drop('Original_Quote_Date', axis=1)
    data['Month'] = data['Date'].apply(lambda x: int(str(x)[5:7]))
    data['weekday'] = data['Date'].dt.dayofweek
    data.drop('Date', axis=1, inplace=True)
    le = LabelEncoder()
    for f in data.columns:
        if data[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(data[f].values))
            data[f] = lbl.transform(list(data[f].values))
    return data



complete_data = pre_processing(complete_data)
print('pre_processing done.....!')
print(complete_data.dtypes)
train_data = complete_data.iloc[0:train_length, ]
test_data = complete_data.iloc[train_length:, ]

def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = train_test_split(
                                       X, y, test_size=.20, 
                                       random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = roc_auc_score(y_cv, preds)
        print("AUC (fold %d/%d): %f" % (i + 1, N, auc))
        mean_auc += auc
    return mean_auc/N
    
print ("Performing greedy feature selection...")
score_hist = []
# class_label_1 = {1: 0.985, 0: (1-0.985)}
N = 10

model = xgb.XGBClassifier(n_estimators=30,
                        nthread=-1,
                        max_depth=10,
                        learning_rate=0.025,
                        silent=True,
                        subsample=0.8,
                        missing=np.nan)


good_features = set([])
# Greedy feature selection loop
while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
    scores = []
    for f in range(len(train_data.columns)):
        if f not in good_features:
            feats = list(good_features) + [f]
            #Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
            Xt = train_data[train_data.columns[feats]]
            score = cv_loop(Xt, y, model, N)
            scores.append((score, f))
            print ("Feature: %i Mean AUC: %f" % (f, score))
    good_features.add(sorted(scores)[-1][1])
    score_hist.append(sorted(scores)[-1])
    print ("Current features: %s" % sorted(list(good_features)))

# Remove last added feature from good_features
good_features.remove(score_hist[-1][1])
good_features = sorted(list(good_features))
print ("Selected features %s" % good_features)

