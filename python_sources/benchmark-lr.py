import numpy as np
import pandas as pd
import scipy.sparse as ssp
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectPercentile,f_classif
from sklearn.metrics import log_loss
seed = 1024
np.random.seed(seed)


path = "../input/"

train = pd.read_json(open(path+"train.json", "r"))
len_train = train.shape[0]

test = pd.read_json(open(path+"test.json", "r"))
test['interest_level']=-1

df = pd.concat([train,test])

df["num_photos"] = df["photos"].apply(len)
df["num_features"] = df["features"].apply(len)
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
df["created"] = pd.to_datetime(df["created"])
df["created_year"] = df["created"].dt.year
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day

num_feats = [
            "bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             ]

cat_feats = ["created_year", "created_month", "created_day",'building_id','display_address','street_address','manager_id']


feat_feats = ['features']

desc_freats = ["description"]


le_dict = dict()
for c in cat_feats:
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].values)
    le_dict[c] = le

X_num = df[num_feats].values
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)


X_cat = df[cat_feats].values

X_feat = []
for row in df[feat_feats].values:
    row = row[0]
    try:
        X_feat.append(''.join(row))
    except:
        X_feat.append('missing')



bow = CountVectorizer()
X_feat = bow.fit_transform(X_feat)

enc = OneHotEncoder()
X_cat = enc.fit_transform(X_cat)

tfidf = TfidfVectorizer((1,3))
X_desc = [] 
for row in df[desc_freats].values:
    row = str(row)
    try:
        X_desc.append(row)
    except:
        X_desc.append('missing')


X_desc = tfidf.fit_transform(X_desc)

data = ssp.hstack([X_num,X_cat,X_feat,X_desc]).tocsr()

X = data[:len_train]
X_t = data[len_train:]
del data

y = train["interest_level"].values


selector = SelectPercentile(f_classif,15,)
selector.fit(X,y)
X = selector.transform(X)
X_t = selector.transform(X_t)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y)
for ind_tr,ind_te in skf:
    X_train = X[ind_tr]
    X_test = X[ind_te]
    y_train = y[ind_tr]
    y_test = y[ind_te]
    break

print(X_train.shape,X_test.shape)


clf = LogisticRegression(C=0.5,dual=True,random_state=seed)

clf.fit(X_train, y_train)
y_test_pred = clf.predict_proba(X_test)
print(log_loss(y_test, y_test_pred))

y = clf.predict_proba(X_t)

labels2idx = {label: i for i, label in enumerate(clf.classes_)}
sub = pd.DataFrame()
sub["listing_id"] = test["listing_id"]
for label in ["high", "medium", "low"]:
    sub[label] = y[:, labels2idx[label]]
sub.to_csv("submission_lr.csv", index=False)
