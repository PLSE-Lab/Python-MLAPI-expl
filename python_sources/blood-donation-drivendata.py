#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


# In[ ]:


train = pd.read_csv("../input/train.csv")


# In[ ]:


train.head(10)


# In[ ]:


train.describe()


# In[ ]:


train.isnull().sum()


# In[ ]:


#It seems that the column "Unnamed: 0" is actually the Donor ID and thus, we can safely assume that it has no connection with the donation , atleast for now


# In[ ]:


train.info()


# In[ ]:


train["DonationsPerMonth"] = train["Number of Donations"]/(train["Months since First Donation"]-train["Months since Last Donation"])    
train["DonationsPerMonth"] = train["DonationsPerMonth"].replace([np.inf, -np.inf], 2)


# In[ ]:


train["MarchWithinInterval"] = np.where(train["Months since Last Donation"]>train["DonationsPerMonth"],1,0)


# In[ ]:


train["MarchWithinInterval"] = train["MarchWithinInterval"].astype('object')


# In[ ]:


train.drop("Unnamed: 0",axis=1, inplace=True)


# In[ ]:


train.info()


# In[ ]:


for col in train.columns:
    train.plot.scatter(x="Made Donation in March 2007",y=col)
    train.plot.hexbin(x="Made Donation in March 2007",y=col,gridsize=30,sharex=False)


# In[ ]:


def bin_last_donation(month):
    bina=""
    if month>20:
        bina="Greater"
    elif month>10:
        bina="Medium"
    elif month<7:
        bina="Lower"
    else:
        bina="Interm"
    return bina

def bin_donation_per_period(interval):
    bina=""
    if interval<0.3:
        bina="Low"
    elif interval<1.0:
        bina="Medium"
    else:
        bina="High"
    return bina

def bin_number_of_donation(num):
    bina=""
    if num<2:
        bina="Novice"
    elif num<4:
        bina="EarlyLearner"
    elif num<7:
        bina="Learner"
    elif num<10:
        bina="Interm"
    elif num<18:
        bina="New"
    elif num<45:
        bina="Seasoned"
    else:
        bina="Seasoned"
    return bina
        
def bin_volume_donated(vol):
    bina=""
    if vol<800:
        bina="Novice"
    elif vol<2000:
        bina="Learner"
    elif vol<4000:
        bina="New"
    elif vol<11000:
        bina="Seasoned"
    else:
        bina="Old"
    return bina
    
    
train["LastDonation"] = train["Months since Last Donation"].apply(bin_last_donation)
train["Freq"] = train["Number of Donations"].apply(bin_number_of_donation)
train["Volume"] = train["Total Volume Donated (c.c.)"].apply(bin_volume_donated)
train["Regularity"] = train["DonationsPerMonth"].apply(bin_donation_per_period)


# In[ ]:


y = train["Made Donation in March 2007"]


# In[ ]:


train.drop("Made Donation in March 2007",axis=1,inplace=True)


# In[ ]:


from scipy.stats import skew

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in train.columns:
    if train[i].dtype in numeric_dtypes: 
        numerics2.append(i)

skew_features = train[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
skews = pd.DataFrame({'skew':skew_features})
skews


# In[ ]:


from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

high_skew = skew_features[skew_features > 0.5]
high_skew = high_skew
skew_index = high_skew.index

for i in skew_index:
    train[i]= boxcox1p(train[i], boxcox_normmax(train[i]+1))

        
skew_features2 = train[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
skews2 = pd.DataFrame({'skew':skew_features2})
skews2


# In[ ]:


train["DonationsPerMonth"]


# In[ ]:


train.drop("Total Volume Donated (c.c.)",axis=1,inplace=True)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train[train.select_dtypes(exclude="object").columns.tolist()] = scaler.fit_transform(train[train.select_dtypes(exclude="object").columns.tolist()])


# In[ ]:


train.describe()


# In[ ]:


y.describe()


# In[ ]:


#train = pd.get_dummies(train)


# In[ ]:


test = pd.read_csv("../input/test.csv")


# In[ ]:


test.describe()


# In[ ]:


test["DonationsPerMonth"] = test["Number of Donations"]/(test["Months since First Donation"]-test["Months since Last Donation"])    
test["DonationsPerMonth"] = test["DonationsPerMonth"].replace([np.inf, -np.inf], 2)


# In[ ]:


test["MarchWithinInterval"] = np.where(test["Months since Last Donation"]>test["DonationsPerMonth"],1,0)


# In[ ]:


test["MarchWithinInterval"] = test["MarchWithinInterval"].astype('object')


# In[ ]:


test.drop("Unnamed: 0",axis=1,inplace=True)
test["LastDonation"] = test["Months since Last Donation"].apply(bin_last_donation)
test["Freq"] = test["Number of Donations"].apply(bin_number_of_donation)
test["Volume"] = test["Total Volume Donated (c.c.)"].apply(bin_volume_donated)
test["Regularity"] = test["DonationsPerMonth"].apply(bin_donation_per_period)


# In[ ]:


from scipy.stats import skew

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in test.columns:
    if test[i].dtype in numeric_dtypes: 
        numerics2.append(i)

skew_features = test[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
skews = pd.DataFrame({'skew':skew_features})
skews


# In[ ]:


from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

high_skew = skew_features[skew_features > 0.5]
high_skew = high_skew
skew_index = high_skew.index

for i in skew_index:
    test[i]= boxcox1p(test[i], boxcox_normmax(test[i]+1))

        
skew_features2 = test[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
skews2 = pd.DataFrame({'skew':skew_features2})
skews2


# In[ ]:


test.drop("Total Volume Donated (c.c.)",axis=1,inplace=True)


# In[ ]:


#test[test.columns.tolist()] = scaler.fit_transform(test[test.columns.tolist()])
test[test.select_dtypes(exclude="object").columns.tolist()] = scaler.fit_transform(test[test.select_dtypes(exclude="object").columns.tolist()])


# In[ ]:


#test = pd.get_dummies(test)


# In[ ]:


CategCols = test.select_dtypes(include="object").columns.tolist()


# In[ ]:


temp = pd.get_dummies(pd.concat([train,test],keys=[0,1]), columns=CategCols, drop_first=True)
train,test = temp.xs(0),temp.xs(1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.25, random_state=28)


# In[ ]:


from sklearn.metrics import log_loss


# In[ ]:


from sklearn.svm import SVC
svcClassif = SVC(gamma="auto")
svcClassif.fit(X_train,y_train)
y_preds = svcClassif.predict(X_test)
#log_loss(y_preds,y_test)
accuracy_score(y_preds,y_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfClassif = RandomForestClassifier()
rfClassif.fit(X_train,y_train)
y_preds = rfClassif.predict(X_test)
y_pred_probs_rf = rfClassif.predict_proba(X_test)
print(log_loss(y_preds,y_test))
print(accuracy_score(y_preds,y_test))


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
adaBoostClassif = AdaBoostClassifier()
adaBoostClassif.fit(X_train,y_train)
y_preds = adaBoostClassif.predict(X_test)
y_pred_probs_ab = adaBoostClassif.predict_proba(X_test)
print(log_loss(y_preds,y_test))
print(accuracy_score(y_preds,y_test))


# In[ ]:


from sklearn.linear_model import LogisticRegression
LRClassif = LogisticRegression(solver="liblinear")
LRClassif.fit(X_train,y_train)
y_preds = LRClassif.predict(X_test)
y_pred_probs_lr = LRClassif.predict_proba(X_test)
print(log_loss(y_preds,y_test))
print(accuracy_score(y_preds,y_test))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbClassif = GradientBoostingClassifier()
gbClassif.fit(X_train,y_train)
y_preds = gbClassif.predict(X_test)
y_pred_probs_gb = gbClassif.predict_proba(X_test)
print(log_loss(y_preds,y_test))
print(accuracy_score(y_preds,y_test))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
kNNClassif = KNeighborsClassifier(n_neighbors=20)
kNNClassif.fit(X_train,y_train)
y_preds = kNNClassif.predict(X_test)
y_pred_probs_kn = kNNClassif.predict_proba(X_test)
print(log_loss(y_preds,y_test))
print(accuracy_score(y_preds,y_test))


# In[ ]:


from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingCVClassifier

# Initializing models

# The StackingCVClassifier uses scikit-learn's check_cv
# internally, which doesn't support a random seed. Thus
# NumPy's random seed need to be specified explicitely for
# deterministic behavior
np.random.seed(42)
sclf = StackingCVClassifier(classifiers=[svcClassif,rfClassif,gbClassif,adaBoostClassif,LRClassif], 
                            meta_classifier=kNNClassif)

params = {'meta-kneighborsclassifier__n_neighbors': [1, 20],
            'randomforestclassifier__n_estimators': [10, 50],
          'logisticregression__C': [0.1, 10.0]}

grid = GridSearchCV(estimator=sclf, 
                    param_grid=params, 
                    cv=5,
                    refit=True)
grid.fit(X_train.values, y_train.values)

cv_keys = ('mean_test_score', 'std_test_score', 'params')

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
             grid.cv_results_[cv_keys[1]][r] / 2.0,
             grid.cv_results_[cv_keys[2]][r]))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)


# In[ ]:


y_pred_probs_stack = grid.predict_proba(X_test.values)


# In[ ]:


y_cum_preds = (0.2*y_pred_probs_rf) + (0.1*y_pred_probs_ab) + (0.1*y_pred_probs_gb) + (0.2*y_pred_probs_lr) + (0.1*y_pred_probs_stack) + (0.3*y_pred_probs_kn)


# In[ ]:


y_cum_preds = np.where(y_cum_preds > 0.5, 1, 0)


# In[ ]:


y_cum_preds = np.argmax(y_cum_preds,axis=1)


# In[ ]:


log_loss(y_cum_preds,y_test)
accuracy_score(y_preds,y_test)


# In[ ]:


y_test_pred_probs_rf = rfClassif.predict_proba(test)
y_test_pred_probs_ab = adaBoostClassif.predict_proba(test)
y_test_pred_probs_gb = gbClassif.predict_proba(test)
y_test_pred_probs_lr = LRClassif.predict_proba(test)
#y_test_pred_probs_nn = nn2.predict_proba(test)
y_test_pred_probs_kn = kNNClassif.predict_proba(test)
y_test_pred_probs_stack = grid.predict_proba(test.values)


# In[ ]:


y_true_cum_preds = (0.2*y_test_pred_probs_rf) + (0.1*y_test_pred_probs_ab) + (0.1*y_test_pred_probs_gb) + (0.2*y_test_pred_probs_lr) + (0.1*y_test_pred_probs_stack) + (0.3*y_test_pred_probs_kn)


# Further course of action : MultiLayerPerceptrons have been added. Now, Take Averages of the models and submit and then make Stacking Classifiers and then make the next submission.

# In[ ]:


len(y_true_cum_preds)


# In[ ]:


y_true_cum_preds = y_true_cum_preds[:,1:]


# In[ ]:


y_true_cum_preds = y_true_cum_preds.flatten()


# In[ ]:


testy = pd.read_csv("../input/test.csv")
submission = pd.DataFrame({'Unnamed: 0':testy["Unnamed: 0"],'Made Donation in March 2007':y_true_cum_preds})
#submission = pd.DataFrame({'Unnamed: 0':testy["Unnamed: 0"],'Made Donation in March 2007':y_test_pred_probs_ab})


# In[ ]:


submission.describe()


# In[ ]:


sub = submission[["Unnamed: 0","Made Donation in March 2007"]]


# In[ ]:


sub.to_csv("submission.csv", encoding='utf-8', index=False)


# In[ ]:




