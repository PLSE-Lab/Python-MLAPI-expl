# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

np.random.seed(42)
import random
random.seed(3)

data_path = '../input/beta-decay'

# Read train data
data_train = pd.read_csv(data_path+'/data_train.csv')
labels_train = pd.read_csv(data_path+'/labels_train.csv')
# Read test data
data_test = pd.read_csv(data_path+'/data_test.csv')

def create_images(data, n_theta_bins=10, n_phi_bins=20, n_time_bins=6):
    images = []
    event_indexes = {}
    event_ids = np.unique(data['EventID'].values)

    # collect event indexes
    data_event_ids = data['EventID'].values
    for i in range(len(data)):
        i_event = data_event_ids[i]
        if i_event in event_indexes:
            event_indexes[i_event].append(i)
        else:
            event_indexes[i_event] = [i]

    # create images
    for i_event in event_ids:
        event = data.iloc[event_indexes[i_event]]
        X = event[['Theta', 'Phi', 'Time']].values
        one_image, edges = np.histogramdd(X, bins=(n_theta_bins, n_phi_bins, n_time_bins))
        images.append(one_image)

    return np.array(images)

# Create train images
X = create_images(data_train, n_theta_bins=10, n_phi_bins=20, n_time_bins=6)
print('train images created')
y = labels_train['Label'].values
# Create test images
X_test = create_images(data_test, n_theta_bins=10, n_phi_bins=20, n_time_bins=6)
print('test images created')

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

def do_train():
    oof_preds = np.zeros((len(X), ))
    preds = None

    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        train_objects = X[trn_]
        valid_objects = X[val_]
        y_train = y[trn_]
        y_valid = y[val_]
        print('train len', len(train_objects))
        print('valid len', len(valid_objects))

        pca = PCA(n_components=20)
        pca.fit(train_objects.reshape(len(train_objects), -1, ))
        trainX_pca = pca.transform(train_objects.reshape(len(train_objects), -1, ))
        validX_pca = pca.transform(valid_objects.reshape(len(valid_objects), -1, ))

        model = ExtraTreesClassifier(n_estimators=1000, max_depth=None, random_state=55)
        model.fit(trainX_pca, y_train)

        y_pred = model.predict_proba(validX_pca)[:, 1]
        print(y_valid.shape)
        print(y_pred.shape)
        current_loss = roc_auc_score(y_valid, y_pred)
        print(current_loss)
        oof_preds[val_] = y_pred

        X_test_pca = pca.transform(X_test.reshape(len(X_test), -1, ))
        test_pred = model.predict_proba(X_test_pca)[:, 1]
        if preds is None:
            preds = test_pred
        else:
            preds += test_pred
        del model

    cv_loss = roc_auc_score(y, oof_preds)
    print('ALL FOLDS AUC: %.5f ' % cv_loss)
    oof_preds_df = pd.DataFrame()
    oof_preds_df['EventID'] = np.unique(data_train['EventID'].values)
    oof_preds_df['Proba'] = oof_preds
    oof_preds_df.to_csv('oof_preds.csv', index=False)
    return cv_loss, preds / folds.n_splits

best_loss, preds = do_train()
print('CV:', best_loss)

submission = pd.DataFrame()
submission['EventID'] = np.unique(data_test['EventID'].values)
submission['Proba'] = preds
submission.to_csv('submission.csv', index=False, float_format='%.6f')
