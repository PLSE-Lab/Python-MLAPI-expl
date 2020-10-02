#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import feather
from sklearn.metrics import log_loss


# In[ ]:


train_label = feather.read_dataframe("../input/creating-a-metadata-dataframe-fastai/labels.fth")
train_meta = feather.read_dataframe("../input/creating-a-metadata-dataframe-fastai/df_trn.fth")
train_label = train_label[train_label.ID != "ID_6431af929"].reset_index(drop=True)
train_meta = train_meta[train_meta.SOPInstanceUID != "ID_6431af929"].reset_index(drop=True)
test_meta = feather.read_dataframe("../input/rsnast2/df_tst_st2.fth")


# In[ ]:


train_result = pd.read_csv("../input/rsnagcp2/exp10_seresnext_train.csv")
train_result16 = pd.read_csv("../input/rsnagcp2/exp16_seres_train.csv")
train_result17 = pd.read_csv("../input/rsnagcp2/exp17_seresnext_train.csv")
train_result18 = pd.read_csv("../input/rsnagcp2/exp18_seres_train.csv")
train_result19 = pd.read_csv("../input/rsnagcp2/exp19_seres_train.csv")
train_result21 = pd.read_csv("../input/rsnagcp2/exp21_seres_train.csv")
train_result22 = pd.read_csv("../input/rsnagcp2/exp22_seres_train.csv")
train_result23 = pd.read_csv("../input/rsnagcp2/exp23_seres_train.csv")
train_result24 = pd.read_csv("../input/rsnagcp2/exp24_seres_train.csv")
train_result["Label"] = train_result["Label"]*0.025+train_result16["Label"]*0.2+train_result17["Label"]*0.1+     train_result18["Label"]*0.025+train_result19["Label"]*0.1+train_result21["Label"]*0.1+train_result23["Label"]*0.025 +     train_result24["Label"]*0.025+train_result22["Label"]*0.2
train_result["Label"] = train_result["Label"] / (0.025+0.2+0.1+0.025+0.1+0.1+0.025+0.025+0.2)


# In[ ]:


df10 = pd.read_csv("../input/rsnagcp2/exp10_seresnext_sub_st2.csv")
df16 = pd.read_csv("../input/rsnagcp2/exp16_seres_sub_st2.csv")
df17 = pd.read_csv("../input/rsna_gcp1000_1_result/exp17_seresnext_sub_st2.csv")
df18 = pd.read_csv("../input/rsnagcp4/exp18_seres_sub_st2.csv")
df19 = pd.read_csv("../input/rsnagcp3/exp19_seres_sub_st2.csv")
df21 = pd.read_csv("../input/rsnagcp2/exp21_seres_sub_st2.csv")
df22 = pd.read_csv("../input/rsnagcp4/exp22_seres_sub_st2.csv")
df23 = pd.read_csv("../input/rsnagcp3/exp23_seres_sub_st2.csv")
df24 = pd.read_csv("../input/rsnagcp2/exp24_seres_sub_st2.csv")
df16["Label"] = df10["Label"]*0.025+df16["Label"]*0.2+df17["Label"]*0.1+     df18["Label"]*0.025+df19["Label"]*0.1+df21["Label"]*0.1+df23["Label"]*0.025+df24["Label"]*0.025+     df22["Label"]*0.2
df16["Label"] = df16["Label"] / (0.025+0.2+0.1+0.025+0.1+0.1+0.025+0.025+0.2)


# In[ ]:


train_result[['ID', 'Image', 'Diagnosis']] = train_result['ID'].str.split('_', expand=True)
train_result = train_result[['Image', 'Diagnosis', 'Label']]
train_result.drop_duplicates(inplace=True)
train_result = train_result.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
train_result['Image'] = 'ID_' + train_result['Image']

train_result = train_result.rename(columns={"any": "pred_any", "epidural": "pred_epidural", "intraparenchymal": "pred_intraparenchymal",
                            "intraventricular": "pred_intraventricular", "subarachnoid": "pred_subarachnoid",
                             "subdural": "pred_subdural"})
train_result = train_result[["Image", "pred_any", "pred_epidural", "pred_intraparenchymal", "pred_intraventricular",
                            "pred_subarachnoid", "pred_subdural"]]
train_result.head()


# In[ ]:


df16[['ID', 'Image', 'Diagnosis']] = df16['ID'].str.split('_', expand=True)
df16 = df16[['Image', 'Diagnosis', 'Label']]
df16.drop_duplicates(inplace=True)
df16 = df16.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
df16['Image'] = 'ID_' + df16['Image']

df16 = df16.rename(columns={"any": "pred_any", "epidural": "pred_epidural", "intraparenchymal": "pred_intraparenchymal",
                            "intraventricular": "pred_intraventricular", "subarachnoid": "pred_subarachnoid",
                             "subdural": "pred_subdural"})
df16 = df16[["Image", "pred_any", "pred_epidural", "pred_intraparenchymal", "pred_intraventricular",
                            "pred_subarachnoid", "pred_subdural"]]
df16.head()


# In[ ]:


y_any = train_label["any"].values
pred_any = train_result["pred_any"].values
log_loss(y_any, pred_any)


# In[ ]:


merge_result = train_result.merge(train_meta, how="left", left_on="Image", right_on="SOPInstanceUID")
merge_label = train_label.merge(merge_result, how="left", left_on="ID", right_on="Image")
merge_test = df16.merge(test_meta, how="left", left_on="Image", right_on="SOPInstanceUID")


# In[ ]:


n_window = 1
pred_cols = ["pred_any", "pred_epidural", "pred_intraparenchymal", "pred_intraventricular", 
             "pred_subarachnoid", "pred_subdural"]
merge_label = merge_label.sort_values(by="ImagePositionPatient2").reset_index(drop=True)
merge_test = merge_test.sort_values(by="ImagePositionPatient2").reset_index(drop=True)
for i in range(1, n_window+1):
    merge_label = pd.concat([merge_label,
                             merge_label.groupby("SeriesInstanceUID")[pred_cols].shift(i).add_prefix("pre{}_".format(i))], axis=1)
    merge_label = pd.concat([merge_label,
                             merge_label.groupby("SeriesInstanceUID")[pred_cols].shift(-1*i).add_prefix("post{}_".format(i))], axis=1)
    merge_test = pd.concat([merge_test,
                             merge_test.groupby("SeriesInstanceUID")[pred_cols].shift(i).add_prefix("pre{}_".format(i))], axis=1)
    merge_test = pd.concat([merge_test,
                             merge_test.groupby("SeriesInstanceUID")[pred_cols].shift(-1*i).add_prefix("post{}_".format(i))], axis=1)


# In[ ]:


n_window = 20
for i in range(2, n_window+1):
    print(i)
    merge_label = pd.concat([merge_label,
                             merge_label.groupby("SeriesInstanceUID")[pred_cols].shift(i).add_prefix("pre{}_".format(i))], axis=1)
    merge_label = pd.concat([merge_label,
                             merge_label.groupby("SeriesInstanceUID")[pred_cols].shift(-1*i).add_prefix("post{}_".format(i))], axis=1)
    merge_test = pd.concat([merge_test,
                             merge_test.groupby("SeriesInstanceUID")[pred_cols].shift(i).add_prefix("pre{}_".format(i))], axis=1)
    merge_test = pd.concat([merge_test,
                             merge_test.groupby("SeriesInstanceUID")[pred_cols].shift(-1*i).add_prefix("post{}_".format(i))], axis=1)
    for c in pred_cols:
        merge_label["prev2_{}_{}".format(i, c)] = merge_label[["pre{}_{}".format(i_, c) for i_ in range(1, i+1)]].mean(axis=1)
        merge_label["postv2_{}_{}".format(i, c)] = merge_label[["post{}_{}".format(i_, c) for i_ in range(1, i+1)]].mean(axis=1)
        merge_test["prev2_{}_{}".format(i, c)] = merge_test[["pre{}_{}".format(i_, c) for i_ in range(1, i+1)]].mean(axis=1)
        merge_test["postv2_{}_{}".format(i, c)] = merge_test[["post{}_{}".format(i_, c) for i_ in range(1, i+1)]].mean(axis=1)


# In[ ]:


dcm_feats = ['ImagePositionPatient',
       'ImageOrientationPatient', 'SamplesPerPixel', 'Rows', 'Columns', 'PixelSpacing',
       'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation',
       'WindowCenter', 'WindowWidth', 'RescaleIntercept', 'RescaleSlope',
       'MultiImagePositionPatient', 'ImagePositionPatient1',
       'ImagePositionPatient2', 'MultiImageOrientationPatient',
       'ImageOrientationPatient1', 'ImageOrientationPatient2',
       'ImageOrientationPatient3', 'ImageOrientationPatient4',
       'ImageOrientationPatient5', 'MultiPixelSpacing', 'PixelSpacing1',
       'img_min', 'img_max', 'img_mean', 'img_std', 'img_pct_window',
       'MultiWindowCenter', 'WindowCenter1', 'MultiWindowWidth',
       'WindowWidth1']
feats = ["pred_any"] + ["pre{}_pred_any".format(i) for i in range(1, n_window+1)] + ["post{}_pred_any".format(i) for i in range(1, n_window+1)]
feats = feats + dcm_feats
x = merge_label[feats]
test_x = merge_test[feats]
y = merge_label["any"]


# In[ ]:


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

def train_lgbm(X_train, y_train, X_valid, y_valid, X_test, categorical_features, feature_name,
               fold_id, lgb_params, fit_params, model_name, score_func, calc_importances=True):
    train = lgb.Dataset(X_train, y_train,
                        categorical_feature=categorical_features,
                        feature_name=feature_name)
    valid = lgb.Dataset(X_valid, y_valid,
                        categorical_feature=categorical_features,
                        feature_name=feature_name)

    evals_result = {}
    model = lgb.train(
        lgb_params,
        train,
        valid_sets=[valid],
        valid_names=['valid'],
        evals_result=evals_result,
        **fit_params
    )
    print('Best Iteration: {}'.format(model.best_iteration))

    y_pred_train = model.predict(X_train)
    y_pred_train[y_pred_train < 0] = 0
    train_score = score_func(y_train, y_pred_train)

    y_pred_valid = model.predict(X_valid)
    y_pred_valid[y_pred_valid < 0] = 0
    valid_score = score_func(y_valid, y_pred_valid)

    model.save_model('{}_fold{}.txt'.format(model_name, fold_id))

    if X_test is not None:
        y_pred_test = model.predict(X_test)
        y_pred_test[y_pred_test < 0] = 0
    else:
        y_pred_test = None

    if calc_importances:
        importances = pd.DataFrame()
        importances['feature'] = feature_name
        importances['gain'] = model.feature_importance(importance_type='gain')
        importances['split'] = model.feature_importance(importance_type='split')
        importances['fold'] = fold_id
    else:
        importances = None

    return y_pred_valid, y_pred_test, train_score, valid_score, importances


def calc_score(y_true, y_pred):
    return log_loss(y_true, y_pred)


LGBM_PARAMS = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': "binary_logloss",
    'learning_rate': 0.02,
    'num_leaves': 15,
    'subsample': 0.9,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'max_depth': 7,
    'max_bin': 255,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'min_child_samples': 20,
    'min_gain_to_split': 0.02,
    'verbose': -1,
    'nthread': -1,
    'seed': 0,
}
LGBM_FIT_PARAMS = {
    'num_boost_round': 50000,
    'early_stopping_rounds': 800,
    'verbose_eval': 50000,
}


def do_ensemble(x, y, test_x, feats):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(x, y)

    y_oof = np.empty(len(x), )
    y_test = np.zeros(len(test_x))
    feature_importances = pd.DataFrame()

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        x_train, y_train = x.loc[train_idx], y[train_idx]
        x_val, y_val = x.loc[val_idx], y[val_idx]

        y_pred_valid, y_pred_test, train_score, valid_score, importances = train_lgbm(
            x_train, y_train, x_val, y_val, X_test=test_x,
            categorical_features=None,
            feature_name=feats,
            fold_id=fold_id,
            lgb_params=LGBM_PARAMS,
            fit_params=LGBM_FIT_PARAMS,
            model_name="model",
            score_func=calc_score,
            calc_importances=True
        )
        print('train score={}'.format(train_score))
        print('val score={}'.format(valid_score))

        y_oof[val_idx] = y_pred_valid
        y_test += y_pred_test / 5
        feature_importances = pd.concat([feature_importances, importances], axis=0, sort=False)
    return y_oof, y_test


# In[ ]:


losses = []
feats = ["pred_any"] + ["pre{}_pred_any".format(i) for i in range(1, n_window+1)] +     ["post{}_pred_any".format(i) for i in range(1, n_window+1)] + ["prev2_{}_pred_any".format(i) for i in range(2, n_window+1)] +    ["postv2_{}_pred_any".format(i) for i in range(2, n_window+1)]
#feats = feats + dcm_feats
x = merge_label[feats]
test_x = merge_test[feats]
y = merge_label["any"]
y_oof_any, y_test_any = do_ensemble(x, y, test_x, feats)
print("loss_any", log_loss(y, merge_label["pred_any"].values), log_loss(y, y_oof_any))
losses.append(log_loss(y, y_oof_any))
losses.append(log_loss(y, y_oof_any))

feats = ["pred_epidural"] + ["pre{}_pred_epidural".format(i) for i in range(1, n_window+1)] +     ["post{}_pred_epidural".format(i) for i in range(1, n_window+1)] + ["prev2_{}_pred_epidural".format(i) for i in range(2, n_window+1)] +    ["postv2_{}_pred_epidural".format(i) for i in range(2, n_window+1)]
#feats = feats + dcm_feats
x = merge_label[feats]
test_x = merge_test[feats]
y = merge_label["epidural"]
y_oof_epidural, y_test_epidural = do_ensemble(x, y, test_x, feats)
print("loss_epidural", log_loss(y, merge_label["pred_epidural"].values), log_loss(y, y_oof_epidural))
losses.append(log_loss(y, y_oof_epidural))

feats = ["pred_intraparenchymal"] + ["pre{}_pred_intraparenchymal".format(i) for i in range(1, n_window+1)] +     ["post{}_pred_intraparenchymal".format(i) for i in range(1, n_window+1)] + ["prev2_{}_pred_intraparenchymal".format(i) for i in range(2, n_window+1)] +    ["postv2_{}_pred_intraparenchymal".format(i) for i in range(2, n_window+1)]
#feats = feats + dcm_feats
x = merge_label[feats]
test_x = merge_test[feats]
y = merge_label["intraparenchymal"]
y_oof_intraparenchymal, y_test_intraparenchymal = do_ensemble(x, y, test_x, feats)
print("loss_intraparenchymal", log_loss(y, merge_label["pred_intraparenchymal"].values), log_loss(y, y_oof_intraparenchymal))
losses.append(log_loss(y, y_oof_intraparenchymal))

feats = ["pred_intraventricular"] + ["pre{}_pred_intraventricular".format(i) for i in range(1, n_window+1)] +     ["post{}_pred_intraventricular".format(i) for i in range(1, n_window+1)] + ["prev2_{}_pred_intraventricular".format(i) for i in range(2, n_window+1)] +    ["postv2_{}_pred_intraventricular".format(i) for i in range(2, n_window+1)]
#feats = feats + dcm_feats
x = merge_label[feats]
test_x = merge_test[feats]
y = merge_label["intraventricular"]
y_oof_intraventricular, y_test_intraventricular = do_ensemble(x, y, test_x, feats)
print("loss_intraventricular", log_loss(y, merge_label["pred_intraventricular"].values), log_loss(y, y_oof_intraventricular))
losses.append(log_loss(y, y_oof_intraventricular))

feats = ["pred_subarachnoid"] + ["pre{}_pred_subarachnoid".format(i) for i in range(1, n_window+1)] +     ["post{}_pred_subarachnoid".format(i) for i in range(1, n_window+1)] + ["prev2_{}_pred_subarachnoid".format(i) for i in range(2, n_window+1)] +    ["postv2_{}_pred_subarachnoid".format(i) for i in range(2, n_window+1)]
#feats = feats + dcm_feats
x = merge_label[feats]
test_x = merge_test[feats]
y = merge_label["subarachnoid"]
y_oof_subarachnoid, y_test_subarachnoid = do_ensemble(x, y, test_x, feats)
print("loss_subarachnoid", log_loss(y, merge_label["pred_subarachnoid"].values), log_loss(y, y_oof_subarachnoid))
losses.append(log_loss(y, y_oof_subarachnoid))

feats = ["pred_subdural"] + ["pre{}_pred_subdural".format(i) for i in range(1, n_window+1)] +     ["post{}_pred_subdural".format(i) for i in range(1, n_window+1)] + ["prev2_{}_pred_subdural".format(i) for i in range(2, n_window+1)] +    ["postv2_{}_pred_subdural".format(i) for i in range(2, n_window+1)]
#feats = feats + dcm_feats
x = merge_label[feats]
test_x = merge_test[feats]
y = merge_label["subdural"]
y_oof_subdural, y_test_subdural = do_ensemble(x, y, test_x, feats)
print("loss_subdural", log_loss(y, merge_label["pred_subdural"].values), log_loss(y, y_oof_subdural))
losses.append(log_loss(y, y_oof_subdural))


# In[ ]:


print(np.mean(losses))


# In[ ]:


sub = pd.DataFrame({
    "any": y_test_any,
    "epidural": y_test_epidural,
    "intraparenchymal": y_test_intraparenchymal,
    "intraventricular": y_test_intraventricular,
    "subarachnoid": y_test_subarachnoid,
    "subdural": y_test_subdural,
})
sub["ID"] = merge_test["Image"].values
sub = sub.set_index("ID")
sub = sub.unstack().reset_index()
sub["ID"] = sub["ID"] + "_" + sub["level_0"]
sub = sub.rename(columns={0: "Label"})
sub = sub.drop("level_0", axis=1)
sub.to_csv("sub_st2.csv", index=False)

