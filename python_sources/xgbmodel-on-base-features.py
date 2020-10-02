#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from xgboost import XGBClassifier

import numpy as np 
import pandas as pd


# In[ ]:


data = pd.read_csv('../input/home-credit-default-risk/application_train.csv')
bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv')
bureau_balance = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv')
credit_card_balance = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv')
installments_payments = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv')
previous_application = pd.read_csv('../input/home-credit-default-risk/previous_application.csv')
# POS_CASH_balance = pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv')

test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')
samp = pd.read_csv('../input/home-credit-default-risk/sample_submission.csv')


# In[ ]:


data.set_index("SK_ID_CURR",
               drop=True,
               inplace=True)

test.set_index("SK_ID_CURR",
               drop=True,
               inplace=True)

full_set = pd.concat([data, test])


# In[ ]:


bureau_mod = bureau.copy()

closed = (bureau_balance
          .groupby("SK_ID_BUREAU")
          .filter(lambda d: (d["STATUS"] == "C").any())["SK_ID_BUREAU"]
          .unique())

bureau_mod["CREDIT_ACTIVE"] = [v if v != "Active" or i not in closed
                               else "Closed" for v, i in
                               list(zip(bureau_mod["CREDIT_ACTIVE"],
                                        bureau_mod["SK_ID_BUREAU"]))]

(bureau_mod["DAYS_CREDIT_ENDDATE"]
 .fillna(bureau_mod["DAYS_ENDDATE_FACT"],
         inplace=True))


# In[ ]:


def get_loan_balance(df):
    out = df.loc[(df["CREDIT_ACTIVE"] == "Active") &
                 (df["DAYS_CREDIT_ENDDATE"] > 0), :].copy()

    return ((out["DAYS_CREDIT_ENDDATE"] /
             (out["DAYS_CREDIT"].abs() +
              out["DAYS_CREDIT_ENDDATE"]) *
             out["AMT_CREDIT_SUM"]).sum())


bureau_mod_gr = bureau_mod.groupby("SK_ID_CURR")

bureau_curr = pd.DataFrame(index=data.index.tolist() + test.index.tolist())

bureau_curr["loan_num"] = bureau_mod_gr.size()

bureau_curr["loan_act"] = (bureau_mod_gr
                           .apply(lambda d:
                                  (d["CREDIT_ACTIVE"] == "Active").sum()))

bureau_curr["loan_ovd"] = (bureau_mod_gr
                           .apply(lambda d:
                                  (d["CREDIT_DAY_OVERDUE"] != 0).sum()))

bureau_curr["loan_bal"] = (bureau_mod_gr
                           .apply(get_loan_balance))

bureau_curr.fillna(0,
                   axis=0,
                   inplace=True)


# In[ ]:


p_apps_mod = (previous_application
              .loc[(previous_application["SK_ID_CURR"]
                      .isin(full_set.index)), :]
              .query("FLAG_LAST_APPL_PER_CONTRACT == 'Y'\
                      and NFLAG_LAST_APPL_IN_DAY == '1'")
              .copy())

overdued = (installments_payments
            .groupby("SK_ID_PREV")
            .apply(lambda d:
                   (d[d["DAYS_ENTRY_PAYMENT"] > d["DAYS_INSTALMENT"]]
                    .sum())
                   .any())
            .rename("is_overdued"))

p_apps_mod = pd.merge(left=p_apps_mod,
                      right=overdued,
                      how="left",
                      left_on="SK_ID_PREV",
                      right_index=True)

exceeded = (credit_card_balance
            .groupby("SK_ID_PREV")
            .apply(lambda d:
                   (d[d["AMT_BALANCE"] > d["AMT_CREDIT_LIMIT_ACTUAL"]]
                    .astype("bool")
                    .sum())
                   .any())
            .rename("limit_exceeded"))

p_apps_mod = pd.merge(left=p_apps_mod,
                      right=exceeded,
                      how="left",
                      left_on="SK_ID_PREV",
                      right_index=True)


# In[ ]:


p_apps_mod_gr = p_apps_mod.groupby("SK_ID_CURR")

p_apps_curr = pd.DataFrame(index=data.index.tolist() + test.index.tolist())

p_apps_curr["hc_loan_num"] = (p_apps_mod_gr
                              .apply(lambda d:
                                     (d["NAME_CONTRACT_STATUS"] == "Approved")
                                     .sum()))

p_apps_curr["hc_loan_num_type"] = (p_apps_mod_gr
                                   .apply(lambda d:
                                          ((d["NAME_CONTRACT_STATUS"] ==
                                            "Approved") & (d["NAME_CONTRACT_TYPE"] ==
                                                              full_set.at[d["SK_ID_CURR"]
                                                                      .iloc[0],
                                                                      "NAME_CONTRACT_TYPE"]))
                                          .sum()))

p_apps_curr["hc_ref_num"] = (p_apps_mod_gr
                             .apply(lambda d:
                                    (d["NAME_CONTRACT_STATUS"] == "Refused")
                                    .sum()))

p_apps_curr["hc_ref_num_type"] = (p_apps_mod_gr
                                  .apply(lambda d:
                                         ((d["NAME_CONTRACT_STATUS"] ==
                                          "Refused") & (d["NAME_CONTRACT_TYPE"] ==
                                                             full_set.at[d["SK_ID_CURR"]
                                                                     .iloc[0],
                                                                     "NAME_CONTRACT_TYPE"]))
                                         .sum()))

p_apps_curr["hc_loan_ovd"] = (p_apps_mod_gr
                              .apply(lambda d:
                                     ((d["NAME_CONTRACT_STATUS"] == "Approved") &
                                      (d["is_overdued"]))
                                     .sum()))

p_apps_curr["hc_loan_ovd_type"] = (p_apps_mod_gr
                                   .apply(lambda d:
                                          ((d["NAME_CONTRACT_STATUS"] ==
                                            "Approved") & (d["is_overdued"]) &
                                           (d["NAME_CONTRACT_TYPE"] ==
                                            full_set.at[d["SK_ID_CURR"]
                                                    .iloc[0],
                                                    "NAME_CONTRACT_TYPE"]))
                                          .sum()))

p_apps_curr["hc_loan_amt"] = (p_apps_mod_gr
                              .apply(lambda d:
                                     d.loc[d["NAME_CONTRACT_STATUS"] == "Approved",
                                                "AMT_CREDIT"].sum()))

p_apps_curr["hc_loan_amt_type"] = (p_apps_mod_gr
                                   .apply(lambda d:
                                          d.loc[(d["NAME_CONTRACT_STATUS"] == "Approved") &
                                                (d["NAME_CONTRACT_TYPE"] ==
                                                 full_set.at[d["SK_ID_CURR"]
                                                         .iloc[0],
                                                         "NAME_CONTRACT_TYPE"]),
                                                "AMT_CREDIT"].sum()))

p_apps_curr["has_lim_exceeded"] = (p_apps_mod_gr
                                   .apply(lambda d:
                                          ((d["NAME_CONTRACT_STATUS"] ==
                                            "Approved") & (d["limit_exceeded"]))
                                          .sum()))

p_apps_curr.fillna(0,
                   axis=0,
                   inplace=True)


# In[ ]:


data = (data
        .join(bureau_curr)
        .join(p_apps_curr))

test = (test
        .join(bureau_curr)
        .join(p_apps_curr))


# In[ ]:


data["DAYS_EMPLOYED"].replace({365243: -1}, inplace=True)
test["DAYS_EMPLOYED"].replace({365243: -1}, inplace=True)


# In[ ]:


cols_to_impute_w_null = ["AMT_GOODS_PRICE",
                         "OBS_30_CNT_SOCIAL_CIRCLE",
                         "DEF_30_CNT_SOCIAL_CIRCLE",
                         "OBS_60_CNT_SOCIAL_CIRCLE",
                         "DEF_60_CNT_SOCIAL_CIRCLE",
                         "OWN_CAR_AGE",
                         "AMT_REQ_CREDIT_BUREAU_HOUR",
                         "AMT_REQ_CREDIT_BUREAU_DAY",
                         "AMT_REQ_CREDIT_BUREAU_WEEK",
                         "AMT_REQ_CREDIT_BUREAU_MON",
                         "AMT_REQ_CREDIT_BUREAU_QRT",
                         "AMT_REQ_CREDIT_BUREAU_YEAR"]

apartments_cols = data.columns[data.columns
                               .str.endswith(("_AVG", "_MODE", "_MEDI"))]

ext_source_cols = data.columns[data.columns
                               .str.startswith("EXT_")]

occu_gr_cols = ["NAME_INCOME_TYPE",
                "NAME_EDUCATION_TYPE",
                "CODE_GENDER",
                "ORGANIZATION_TYPE"]

exp_gr_cols = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]

cols_to_discretize = ["AMT_INCOME_TOTAL",
                      "AMT_CREDIT",
                      "AMT_ANNUITY",
                      "AMT_GOODS_PRICE",
                      "loan_bal",
                      "loan_act,"
                      "hc_loan_amt",
                      "hc_loan_amt_type"]

cols_to_drop = (apartments_cols.tolist() +
                ext_source_cols.tolist() +
                ["mean", "std", "zscore"])


# In[ ]:


class SimpleColumnsAdder(BaseEstimator):

    def __init__(self, apartment_cols, cols_to_drop):
        self.apartment_cols = apartment_cols
        self.cols_to_drop = cols_to_drop


    def fit(self, X, y=None, **fit_params):
        return self


    def transform(self, X, **transform_params):
        X.loc[:, "APART_DESC_INTEGRITY"] = (1 - X[self.apartment_cols]
                                            .isnull().mean(axis=1))
        X.loc[:, "GOODS_RATIO"] = (X.apply(lambda row:
                                           0 if np.isnan(row["AMT_GOODS_PRICE"])
                                           else (row["AMT_GOODS_PRICE"] /
                                                 row["AMT_CREDIT"]), axis=1))
        X = X.eval("ANNUITY_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL")
        X.drop(self.cols_to_drop, axis=1, inplace=True)
        return X.copy()


class ExtSourceIntegrity(BaseEstimator):

    def __init__(self, ext_source_cols):
        self.ext_source_cols = ext_source_cols


    def fit(self, X, y=None, **fit_params):
        return self


    def transform(self, X, **transform_params):
        X.loc[:, "EXT_SOURCE_INTEGRITY"] = (X.apply(lambda row: 0
                                                    if (row[self.ext_source_cols]
                                                        .isnull().sum() == 3)
                                                    else (row[self.ext_source_cols]
                                                          .mean() *
                                                          (1 - row[self.ext_source_cols]
                                                      .isnull().mean())), axis=1))
        return X.copy()


class OccupationsImputer(BaseEstimator):

    def __init__(self, occu_gr_cols, occupations=None):
        self.occu_gr_cols = occu_gr_cols
        self.occupations = occupations


    def fit(self, X, y=None, **fit_params):
        self.occupations = (X
                            .groupby(self.occu_gr_cols)["OCCUPATION_TYPE"]
                            .apply(lambda d:
                                   d.value_counts(dropna=False).index[0])
                            .replace(np.nan, "Missed"))
        return self


    def transform(self, X, **transform_params):
        X.reset_index(inplace=True)
        X.set_index(self.occu_gr_cols, inplace=True)
        X.update(self.occupations)
        X.reset_index(inplace=True)
        X.set_index(["SK_ID_CURR"], inplace=True)
        return X.copy()


class CustomOHEncoder(BaseEstimator):

    def __init__(self, cols=None):
        self.cols = cols


    def fit(self, X, y=None, **fit_params):
        self.cols = [re.compile(r"\[|\]|<", re.IGNORECASE).sub("_", col)
                     if any(x in str(col) for x in set(("[", "]", "<")))
                     else col for col in pd.get_dummies(X).columns.values]
        return self


    def transform(self, X, **transform_params):
        df = pd.get_dummies(X)
        df.columns = [re.compile(r"\[|\]|<", re.IGNORECASE).sub("_", col)
                      if any(x in str(col) for x in set(("[", "]", "<")))
                      else col for col in df.columns.values]
        for c in self.cols:
            if c not in df:
                df[c] = 0
        return df[self.cols].copy()


class CustomImputer(BaseEstimator):

    def __init__(self, cols, strategy="constant", fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.cols = cols


    def fit(self, X, y=None, **fit_params):
        if self.strategy == "constant":
            self.fill_value = 0
        elif self.strategy == "mode":
            self.fill_value = X[self.cols].mode()[0]
        return self


    def transform(self, X, **transform_params):
        X.loc[:, self.cols] = X[self.cols].fillna(self.fill_value)
        return X.copy()


class CustomQuantileDiscretizer(BaseEstimator):

    def __init__(self, cols, col_bins=None, q=50):
        self.cols = cols
        self.col_bins = col_bins
        self.q = q


    def fit(self, X, y=None, **fit_params):
        self.col_bins = {}
        for col in self.cols:
            bins = pd.qcut(x=X[col],
                           q=self.q,
                           duplicates="drop",
                           precision=0,
                           retbins=True)[1]
            self.col_bins.update({col: bins})
        return self


    def transform(self, X, **transform_params):
        for col in self.cols:
            X.loc[:, col] = pd.cut(X[col], self.col_bins[col])
        return X.copy()


class ZscoreQuantileDiscretizer(BaseEstimator):

    def __init__(self, bins=None, q=7):
        self.bins = bins
        self.q = q


    def fit(self, X, y=None, **fit_params):
        self.bins = pd.qcut(x=X["zscore"],
                            q=self.q,
                            duplicates="drop",
                            precision=0,
                            retbins=True)[1]
        return self


    def transform(self, X, **transform_params):
        X["zscore"] = [min(self.bins) if v < min(self.bins)
                       else max(self.bins) if v > max(self.bins)
                       else v for v in X["zscore"]]

        X["zscore_disc"] = pd.cut(X["zscore"],
                                  self.bins,
                                  include_lowest=True)
        return X.copy()


class DaysEmployedZscore(BaseEstimator):

    def __init__(self, exp_gr_cols, mean_std=None):
        self.exp_gr_cols = exp_gr_cols
        self.mean_std = mean_std


    def fit(self, X, y=None, **fit_params):
        self.mean_std = (X
                         .groupby(self.exp_gr_cols)["DAYS_EMPLOYED"]
                         .agg(["mean", "std"]))
        return self


    def transform(self, X, **transform_params):
        X = (X
             .merge(self.mean_std,
                    how="left",
                    left_on=self.exp_gr_cols,
                    right_index=True))
        X = X.eval("zscore = (DAYS_EMPLOYED - mean) / std")
        X["zscore"].fillna(0, inplace=True)
        return X.copy()


# In[ ]:


target = data["TARGET"].value_counts()

spw = target[0] / target[1]

xgb_model = XGBClassifier(random_state=1234,
                          objective="binary:logistic",
                          scale_pos_weight=spw,                          
                          n_jobs=-1)

model_pipe = Pipeline(
    steps=[
        ("impute_nums", CustomImputer(strategy="constant",
                                      cols=cols_to_impute_w_null)),
        ("impute_cats", CustomImputer(strategy="mode",
                                      cols="NAME_TYPE_SUITE")),
        ("get_ext_source_integrity", ExtSourceIntegrity(ext_source_cols)),
        ("impute_occupations", OccupationsImputer(occu_gr_cols)),
        ("normalize_days_employed", DaysEmployedZscore(exp_gr_cols)),
        ("discretizing_zscore", ZscoreQuantileDiscretizer()),
        ("get_apart_desc_integrity", SimpleColumnsAdder(apartments_cols,
                                                        cols_to_drop)),
#         ("discretizing_amt", CustomQuantileDiscretizer(cols_to_discretize)),
        ("oh_encoding", CustomOHEncoder()),
        ("xgb_model", xgb_model)])


# In[ ]:


y = data["TARGET"]
X = data.drop(["TARGET"], axis=1)

model_pipe.fit(X, y)


# In[ ]:


samp.loc[:, "TARGET"] = model_pipe.predict_proba(test)[:, 1]
samp.to_csv("baseline_model_disc.csv", index=False)


# In[ ]:


xgb_fea_imp = (pd.DataFrame(list(model_pipe.steps[-1][-1].get_booster()
                                 .get_fscore().items()),
                            columns=["feature", "importance"])
               .sort_values("importance", ascending=False))

xgb_fea_imp.head(15)

