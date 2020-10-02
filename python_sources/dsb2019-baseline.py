#!/usr/bin/env python
# coding: utf-8

# ## Description
# 
# 
# This kernel is based on the result of [Catboost - Some more features](https://www.kaggle.com/braquino/catboost-some-more-features).  
# The largest difference between this kernel and the aforementioned reference kernel is the use of [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html). It prevents us from overfitting, and eventually bump up the score.

# ## Libraries

# In[ ]:


import abc
import codecs
import inspect
import json
import logging
import gc
import pickle
import sys
import time
import warnings

import catboost as cat
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from numba import jit
from typing import List, Optional, Union, Tuple, Dict

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split, GroupKFold
from tqdm import tqdm_notebook


# ## Config
# 
# It seems a bit strange but I give configuration for this model with 'yaml like' string, since I usually work on data pipeline which takes yaml config file as input. I got this data pipeline idea from the repository [pudae/kaggle-hpa](https://github.com/pudae/kaggle-hpa).

# In[ ]:


conf_string = '''
dataset:
  dir: "../input/data-science-bowl-2019/"
  feature_dir: "features"
  params:

features:
  - Basic

av:
  split_params:
    test_size: 0.33
    random_state: 42

  model_params:
    objective: "binary"
    metric: "auc"
    boosting: "gbdt"
    max_depth: 7
    num_leaves: 75
    learning_rate: 0.01
    colsample_bytree: 0.7
    subsample: 0.1
    subsample_freq: 1
    seed: 111
    feature_fraction_seed: 111
    drop_seed: 111
    verbose: -1
    first_metric_only: True

  train_params:
    num_boost_round: 1000
    early_stopping_rounds: 100
    verbose_eval: 100

model:
  name: "catboost"
  model_params:
    loss_function: "MultiClass"
    eval_metric: "WKappa"
    task_type: "CPU"
    iterations: 6000
    early_stopping_rounds: 500
    random_seed: 42

  train_params:
    mode: "classification"

val:
  name: "group_kfold"
  params:
    n_splits: 5

output_dir: "output"
'''


# In[ ]:


config = dict(yaml.load(conf_string, Loader=yaml.SafeLoader))


# ## Functions and Classes

# ### utils

# #### checker

# In[ ]:


def feature_existence_checker(feature_path: Path,
                              feature_names: List[str]) -> bool:
    features = [f.name for f in feature_path.glob("*.ftr")]
    for f in feature_names:
        if f + "_train.ftr" not in features:
            return False
        if f + "_test.ftr" not in features:
            return False
    return True


# #### jsonutil

# In[ ]:


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def save_json(config: dict, save_path: Union[str, Path]):
    f = codecs.open(str(save_path), mode="w", encoding="utf-8")
    json.dump(config, f, indent=4, cls=MyEncoder, ensure_ascii=False)
    f.close()


# #### logger

# In[ ]:


def configure_logger(config_name: str, log_dir: Union[Path, str], debug: bool):
    if isinstance(log_dir, str):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    else:
        log_dir.mkdir(parents=True, exist_ok=True)

    log_filename = config_name.split("/")[-1].replace(".yml", ".log")
    log_filepath = log_dir / log_filename         if isinstance(log_dir, Path) else Path(log_dir) / log_filename

    # delete the old log
    if log_filepath.exists():
        with open(log_filepath, mode="w"):
            pass

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        filename=str(log_filepath),
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p")


# #### timer

# In[ ]:


@contextmanager
def timer(name: str, log: bool = False):
    t0 = time.time()
    msg = f"[{name}] start"
    if not log:
        print(msg)
    else:
        logging.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if not log:
        print(msg)
    else:
        logging.info(msg)


# ### validation
# 
# This kernel uses GroupKFold as validation strategy.  
# In this kernel, I grouped up the training sample with `installation_id` so that samples with certain `installation_id` do not exist in both train and val set in the same fold.

# In[ ]:


def group_kfold(df: pd.DataFrame, groups: pd.Series,
                config: dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config["val"]["params"]
    kf = GroupKFold(n_splits=params["n_splits"])
    split = list(kf.split(df, groups=groups))
    return split


def get_validation(df: pd.DataFrame,
                   config: dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    name: str = config["val"]["name"]

    func = globals().get(name)
    if func is None:
        raise NotImplementedError

    if "group" in name:
        cols = df.columns.tolist()
        cols.remove("group")
        groups = df["group"]
        return func(df[cols], groups, config)
    else:
        return func(df, config)


# ### evaluation
# 
# Code from [Ultra Fast QWK Calc Method](https://www.kaggle.com/cpmpml/ultra-fast-qwk-calc-method).

# In[ ]:


@jit
def qwk(y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list],
        max_rat: int = 3) -> float:
    y_true_ = np.asarray(y_true, dtype=int)
    y_pred_ = np.asarray(y_pred, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    numerator = 0
    for k in range(y_true_.shape[0]):
        i, j = y_true_[k], y_pred_[k]
        hist1[i] += 1
        hist2[j] += 1
        numerator += (i - j) * (i - j)

    denominator = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            denominator += hist1[i] * hist2[j] * (i - j) * (i - j)

    denominator /= y_true_.shape[0]
    return 1 - numerator / denominator


def calc_metric(y_true: Union[np.ndarray, list],
                y_pred: Union[np.ndarray, list]) -> float:
    return qwk(y_true, y_pred)


# ### models

# #### base
# 
# 
# Code taken from [hakubishin/kaggle_ieee](https://github.com/hakubishin3/kaggle_ieee).

# In[ ]:


# type alias
AoD = Union[np.ndarray, pd.DataFrame]
AoS = Union[np.ndarray, pd.Series]
CatModel = Union[cat.CatBoostClassifier, cat.CatBoostRegressor]
LGBModel = Union[lgb.LGBMClassifier, lgb.LGBMRegressor]
Model = Union[CatModel, LGBModel]


class BaseModel(object):
    @abstractmethod
    def fit(self, x_train: AoD, y_train: AoS, x_valid: AoD, y_valid: AoS,
            config: dict) -> Tuple[Model, dict]:
        raise NotImplementedError

    @abstractmethod
    def get_best_iteration(self, model: Model) -> int:
        raise NotImplementedError

    @abstractmethod
    def predict(self, model: Model, features: AoD) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_feature_importance(self, model: Model) -> np.ndarray:
        raise NotImplementedError

    def cv(self,
           y_train: AoS,
           train_features: AoD,
           test_features: AoD,
           feature_name: List[str],
           folds_ids: List[Tuple[np.ndarray, np.ndarray]],
           config: dict,
           log: bool = True
           ) -> Tuple[List[Model], np.ndarray, np.ndarray, pd.DataFrame, dict]:
        # initialize
        test_preds = np.zeros(len(test_features))
        oof_preds = np.zeros(len(train_features))
        importances = pd.DataFrame(index=feature_name)
        best_iteration = 0.0
        cv_score_list: List[dict] = []
        models: List[Model] = []

        X = train_features.values if isinstance(train_features, pd.DataFrame)             else train_features
        y = y_train.values if isinstance(y_train, pd.Series)             else y_train

        for i_fold, (trn_idx, val_idx) in enumerate(folds_ids):
            # get train data and valid data
            x_trn = X[trn_idx]
            y_trn = y[trn_idx]
            x_val = X[val_idx]
            y_val = y[val_idx]

            # train model
            model, best_score = self.fit(x_trn, y_trn, x_val, y_val, config)
            cv_score_list.append(best_score)
            models.append(model)
            best_iteration += self.get_best_iteration(model) / len(folds_ids)

            # predict oof and test
            oof_preds[val_idx] = self.predict(model, x_val).reshape(-1)
            test_preds += self.predict(
                model, test_features).reshape(-1) / len(folds_ids)

            # get feature importances
            importances_tmp = pd.DataFrame(
                self.get_feature_importance(model),
                columns=[f"gain_{i_fold+1}"],
                index=feature_name)
            importances = importances.join(importances_tmp, how="inner")

        # summary of feature importance
        feature_importance = importances.mean(axis=1)

        # print oof score
        oof_score = calc_metric(y_train, oof_preds)
        print(f"oof score: {oof_score:.5f}")

        if log:
            logging.info(f"oof score: {oof_score:.5f}")

        evals_results = {
            "evals_result": {
                "oof_score":
                oof_score,
                "cv_score": {
                    f"cv{i + 1}": cv_score
                    for i, cv_score in enumerate(cv_score_list)
                },
                "n_data":
                len(train_features),
                "best_iteration":
                best_iteration,
                "n_features":
                len(train_features.columns),
                "feature_importance":
                feature_importance.sort_values(ascending=False).to_dict()
            }
        }

        return models, oof_preds, test_preds, feature_importance, evals_results


# #### cat

# In[ ]:


CatModel = Union[CatBoostClassifier, CatBoostRegressor]


class CatBoost(BaseModel):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_valid: np.ndarray, y_valid: np.ndarray,
            config: dict) -> Tuple[CatModel, dict]:
        model_params = config["model"]["model_params"]
        mode = config["model"]["train_params"]["mode"]
        if mode == "regression":
            model = CatBoostRegressor(**model_params)
        else:
            model = CatBoostClassifier(**model_params)

        model.fit(
            x_train,
            y_train,
            eval_set=(x_valid, y_valid),
            use_best_model=True,
            verbose=model_params["early_stopping_rounds"])
        best_score = model.best_score_
        return model, best_score

    def get_best_iteration(self, model: CatModel):
        return model.best_iteration_

    def predict(self, model: CatModel,
                features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return model.predict(features)

    def get_feature_importance(self, model: CatModel) -> np.ndarray:
        return model.feature_importances_


# #### factory

# In[ ]:


def catboost() -> CatBoost:
    return CatBoost()


def get_model(config: dict):
    model_name = config["model"]["name"]
    func = globals().get(model_name)
    if func is None:
        raise NotImplementedError
    return func()


# ### features

# #### base

# In[ ]:


class Feature(metaclass=abc.ABCMeta):
    prefix = ""
    suffix = ""
    save_dir = "features"
    is_feature = True

    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.train_path = Path(self.save_dir) / f"{self.name}_train.ftr"
        self.test_path = Path(self.save_dir) / f"{self.name}_test.ftr"

    def run(self,
            train_df: pd.DataFrame,
            test_df: Optional[pd.DataFrame] = None,
            log: bool = False):
        with timer(self.name, log=log):
            self.create_features(train_df, test_df)
            prefix = self.prefix + "_" if self.prefix else ""
            suffix = self.suffix + "_" if self.suffix else ""
            self.train.columns = [str(c) for c in self.train.columns]
            self.test.columns = [str(c) for c in self.test.columns]
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self

    @abc.abstractmethod
    def create_features(self, train_df: pd.DataFrame,
                        test_df: Optional[pd.DataFrame]):
        raise NotImplementedError

    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))


class PartialFeature(metaclass=abc.ABCMeta):
    def __init__(self):
        self.df = pd.DataFrame

    @abc.abstractmethod
    def create_features(self, df: pd.DataFrame, test: bool = False):
        raise NotImplementedError


def is_feature(klass) -> bool:
    return "is_feature" in set(dir(klass))


def get_features(namespace: dict):
    for v in namespace.values():
        if inspect.isclass(v) and is_feature(v) and not inspect.isabstract(v):
            yield v()


def generate_features(train_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      namespace: dict,
                      overwrite: bool,
                      log: bool = False):
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            if not log:
                print(f.name, "was skipped")
            else:
                logging.info(f"{f.name} was skipped")
        else:
            f.run(train_df, test_df, log).save()


def load_features(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feather_path = config["dataset"]["feature_dir"]

    dfs = [
        pd.read_feather(f"{feather_path}/{f}_train.ftr", nthreads=-1)
        for f in config["features"]
        if Path(f"{feather_path}/{f}_train.ftr").exists()
    ]
    x_train = pd.concat(dfs, axis=1)

    dfs = [
        pd.read_feather(f"{feather_path}/{f}_test.ftr", nthreads=-1)
        for f in config["features"]
        if Path(f"{feather_path}/{f}_test.ftr").exists()
    ]
    x_test = pd.concat(dfs, axis=1)
    return x_train, x_test


# #### basic
# 
# 
# The features used in this kernel is all the same as those used in [Catboost - Some more features](https://www.kaggle.com/braquino/catboost-some-more-features).

# In[ ]:


IoF = Union[int, float]
IoS = Union[int, str]


class Basic(Feature):
    def create_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        all_activities = set(train_df["title"].unique()).union(
            set(test_df["title"].unique()))
        all_event_codes = set(train_df["event_code"].unique()).union(
            test_df["event_code"].unique())
        activities_map = dict(
            zip(all_activities, np.arange(len(all_activities))))
        inverse_activities_map = dict(
            zip(np.arange(len(all_activities)), all_activities))

        compiled_data_train: List[List[IoF]] = []
        compiled_data_test: List[List[IoF]] = []

        installation_ids_train = []
        installation_ids_test = []

        train_df["title"] = train_df["title"].map(activities_map)
        test_df["title"] = test_df["title"].map(activities_map)

        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

        for ins_id, user_sample in tqdm_notebook(
                train_df.groupby("installation_id", sort=False),
                total=train_df["installation_id"].nunique(),
                desc="train features"):
            if "Assessment" not in user_sample["type"].unique():
                continue
            feats = KernelFeatures(all_activities, all_event_codes,
                                   activities_map, inverse_activities_map)
            feat_df = feats.create_features(user_sample, test=False)
            installation_ids_train.extend([ins_id] * len(feat_df))
            compiled_data_train.append(feat_df)
        self.train = pd.concat(compiled_data_train, axis=0, sort=False)
        self.train["installation_id"] = installation_ids_train
        self.train.reset_index(drop=True, inplace=True)

        for ins_id, user_sample in tqdm_notebook(
                test_df.groupby("installation_id", sort=False),
                total=test_df["installation_id"].nunique(),
                desc="test features"):
            feats = KernelFeatures(all_activities, all_event_codes,
                                   activities_map, inverse_activities_map)
            feat_df = feats.create_features(user_sample, test=True)
            installation_ids_test.extend([ins_id] * len(feat_df))
            compiled_data_test.append(feat_df)
        self.test = pd.concat(compiled_data_test, axis=0, sort=False)
        self.test["installation_id"] = installation_ids_test
        self.test.reset_index(drop=True, inplace=True)


class KernelFeatures(PartialFeature):
    def __init__(self, all_activities: set, all_event_codes: set,
                 activities_map: Dict[str, float],
                 inverse_activities_map: Dict[float, str]):
        self.all_activities = all_activities
        self.all_event_codes = all_event_codes
        self.activities_map = activities_map
        self.inverse_activities_map = inverse_activities_map

        win_code = dict(
            zip(activities_map.values(),
                (4100 * np.ones(len(activities_map))).astype(int)))
        win_code[activities_map["Bird Measurer (Assessment)"]] = 4110
        self.win_code = win_code

        super().__init__()

    def create_features(self, df: pd.DataFrame, test: bool = False):
        time_spent_each_act = {act: 0 for act in self.all_activities}
        event_code_count = {ev: 0 for ev in self.all_event_codes}
        user_activities_count: Dict[IoS, IoF] = {
            "Clip": 0,
            "Activity": 0,
            "Assessment": 0,
            "Game": 0
        }

        all_assesments = []

        accumulated_acc_groups = 0
        accumulated_acc = 0
        accumulated_correct_attempts = 0
        accumulated_failed_attempts = 0
        accumulated_actions = 0

        counter = 0

        accuracy_group: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

        durations: List[float] = []
        last_activity = ""

        for i, sess in df.groupby("game_session", sort=False):
            sess_type = sess["type"].iloc[0]
            sess_title = sess["title"].iloc[0]

            if sess_type != "Assessment":
                time_spent = int(sess["game_time"].iloc[-1] / 1000)
                time_spent_each_act[
                    self.inverse_activities_map[sess_title]] += time_spent

            if sess_type == "Assessment" and (test or len(sess) > 1):
                all_attempts: pd.DataFrame = sess.query(
                    f"event_code == {self.win_code[sess_title]}")
                true_attempt = all_attempts["event_data"].str.contains(
                    "true").sum()
                false_attempt = all_attempts["event_data"].str.contains(
                    "false").sum()

                features = user_activities_count.copy()
                features.update(time_spent_each_act.copy())
                features.update(event_code_count.copy())

                features["session_title"] = sess_title

                features["accumulated_correct_attempts"] =                     accumulated_correct_attempts
                features["accumulated_failed_attempts"] =                     accumulated_failed_attempts

                accumulated_correct_attempts += true_attempt
                accumulated_failed_attempts += false_attempt

                features["duration_mean"] = np.mean(
                    durations) if durations else 0
                durations.append((sess.iloc[-1, 2] - sess.iloc[0, 2]).seconds)

                features["accumulated_acc"] =                     accumulated_acc / counter if counter > 0 else 0

                acc = true_attempt / (true_attempt + false_attempt)                     if (true_attempt + false_attempt) != 0 else 0
                accumulated_acc += acc

                if acc == 0:
                    features["accuracy_group"] = 0
                elif acc == 1:
                    features["accuracy_group"] = 3
                elif acc == 0.5:
                    features["accuracy_group"] = 2
                else:
                    features["accuracy_group"] = 1

                features.update(accuracy_group.copy())
                accuracy_group[features["accuracy_group"]] += 1

                features["accumulated_accuracy_group"] =                     accumulated_acc_groups / counter if counter > 0 else 0
                accumulated_acc_groups += features["accuracy_group"]

                features["accumulated_actions"] = accumulated_actions

                if test:
                    all_assesments.append(features)
                elif true_attempt + false_attempt > 0:
                    all_assesments.append(features)

                counter += 1

            num_event_codes: dict = sess["event_code"].value_counts().to_dict()
            for k in num_event_codes.keys():
                event_code_count[k] += num_event_codes[k]

            accumulated_actions += len(sess)
            if last_activity != sess_type:
                user_activities_count[sess_type] + +1
                last_activity = sess_type

        if test:
            self.df = pd.DataFrame([all_assesments[-1]])
        else:
            self.df = pd.DataFrame(all_assesments)

        return self.df


# ## main

# ### Settings

# In[ ]:


warnings.filterwarnings("ignore")

debug = True
config_path = "../config/cat_0.yml"
log_dir = "../log/"

configure_logger(config_path, log_dir, debug)

logging.info(f"config: {config_path}")
logging.info(f"debug: {debug}")

config["args"] = dict()
config["args"]["config"] = config_path

# make output dir
output_root_dir = Path(config["output_dir"])
feature_dir = Path(config["dataset"]["feature_dir"])

config_name: str = config_path.split("/")[-1].replace(".yml", "")
output_dir = output_root_dir / config_name
output_dir.mkdir(parents=True, exist_ok=True)

logging.info(f"model output dir: {str(output_dir)}")

config["model_output_dir"] = str(output_dir)


# ### Data/Feature Loading

# In[ ]:


input_dir = Path(config["dataset"]["dir"])

if not feature_existence_checker(feature_dir, config["features"]):
    with timer(name="load data", log=True):
        train = pd.read_csv(input_dir / "train.csv")
        test = pd.read_csv(input_dir / "test.csv")
        specs = pd.read_csv(input_dir / "specs.csv")
        
    generate_features(
        train, test, namespace=globals(), overwrite=False, log=True)

    del train, test
    gc.collect()

with timer("feature laoding", log=True):
    x_train = pd.concat([
        pd.read_feather(feature_dir / (f + "_train.ftr"), nthreads=-1)
        for f in config["features"]
    ],
                        axis=1,
                        sort=False)
    x_test = pd.concat([
        pd.read_feather(feature_dir / (f + "_test.ftr"), nthreads=-1)
        for f in config["features"]
    ])

groups = x_train["installation_id"].values
y_train = x_train["accuracy_group"].values.reshape(-1)
cols: List[str] = x_train.columns.tolist()
cols.remove("installation_id")
cols.remove("accuracy_group")
x_train, x_test = x_train[cols], x_test[cols]

assert len(x_train) == len(y_train)
logging.debug(f"number of features: {len(cols)}")
logging.debug(f"number of train samples: {len(x_train)}")
logging.debug(f"numbber of test samples: {len(x_test)}")


# ### Adversarial Validation

# In[ ]:


logging.info("Adversarial Validation")
train_adv = x_train.copy()
test_adv = x_test.copy()

train_adv["target"] = 0
test_adv["target"] = 1
train_test_adv = pd.concat([train_adv, test_adv], axis=0,
                           sort=False).reset_index(drop=True)

split_params: dict = config["av"]["split_params"]
train_set, val_set = train_test_split(
    train_test_adv,
    random_state=split_params["random_state"],
    test_size=split_params["test_size"])
x_train_adv = train_set[cols]
y_train_adv = train_set["target"]
x_val_adv = val_set[cols]
y_val_adv = val_set["target"]

logging.debug(f"The number of train set: {len(x_train_adv)}")
logging.debug(f"The number of valid set: {len(x_val_adv)}")

train_lgb = lgb.Dataset(x_train_adv, label=y_train_adv)
valid_lgb = lgb.Dataset(x_val_adv, label=y_val_adv)

model_params = config["av"]["model_params"]
train_params = config["av"]["train_params"]
clf = lgb.train(
    model_params,
    train_lgb,
    valid_sets=[train_lgb, valid_lgb],
    valid_names=["train", "valid"],
    **train_params)

# Check the feature importance
feature_imp = pd.DataFrame(
    sorted(zip(clf.feature_importance(importance_type="gain"), cols)),
    columns=["value", "feature"])

plt.figure(figsize=(20, 10))
sns.barplot(
    x="value",
    y="feature",
    data=feature_imp.sort_values(by="value", ascending=False).head(50))
plt.title("LightGBM Features")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_adv.png")

config["av_result"] = dict()
config["av_result"]["score"] = clf.best_score
config["av_result"]["feature_importances"] =     feature_imp.set_index("feature").sort_values(
        by="value",
        ascending=False
    ).head(100).to_dict()["value"]


# ### Train model

# In[ ]:


logging.info("Train model")

# get folds
x_train["group"] = groups
splits = get_validation(x_train, config)
x_train.drop("group", axis=1, inplace=True)

model = get_model(config)
models, oof_preds, test_preds, feature_importance, eval_results = model.cv(
    y_train, x_train, x_test, cols, splits, config, log=True)

config["eval_results"] = dict()
for k, v in eval_results.items():
    config["eval_results"][k] = v

feature_imp = feature_importance.reset_index().rename(columns={
    "index": "feature",
    0: "value"
})
plt.figure(figsize=(20, 10))
sns.barplot(
    x="value",
    y="feature",
    data=feature_imp.sort_values(by="value", ascending=False).head(50))
plt.title("Model Features")
plt.tight_layout()
plt.savefig(output_dir / "feature_importance_model.png")


# ### Save

# In[ ]:


save_path = output_dir / "output.json"
save_json(config, save_path)
np.save(output_dir / "oof_preds.npy", oof_preds)

with open(output_dir / "model.pkl", "wb") as m:
    pickle.dump(models, m)


# ### Make submission

# In[ ]:


sample_submission = pd.read_csv(
    input_dir / "sample_submission.csv")
sample_submission["accuracy_group"] = np.round(test_preds).astype('int')
sample_submission.to_csv('submission.csv', index=None)
sample_submission.head()

