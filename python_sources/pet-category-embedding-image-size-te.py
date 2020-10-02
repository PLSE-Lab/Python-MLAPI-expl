#!/usr/bin/env python
# coding: utf-8

# ## Libraries

# In[ ]:


import os
import cv2
import json
import pickle
import random
import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision.models as models
import torchvision.transforms as transforms

from pathlib import Path
from datetime import datetime as dt
from functools import partial
from collections import Counter

from PIL import Image

from joblib import Parallel, delayed

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

pd.options.display.max_columns = 128
torch.multiprocessing.set_start_method("spawn")


# ## Image model loading

# In[ ]:


get_ipython().system('cp ../input/pytorch-pretrained-image-models/* ./')
get_ipython().system('ls ')


# ## Metadata and Sentiment data

# In[ ]:


def jopen(path):
    with open(path, "r", encoding="utf-8") as f:
        json_file = json.load(f)
    return json_file


def parse_sentiment_file(path):
    file = jopen(path)
    language: str = file["language"]

    sentiment: list = file["documentSentiment"]
    entities: list = [x["name"] for x in file["entities"]]
    entity = " ".join(entities)

    sentence_sentiment: list = [x["sentiment"] for x in file["sentences"]]
    magnitude: np.ndarray = np.array(
        [x["magnitude"] for x in sentence_sentiment])
    score: np.ndarray = np.array([x["score"] for x in sentence_sentiment])

    return_js = {
        "magnitude_sum": magnitude.sum(),
        "magnitude_mean": magnitude.mean(),
        "magnitude_var": magnitude.var(),
        "score_sum": score.sum(),
        "score_mean": score.mean(),
        "score_var": score.var(),
        "language": language,
        "entity": entity,
        "document_magnitude": sentiment["magnitude"],
        "document_score": sentiment["score"]
    }
    return return_js


def parse_metadata(path):
    file: dict = jopen(path)
    file_keys = list(file.keys())
    if "labelAnnotations" in file_keys:
        file_annots = file["labelAnnotations"]
        file_mean_score = np.asarray([x["score"] for x in file_annots]).mean()
        file_desc = " ".join([x["description"] for x in file_annots])
    else:
        file_mean_score = np.nan
        file_desc = ""

    file_colors: list = file["imagePropertiesAnnotation"]["dominantColors"][
        "colors"]
    file_crops: list = file["cropHintsAnnotation"]["cropHints"]

    color_score = np.asarray([x["score"] for x in file_colors]).mean()
    pixel_frac = np.asarray([x["pixelFraction"] for x in file_colors]).mean()
    crop_conf = np.asarray([x["confidence"] for x in file_crops]).mean()

    if "importanceFraction" in file_crops[0].keys():
        crop_importance = np.asarray(
            [x["importanceFraction"] for x in file_crops]).mean()
    else:
        crop_importance = np.nan
    metadata = {
        "annot_score": file_mean_score,
        "color_score": color_score,
        "pixel_frac": pixel_frac,
        "crop_conf": crop_conf,
        "crop_importance": crop_importance,
        "desc": file_desc
    }
    return metadata


def additinal_features_per_id(pet_id, sentiment_path: Path, meta_path: Path):
    sentiment_path = sentiment_path / f"{pet_id}.json"
    try:
        sentiment = parse_sentiment_file(sentiment_path)
        sentiment["pet_id"] = pet_id
    except FileNotFoundError:
        sentiment = {}

    meta_files = sorted(meta_path.glob(f"{pet_id}*.json"))
    metadata_list = []
    if len(meta_files) > 0:
        for f in meta_files:
            metadata = parse_metadata(f)
            metadata["pet_id"] = pet_id
            metadata_list.append(metadata)
    return sentiment, metadata_list


def load_additional_features(ped_ids: list, sentiment_path: Path,
                             meta_path: Path):
    features = Parallel(
        n_jobs=-1, verbose=1)(
            delayed(additinal_features_per_id)(i, sentiment_path, meta_path)
            for i in ped_ids)
    sentiments = [x[0] for x in features if len(x[0]) > 0]
    metadatas = [x[1] for x in features if len(x[1]) > 0]
    sentiment_keys = sentiments[0].keys()
    metadata_keys = metadatas[0][0].keys()
    sentiment_dict = {}
    metadata_dict = {}
    for key in sentiment_keys:
        sentiment_dict[key] = [x[key] for x in sentiments]

    for key in metadata_keys:
        meta_list = []
        for meta_per_pid in metadatas:
            meta_list += [meta[key] for meta in meta_per_pid]
        metadata_dict[key] = meta_list

    sentiment_df = pd.DataFrame(sentiment_dict)
    metadata_df = pd.DataFrame(metadata_dict)
    return sentiment_df, metadata_df


def aggregate_metadata(metadata_df: pd.DataFrame,
                       aggregates=["sum", "mean", "var"]):
    meta_desc: pd.DataFrame = metadata_df.groupby(["pet_id"])["desc"].unique()
    meta_desc = meta_desc.reset_index()
    meta_desc["desc"] = meta_desc["desc"].apply(lambda x: " ".join(x))

    meta_gr: pd.DataFrame = metadata_df.drop(["desc"], axis=1)
    for i in meta_gr.columns:
        if "pet_id" not in i:
            meta_gr[i] = meta_gr[i].astype(float)
    meta_gr = meta_gr.groupby(["pet_id"]).agg(aggregates)
    meta_gr.columns = pd.Index(
        [f"{c[0]}_{c[1].upper()}" for c in meta_gr.columns.tolist()])
    meta_gr = meta_gr.reset_index()
    return meta_gr, meta_desc


def aggregate_sentiment(sentiment_df: pd.DataFrame, aggregates=["sum"]):
    sentiment_desc: pd.DataFrame = sentiment_df.groupby(
        ["pet_id"])["entity"].unique()
    sentiment_desc = sentiment_desc.reset_index()
    sentiment_desc["entity"] = sentiment_desc["entity"].apply(
        lambda x: " ".join(x))
    sentiment_lang = sentiment_df.groupby(
        ["pet_id"])["language"].unique()
    sentiment_lang = sentiment_lang.reset_index()
    sentiment_lang["language"] = sentiment_lang["language"].apply(
        lambda x: " ".join(x))
    sentiment_desc = sentiment_desc.merge(
        sentiment_lang, how="left", on="pet_id")
    

    sentiment_gr: pd.DataFrame = sentiment_df.drop(["entity", "language"],
                                                   axis=1)
    for i in sentiment_gr.columns:
        if "pet_id" not in i:
            sentiment_gr[i] = sentiment_gr[i].astype(float)
    sentiment_gr = sentiment_gr.groupby(["pet_id"]).agg(aggregates)
    sentiment_gr.columns = pd.Index(
        [f"{c[0]}" for c in sentiment_gr.columns.tolist()])
    sentiment_gr = sentiment_gr.reset_index()
    return sentiment_gr, sentiment_desc


# ## Load data

# In[ ]:


input_dir = Path("../input/petfinder-adoption-prediction/")
train = pd.read_csv(input_dir / "train/train.csv")
test = pd.read_csv(input_dir / "test/test.csv")
sample_submission = pd.read_csv(input_dir / "test/sample_submission.csv")


# In[ ]:


sp_train = input_dir / Path("train_sentiment/")
mp_train = input_dir / Path("train_metadata/")
sp_test = input_dir / Path("test_sentiment/")
mp_test = input_dir / Path("test_metadata/")


# In[ ]:


train_pet_ids = train.PetID.unique()
test_pet_ids = test.PetID.unique()


# In[ ]:


train_sentiment_df, train_metadata_df = load_additional_features(
    train_pet_ids, sp_train, mp_train)

test_sentiment_df, test_metadata_df = load_additional_features(
    test_pet_ids, sp_test, mp_test)


# ## Aggregate sentiment data and metadata

# In[ ]:


train_meta_gr, train_meta_desc = aggregate_metadata(train_metadata_df)
test_meta_gr, test_meta_desc = aggregate_metadata(test_metadata_df)
train_sentiment_gr, train_sentiment_desc =     aggregate_sentiment(train_sentiment_df)
test_sentiment_gr, test_sentiment_desc =     aggregate_sentiment(test_sentiment_df)


# ## Merge processed DataFrames with base train/test DataFrame

# In[ ]:


train_proc = train.copy()
train_proc = train_proc.merge(
    train_sentiment_gr, how="left", left_on="PetID", right_on="pet_id")
train_proc = train_proc.merge(
    train_meta_gr, how="left", left_on="PetID", right_on="pet_id")
train_proc = train_proc.merge(
    train_sentiment_desc, how="left", left_on="PetID", right_on="pet_id")
train_proc = train_proc.merge(
    train_meta_desc, how="left", left_on="PetID", right_on = "pet_id")

test_proc = test.copy()
test_proc = test_proc.merge(
    test_sentiment_gr, how="left", left_on="PetID", right_on="pet_id")
test_proc = test_proc.merge(
    test_meta_gr, how="left", left_on="PetID", right_on="pet_id")
test_proc = test_proc.merge(
    test_sentiment_desc, how="left", left_on="PetID", right_on="pet_id")
test_proc = test_proc.merge(
    test_meta_desc, how="left", left_on="PetID", right_on = "pet_id")


# In[ ]:


print(train_proc.shape, test_proc.shape)
assert train_proc.shape[0] == train.shape[0]
assert test_proc.shape[0] == test.shape[0]


# In[ ]:


train_proc.drop(train_proc.filter(
    regex="pet_id", axis=1).columns.tolist(), 
    axis=1, 
    inplace=True)

test_proc.drop(test_proc.filter(
    regex="pet_id", axis=1).columns.tolist(),
    axis=1,
    inplace=True)

train_proc.head()


# In[ ]:


train_proc.language.fillna("", inplace=True)
test_proc.language.fillna("", inplace=True)

langs = train_proc.language.unique()
encode_dict = {k: i for i, k in enumerate(langs)}

train_proc.language = train_proc.language.map(encode_dict)
test_proc.language = test_proc.language.map(encode_dict)


# ## Feature engineering

# In[ ]:


X = pd.concat([train_proc, test_proc], ignore_index=True, sort=False)
X_temp = X.copy()

text_columns = [
    "Description",
    "entity",
    "desc"]
categorical_columns = [
    "Type", "Breed1", "Breed2", "Gender",
    "Color1", "Color2", "Color3", "MaturitySize",
    "FurLength", "Vaccinated", "Dewormed", "Sterilized",
    "State", "language"
]
drop_columns = [
    "PetID", "Name", "RescuerID"
]


# In[ ]:


rescuer_count = X.groupby(["RescuerID"])["PetID"].count().reset_index()
rescuer_count.columns = ["RescuerID", "RescuerID_COUNT"]

X_temp = X_temp.merge(rescuer_count, how="left", on="RescuerID")


# In[ ]:


X_text = X_temp[text_columns]
for i in X_text.columns:
    X_text.loc[:, i] = X_text.loc[:, i].fillna("none")


# In[ ]:


X_temp["len_description"] = X_text["Description"].map(len)
X_temp["len_meta_desc"] = X_text["desc"].map(len)
X_temp["len_entity"] = X_text["entity"].map(len)


# ## Tfidf

# In[ ]:


n_components = 16
text_features = []

for i in X_text.columns:
    print(f"generating features from: {i}")
    tfv = TfidfVectorizer(
        min_df=2,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 3),
        use_idf=1,
        smooth_idf=1,
        sublinear_tf=1)
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=1337)
    tfidf_col = tfv.fit_transform(X_text.loc[:, i].values)
    svd_col = svd.fit_transform(tfidf_col)
    svd_col = pd.DataFrame(svd_col)
    svd_col = svd_col.add_prefix("Tfidf_{}_".format(i))
    
    text_features.append(svd_col)

text_features = pd.concat(text_features, axis=1)
X_temp = pd.concat([X_temp, text_features], axis=1)

for i in X_text.columns:
    X_temp.drop(i, axis=1, inplace=True)


# ## Image size features

# In[ ]:


import os
import glob

train_image_files = sorted(
    glob.glob("../input/petfinder-adoption-prediction/train_images/*.jpg"))
test_image_files = sorted(
    glob.glob("../input/petfinder-adoption-prediction/test_images/*.jpg"))

train_df_imgs = pd.DataFrame(train_image_files)
test_df_imgs = pd.DataFrame(test_image_files)
train_df_imgs.columns = ["image_file_name"]
test_df_imgs.columns = ["image_file_name"]

train_imgs_pets = train_df_imgs["image_file_name"].apply(
    lambda x: x.split("/")[-1].split("-")[0])
test_imgs_pets = test_df_imgs["image_file_name"].apply(
    lambda x: x.split("/")[-1].split("-")[0])
train_df_imgs = train_df_imgs.assign(PetID=train_imgs_pets)
test_df_imgs = test_df_imgs.assign(PetID=test_imgs_pets)

def get_size(filename):
    st = os.stat(filename)
    return st.st_size

def get_dimensions(filename):
    img_size = Image.open(filename).size
    return img_size

train_df_imgs["image_size"] = train_df_imgs["image_file_name"].apply(get_size)
test_df_imgs["image_size"] = test_df_imgs["image_file_name"].apply(get_size)
train_df_imgs["temp_size"] = train_df_imgs["image_file_name"].apply(get_dimensions)
test_df_imgs["temp_size"] = test_df_imgs["image_file_name"].apply(get_dimensions)
train_df_imgs["width"] = train_df_imgs["temp_size"].apply(lambda x: x[0])
test_df_imgs["width"] = test_df_imgs["temp_size"].apply(lambda x: x[0])
train_df_imgs["height"] = train_df_imgs["temp_size"].apply(lambda x: x[1])
test_df_imgs["height"] = test_df_imgs["temp_size"].apply(lambda x: x[1])
train_df_imgs.drop(["temp_size"], axis=1, inplace=True)
test_df_imgs.drop(["temp_size"], axis=1, inplace=True)

aggs = {
    "image_size": ["sum", "mean", "var"],
    "width": ["sum", "mean", "var"],
    "height": ["sum", "mean", "var"]
}
agg_train_imgs = train_df_imgs.groupby("PetID").agg(aggs)
new_columns = [
    k + "_" + agg for k in aggs.keys() for agg in aggs[k]
]
agg_test_imgs = test_df_imgs.groupby("PetID").agg(aggs)
agg_train_imgs.columns = new_columns
agg_train_imgs = agg_train_imgs.reset_index()

agg_test_imgs.columns = new_columns
agg_test_imgs = agg_test_imgs.reset_index()


# In[ ]:


agg_imgs = pd.concat([agg_train_imgs, agg_test_imgs], axis=0).reset_index(drop=True)
X_temp = X_temp.merge(agg_imgs, how="left", on="PetID")
X_temp.head()


# ## Drop columns

# In[ ]:


X_temp.drop(drop_columns, axis=1, inplace=True)
X_temp.head()


# In[ ]:


for c in new_columns:
    X_temp.loc[:, c] = X_temp[c].map(lambda x: np.log1p(x))
    
X_temp.head()


# In[ ]:


X_train = X_temp.loc[np.isfinite(X_temp.AdoptionSpeed), :]
X_test = X_temp.loc[~np.isfinite(X_temp.AdoptionSpeed), :]
X_test.drop(["AdoptionSpeed"], axis=1, inplace=True)

assert X_train.shape[0] == train.shape[0]
assert X_test.shape[0] == test.shape[0]


# In[ ]:


train_cols = X_train.columns.tolist()
train_cols.remove("AdoptionSpeed")

test_cols = X_test.columns.tolist()
assert np.all(train_cols == test_cols)


# In[ ]:


X_train_non_null = X_train.fillna(-1)
X_test_non_null = X_test.fillna(-1)

X_train_non_null.isnull().any().any(), X_test_non_null.isnull().any().any()


# In[ ]:


X_train_non_null.Fee = X_train_non_null.Fee.map(lambda x: np.log1p(x))
X_test_non_null.Fee = X_test_non_null.Fee.map(lambda x: np.log1p(x))

X_train_non_null.RescuerID_COUNT = X_train_non_null.RescuerID_COUNT.map(lambda x: np.log(x))
X_test_non_null.RescuerID_COUNT = X_test_non_null.RescuerID_COUNT.map(lambda x: np.log(x))


# In[ ]:


cat_features = ["Type", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3",
                "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized",
                "Health", "Quantity", "State", "language"]

X_train_cat = X_train_non_null.loc[:, cat_features]
X_test_cat = X_test_non_null.loc[:, cat_features]


# ## Categorical Target Encoding

# In[ ]:


y = X_train_non_null.AdoptionSpeed
X_train_cat_y = X_train_cat.copy()
X_train_cat_y["AdoptionSpeed"] = y

kfold = GroupKFold(n_splits=5)
X_train_cat_encoded = np.zeros((X_train_cat.shape[0], len(cat_features) * 2))
X_test_cat_encoded = np.zeros((X_test_cat.shape[0], len(cat_features) * 2))

for trn_idx, val_idx in kfold.split(X_train_cat, y, train_pet_ids):
    X_trn, X_val = X_train_cat_y.loc[trn_idx, :], X_train_cat_y.loc[val_idx, :]
    for j, c in enumerate(cat_features):
        X_trn_enc = X_trn.groupby(c).agg({
            "AdoptionSpeed": ["mean", "std"]
        })
        cte_columns = [f"{c}_{x}" for x in ["mean", "std"]]
        X_trn_enc.columns = cte_columns
        X_temp = np.zeros((X_test_cat.shape[0], 2))
        X_temp_df = pd.DataFrame(data=X_temp, columns=cte_columns)
        for x in X_trn_enc.columns:
            X_val[x] = X_val[c].map(X_trn_enc[x])
            X_temp_df[x] = X_test_cat[c].map(X_trn_enc[x]).reset_index(drop=True)
        X_train_cat_encoded[val_idx, 2 * j:2 * (j + 1)] = X_val[X_trn_enc.columns].values
        X_test_cat_encoded[:, 2 * j: 2 * (j + 1)] += X_temp_df.values / 5


# In[ ]:


X_train_cat_encoded.shape, X_test_cat_encoded.shape


# In[ ]:


columns = [f"{c}_{x}" for c in cat_features for x in ["mean", "std"]]
X_train_cat_encoded_df = pd.DataFrame(data=X_train_cat_encoded, columns=columns)
X_test_cat_encoded_df = pd.DataFrame(data=X_test_cat_encoded, columns=columns)
X_test_cat_encoded_df.head()


# ## Numerical Features

# In[ ]:


X_train_num = X_train_non_null.drop(cat_features + ["AdoptionSpeed"], axis=1)
X_test_num = X_test_non_null.drop(cat_features, axis=1)

target = X_train_non_null["AdoptionSpeed"]


# In[ ]:


X_train_num = pd.concat([X_train_num, X_train_cat_encoded_df], axis=1)

X_test_cat_encoded_df.index = X_test_num.index
X_test_num = pd.concat([X_test_num, X_test_cat_encoded_df], axis=1)


# In[ ]:


X_all_num = pd.concat([X_train_num, X_test_num], axis=0)
ss = StandardScaler()
ss.fit(X_all_num)

X_train_ss = ss.transform(X_train_num)
X_test_ss = ss.transform(X_test_num)


# In[ ]:


X_train_ss[np.isnan(X_train_ss)] = 0.0
X_test_ss[np.isnan(X_test_ss)] = 0.0


# In[ ]:


cat_cat = pd.concat([X_train_cat, X_test_cat])

n_breed1 = cat_cat["Breed1"].nunique()
n_breed2 = cat_cat["Breed2"].nunique()
n_langs = cat_cat["language"].nunique()
n_color1 = cat_cat["Color1"].nunique()
n_color2 = cat_cat["Color2"].nunique()
n_color3 = cat_cat["Color3"].nunique()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
for c in X_train_cat.columns:
        le = LabelEncoder()
        le.fit(cat_cat[c])
        X_train_cat[c] = le.transform(X_train_cat[c])
        X_test_cat[c] = le.transform(X_test_cat[c])


# ## Metrics

# In[ ]:


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0
    
    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return -cohen_kappa_score(y, preds, weights='quadratic')
    
    def fit(self, X, y, initial_coef=[]):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = initial_coef
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
    
    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return preds
    
    def coefficients(self):
        return self.coef_['x']
    
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    assert len(rater_a) == len(rater_b)

    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_rating = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_rating)] for j in range(num_rating)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    rater_a = y
    rater_b = y_pred
    min_rating = None
    max_rating = None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)

    assert len(rater_a) == len(rater_b)

    min_rating = min(min(rater_a), min(rater_b))
    max_rating = max(max(rater_a), max(rater_b))

    conf_mat = confusion_matrix(rater_a, rater_b, min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (
                hist_rater_a[i] * hist_rater_b[j]) / num_scored_items
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


# ## DataLoader

# In[ ]:


class PetDataset(data.Dataset):
    def __init__(self, 
                 pet_ids, 
                 cat_features, 
                 num_features, 
                 labels, 
                 root_dir, 
                 subset=False, 
                 transform=None):
        self.pet_ids = pet_ids
        self.cat_features = cat_features
        self.num_features = num_features
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.pet_ids)
    
    def __getitem__(self, idx):
        img_name = f"{self.pet_ids[idx]}-1.jpg"
        fullname = self.root_dir / Path(img_name)
        try:
            image = Image.open(fullname).convert("RGB")
        except FileNotFoundError:
            image = np.zeros((3, 224, 224), dtype=np.uint8).transpose(1, 2, 0)
            image = Image.fromarray(np.uint8(image))
        cat_feature = self.cat_features[idx]
        num_feature = self.num_features[idx]
        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return [image, cat_feature, num_feature, label]
        else:
            return [image, cat_feature, num_feature]


# In[ ]:


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
ds_trans = transforms.Compose([transforms.Resize(224),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize])


# ## Trainer

# In[ ]:


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def run_xgb(params,
            X,
            y,
            X_test,
            resc,
            n_splits=10,
            num_rounds=60000,
            early_stop=500,
            verbose_eval=1000):
    fold = GroupKFold(n_splits=n_splits)
    oof_train = np.zeros((X.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))

    for i, (trn_index, val_index) in enumerate(fold.split(X, y, train_pet_ids)):
        X_tr = X.iloc[trn_index, :]
        X_val = X.iloc[val_index, :]

        y_tr = y[trn_index]
        y_val = y[val_index]
        d_train = xgb.DMatrix(
            data=X_tr, label=y_tr, feature_names=X_tr.columns)
        d_valid = xgb.DMatrix(
            data=X_val, label=y_val, feature_names=X_val.columns)

        watchlist = [(d_train, "train"), (d_valid, "valid")]
        model = xgb.train(
            params=params,
            dtrain=d_train,
            num_boost_round=num_rounds,
            evals=watchlist,
            early_stopping_rounds=early_stop,
            verbose_eval=verbose_eval)
        valid_pred = model.predict(
            xgb.DMatrix(X_val, feature_names=X_val.columns),
            ntree_limit=model.best_ntree_limit)
        test_pred = model.predict(
            xgb.DMatrix(X_test, feature_names=X_test.columns),
            ntree_limit=model.best_ntree_limit)
        oof_train[val_index] = valid_pred
        oof_test[:, i] = test_pred
    return model, oof_train, oof_test


class Trainer:
    def __init__(self, 
                 model,
                 resc,
                 n_splits=5, 
                 seed=42, 
                 device="cuda:0", 
                 train_batch=16,
                 val_batch=32,
                 kwargs={}):
        self.model = model
        self.resc = resc
        self.n_splits = n_splits
        self.seed = seed
        self.device = device
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.kwargs = kwargs
        
        self.fold = GroupKFold(
            n_splits=n_splits)
        self.best_score = None
        self.tag = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        self.loss_fn = nn.MSELoss(reduction="mean").to(self.device)
        path = Path(f"bin/{self.tag}")
        path.mkdir(exist_ok=True, parents=True)
        self.path = path
        
    def fit(self, pet_ids, cat_feats, num_feats, answer, n_epochs=30):
        self.train_preds = np.zeros((train.shape[0]))
        answer = answer.values
        cat_feats = cat_feats.values
        for i, (trn_idx, val_idx) in enumerate(self.fold.split(pet_ids, answer, self.resc)):
            self.fold_num = i
            print(f"Fold: {i+1}")
            pid_train, pid_val = pet_ids[trn_idx], pet_ids[val_idx]
            cat_train, cat_val = cat_feats[trn_idx], cat_feats[val_idx]
            num_train, num_val = num_feats[trn_idx], num_feats[val_idx]
            y_train, y_val = answer[trn_idx] / 4, answer[val_idx] / 4
            
            valid_preds = self._fit(pid_train, 
                                    cat_train, 
                                    num_train, 
                                    y_train,
                                    n_epochs,
                                    pid_val,
                                    cat_val,
                                    num_val,
                                    y_val
                                   )
            self.train_preds[val_idx] = valid_preds
        
    def _fit(self, pid, cat, num, y, n_epochs, pid_val, cat_val, num_val, y_val):
        seed_torch(self.seed)
        cat_tensor = torch.tensor(cat, dtype=torch.long).to(self.device)
        num_tensor = torch.tensor(num, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y[:, np.newaxis], dtype=torch.float32).to(self.device)
        train = PetDataset(pid, 
                           cat_tensor, 
                           num_tensor, 
                           y_tensor,
                           "../input/petfinder-adoption-prediction/train_images/",
                           transform=ds_trans)
        train_loader = data.DataLoader(train, 
                                       batch_size=self.train_batch, shuffle=True)
        cat_eval = torch.tensor(cat_val, dtype=torch.long).to(self.device)
        num_eval = torch.tensor(num_val, dtype=torch.float32).to(self.device)
        y_eval = torch.tensor(y_val[:, np.newaxis], dtype=torch.float32).to(self.device)
        eval_ = PetDataset(pid_val,
                           cat_eval,
                           num_eval,
                           y_eval,
                           "../input/petfinder-adoption-prediction/train_images/",
                           transform=ds_trans)
        eval_loader = data.DataLoader(eval_,
                                      batch_size=self.val_batch, shuffle=False)
        
        model = self.model(**self.kwargs)
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters())
        best_score = np.inf
        
        for epoch in range(n_epochs):
            model.train()
            avg_loss = 0.
            for i_batch, c_batch, n_batch, y_batch in tqdm(train_loader):
                i_batch = i_batch.to(self.device)
                y_pred = model(i_batch, c_batch, n_batch)
                loss = self.loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
            valid_preds, avg_val_loss = self._val(eval_loader, model)
            print(f"epoch {epoch+1}/{n_epochs}")
            print(f"avg_loss: {avg_loss:.4f}")
            print(f"avg_val_loss: {avg_val_loss:.4f}")
            if best_score > avg_val_loss:
                torch.save(model.state_dict(),
                           self.path / f"best{self.fold_num}.pt")
                print(f"Save model on epoch {epoch + 1}")
                best_score = avg_val_loss
        model.load_state_dict(torch.load(self.path / f"best{self.fold_num}.pt"))
        valid_preds, avg_val_loss = self._val(eval_loader, model)
        print(f"Validation loss: {avg_val_loss}")
        return valid_preds
    
    def _val(self, loader, model):
        model.eval()
        valid_preds = np.zeros(loader.dataset.cat_features.size(0))
        avg_val_loss = 0.

        for i, (i_batch, c_batch, n_batch, y_batch) in enumerate(loader):
            with torch.no_grad():
                i_batch = i_batch.to(self.device)
                y_pred = model(i_batch, c_batch, n_batch).detach()
                avg_val_loss += self.loss_fn(y_pred,
                                             y_batch).item() / len(loader)
                valid_preds[i * self.val_batch:(i + 1) * self.val_batch] = y_pred.cpu().numpy()[:, 0]
        return valid_preds, avg_val_loss
        


# In[ ]:


class NeuralNet(nn.Module):
    def __init__(self, path, emb_dims, num_dims, img_linear, linear_size):
        super(NeuralNet, self).__init__()
        self.densenet121 = models.densenet121()
        self.densenet121.load_state_dict(torch.load(path))
        self.densenet121.classifier = nn.Linear(1024, img_linear)
        dense = nn.Sequential(*list(self.densenet121.children())[:-1])
        for param in dense.parameters():
            param.requires_grad = False

        self.embeddings = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in emb_dims])
        n_emb_out = sum([y for x, y in emb_dims])
        self.fc1 = nn.Linear(img_linear + n_emb_out + num_dims, linear_size)
        self.bn1 = nn.BatchNorm1d(linear_size)
        self.fc2 = nn.Linear(linear_size, 1)
        self.drop = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, i, c, n):
        img_feats = self.densenet121(i)
        emb = [
            emb_layer(c[:, j]) for j, emb_layer in enumerate(self.embeddings)
        ]
        emb = torch.cat(emb, 1)
        data = torch.cat([img_feats, emb, n], 1)
        out = self.relu(self.fc1(data))
        out = self.bn1(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out


# ## Train

# In[ ]:


emb_dims = [(2, 1), (n_breed1, 3), (n_breed2, 3), (3, 1), (n_color1, 1),
            (n_color2, 1), (n_color3, 1), (4, 1), (3, 1), (3, 1),
            (3, 1), (3, 1), (3, 1), (19, 1), (14, 1),(n_langs, 1)]
num_dims = X_test_ss.shape[1]
trainer = Trainer(
    NeuralNet,
    resc=train_pet_ids,
    n_splits=4,
    train_batch=128,
    val_batch=128,
    kwargs={
        "path": "densenet121.pth",
        "emb_dims": emb_dims,
        "num_dims": num_dims,
        "img_linear": 48,
        "linear_size": 120
    })


# In[ ]:


trainer.fit(train_pet_ids, X_train_cat, X_train_ss, target, 5)


# In[ ]:


bin_path = trainer.path
test_preds = np.zeros((X_test_cat.shape[0]))
c_tensor = torch.tensor(X_test_cat.values, dtype=torch.long).to(trainer.device)
n_tensor = torch.tensor(X_test_ss, dtype=torch.float32).to(trainer.device)
test_dataset = PetDataset(test_pet_ids, 
                          c_tensor, 
                          n_tensor, 
                          labels=None, 
                          root_dir="../input/petfinder-adoption-prediction/test_images/",
                          transform=ds_trans)
test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False)

for path in bin_path.iterdir():
    print(f"using {str(path)}")
    model = NeuralNet(**trainer.kwargs)
    model.to("cuda:0")
    model.load_state_dict(torch.load(path))

    model.eval()
    temp = np.zeros((X_test_cat.shape[0]))
    for i, (i_batch, c_batch, n_batch) in enumerate(test_loader):
        i_batch = i_batch.to(trainer.device)
        with torch.no_grad():
            y_pred = model(i_batch, c_batch, n_batch).detach()
            temp[i * 128:(i + 1) * 128] = y_pred.cpu().numpy()[:, 0]
    test_preds += temp / trainer.n_splits


# ## Image

# In[ ]:


class ImageDataset(data.Dataset):
    def __init__(self, pet_ids, root_dir, transform=None):
        self.pet_ids = pet_ids
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.pet_ids)
    
    def __getitem__(self, idx):
        img_name = f"{self.pet_ids[idx]}-1.jpg"
        fullname = self.root_dir / Path(img_name)
        try:
            image = Image.open(fullname).convert("RGB")
        except FileNotFoundError:
            image = np.zeros((3, 224, 224), dtype=np.uint8).transpose(1, 2, 0)
            image = Image.fromarray(np.uint8(image))
        if self.transform:
            image = self.transform(image)
            
        return [image]


# In[ ]:


train_dataset = ImageDataset(
    train_pet_ids,
    "../input/petfinder-adoption-prediction/train_images/",
    transform=ds_trans)
batch = 256
n_img_dim = 48
train_loader = data.DataLoader(train_dataset,
                               batch_size=batch,
                               shuffle=False)
X_train_img = np.zeros((len(train_pet_ids), n_img_dim))

test_dataset = ImageDataset(
    test_pet_ids,
    "../input/petfinder-adoption-prediction/test_images/",
    transform=ds_trans)
test_loader = data.DataLoader(test_dataset,
                              batch_size=batch,
                              shuffle=False)
X_test_img = np.zeros((len(test_pet_ids), n_img_dim))
bin_path = trainer.path
for path in bin_path.iterdir():
    model = NeuralNet(**trainer.kwargs)
    model.to("cuda:0")
    model.load_state_dict(torch.load(path))
    model.eval()
    temp = np.zeros((len(train_pet_ids), n_img_dim))
    
    for i, (i_batch, ) in tqdm(enumerate(train_loader)):
        with torch.no_grad():
            i_batch = i_batch.to("cuda:0")
            y_pred = model.densenet121(i_batch).detach()
            temp[i * batch:(i + 1) * batch, :] = y_pred.cpu().numpy()
    X_train_img += temp / trainer.n_splits
    
    temp = np.zeros((len(test_pet_ids), n_img_dim))
    for i, (i_batch, ) in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            i_batch = i_batch.to("cuda:0")
            y_pred = model.densenet121(i_batch).detach()
            temp[i * batch:(i + 1) * batch, :] = y_pred.cpu().numpy()
    X_test_img += temp / trainer.n_splits


# In[ ]:


X_train_img.shape, X_test_img.shape


# In[ ]:


train_img = pd.DataFrame(data=X_train_img, columns=[
    f"img{i}" for i in range(X_train_img.shape[1])
])
test_img = pd.DataFrame(data=X_test_img, columns=[
    f"img{i}" for i in range(X_test_img.shape[1])
])


# In[ ]:


num_columns = X_train_num.columns
X_train_num_df = pd.DataFrame(data=X_train_ss, columns=num_columns)
X_test_num_df = pd.DataFrame(data=X_test_ss, columns=num_columns)

X_test_num_df.index = X_test_cat.index
test_img.index = X_test_cat.index

X_train_all = pd.concat([X_train_num_df, X_train_cat, train_img], axis=1)
X_test_all = pd.concat([X_test_num_df, X_test_cat, test_img], axis=1)

print(X_train_all.shape, X_test_all.shape)
X_train_all.head()


# In[ ]:


"AdoptionSpeed" in X_train_all.columns, "AdoptionSpeed" in X_test_all.columns


# In[ ]:


X_train_all.columns.tolist() == X_test_all.columns.tolist()


# ## Category Embedding

# In[ ]:


class CategoryDataset(data.Dataset):
    def __init__(self, category):
        self.category = category
        
    def __len__(self):
        return len(self.category)
    
    def __getitem__(self, idx):
        category = self.category[idx, :]
        return [category]


# In[ ]:


c_train = torch.tensor(X_train_cat.values, dtype=torch.long).to("cuda:0")
c_test = torch.tensor(X_test_cat.values, dtype=torch.long).to("cuda:0")
train_dataset = CategoryDataset(c_train)
test_dataset = CategoryDataset(c_test)
train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False)

X_train_cat_ = np.zeros((len(train_pet_ids), 20))
for i, (c_batch, ) in tqdm(enumerate(train_loader)):
    with torch.no_grad():
        y_pred = [model.embeddings[i](c_batch[:, i]) for i in range(len(model.embeddings))]
        y_pred =torch.cat(y_pred, 1).detach()
        X_train_cat_[i * 128:(i + 1) * 128, :] = y_pred.cpu().numpy()
        
X_test_cat_ = np.zeros((len(test_pet_ids), 20))
for i, (c_batch, ) in tqdm(enumerate(test_loader)):
    with torch.no_grad():
        y_pred = [model.embeddings[i](c_batch[:, i]) for i in range(len(model.embeddings))]
        y_pred = torch.cat(y_pred, 1).detach()
        X_test_cat_[i * 128:(i + 1) * 128, :] = y_pred.cpu().numpy()


# In[ ]:


X_train_cat_.shape, X_test_cat_.shape


# In[ ]:


train_emb = pd.DataFrame(data=X_train_cat_, columns=[
    f"emb{i}" for i in range(X_train_cat_.shape[1])
])
test_emb = pd.DataFrame(data=X_test_cat_, columns=[
    f"emb{i}" for i in range(X_test_cat_.shape[1])
])


# In[ ]:


test_emb.index = X_test_all.index

X_train_all = pd.concat([X_train_all, train_emb], axis=1)
X_test_all = pd.concat([X_test_all, test_emb], axis=1)

print(X_train_all.shape, X_test_all.shape)
X_train_all.head()


# In[ ]:


xgb_params = {
    "eval_metric": "rmse",
    "seed": 1337,
    "eta": 0.01,
    "subsample": 0.75,
    "colsample_bytree": 0.85,
    "tree_method": "gpu_hist",
    "device": "gpu",
    "silent": 1
}

xgb_X = X_train_all
xgb_y = target
xgb_X_test = X_test_all

model, oof_train_xgb, oof_test_xgb= run_xgb(
    xgb_params, 
    xgb_X, 
    xgb_y, 
    xgb_X_test,
    resc=train_pet_ids,
    n_splits=5,
    num_rounds=10000)


# ## Run LGBM

# In[ ]:


def run_lgb(params,
            X,
            y,
            X_test,
            resc,
            cat_features,
            n_splits=10,
            early_stop=500):
    fold = GroupKFold(n_splits=n_splits)
    oof_train = np.zeros((X.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_splits))

    for i, (trn_index, val_index) in enumerate(fold.split(X, y, resc)):
        X_tr = X.iloc[trn_index, :]
        X_val = X.iloc[val_index, :]

        y_tr = y[trn_index]
        y_val = y[val_index]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, 
                  y_tr, 
                  eval_set=(X_val, y_val),
                  verbose=500,
                  early_stopping_rounds=early_stop,
                  categorical_feature=cat_features)
        valid_pred = model.predict(X_val, num_iteration=model.best_iteration_)
        test_pred = model.predict(X_test, num_iteration=model.best_iteration_)
        oof_train[val_index] = valid_pred
        oof_test[:, i] = test_pred
    return model, oof_train, oof_test


# In[ ]:


lgb_params = {
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.01,
    "n_estimators": 10000,
    "subsample": 0.75,
    "subsample_freq": 8,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.01,
    "reg_lambda": 0.01,
    "n_jobs": -1
}

model, oof_train, oof_test = run_lgb(
    lgb_params, 
    xgb_X, 
    xgb_y, 
    xgb_X_test,
    train_pet_ids,
    cat_features,
    n_splits=5,
    early_stop=500)


# ## Post process

# In[ ]:


def plot_pred(pred):
    sns.distplot(pred, kde=True, hist_kws={"range": [0, 5]})


# In[ ]:


plot_pred(oof_train)


# In[ ]:


plot_pred(oof_train_xgb)


# In[ ]:


oof_train.shape


# In[ ]:


plot_pred(oof_test.mean(axis=1))


# In[ ]:


plot_pred(oof_test_xgb.mean(1))


# In[ ]:


plot_pred(trainer.train_preds * 4)


# In[ ]:


plot_pred(test_preds * 4)


# In[ ]:


lgb_xgb = 0.5 * oof_train + 0.5 * oof_train_xgb
lgb_xgb_test = 0.5 * oof_test + 0.5 * oof_test_xgb
plot_pred(lgb_xgb)
plot_pred(lgb_xgb_test.mean(1))


# In[ ]:


nn_preds = np.clip(trainer.train_preds, a_min=0.0, a_max=1.0)
nn_preds_test = np.clip(test_preds, a_min=0.0, a_max=1.0)
plot_pred(nn_preds * 4)
plot_pred(nn_preds_test * 4)


# In[ ]:


lgb_xgb_nn = 0.6 * lgb_xgb + 0.4 * (nn_preds * 4)
lgb_xgb_nn_test = 0.6 * lgb_xgb_test.mean(1) + 0.4 * nn_preds_test * 4


# In[ ]:


plot_pred(lgb_xgb_nn)


# In[ ]:


plot_pred(lgb_xgb_nn_test)


# In[ ]:


opt = OptimizedRounder()
opt.fit(lgb_xgb, target, [1.5, 2.0, 2.5, 3.5])
coeff = opt.coefficients()
valid_pred = opt.predict(lgb_xgb, coeff)
qwk = quadratic_weighted_kappa(xgb_y, valid_pred)
print("QWK = ", qwk)
coeffs = coeff.copy()
train_predictions = opt.predict(lgb_xgb, coeffs).astype(np.int8)
print(f"train_preds: {Counter(train_predictions)}")
test_predictions = opt.predict(lgb_xgb_test.mean(1), coeffs).astype(np.int8)
print(f"test_preds: {Counter(test_predictions)}")
submission = pd.DataFrame({"PetID": test.PetID.values, "AdoptionSpeed": test_predictions})
submission.to_csv("submission_xgb_lgb.csv", index=False)
submission.head()


# In[ ]:


opt = OptimizedRounder()
opt.fit(lgb_xgb_nn, target, [1.5, 2.0, 2.5, 3.5])
coeff = opt.coefficients()
valid_pred = opt.predict(lgb_xgb_nn, coeff)
qwk = quadratic_weighted_kappa(xgb_y, valid_pred)
print("QWK = ", qwk)
coeffs = coeff.copy()
train_predictions = opt.predict(lgb_xgb_nn, coeffs).astype(np.int8)
print(f"train_preds: {Counter(train_predictions)}")
test_predictions = opt.predict(lgb_xgb_nn_test, coeffs).astype(np.int8)
print(f"test_preds: {Counter(test_predictions)}")
submission = pd.DataFrame({"PetID": test.PetID.values, "AdoptionSpeed": test_predictions})
submission.to_csv("submission.csv", index=False)
submission.head()


# In[ ]:


opt = OptimizedRounder()
opt.fit(nn_preds * 4, target, [1.5, 2.0, 2.5, 3.5])
coeff = opt.coefficients()
valid_pred = opt.predict(nn_preds * 4, coeff)
qwk = quadratic_weighted_kappa(target, valid_pred)
print("QWK = ", qwk)
coeffs = coeff.copy()
train_predictions = opt.predict(nn_preds * 4, coeffs).astype(np.int8)
print(f"train_preds: {Counter(train_predictions)}")
test_predictions = opt.predict(nn_preds_test * 4, coeffs).astype(np.int8)
print(f"test_preds: {Counter(test_predictions)}")
submission = pd.DataFrame({"PetID": test.PetID.values, "AdoptionSpeed": test_predictions})
submission.to_csv("submission_nn.csv", index=False)
submission.head()


# In[ ]:




