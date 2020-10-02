"""
Based on
https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s/code
"""
import os; os.environ['OMP_NUM_THREADS'] = '1'
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
from typing import List, Dict

import keras as ks
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.model_selection import KFold

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(
        FunctionTransformer(itemgetter(f), validate=False), *vec
    )


def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient="records")


def fit_predict(xs, params, train_filled_cols, o_y_train, dropout=True):
    X_train, X_val, X_test = xs

    config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        use_per_session_threads=1,
        inter_op_parallelism_threads=1,
    )

    y_train = ks.utils.np_utils.to_categorical(o_y_train)
    
    # Weight correction
    sample_weight = np.array([params["blank_sample_weight"] if y == 0 else 1.0 for y in o_y_train])
    sample_weight = np.multiply(sample_weight, train_filled_cols)

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype="float32", sparse=True)
        out = ks.layers.Dense(params["nn_layer_1_nodes"], activation="relu")(model_in)

        if dropout:
            out = ks.layers.Dropout(params["nn_layer_1_2_dropout"])(out)

        out = ks.layers.Dense(params["nn_layer_2_nodes"], activation="relu")(out)
        out = ks.layers.Dense(params["nn_layer_3_nodes"], activation="relu")(out)
        out = ks.layers.Dense(len(y_train[0]), activation="softmax")(out)
        
        model = ks.Model(model_in, out)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=ks.optimizers.Adam(lr=params["adam_lr"]),
        )
        
        for i in range(params["epochs"]):
            model.fit(
                x=X_train,
                y=y_train,
                batch_size=2 ** (params["starting_batch_size"] + i),
                epochs=1,
                verbose=0,
                sample_weight=sample_weight,
            )

        return model.predict(X_val), model.predict(X_test)


def convert_list_to_space_sep(items):
    items = [str(int(float(x))) for x in items]
    return " ".join(items)
    

def get_top_k_preds(preds, predict=False, k=2):

    preds = (-preds).argsort(axis=1)[:, : (k + 1)]
    return [
        [item - 1 if predict else item for item in y_p if item != 0][:2]
        for y_p in preds
    ]
    
    
def create_features(df):
    titles = df["title"].str.split()
    df["title_length"] = titles.apply(len)
    df["title_uniques"] = titles.apply(set).apply(len)
    df["id_group"] = df["itemid"] // 1e7
    df["id_group"] = df["id_group"].astype("int32")
    return df

def nn_predict(
    o_train, o_test, params, cat=None, column=None,
):
    
    o_train = create_features(o_train)
    o_test = create_features(o_test)
    
    vectorizer = make_union(
        on_field(
            "title",
            Tfidf(
                max_features=params["tfidf_max_features"],
                token_pattern="\w+",
                ngram_range=(1, 2),
                max_df=0.3,
                use_idf=False,
                norm=None,
            ),
        ),
        on_field(
            "title",
            Tfidf(
                max_features=params["tfidf_3_grams"],
                token_pattern="\w+",
                ngram_range=(3, 3),
                use_idf=False,
                norm=None,
            ),
        ),
        on_field(
            ["title_length", "title_uniques", "id_group"],
            FunctionTransformer(to_records, validate=False),
            DictVectorizer(sparse=False),
            OneHotEncoder(),
        ),
        n_jobs=3,
    )

    with timer("process train"):
        print(f"Original: {o_train.shape}")
        cv = KFold(n_splits=500, shuffle=True, random_state=42)
        train_ids, valid_ids = next(cv.split(o_train))
        train, valid = o_train.iloc[train_ids], o_train.iloc[valid_ids]
        training_cols = [
                "itemid",
                "title",
                "image_path",
                "title_length",
                "title_uniques",
                "id_group",
            ]
        Y_train = train[list(set(train.columns) - set(training_cols))]
        Y_valid = valid[list(set(valid.columns) - set(training_cols))]

        # Adjust weight based on filled columns
        train_filled_cols = Y_train.count(axis=1)
        train_filled_cols_mean = train_filled_cols.mean()
        train_filled_cols = train_filled_cols / train_filled_cols_mean
        train_filled_cols = np.array(train_filled_cols)

        # Fill with 0
        Y_train, Y_valid = Y_train + 1, Y_valid + 1
        Y_train, Y_valid = Y_train.fillna(0), Y_valid.fillna(0)

        c_train = o_train[training_cols]
        combined = pd.concat([c_train, o_test], ignore_index=True, sort=False)

        vectorizer.fit(combined)
        X_train = vectorizer.transform(train).astype(np.float32)
        print(f"X_train: {X_train.shape} of {X_train.dtype}")

    with timer("process valid"):
        X_valid = vectorizer.transform(valid).astype(np.float32)
        print(f"X_valid: {X_valid.shape} of {X_valid.dtype}")

    with timer("process test"):
        X_test = vectorizer.transform(o_test).astype(np.float32)
        print(f"X_test: {X_test.shape} of {X_test.dtype}")

    xs = [[X_train, X_valid, X_test]] * 8
    output_tuples = []
    
    poor_filled_cols = set(["Camera", "Features", "Network Connections", "Phone Screen Size", "Skin_type"])
    cols = Y_train.columns if not column else [column]

    for col in sorted(cols):
        with timer(f"fit predict for {col}"):
            dropout = False if col in poor_filled_cols else True
            with ThreadPool(processes=4) as pool:
                preds = pool.map(
                    partial(
                        fit_predict,
                        params=params,
                        train_filled_cols=train_filled_cols,
                        o_y_train=Y_train[col],
                        dropout=dropout,
                    ),
                    xs,
                )
                y_pred = np.mean([p[1] for p in preds], axis=0)
    
                # Get top 2 preds
                y_pred = get_top_k_preds(y_pred, predict=True)
    
                output_tuples.extend([
                    (f"{itemid}_{col}", convert_list_to_space_sep(pred))
                    for itemid, pred in zip(o_test["itemid"], y_pred)
                ])

    o_df = pd.DataFrame(data=output_tuples, columns=["id", "n_tagging"])
    return o_df, None


params = {
    "tfidf_max_features": 100_000,
    "tfidf_3_grams": 500,
    "nn_layer_1_nodes": 256,
    "nn_layer_1_2_dropout": 0.3,
    "nn_layer_2_nodes": 64,
    "nn_layer_3_nodes": 64,
    "adam_lr": 5e-3,
    "epochs": 2,
    "starting_batch_size": 11,  # 2 ^ 11
    "blank_sample_weight": 0.5,
    "sample_filled_adjustment": 0.5,
}

mobile_df, _ = nn_predict(
    pd.read_csv("../input/mobile_data_info_train_competition.csv"),
    pd.read_csv("../input/mobile_data_info_val_competition.csv"),
    params=params,
    cat="mobile",
)

beauty_df, _ = nn_predict(
    pd.read_csv("../input/beauty_data_info_train_competition.csv"),
    pd.read_csv("../input/beauty_data_info_val_competition.csv"),
    params=params,
    cat="beauty",
)

fashion_df, _ = nn_predict(
    pd.read_csv("../input/fashion_data_info_train_competition.csv"),
    pd.read_csv("../input/fashion_data_info_val_competition.csv"),
    params=params,
    cat="fashion",
)

submit_df = pd.concat(
    [beauty_df, mobile_df, fashion_df],
    axis=0,
    ignore_index=True,
    sort=False,
)

submit_df["tagging"] = submit_df["n_tagging"]
submit_df = submit_df.drop(["n_tagging"], axis=1, errors="ignore")
submit_df.info()
submit_df.to_csv("nn_submit.csv", index=False)
