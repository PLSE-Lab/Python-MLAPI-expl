import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb


def fix_tiempo(t):
    if "día" not in t and "dia" not in t:
        return 1
    return int(t.split()[0])


def fix(df):
    df.Tiempo = df.Tiempo.apply(fix_tiempo)

    cv = CountVectorizer(strip_accents="unicode",
                         min_df=0.001)

    X = cv.fit_transform(df.pop("Modelo"))
    feat = pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
    feat.index = df.index
    df = pd.concat([df, feat], axis=1, sort=False)

    df.drop(["Localidad"], axis=1, inplace=True)
    df.rename(columns={"Año": "Anyo"}, inplace=True)

    # Label encoder
    cat = df.select_dtypes(include=[object]).columns
    le = LabelEncoder()
    df[cat] = df[cat].apply(lambda col: le.fit_transform(col.astype(str)))

    return df

DATA = "../input/murcia-car-challenge/"
print("Preparando Datos")
train = pd.read_csv(f"{DATA}train.csv", index_col="Id")
test = pd.read_csv(f"{DATA}test.csv", index_col="Id")
df = pd.concat([train, test], keys=[0, 1], sort=False)
df = fix(df)

x = df.xs(0)
test = df.xs(1)
test.pop("Precio")
y = np.log1p(x.pop("Precio"))


print(x.shape, y.shape, test.shape, x.head(), test.head())

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)
    
model = lgb.LGBMRegressor(n_jobs=8)

print("Entrenando PRUEBA")
model.fit(x_train, y_train)
preds = model.predict(x_valid)

acc = np.sqrt(mean_squared_log_error(np.expm1(y_valid), np.expm1(preds)))
print(acc)


print("Entrenando FINAL")
model.fit(x, y)
preds = model.predict(test)
preds = np.expm1(preds)

df = pd.read_csv(f"{DATA}sampleSubmission.csv")
df["Precio"] = preds
df.to_csv("{}.csv".format("predict"), index=False)
