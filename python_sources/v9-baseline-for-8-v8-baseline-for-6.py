import json
import string
from multiprocessing import Pool, cpu_count

import lightfm
import numpy as np
import pandas as pd
import pymorphy2
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(s):
    return " ".join(
        [morph.normal_forms(word)[0] for word in s.translate(table).lower().split()]
    )


def process_item_line(line):
    item = json.loads(line)
    if isinstance(item["image"], float):
        item["image"] = np.full((96,),0)
    else:
        item["image"] = np.array(item["image"])
    item["content"] = clean_text(item["content"])
    item["title"] = clean_text(item["title"])
    return item

chars_to_replace=string.punctuation + "«»\n—–"
table = str.maketrans(chars_to_replace, " "*len(chars_to_replace))
morph = pymorphy2.MorphAnalyzer()
items_list = []

with open("../input/items.json/items.json") as inf:
    with Pool(cpu_count()) as p:
        items_list = list(p.imap(process_item_line, inf))

items = pd.DataFrame(items_list).set_index("itemId")
num_users = 42977
num_items = len(items)
data = []
row = []
col = []
with open("../input/train.json/train.json") as inf:
    for i, line in enumerate(inf):
        j = json.loads(line)
        for item, rating in j["trainRatings"].items():
            data.append((-1) ** (int(rating) + 1))
            row.append(i)
            col.append(int(item))
train_int = scipy.sparse.coo_matrix((data, (row, col)))
print("created train interactions")
del data, row, col
vect_content = TfidfVectorizer(min_df=90, max_df=0.01, lowercase=False)
tfidf_content = vect_content.fit_transform(items.content)
print("transformed content")
vect_title = TfidfVectorizer(min_df=90, max_df=0.01, lowercase=False)
tfidf_title = vect_title.fit_transform(items.title)
print("transformed title")
identity_items = scipy.sparse.eye(num_items)
item_features = scipy.sparse.hstack(
    [identity_items, tfidf_content, tfidf_title], format="csr"
)
model = lightfm.LightFM(no_components=128, loss="logistic", random_state=0)
print("start training")
model.fit(train_int, epochs=7, num_threads=cpu_count(), item_features=item_features)
print("end training")
sample = pd.read_csv("../input/random_benchmark.csv")
sample["pred"] = model.predict(
    sample.userId.values,
    sample.itemId.values,
    item_features=item_features,
    num_threads=cpu_count(),
)
sample.sort_values(["userId", "pred"], ascending=[True, False], inplace=True)
sample.drop(columns=["pred"], inplace=True)
sample.to_csv("lightfm_tfidf.csv", index=False)
