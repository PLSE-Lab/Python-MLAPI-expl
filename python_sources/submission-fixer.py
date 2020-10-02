import json
import pandas

data_dir = "../input/ndsc-advanced"

# Transforms the train rows into a list of bad ids
category = "fashion"
train = pandas.read_csv("%s/%s_data_info_train_competition.csv" % (data_dir, category))
with open("%s/%s_profile_train.json" % (data_dir, category)) as stream:
    metadata = json.load(stream)
bad_ids = list()
for label in metadata.keys():
    train["suffix"] = ("_%s" % label)
    train["id"] = train["itemid"].astype("str") + train["suffix"]
    bad_ids.append(train[["id"]].copy())
bad_ids = pandas.concat(bad_ids, axis=0).set_index("id")

# Deletes the leaked rows.
submission = pandas.read_csv("../input/fashion-data-leak-exploit/submission-out.csv", index_col="id")
bad_ids = list(set(bad_ids.index.values) & set(submission.index.values))
submission.drop(bad_ids, axis=0, inplace=True)
submission.to_csv("submission-fixed.csv")
print(submission.shape)