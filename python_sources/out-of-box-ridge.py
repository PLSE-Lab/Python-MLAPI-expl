import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import Ridge

train = pd.read_csv("../input/train.csv")
y = train["loss"].values
train.drop(labels = ["id", "loss"], axis = 1, inplace = True)
num_train = len(y)

test = pd.read_csv("../input/test.csv")
test_id = test["id"]
test.drop(labels = "id", axis = 1, inplace = True)

data = pd.concat([train, test], axis = 0)

categorical_columns = [colname for colname in data.columns.tolist() if colname.startswith('cat')]
numeric_columns = [colname for colname in data.columns.tolist() if colname.startswith('cont')]

cat_col_encoders = {}
for i, cat_col in enumerate(categorical_columns):
	cat_col_encoder = LabelEncoder()
	data[cat_col] = cat_col_encoder.fit_transform(data[cat_col])
	cat_col_encoders[cat_col] = cat_col_encoder

cat_encoder = OneHotEncoder()
cat_data = cat_encoder.fit_transform(data[categorical_columns])

pdata = csr_matrix(hstack([cat_data, data[numeric_columns]]))

train = pdata[:num_train, :]
test = pdata[num_train:, :]

RidgeReg = Ridge()
RidgeReg.fit(X = train, y = y)

pd.DataFrame({"id" : test_id.values, "loss" : RidgeReg.predict(test)}).to_csv("Ridge.csv", index = False)
