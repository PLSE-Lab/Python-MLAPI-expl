# imports
import pandas as pd
import sklearn.linear_model as skl
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as select
from sklearn.metrics import f1_score

# read data
DS = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

# remove Nan from data
DS = DS.dropna(axis='columns')

# split DS to x, y ,id
DS_x = DS.drop(columns=["diagnosis", "id"], axis=0)

# list top
top = list(DS.columns)
top_x = np.array(pd.DataFrame(DS_x.columns))


# -much the data-
# organize y
change = {"M": 1, "B": 0}
DS.diagnosis = [change[item] for item in DS.diagnosis]

# feature scaling
scaler = StandardScaler()
scaler.fit(DS_x)
DS_x = scaler.transform(DS_x)


# separate to train & test
x_train, x_test, y_train, y_test = select.train_test_split(DS_x, DS.diagnosis, train_size=0.5, test_size=0.5)
y_test = np.array(y_test)

# model
model = skl.LogisticRegression(solver="liblinear", penalty="l1")
model.fit(x_train, y_train)
coef = model.coef_


# -Diagnoses-

# score
score = model.score(x_test, y_test)
print("score : ")
print(score)

# f1 score
y_predict_test = model.predict(x_test)
f1_score = f1_score(y_test,y_predict_test)
print("f1 score : ")
print(f1_score)


# useless feature
zero_place = np.where(coef == 0.0,)[1]

useless_feature = []
for i in zero_place:
    useless_feature.append(top_x[i][0])

print("\nuseless feature : ")
print(useless_feature)