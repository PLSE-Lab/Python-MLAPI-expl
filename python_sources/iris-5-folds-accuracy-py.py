import numpy as np
from sklearn import preprocessing, pipeline, linear_model, model_selection

X = []
y = []
with open("../input/Iris.csv", "r") as fp:
    fp.readline()
    for row in fp:
        cols = row.strip().split(",")
        label = cols[-1]
        x = np.array(cols[:-1], dtype=np.float64)
        X.append(x)
        y.append(label)

X = np.array(X)
y = preprocessing.LabelEncoder().fit_transform(y)

scaler = preprocessing.StandardScaler()
clf = linear_model.LogisticRegression(penalty="l2", random_state=1)
base_model = pipeline.Pipeline([
    ("scaler", scaler),
    ("clf", clf),
])
param_grid = {
    "clf__C": np.logspace(0, 3, 20)
}

gs = model_selection.GridSearchCV(base_model, param_grid, cv=5, verbose=2, scoring="accuracy")
gs.fit(X, y)

print("accuracy:", gs.best_score_)
print(gs.best_params_)

model = gs.best_estimator_

