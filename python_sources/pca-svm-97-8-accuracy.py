import pandas as pd
from sklearn import linear_model, svm, preprocessing, decomposition, model_selection
import numpy as np

def main():
    print("Data loading...")
    pca, (X, X_test, y, y_test) = read_train_data("../input/train.csv")
    X_test_set = read_test_data("../input/test.csv", pca)

    print("Training SVM...")
    svm = train_svm(X, y)
    print("SVM accuracy on training set: ", svm.score(X_test, y_test))

    ans = svm.predict(X_test_set)
    write_prediction(ans)
    print("Prediction writed as predict.csv")


def train_svm(X, y):
    svc = svm.SVC(kernel="rbf")

    print("Finding better regularization parameter and gamma...")
    Cs = np.logspace(-6, 6, 30)
    gammas = np.logspace(-6, 1, 10)
    cv = model_selection.ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    clf = model_selection.GridSearchCV(estimator=svc, cv=cv, param_grid=dict(C=Cs, gamma=gammas), n_jobs=-1)
    # small part of data set for finding parameters because svm training too long
    clf.fit(X[:2000], y[:2000])

    print("Found best reg parameter: ", clf.best_estimator_.C)
    print("Found best gamma: ", clf.best_estimator_.gamma)

    print("Training estimator on full dataset")
    est = svm.SVC(C=clf.best_estimator_.C, gamma=clf.best_estimator_.gamma)
    est.fit(X, y)

    return est


def read_train_data(train_file):
    print("Loading train data...")
    train = pd.read_csv(train_file)

    Y = train.ix[:,'label':'label'].values.flatten()
    X = preprocessing.normalize(train.ix[:,'pixel0':].values)

    pca = fit_pca(X)
    X = pca.transform(X)

    return pca, model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)


def read_test_data(test_file, pca):
    print("Loading test data...")
    test = pd.read_csv(test_file)

    X = preprocessing.normalize(test.values)

    return pca.transform(X)


def fit_pca(X):
    print("Reduction...")
    pca = decomposition.PCA(n_components=50, whiten=True)
    pca.fit(X)
    return pca


def write_prediction(answer):
    # ids column
    ids = np.reshape(np.arange(1, answer.shape[0] + 1), (-1, 1))
    answer_column = np.reshape(answer, (-1, 1))
    answer_matrix = np.append(ids, answer_column, axis=1)

    df = pd.DataFrame(answer_matrix, columns=["ImageId", "Label"])
    df.to_csv("predict.csv", sep=",", index=False)


if __name__ == '__main__':
    main()
