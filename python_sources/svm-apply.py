import pandas as pd
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


def learn_model(label,data):
    data_train, data_test, target_train, target_test = cross_validation.train_test_split(data, label,
                                                                                         test_size=0.4, random_state=43)
    pca = PCA(n_components=0.8,whiten=True)
    data_train = pca.fit_transform(data_train)
    data_test = pca.transform(data_test)
    model = ExtraTreesClassifier(n_estimators=9,
                              n_jobs=-1,
                              random_state=0)

    model.fit(data_train,target_train)
    predicted = model.predict(data_test)
    print(accuracy_score(target_test,predicted))
    

def main():
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    label = train_df.ix[:,0]
    data = train_df.values[:,1:]
    data_test = test_df.values
    learn_model(label,data)

# main function
main()