import numpy as np
import nltk
import csv
import random
import math
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import savefig
from collections import defaultdict

random.seed("123")


class Titanic(object):

    def __init__(self, input_file):
        self.df = pd.read_csv(input_file)
        self.embarked_predictor = None

    def fix_data(self, df=None):
        if not isinstance(df, pd.DataFrame):
            df = self.df
        df['Title'] = df['Name'].apply(
            lambda x: x.lower().split(', ')[1].split('. ')[0])

        def tg(r):
            if r in ['lady', 'mile', 'mme', 'ms', 'sir', 'the countess']:
                return 1
            elif r in ['capt', 'don', 'jonkheer', 'mr', 'rev']:
                return 3
            else:
                return 2
        df['TitleGroup'] = df['Title'].apply(tg)

        def a(r):
            if not math.isnan(r):
                return r
            return int(df[df.Age.notnull()].Age.sample(n=1))
        df['Age'] = df['Age'].apply(a)

        df.loc[df.Fare.isnull(), 'Fare'] = df['Fare'].mean()
        # standartize due to big differenc in density
        df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
        df['Fare'] = (df['Fare'] - df['Fare'].mean()) / df['Fare'].std()

        df['Family'] = df['Parch'] + df['SibSp']

        return df

    def fix_embarked(self, df=None):
        if not isinstance(df, pd.DataFrame):
            df = self.df
        for i, r in df.iterrows():
            if isinstance(r['Embarked'], str):
                continue
            emb = self.embarked_predictor.predict(
                [self._get_embarked_feature(r)])[0]
            df.set_value(i, 'Embarked', emb)
        return df

    def _get_embarked_feature(self, r):
        def get_cabin_class(r):
            didx = dict(zip("ABCDEFGTO", [1, 2, 3, 4, 5, 6, 7, 8, 9]))
            if isinstance(r['Cabin'], float):
                return 0
            cab = r['Cabin'][0]
            return didx[cab]

        def get_num_cabins(r):
            if isinstance(r['Cabin'], float):
                return 0
            return len(r['Cabin'].split(' '))

        o = {
            'class': r['Pclass'],
            'family': r['Family'],
            'fare': r['Fare'] if not math.isnan(r['Fare']) else 28.0,
            'cabin_class': get_cabin_class(r),
            'num_cabins': get_num_cabins(r)
        }
        return list(o.values())

    def get_embarked_predictor(self, test_input):
        df = pd.read_csv(test_input)
        test = self.fix_data(df)
        test = test[test.Embarked.notnull()].to_dict('records')
        data = self.df[self.df.Embarked.notnull()].to_dict('records')
        data += test

        self.embarked_predictor = SVC()
        train_set = [self._get_embarked_feature(r) for r in data]
        train_labels_set = [r.get("Embarked") for r in data]

        self.embarked_predictor.fit(train_set, train_labels_set)

    def get_feature(self, r, is_list=False):
        emidx = {"S": 1, "Q": 3, "C": 2}

        def get_cabin_class(r):
            didx = dict(zip("ABCDEFGTO", [1, 2, 3, 4, 5, 6, 7, 8, 9]))
            if isinstance(r['Cabin'], float):
                return 0
            cab = r['Cabin'][0]
            return didx[cab]

        def get_num_cabins(r):
            if isinstance(r['Cabin'], float):
                return 0
            return len(r['Cabin'].split(' '))

        def get_age_group(r):
            if float(r.get("Age")) < 10:
                return 1
            if float(r.get("Age")) > 60:
                return 3
            return 2

        def get_fare_group(r):
            if math.isnan(r.get('Fare', 0)):
                return 1
            if float(r.get("Fare")) < 15:
                return 1
            if float(r.get("Fare")) < 40:
                return 2
            if float(r.get("Fare")) >= 40 and float(r.get("Fare")) < 400:
                return 3
            if float(r.get("Fare")) >= 400:
                return 4
        o = {
            # "age": r.get('Age'),
            # "fare": r.get('Fare') if not math.isnan(r.get('Fare', 0)) else 15,
            # "fare_group": get_fare_group(r),
            # "sex": 1 if(r.get("Sex") == "male") else 0,
            # "embarked": emidx[r.get("Embarked")],
            # "family_size": r['Family'],
            # "status_group": r['TitleGroup'],
            # "num_cabs": get_num_cabins(r),
            # "class": int(r.get("Pclass")) if(r.get("Pclass")) else 3,
            # "cab_class": get_cabin_class(r),
            # "age_group": get_age_group(r),
            # "ticket": bool(r.get("Ticket")),


            "class": int(r.get("Pclass")) if(r.get("Pclass")) else 3,
            "sex": 1 if(r.get("Sex") == "male") else 0,
            "age_group": get_age_group(r),
            "embarked": emidx.get(r.get("Embarked")),
            "status_group": r['TitleGroup'],
            "ticket": bool(r.get("Ticket")),
            "fare_group": get_fare_group(r),
            "family_size": r['Family'],
        }
        if is_list:
            return list(o.values())
        return o

    def confusion_matrix(self):
        train_set = self.df.to_dict('records')
        # random.shuffle(train_set)
        st = int(len(train_set) / 5)
        featuresets = [self.get_feature(r, True) for r in train_set]
        labels = [r.get("Survived") for r in train_set]

        test_set = featuresets[0:st]
        train_set = featuresets[st:]

        ground_true = labels[0:st]
        train_labels_set = labels[st:]

        clf = SVC(C=50)
        clf.fit(train_set, train_labels_set)
        predicted = clf.predict(test_set)

        total_pos = 0
        total_neg = 0
        g_pos_p_neg = 0
        g_neg_p_pos = 0
        for i in xrange(len(ground_true)):
            if ground_true[i] == predicted[i] == 1:
                total_pos += 1
            if ground_true[i] == predicted[i] == 0:
                total_neg += 1
            if ground_true[i] != predicted[i] and ground_true[i] == 1:
                g_pos_p_neg += 1
            if ground_true[i] != predicted[i] and ground_true[i] == 0:
                g_neg_p_pos += 1

        print (clf.score(test_set, ground_true))

        print ('{:>10}|{:<30}'.format('Ground', 'Predicted'))
        print ('{:>10}|{:>5}|{:>5}'.format('', '1', '0'))
        print ('{:>10}|{:5d}|{:5d}'.format('1', total_pos, g_pos_p_neg))
        print ('{:>10}|{:5d}|{:5d}'.format('0', g_neg_p_pos, total_neg))

    def roc_curve(self):

        train_set = self.df.to_dict('records')
        random.shuffle(train_set)
        st = int(len(train_set) / 5)
        featuresets = [self.get_feature(r, True) for r in train_set]
        labels = [r.get("Survived") for r in train_set]

        test_set = featuresets[0:st]
        train_set = featuresets[st:]

        ground_true = labels[0:st]
        train_labels_set = labels[st:]

        clf = SVC()
        y_score = clf.fit(
            train_set, train_labels_set).decision_function(test_set)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(1):
            fpr[i], tpr[i], _ = roc_curve(ground_true, y_score)
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        lw = 2
        plt.plot(fpr[0], tpr[0], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def cross_validate(self):
        train_set = self.df.to_dict('records')
        st = int(len(train_set) / 5)
        # random.shuffle(train_set)
        featuresets = [self.get_feature(r, True) for r in train_set]
        labels = [r.get("Survived") for r in train_set]
        accuracy = 0
        for i in range(5):
            test_set = featuresets[st * i:st * i + st]
            train_set = featuresets[:st * i] + featuresets[st * i + st:]

            test_labels_set = labels[st * i:st * i + st]
            train_labels_set = labels[:st * i] + labels[st * i + st:]
            clf = SVC(C=50)
            clf.fit(train_set, train_labels_set)
            score = clf.score(test_set, test_labels_set)
            accuracy += score
        print(accuracy / 5.0)

    def build_prediction(self, test_input):
        data = self.df.to_dict('records')
        train_set = [self.get_feature(r, True) for r in data]
        train_labels_set = [r.get("Survived") for r in data]

        clf = SVC(C=50)
        clf.fit(train_set, train_labels_set)

        df = pd.read_csv(test_input)
        test = self.fix_data(df)
        test = t.fix_embarked(test)
        pf = open("titanic_out.csv", "w")
        predict_set = test.to_dict('records')
        pfw = csv.DictWriter(pf, fieldnames=["PassengerId", "Survived"])
        pfw.writeheader()
        featuresets = [self.get_feature(r, True) for r in predict_set]
        out = clf.predict(featuresets)
        for i, r in enumerate(predict_set):
            pfw.writerow({"PassengerId": r.get(
                "PassengerId"), "Survived": out[i]})

        pf.close()


if __name__ == "__main__":
    t = Titanic("../input/train.csv")
    t.fix_data()
    t.get_embarked_predictor("../input/test.csv")
    t.fix_embarked()
    # t.roc_curve()
    #t.confusion_matrix()
    #t.cross_validate()
    t.build_prediction("../input/test.csv")