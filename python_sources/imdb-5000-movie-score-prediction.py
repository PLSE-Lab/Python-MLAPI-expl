# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection, metrics, feature_extraction, decomposition
from sklearn import linear_model, ensemble
from collections import Counter
import seaborn as sns
import copy


FILE_NAME = '../input/movie_metadata.csv'
# FILE_NAME = 'movie_metadata_sample.csv'
LABEL_NAME = 'imdb_score'


def readMovieMetaDataRaw(filename):
    # import data
    print("\nImporting data...")
    dataframe = pd.read_csv(filename)
    return dataframe

# not used
def doFeatureSelection():
    dataframe = readMovieMetaDataRaw(FILE_NAME)
    dataframe = dataframe.dropna()  # drop all the objects with nan

    le = preprocessing.LabelEncoder()
    for column in dataframe.columns:
        # if column != LABEL_NAME:
        dataframe[column] = le.fit_transform(dataframe[column])

    data = dataframe.values
    # print data

    (Xdata, ydata) = getXYdata(dataframe)
    X_Train, X_Test, y_Train, y_Test = model_selection.train_test_split(Xdata, ydata, test_size=0.3, random_state=1)

    rf = ensemble.RandomForestClassifier(n_estimators=500, max_depth=4, n_jobs=-1)
    print("=== y_Train ===")
    print(y_Train)
    rf.fit(X_Train, y_Train)

    names = ["color",
    "director_name",
    "num_critic_for_reviews",
    "duration",
    "director_facebook_likes",
    "actor_3_facebook_likes",
    "actor_2_name",
    "actor_1_facebook_likes",
    "gross",
    "genres",
    "actor_1_name",
    "movie_title",
    "num_voted_users",
    "cast_total_facebook_likes",
    "actor_3_name",
    "facenumber_in_poster",
    "plot_keywords",
    "movie_imdb_link",
    "num_user_for_reviews",
    "language",
    "country",
    "content_rating",
    "budget",
    "title_year",
    "actor_2_facebook_likes",
    "aspect_ratio",
    "movie_facebook_likes"]

    featureScores = rf.feature_importances_
    # SortedFeatureScores = np.ndarray.sort(featureScores)
    yPos = np.arange(len(names))

    plt.bar(yPos, featureScores, align='center', alpha=0.5)
    plt.xticks(yPos, names, rotation=90)
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()


def numPerCountry(dataframe):
    country = []
    for i in dataframe['country']:
        country.extend(i.split(","))
    country_counts = Counter(country)

    country_df = pd.DataFrame([(i, j) for i, j in country_counts.items()], columns=["country", "Count"])
    country_df.sort_values(by="Count", ascending=False, inplace=True)
    more_than_2 = country_df[country_df["Count"] > 2].reset_index(drop=True)
    plt.subplots(figsize=(14, 8))

    # plt.gca().set_yscale('log', basey=10)
    g = sns.barplot(x="country", y="Count", data=more_than_2)
    # g.set_yscale('log')
    plt.xticks([i - 0.2 for i in range(len(more_than_2))], more_than_2["country"], rotation=45)
    plt.ylabel("Number of movies")
    plt.title("Number of movies per country")
    # plt.gca().set_yscale('log', basey=2)
    plt.tight_layout()
    plt.show()


def medainRatPerCountry(dataframe):
    country = []
    for i in dataframe['country']:
        # print i
        country.extend(i.split(','))
    country_counts = Counter(country)
    top_10_countries = [i[0] for i in country_counts.most_common(10)]
    for i in top_10_countries:
        dataframe[i] = dataframe["country"].map(lambda x: 1 if i in str(x) else 0)
    country_rating = []
    for i in top_10_countries:
        country_rating.append([i, dataframe[LABEL_NAME][dataframe[i] == 1].median()])

    country_rating = pd.DataFrame(country_rating, columns=["country", "median_rating"])
    country_rating.sort_values(by="median_rating", ascending=False, inplace=True)

    scaler2 = preprocessing.MinMaxScaler(feature_range=(0, 10))
    # print(scaler2)
    country_rating["scaled_rating"] = scaler2.fit_transform(country_rating["median_rating"].values.reshape(-1, 1))

    sns.barplot(x="country", y="scaled_rating", data=country_rating)
    plt.title("Scaled median rating per Country")
    plt.ylabel("Scaled median rating")
    plt.xticks(rotation="vertical")
    plt.tight_layout()
    plt.show()


def numPerGenre(dataframe):
    genre_list = []
    for i in dataframe['genres']:
        genre_list.extend(i.split('|'))
    genre_counts = Counter(genre_list)
    genre_df = pd.DataFrame([(i, j) for i, j in genre_counts.items()], columns=["genres", "Count"])
    genre_df.sort_values(by="Count", ascending=False, inplace=True)

    plt.subplots(figsize=(8, 5))
    sns.barplot(x="genres", y="Count", data=genre_df)
    plt.xticks(rotation="vertical")
    plt.ylabel("Number of movies")
    plt.title("Number of movies per Genre")
    plt.ylim((0, 2000))
    plt.tight_layout()
    plt.show()


def medainRatPerGenre(dataframe):
    genre_list = []
    for i in dataframe["genres"]:
        genre_list.extend([x.strip() for x in i.split("|")])
    genre_list = list(set(genre_list))
    genre_list.sort()

    for i in genre_list:
        dataframe[i] = dataframe["genres"].apply(lambda x: 1 if i in str(x) else 0)
    genre_rating = []

    for i in genre_list:
        genre_rating.append([i, dataframe[LABEL_NAME][dataframe[i] == 1].median()])

    genre_rating = pd.DataFrame(genre_rating, columns=["genres", "median_rating"])
    genre_rating.sort_values(by="median_rating", ascending=False, inplace=True)

    genre_scaler = preprocessing.MinMaxScaler()
    genre_rating["scaled_rating"] = genre_scaler.fit_transform(genre_rating["median_rating"].values.reshape(-1, 1))

    sns.barplot(x="genres", y="scaled_rating", data=genre_rating)
    plt.title("Scaled median rating per Genre")
    plt.ylabel("Scaled median rating")
    plt.xticks(rotation="vertical")
    plt.tight_layout()
    plt.show()


def numPerActor(dataframe):
    vectorizer = feature_extraction.text.CountVectorizer(token_pattern=u'(?u)\w+.?\w?.? \w+')
    actors_df = pd.DataFrame(vectorizer.fit_transform(dataframe["actor_1_name"]).todense(),
                             columns=vectorizer.get_feature_names())
    top_15_actors = actors_df.sum().sort_values(ascending=False).head(15)
    actors_count = pd.DataFrame(top_15_actors).reset_index()
    actors_count.columns = ["actor_1_name", "Count"]
    actors_count.sort_values(by="Count", ascending=False, inplace=True)

    more_than = actors_count[actors_count["Count"] > 3].reset_index(drop=True)
    plt.subplots(figsize=(8, 5))
    sns.barplot(x="actor_1_name", y="Count", data=more_than)
    plt.xticks(rotation="vertical")
    plt.ylabel("Number of movies")
    plt.title("Number of movies per actor")
    plt.ylim((0, 50))
    plt.tight_layout()
    plt.show()


def medainRatPerActor(dataframe):
    vectorizer = feature_extraction.text.CountVectorizer(token_pattern=u'(?u)\w+.?\w?.? \w+')
    actors_df = pd.DataFrame(vectorizer.fit_transform(dataframe["actor_1_name"]).todense(),
                             columns=vectorizer.get_feature_names())
    top_15_actors = actors_df.sum().sort_values(ascending=False).head(15).index
    top_15_actors = [i.replace(" ", "_") for i in top_15_actors]
    for i in top_15_actors:
        dataframe[i.replace(" ", "_")] = actors_df.loc[:, i.replace("_", " ")]
    actor_rating = []
    for i in top_15_actors:
        actor_rating.append([i, dataframe[LABEL_NAME][dataframe[i] == 1].median()])

    actor_rating = pd.DataFrame(actor_rating, columns=["actor_1_name", "median_rating"])
    actor_rating.sort_values(by="median_rating", ascending=False, inplace=True)

    scaler2 = preprocessing.MinMaxScaler()
    actor_rating["scaled_rating"] = scaler2.fit_transform(actor_rating["median_rating"].values.reshape(-1, 1))

    sns.barplot(x="actor_1_name", y="scaled_rating", data=actor_rating)
    plt.title("Scaled median rating per Actor")
    plt.ylabel("Scaled median rating")
    plt.xticks(rotation="vertical")
    plt.tight_layout()
    plt.show()


def numPerDirctor(dataframe):
    vectorizer = feature_extraction.text.CountVectorizer(token_pattern=u'(?u)\w+.?\w?.? \w+')
    directors_df = pd.DataFrame(vectorizer.fit_transform(dataframe["director_name"]).todense(),
                                columns=vectorizer.get_feature_names())
    top_15_directors = directors_df.sum().sort_values(ascending=False).head(15)
    director_count = pd.DataFrame(top_15_directors).reset_index()
    director_count.columns = ["director_name", "Count"]
    director_count.sort_values(by="Count", ascending=False, inplace=True)

    more_than = director_count[director_count["Count"] > 3].reset_index(drop=True)
    plt.subplots(figsize=(8, 5))
    sns.barplot(x="director_name", y="Count", data=more_than)
    plt.xticks(rotation="vertical")
    plt.ylabel("Number of movies")
    plt.title("Number of movies per director")
    plt.ylim((0, 30))
    plt.tight_layout()
    plt.show()


def medainRatPerDirector(dataframe):
    vectorizer = feature_extraction.text.CountVectorizer(token_pattern=u'(?u)\w+.?\w?.? \w+')
    directors_df = pd.DataFrame(vectorizer.fit_transform(dataframe["director_name"]).todense(),
                                columns=vectorizer.get_feature_names())
    top_15_directors = directors_df.sum().sort_values(ascending=False).head(15).index
    top_15_directors = [i.replace(" ", "_") for i in top_15_directors]
    for i in top_15_directors:
        dataframe[i.replace(" ", "_")] = directors_df.loc[:, i.replace("_", " ")]
    director_rating = []
    for i in top_15_directors:
        director_rating.append([i, dataframe[LABEL_NAME][dataframe[i] == 1].median()])

    director_rating = pd.DataFrame(director_rating, columns=["director_name", "median_rating"])
    director_rating.sort_values(by="median_rating", ascending=False, inplace=True)

    scaler2 = preprocessing.MinMaxScaler()
    director_rating["scaled_rating"] = scaler2.fit_transform(director_rating["median_rating"].values.reshape(-1, 1))

    sns.barplot(x="director_name", y="scaled_rating", data=director_rating)
    plt.title("Scaled median rating per Director")
    plt.ylabel("Scaled median rating")
    plt.xticks(rotation="vertical")
    plt.tight_layout()
    plt.show()


def dataEDA(dataframe):
    print("\nInspecting data...")
    print("\n=== dataframe.index ===")
    print(dataframe.index)
    print()
    # print dataframe.columns
    # print
    print("\n=== dataframe.head() ===")
    print(dataframe.head())
    print("\n=== dataframe.info() ===")
    dataframe.info()
    print()
    top = dataframe.head(1)
    headers = top.columns.values
    print("\n=== data columns ===")
    print(headers)
    index = headers.tolist().index(LABEL_NAME)
    print("\n=== feature index: {} ===".format(index))
    print()

    dataframe = dataframe.dropna()
    numPerCountry(dataframe)
    medainRatPerCountry(dataframe)
    numPerGenre(dataframe)
    medainRatPerGenre(dataframe)
    numPerActor(dataframe)
    medainRatPerActor(dataframe)
    numPerDirctor(dataframe)
    medainRatPerDirector(dataframe)


def cleanData(dataframe):
    print("Cleaning data...")
    dataframe = dataframe.dropna()  # drop all the objects with nan

    le = preprocessing.LabelEncoder()
    for column in dataframe.columns:
        if column != LABEL_NAME:
            # dataframe[column] = le.fit_transform(dataframe[column])
            dataframe.loc[:, column] = le.fit_transform(dataframe.loc[:, column])
    return dataframe


def getXYdata(dataframe):
    # todo change to k fold?
    # featureColumns = [col for col in dataframe.columns if col not in ['imdb_score', 'budget', ]]
    featureColumns = [col for col in dataframe.columns if col != LABEL_NAME]
    X = dataframe[featureColumns]
    y = dataframe[LABEL_NAME]
    print("\n=== y.describle() ===")
    print(y.describe())
    # sns.boxplot(y, vert=True)
    # plt.show()

    Xdata = X.values
    ydata = y.values

    return (Xdata, ydata)

# not used
def splitData(dataframe):
    data = dataframe.values
    dataRows = data.shape[0]

    (Xdata, ydata) = getXYdata(dataframe)
    # use even rows for testing
    X_train = Xdata[0:dataRows:2]
    y_train = ydata[0:dataRows:2]
    X_test = Xdata[1:dataRows:2]
    y_test = ydata[1:dataRows:2]
    return (X_train, y_train, X_test, y_test)


# def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
#     """pretty print for confusion matrixes"""
#     columnwidth = max([10])  # 5 is value length
#     empty_cell = " " * columnwidth
#     # Print header
#     print "    " + empty_cell,
#     for label in labels:
#         print "%{0}s".format(columnwidth) % label,
#     print
#     # Print rows
#     for i, label1 in enumerate(labels):
#         print "    %{0}s".format(columnwidth) % label1,
#         for j in range(len(labels)):
#             cell = "%{0}.1f".format(columnwidth) % cm[i, j]
#             if hide_zeroes:
#                 cell = cell if float(cm[i, j]) != 0 else empty_cell
#             if hide_diagonal:
#                 cell = cell if i != j else empty_cell
#             if hide_threshold:
#                 cell = cell if cm[i, j] > hide_threshold else empty_cell
#             print cell,
#         print


# def printConfusionMatrix(y_pred, y_test):
#     # confusion matrix
#     print "\n=== confusion matrix ==="
#     labels = [0., 1.]
#     print y_test
#     print y_pred
#     cm = metrics.confusion_matrix(y_test, y_pred, labels)
#     print_cm(cm, labels)


def fit_model(model, name, X_train, X_test, y_train, y_test, classification_threshold, mtype="r"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    print("{} Score: {:.2f}".format(name, score))

    if mtype == "r":
        print("{} MSE: {:.2f}".format(name, metrics.mean_squared_error(y_test, y_pred)))
        print("{} RMSE: {:.2f}".format(name, np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
    elif mtype == "c":
        print("classification threshold = {}".format(classification_threshold))
    return model, y_pred, score


def evaluate_model(Xdata, ydata, model, name, mtype="r", classification_threshold=8.5):
    print("\n=== {} ===".format(name))

    if mtype == "r":  # "=== do regression ==="
        X_train, X_test, y_train, y_test = model_selection.train_test_split(Xdata, ydata, test_size=0.3, random_state=77)

        print("=== len ===")
        print(len(X_train))
        print(len(X_test))
        print(len(y_train))
        print(len(y_test))
        # fit the model
        model, y_pred, score = fit_model(model, name, X_train, X_test, y_train, y_test, classification_threshold, mtype)

        # Plotting
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '-')
        plt.scatter(y_pred, y_test)
        plt.title("{}\nActual and predicted ratings".format(name))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
        return model
    elif mtype == "c":  # "=== do classification ==="
        print("=== ydata ===")
        print (ydata)
        ydata[ydata < classification_threshold] = 0
        ydata[ydata >= classification_threshold] = 1
        print (ydata)
        print()
        X_train, X_test, y_train, y_test = model_selection.train_test_split(Xdata, ydata, stratify=ydata,
                                                                            test_size=0.3, random_state=77)
        # print "=== len ==="
        # print len(X_train)
        # print len(X_test)
        # print len(y_train)
        # print len(y_test)

        # fit the model
        model, y_pred, score = fit_model(model, name, X_train, X_test, y_train, y_test, classification_threshold, mtype)

        # confusion matrix
        print("=== confusion matrix ===")
        conmat = metrics.confusion_matrix(y_test, y_pred)
        conmat = pd.DataFrame(conmat)
        print(conmat)

        # classification report
        print(metrics.classification_report(y_test, y_pred))

        # plotting
        try:
            y_score = model.decision_function(X_test)
        except:
            y_score = model.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        # plt.plot(fpr, tpr)
        # plt.title("{}\nROC curve, classification threshold = {}".format(name, classification_threshold))
        # plt.ylim((-0.1, 1.1))
        # plt.xlim((-0.1, 1.1))
        # plt.xlabel("FPR")
        # plt.ylabel("TPR")
        # plt.text(0.8, 0.05, "AUC: {:0.2f}".format(metrics.roc_auc_score(y_test, y_score)))
        # plt.show()
        return model, conmat, Xdata, ydata, score
    else:
        return "Wrong type"


def main():
    dataframe = readMovieMetaDataRaw(FILE_NAME)
    dataEDA(dataframe)
    dataframe = cleanData(dataframe)


    # todo, change to k fold?
    # doFeatureSelection()
    # (X_train, y_train, X_test, y_test) = splitData(dataframe)  # this is not really used

    (Xdata, ydata) = getXYdata(dataframe)


    # _ = evaluate_model(Xdata, ydata, linear_model.LinearRegression(), "Linear Regression")

    # p = decomposition.PCA(n_components=5)
    # p.fit(Xdata)
    # Xdata = p.fit_transform(Xdata)
    # _ = evaluate_model(Xdata, ydata, linear_model.LogisticRegressionCV(), "Logistic Regression")
    # _ = evaluate_model(Xdata, ydata, linear_model.LogisticRegressionCV(cv=3), "Linear Regression")

    # _ = evaluate_model(Xdata, ydata, model_selection.GridSearchCV(ensemble.GradientBoostingRegressor(random_state=1),
    #     {"n_estimators": np.arange(50, 100, 10)}, cv=5), "GridSearched Gradient Boosting Regressor")

    # rf, rf_conmat, X, y, score = evaluate_model(Xdata, ydata, model_selection.GridSearchCV(ensemble.RandomForestClassifier(),
    #                       {"n_estimators": np.arange(50, 100, 10), "min_samples_split": np.arange(5, 10, 1),
    #                       "min_samples_leaf": np.arange(5, 10, 1)}, cv=5), "GridSearched Random Forest", mtype="c")

    scores = []
    thresholds = [x / 10.0 for x in range(40, 92, 2)]
    # thresholds = [5.9]
    # thresholds = [6.5]
    # thresholds = [7.2]
    print(thresholds)
    for i in thresholds:
        ydata_temp = copy.deepcopy(ydata)
        logreg, logreg_conmat, X, y, score = evaluate_model(Xdata, ydata_temp, linear_model.LogisticRegressionCV(cv=3),
                                                    "Logistic Regression", mtype="c", classification_threshold=i)
    #
        scores.append(score)
    print(thresholds)
    print(scores)
    # plt.title("Score / Threshold relation")
    plt.title("Threshold / Score relation")
    plt.xlabel("Classification threshold")
    plt.ylabel("Logistic Regression Score")
    # plt.plot(thresholds, scores, 'o')
    plt.plot(thresholds, scores)
    plt.show()

    # x = [4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8, 9.0]
    # y = [0.97604259094942325, 0.97071872227151734, 0.96273291925465843, 0.9520851818988465, 0.93522626441881096,
    #  0.92102928127772843, 0.89352262644188107, 0.87400177462289264, 0.83850931677018636, 0.81455190771960961,
    #  0.77905944986690323, 0.75953859804791479, 0.74800354924578527, 0.7630878438331854, 0.78793256433007985,
    #  0.80301685891747998, 0.83673469387755106, 0.86779059449866902, 0.90150842945873999, 0.93256433007985806,
    #  0.94853593611357589, 0.97515527950310554, 0.98402839396628217, 0.99378881987577639, 0.99201419698314108,
    #  0.9982253771073647]

    # fig, ax = plt.plot(x, y, 'o')
    # fig, ax = plt.plot(x, y)
    # xticks = x
    # ax.set_xticks(xticks)
    # plt.show()


if __name__ == '__main__':
    print("IMDB score prediction")
    main()

