import json
import os
import re
from datetime import datetime
from profilehooks import timecall

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline, \
                             make_pipeline

from sklearn.metrics import accuracy_score, \
                            classification_report, \
                            confusion_matrix, \
                            mean_squared_log_error

from sklearn.linear_model import SGDClassifier, \
                                 SGDRegressor

from sklearn.ensemble import RandomForestClassifier, \
                             RandomForestRegressor                             

from sklearn.preprocessing import Normalizer, \
                                  LabelEncoder

from sklearn.model_selection import train_test_split, \
                                    GridSearchCV


@timecall(immediate=True)
def load_words():
    try:
        flatten = lambda l: [item for sublist in l for item in sublist]

        valid_words = set()
        filename = "../input/words_dictionary.json"
        with open(filename, "r") as english_dictionary:
            eng_words = json.load(english_dictionary)

        filename = "../input/comp_terms.txt"
        with open(filename, "r") as tech_dictinary:
            words = set()
            for wrd in tech_dictinary:
                words.add(wrd)

        laptops = set()
        filename = "../input/laptop_models.json"
        with open(filename, "r") as laptops_dictionary:
            laptops = set(flatten(
                map(lambda ele: ele['title'].lower().split(' '),
                    json.load(laptops_dictionary))))

        valid_words = set(eng_words).union(set(words))
        valid_words = valid_words.union(laptops)

        return valid_words
    except Exception as e:
        return str(e)


def text_processor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


@timecall(immediate=True)
def get_abbreviations():
    abbr_dict = dict()
    with open('../input/cs_words_abbrv.txt', 'r') as fp:
        for abrv_data in fp.readlines():
            pattern = r'\[\[(.*?)\]\]'
            matched = re.findall(pattern, abrv_data)
            if matched and len(abrv_data.split(']]-')) > 1:
                for acronym in matched[0].split('|'):
                    acronym = acronym.strip()
                    abbr = abrv_data.split(']]-')[1].split('{{')[0]
                    abbr_dict[acronym] = abbr.lower()
    return abbr_dict


@timecall(immediate=True)
def postprocessing(dataset, classification=True):
    # Cleaning Dataset

    dataset.price = dataset.price.replace('[Rs,]', '', regex=True).astype(float)
    dataset = dataset.query('price <= 200000 and price >= 5000')
    dataset = dataset.dropna()

    if classification:
        labels = ["{0} - {1}".format(i, i + 2000) for i in range(2000, 200000, 2000)]
        dataset.loc[:, 'price_group'] = pd.cut(dataset.price,
                                               range(2001, 202000, 2000), 
                                               right=False, labels=labels)
        le = LabelEncoder()
        price_group_le = le.fit(dataset.price_group)
        dataset.price_group = price_group_le.transform(dataset.price_group)
    else:
        dataset.price = np.log1p(dataset.price)

    dataset['product_info'] = dataset['title'] + ' ' + dataset['description']

    return dataset


@timecall(immediate=True)
def perform_tfidf(X_train, english_words):
    stop_words = stopwords.words("english")

    wordnet_lemmatizer = WordNetLemmatizer()
    abbreviations = get_abbreviations()
    stop_words.remove('i')

    def tokenizer_porter(text):
        word_vec = list()
        for word in text_processor(text).split():
            word = word.strip()
            if (word not in stop_words and word in english_words):
                word = abbreviations.get(word, word)
                word_vec.append(wordnet_lemmatizer.lemmatize(word))
        return word_vec

    print("****** Title and Description ******")
    vect_pinfo = TfidfVectorizer(min_df=20,
                                 max_features=100000,
                                 tokenizer=tokenizer_porter,
                                 ngram_range=(2, 3),
                                 lowercase=True)
    vz_pinfo = vect_pinfo.fit_transform(X_train)

    # TFIDF Matrix

    tfidf_pinfo = dict(zip(vect_pinfo.get_feature_names(), vect_pinfo.idf_))
    tfidf_pinfo = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf_pinfo), orient='index')

    tfidf_pinfo.columns = ['tfidf']
    print(tfidf_pinfo.sort_values(by=['tfidf'], ascending=True).head(10))
    print(tfidf_pinfo.sort_values(by=['tfidf'], ascending=False).head(10))

    X_train = pd.DataFrame(vz_pinfo.toarray())

    return X_train, vect_pinfo


@timecall(immediate=True)
def perform_clustering(dataset, english_words, true_k=30, show_top_terms=False, show_clusters_map=False):
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=500,
                         init_size=500, batch_size=1000,
                         max_iter=50, random_state=10)

    train = dataset['product_info'].values

    print("Performing TfIdf of complete dataset")
    train, vectorizer = perform_tfidf(train, english_words)

    print("Performing PCA of complete dataset")
    train, svd, normalizer = apply_pca(train, 300, 20)

    print("Complete dataset cluster traning")
    km.fit(train)

    clusters = km.labels_.tolist()
    dataset['cluster'] = clusters

    if show_top_terms:
        print(dataset['cluster'].value_counts())

        grouped = dataset['price'].groupby(dataset['cluster'])

        print(grouped.mean())

        print("Top terms per cluster:\n")

        # sort cluster centers by proximity to centroid
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :20]:
                print(' %s' % terms[ind], end='')
            print("\n")

        if show_clusters_map:
            tfs_embedded = TSNE(random_state=10, n_components=2, perplexity=40, verbose=2, n_iter=500).fit_transform(
                train)
            fig = plt.figure(figsize=(20, 20))
            ax = plt.axes()
            plt.scatter(tfs_embedded[:, 0], tfs_embedded[:, 1], marker="x", c=km.labels_)
            plt.show()

    return km, dataset


@timecall(immediate=True)
def apply_pca(ds, n_components=500, n_iter=50):
    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=10)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    ds = lsa.fit_transform(ds)

    explained_variance = svd.explained_variance_ratio_.sum()

    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    return ds, svd, normalizer


@timecall(immediate=True)
def apply_model(X_train, y_train, X_test, y_test,
                cluster_id=None, classification_run=False):
    if not classification_run:
        model = SGDRegressor(loss='squared_loss',
                             penalty='l2', random_state=10, max_iter=60)
        params = {'penalty': ['none', 'l2', 'l1'],
                  'alpha': [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]}

        gs = GridSearchCV(estimator=model,
                          param_grid=params,
                          scoring='neg_mean_squared_error',
                          n_jobs=8,
                          cv=10,
                          verbose=3)

        gs.fit(X_train, y_train)

        model = gs.best_estimator_
        print(gs.best_params_)
        print(gs.best_score_)

        pipe = Pipeline([('model', model)])

        y_pred = pipe.predict(X_test)
        sgd_mse = np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(y_pred)))

        print("SGD MSE:", sgd_mse)
        plot_regression(y_test, y_pred, sgd_mse, cluster_id, 'SGD')

        forest_reg = RandomForestRegressor(random_state=10)
        forest_reg.fit(X_train, y_train)
        pipe = Pipeline([('model', forest_reg)])

        y_pred = pipe.predict(X_test)
        rnf_mse = np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(y_pred)))

        print("RNF MSE:", rnf_mse)
        plot_regression(y_test, y_pred, rnf_mse, cluster_id, 'RNF')

        return min(sgd_mse, rnf_mse)

    else:
        clf = RandomForestClassifier(n_estimators=20, random_state=10)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        print("RNF:", accuracy_score(y_test, predictions))

        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))

        knn = KNeighborsClassifier(500)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        print("KNN:", accuracy_score(y_test, predictions))

        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))

        sgd = SGDClassifier()
        sgd.fit(X_train, y_train)
        predictions = sgd.predict(X_test)
        print("SGD", accuracy_score(y_test, predictions))

        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))


@timecall(immediate=True)
def plot_regression(y_test, y_pred, mse, cluster_id, algo_name):
    plt.clf()
    plt.ioff()
    fig = plt.figure()

    fit = np.polyfit(y_test, y_pred, 1)
    fit_fn = np.poly1d(fit)

    plt.xlabel("True Values")
    plt.ylabel("Predictions")

    patch = mpatches.Patch(color='green', label='With %s MSE: %f' % (algo_name, mse))
    plt.legend(handles=[patch])

    plt.plot(y_test, y_pred, 'yo', y_test, fit_fn(y_test), '--k')
    plt.show()
    plt.close(fig)


@timecall(immediate=True)
def perform_analytics(classification_run=False):
    english_words = load_words()

    print("****** Data Import ******")
    dataset_list = list()
    for ds_date in ['2018-03-27', '2018-04-15', '2018-04-30']:
        path = os.path.join('../input', ds_date + '.json')
        dataset_list.append(pd.read_json(path))

    print("****** Data Joining ******")
    dataset = pd.concat(dataset_list)
    dataset = dataset.reset_index(drop=True)

    print("****** Data Cleaning ******")
    dataset = postprocessing(dataset, classification_run)

    if classification_run:
        print("****** Split and Fit ******")
        X = dataset['product_info'].values
        y = dataset['price_group'].values

        print("****** Vectorization ******")
        X, vectorizer = perform_tfidf(X, english_words)

        print("****** Dimension Compression ******")
        X, svd, normalizer = apply_pca(X, 300, 20)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.3,
                                                            random_state=10)

        print("****** Applying Model ******")
        mse = apply_model(X_train, y_train, X_test,
                          y_test, classification_run=classification_run)

    else:
        print("****** Generate Clusters ******")
        kmeans, dataset = perform_clustering(dataset, english_words)

        print("****** Regression for each cluster ******")
        seed = 10
        mmse = list()
        grouped = dataset.groupby(dataset['cluster'])
        for cluster_id, group in grouped:
            print("****** Working For Cluster With Size ******")
            print(cluster_id, group.shape)

            print("****** Split and Fit ******")
            X = group['product_info'].values
            if classification_run:
                y = group['price_group'].values
            else:
                y = group['price'].values

            print("****** Vectorization Sub Cluster ******")
            X, vectorizer = perform_tfidf(X, english_words)

            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=0.3,
                                                                random_state=seed)

            print("****** Applying Model  Sub Cluster ******")
            mse = apply_model(X_train, y_train, X_test, y_test, cluster_id, classification_run)
            mmse.append(mse)

        print("Avg MSE: ", np.mean(np.array(mmse)))


if __name__ == '__main__':
    for run_type in [True, False]:
        perform_analytics(run_type)
