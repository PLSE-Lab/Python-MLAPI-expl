#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn import multiclass as skmc
from sklearn import naive_bayes as sknb
from sklearn import ensemble as sken
from sklearn import svm as sksvm
from sklearn import tree as sktr

import os
import pickle as pkl

I_AM_ON_KAGGLE = True

if I_AM_ON_KAGGLE:
    IN_FILES_DIR = os.path.join('..', 'input')
    CACHES_DIR = '.'
    BREED_MAP = None
else:
    IN_FILES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    CACHES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'caches')
    BREED_MAP = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'breeds_map.csv')

class AbstractDataSet(object):

    @staticmethod
    def ensureCacheDir():
        # We check if this is a directory, and if this is not the case, we create it.
        # This will fail if CACHES_DIR exists but is not a directory ... and we
        # expect it to fail in order to prevent unintended files replacements.
        if not os.path.isdir(CACHES_DIR):
            os.makedirs(CACHES_DIR)

    def __init__(self, which):
        assert which in ('train', 'test')
        AbstractDataSet.ensureCacheDir()
        self._which = which
        self._fileName = os.path.join(IN_FILES_DIR, "%s.csv" % which)
        self._cacheName = os.path.join(CACHES_DIR, "%s_%s.pkl" % (self.__class__.__name__, which))
        self._data = None

    def get(self, dropNa, withTimestamp):
        if self._data is None:
            if os.path.exists(self._cacheName):
                print("Loading %s from cache" % self._which)
                self._data = pkl.load(open(self._cacheName, "rb"))
            else:
                self._data = self._getExtracted()
                print("Caching %s" % self._which)
                pkl.dump(self._data, open(self._cacheName, "wb"))
        data = self._data.copy()
        if dropNa:
            print("Removing NaN")
            data.dropna(inplace=True)
        if not withTimestamp:
            print("Removing timestamp")
            del data['timestamp']
        return self._splitFeaturesReply(data)

    def _getExtracted(self):
        print("Extracting %s" % self._which)
        ret = pd.read_csv(self._fileName)
        print("Munging %s" % self._which)
        return self._mungle(ret)
        data = self._splitFeaturesReply(data)
        return data

    def _splitFeaturesReply(self, data):
        print("Splitting X and y")
        if self._which == 'train':
            y = data['OutcomeType']
            y.name = None
            y = pd.get_dummies(y)
            y = y.astype(np.float16)
            del data['OutcomeType']
        else:
            y = None
        X = data.copy()
        return (X, y)

    def _mungle(self, data):
        raise NotImplementedError("Abstract method called")


class DataSetTrial1(AbstractDataSet):

    def _mungle(self, data):
        if self._which == 'test':  # Inconsistency between train and test set layout
            data.rename(columns={'ID': 'AnimalID'}, inplace=True)
            data['AnimalID'] = data['AnimalID'].astype(np.uint32)
        elif self._which == 'train':
            # Animal ID always starts with A, and then that's a numeric value.
            # When we remove the leading A, the max value is 721113, so we convert
            # it to uint32.
            data['AnimalID'] = data['AnimalID'].str[1:].astype(np.uint32)
        else:
            assert False

        # DateTime: obvious conversion, but some components might be interesting
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data['month'] = data['DateTime'].dt.month.astype(np.uint8)
        data['week'] = data['DateTime'].dt.week.astype(np.uint8)
        data['weekDay'] = data['DateTime'].dt.dayofweek.astype(np.uint8)

        # AgeuponOutcome: converting it to years
        tmp = data['AgeuponOutcome'].str.split(' ', expand=True)
        tmp['years'] = np.nan
        tmp.loc[tmp[1].isin(('year', 'years')), 'years'] = tmp[0].astype(np.float16)
        tmp.loc[tmp[1].isin(('month', 'months')), 'years'] = tmp[0].astype(np.float16) / 12
        tmp.loc[tmp[1].isin(('week', 'weeks')), 'years'] = tmp[0].astype(np.float16) / 52
        tmp.loc[tmp[1].isin(('day', 'days')), 'years'] = tmp[0].astype(np.float16) / 365.25
        del tmp[0], tmp[1], data['AgeuponOutcome']
        data = pd.merge(data, tmp, left_index=True, right_index=True)
        del tmp

        # Name: I don't care about the value, but I care if the animal has one:
        # this might impact the outcome
        data['hasName'] = ~ data['Name'].isnull()
        del data['Name']

        # Outcome subtype: this value depends of the Y, and we don't have to
        # predict it. So, we can drop it.
        if self._which == 'train':
            del data['OutcomeSubtype']

        # Sex: splitting between sex and canReproduce
        tmp = data['SexuponOutcome'].str.split(' ', expand=True)
        tmp['isMale'] = tmp[1] == 'Male'
        tmp.loc[tmp[0] == 'Unknown', 'isMale'] = np.nan
        tmp['canReproduce'] = tmp[0] == 'Intact'
        tmp.loc[tmp[0] == 'Unknown', 'canReproduce'] = np.nan
        del tmp[0], tmp[1], data['SexuponOutcome']
        data = pd.merge(data, tmp, left_index=True, right_index=True)
        del tmp

        # Breed: pure or cross or mix
        # Mixity = fraction of components which are mix
        # Purity = fraction of components which are not the same (number of slashes)
        # Then, one column for the percentage of every breeds, no matter if mix or not
        #Extracting breeds - Attention: sometimes it is written as two different, but it is not the case!
        tmp = data['Breed'].str.split('/').apply(lambda x: '/'.join(set(x))).str.split('/', expand=True)
        # Computing purity and mixity percentages
        tmp['nBreeds'] = (~tmp.isnull()).sum(axis=1)
        nBreedsMax = tmp['nBreeds'].max()
        tmp['purityBreeds'] = (nBreedsMax + 1 - tmp['nBreeds']) / nBreedsMax
        tmp['nMixesBreeds'] = tmp.loc[:, [c for c in range(nBreedsMax)]].fillna("").apply(lambda x: x.str.contains('Mix'), axis=0).astype(int).sum(axis=1)
        tmp['mixityBreeds'] = tmp['nMixesBreeds'] / tmp['nBreeds']
        for c in range(nBreedsMax):
            tmp[c] = tmp[c].str.replace("Mix", "").str.strip()
        data = pd.merge(tmp.loc[: , ['purityBreeds', 'mixityBreeds']], data, left_index=True, right_index=True)
        #Extracting composing breeds
        tmp2 = tmp.copy()
        tmp2.reset_index(inplace=True, drop=False)
        tmp3 = pd.concat([tmp2[['index', i]].rename(columns={i: 'breed'}, inplace=False) for i in range(nBreedsMax)])
        tmp3.dropna(inplace=True)
        tmp3 = pd.merge(tmp3, tmp.loc[:, ['nBreeds', ]], left_on='index', right_index=True)
        tmp3['has'] = 1 / tmp3['nBreeds']
        del tmp3['nBreeds']
        tmp3['breed'] = 'breed' + tmp3['breed']
        tmp3 = tmp3.groupby(['index', 'breed']).agg(np.sum) # Because sometimes the components are the same drop_duplicates(inplace=True)
        tmp3 = tmp3.unstack()['has']
        tmp3.fillna(0, inplace=True)
        tmp3.columns.name = None
        tmp3.index.name = None
        tmp3 = tmp3.astype(np.float16)
        if 'breedUnknown' in tmp3.columns:
            del tmp3['breedUnknown']  # unknown breed is useless information...
        data = pd.merge(data, tmp3, left_index=True, right_index=True)
        del data['Breed']

        #Color:
        # Color mix
        # One column per color, giving her "purity" in fraction
        tmp = data['Color'].str.split('/').apply(lambda x: '/'.join(set(x))).str.split('/', expand=True)
        tmp['nColors'] = (~tmp.isnull()).sum(axis=1)
        nColorsMax = tmp['nColors'].max()
        tmp['purityColor'] = (nColorsMax + 1 - tmp['nColors']) / nColorsMax
        data = pd.merge(tmp.loc[:, ['purityColor', ]], data, left_index=True, right_index=True)
        tmp2 = tmp.copy()
        tmp2.reset_index(inplace=True, drop=False)
        tmp3 = pd.concat([tmp2[['index', i]].rename(columns={i: 'color'}, inplace=False) for i in range(nColorsMax)])
        tmp3.dropna(inplace=True)
        tmp3 = pd.merge(tmp3, tmp.loc[:, ['nColors', ]], left_on='index', right_index=True)
        tmp3['has'] = 1 / tmp3['nColors']
        del tmp3['nColors']
        tmp3['color'] = 'color' + tmp3['color']
        tmp3 = tmp3.groupby(['index', 'color']).agg(np.sum) # Because sometimes the components are the same drop_duplicates(inplace=True)
        tmp3 = tmp3.unstack()['has']
        tmp3.fillna(0, inplace=True)
        tmp3.columns.name = None
        tmp3.index.name = None
        tmp3 = tmp3.astype(np.float16)
        data = pd.merge(data, tmp3, left_index=True, right_index=True)
        del data['Color']

        # Animal type: getting dummies
        data = pd.get_dummies(data, columns=['AnimalType', ])

        # Renaming columns (my habits...), including dummies animal types
        newNames = {x: 'is%s' % x[11:] for x in data.columns if x[:10] == 'AnimalType'}
        newNames['DateTime'] = 'timestamp'
        newNames['AnimalID'] = 'animalId'
        data.rename(columns=newNames, inplace=True)

        # Converting the remaining objects to categories
        for c in data.columns:
            if data[c].dtype == 'object':
                data[c] = data[c].astype('category')

        # This is an index, not a meaningful feature...
        data.set_index('animalId', inplace=True)

        # So data type fixing:
        for c in 'hasName', 'isCat', 'isDog':
            data[c] = data[c].astype(np.float16)
        return data.copy()


class DataSetTrial2(object):
    def __init__(self, which):
        self._which = which
    def get(self, dropNa, withTimestamp):
        X, y = DataSetTrial1(self._which).get(dropNa, withTimestamp)
        for c in X.columns:
            if c.startswith('breed') or c.startswith('color'):
                del X[c]
        return X, y


class DataSetTrial3(AbstractDataSet):
    # Well, that's a stupid copy/paste of most of the DataSetTrial1 code, make
    # "just to try". No time to make it abstract and re-usable.

    def _breedToGroup(self, breeds):
        if I_AM_ON_KAGGLE:
            raise Exception("Cannot run this on Kaggle platform because an additional input file is needed")
        breedMap = pd.read_csv(BREED_MAP)
        breedMap.dropna(inplace=True)
        def backend(x):
            ret = []
            for b in x.split('/'):
                isMix = b.endswith('Mix')
                if isMix:
                    b = b[:-3].strip()
                filt = breedMap['breed'] == b
                nb = breedMap.loc[filt, :]['category']
                if nb.shape[0] > 0:
                    nb = list(nb)[0]
                else:
                    nb = 'Others'
                if isMix:
                    nb = '%s Mix' % nb
                ret.append(nb)
            return '/'.join(set(ret))
        vals = pd.DataFrame({'from': breeds.unique()})
        vals['to'] = vals['from'].apply(backend)
        ret = pd.DataFrame({'from': breeds})
        ret = pd.merge(ret, vals, on='from')
        ret.fillna("Others")
        ret = ret['to']
        ret.name='Breed'
        return ret

    def _mungle(self, data):
        if self._which == 'test':  # Inconsistency between train and test set layout
            data.rename(columns={'ID': 'AnimalID'}, inplace=True)
            data['AnimalID'] = data['AnimalID'].astype(np.uint32)
        elif self._which == 'train':
            # Animal ID always starts with A, and then that's a numeric value.
            # When we remove the leading A, the max value is 721113, so we convert
            # it to uint32.
            data['AnimalID'] = data['AnimalID'].str[1:].astype(np.uint32)
        else:
            assert False

        # DateTime: obvious conversion, but some components might be interesting
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data['month'] = data['DateTime'].dt.month.astype(np.uint8)
        data['week'] = data['DateTime'].dt.week.astype(np.uint8)
        data['weekDay'] = data['DateTime'].dt.dayofweek.astype(np.uint8)

        # AgeuponOutcome: converting it to years
        tmp = data['AgeuponOutcome'].str.split(' ', expand=True)
        tmp['years'] = np.nan
        tmp.loc[tmp[1].isin(('year', 'years')), 'years'] = tmp[0].astype(np.float16)
        tmp.loc[tmp[1].isin(('month', 'months')), 'years'] = tmp[0].astype(np.float16) / 12
        tmp.loc[tmp[1].isin(('week', 'weeks')), 'years'] = tmp[0].astype(np.float16) / 52
        tmp.loc[tmp[1].isin(('day', 'days')), 'years'] = tmp[0].astype(np.float16) / 365.25
        del tmp[0], tmp[1], data['AgeuponOutcome']
        data = pd.merge(data, tmp, left_index=True, right_index=True)
        del tmp

        # Name: I don't care about the value, but I care if the animal has one:
        # this might impact the outcome
        data['hasName'] = ~ data['Name'].isnull()
        del data['Name']

        # Outcome subtype: this value depends of the Y, and we don't have to
        # predict it. So, we can drop it.
        if self._which == 'train':
            del data['OutcomeSubtype']

        # Sex: splitting between sex and canReproduce
        tmp = data['SexuponOutcome'].str.split(' ', expand=True)
        tmp['isMale'] = tmp[1] == 'Male'
        tmp.loc[tmp[0] == 'Unknown', 'isMale'] = np.nan
        tmp['canReproduce'] = tmp[0] == 'Intact'
        tmp.loc[tmp[0] == 'Unknown', 'canReproduce'] = np.nan
        del tmp[0], tmp[1], data['SexuponOutcome']
        data = pd.merge(data, tmp, left_index=True, right_index=True)
        del tmp

        # Converting Breed to Breed group
        data['Breed'] = self._breedToGroup(data['Breed'])

        # Breed: pure or cross or mix
        # Mixity = fraction of components which are mix
        # Purity = fraction of components which are not the same (number of slashes)
        # Then, one column for the percentage of every breeds, no matter if mix or not
        #Extracting breeds - Attention: sometimes it is written as two different, but it is not the case!
        tmp = data['Breed'].str.split('/').apply(lambda x: '/'.join(set(x))).str.split('/', expand=True)
        # Computing purity and mixity percentages
        tmp['nBreeds'] = (~tmp.isnull()).sum(axis=1)
        nBreedsMax = tmp['nBreeds'].max()
        tmp['purityBreeds'] = (nBreedsMax + 1 - tmp['nBreeds']) / nBreedsMax
        tmp['nMixesBreeds'] = tmp.loc[:, [c for c in range(nBreedsMax)]].fillna("").apply(lambda x: x.str.contains('Mix'), axis=0).astype(int).sum(axis=1)
        tmp['mixityBreeds'] = tmp['nMixesBreeds'] / tmp['nBreeds']
        for c in range(nBreedsMax):
            tmp[c] = tmp[c].str.replace("Mix", "").str.strip()
        data = pd.merge(tmp.loc[: , ['purityBreeds', 'mixityBreeds']], data, left_index=True, right_index=True)
        #Extracting composing breeds
        tmp2 = tmp.copy()
        tmp2.reset_index(inplace=True, drop=False)
        tmp3 = pd.concat([tmp2[['index', i]].rename(columns={i: 'breed'}, inplace=False) for i in range(nBreedsMax)])
        tmp3.dropna(inplace=True)
        tmp3 = pd.merge(tmp3, tmp.loc[:, ['nBreeds', ]], left_on='index', right_index=True)
        tmp3['has'] = 1 / tmp3['nBreeds']
        del tmp3['nBreeds']
        tmp3['breed'] = 'breed' + tmp3['breed']
        tmp3 = tmp3.groupby(['index', 'breed']).agg(np.sum) # Because sometimes the components are the same drop_duplicates(inplace=True)
        tmp3 = tmp3.unstack()['has']
        tmp3.fillna(0, inplace=True)
        tmp3.columns.name = None
        tmp3.index.name = None
        tmp3 = tmp3.astype(np.float16)
        if 'breedUnknown' in tmp3.columns:
            del tmp3['breedUnknown']  # unknown breed is useless information...
        data = pd.merge(data, tmp3, left_index=True, right_index=True)
        del data['Breed']

        #Color:
        # Color mix
        # One column per color, giving her "purity" in fraction
        tmp = data['Color'].str.split('/').apply(lambda x: '/'.join(set(x))).str.split('/', expand=True)
        tmp['nColors'] = (~tmp.isnull()).sum(axis=1)
        nColorsMax = tmp['nColors'].max()
        tmp['purityColor'] = (nColorsMax + 1 - tmp['nColors']) / nColorsMax
        data = pd.merge(tmp.loc[:, ['purityColor', ]], data, left_index=True, right_index=True)
        tmp2 = tmp.copy()
        tmp2.reset_index(inplace=True, drop=False)
        tmp3 = pd.concat([tmp2[['index', i]].rename(columns={i: 'color'}, inplace=False) for i in range(nColorsMax)])
        tmp3.dropna(inplace=True)
        tmp3 = pd.merge(tmp3, tmp.loc[:, ['nColors', ]], left_on='index', right_index=True)
        tmp3['has'] = 1 / tmp3['nColors']
        del tmp3['nColors']
        tmp3['color'] = 'color' + tmp3['color']
        tmp3 = tmp3.groupby(['index', 'color']).agg(np.sum) # Because sometimes the components are the same drop_duplicates(inplace=True)
        tmp3 = tmp3.unstack()['has']
        tmp3.fillna(0, inplace=True)
        tmp3.columns.name = None
        tmp3.index.name = None
        tmp3 = tmp3.astype(np.float16)
        data = pd.merge(data, tmp3, left_index=True, right_index=True)
        del data['Color']

        # Animal type: getting dummies
        data = pd.get_dummies(data, columns=['AnimalType', ])

        # Renaming columns (my habits...), including dummies animal types
        newNames = {x: 'is%s' % x[11:] for x in data.columns if x[:10] == 'AnimalType'}
        newNames['DateTime'] = 'timestamp'
        newNames['AnimalID'] = 'animalId'
        data.rename(columns=newNames, inplace=True)

        # Converting the remaining objects to categories
        for c in data.columns:
            if data[c].dtype == 'object':
                data[c] = data[c].astype('category')

        # This is an index, not a meaningful feature...
        data.set_index('animalId', inplace=True)

        # So data type fixing:
        for c in 'hasName', 'isCat', 'isDog':
            data[c] = data[c].astype(np.float16)
        return data.copy()


def genericPredict(DataSet, predictor):
    # See the ml_exploration notebook for the choice of the algorithm (predictor)
    # We know the best algorithm, so we learn on all the available data.
    X_train, y_train = DataSet('train').get(dropNa=True, withTimestamp=False)
    X_test = DataSetTrial1('test').get(dropNa=False, withTimestamp=False)[0]
    print("Learning")
    clf = skmc.OneVsRestClassifier(predictor, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Aligning columns in train and test set")
    # The columns missing in test are about missing Breeds and Colors. So, it is 0.
    for c in [x for x in X_train.columns if (x not in X_test.columns and (x.startswith('color') or x.startswith('breed')))]:
        X_test[c] = np.float16(0)
    # The columns missing in train are about missing Breeds and Colors. Since we have not learnt with them, we remove it.
    for c in [x for x in X_test.columns if (x not in X_train.columns and (x.startswith('color') or x.startswith('breed')))]:
        del X_test[c]
    #del X_train, y_train
    print("Cleansing test set")
    # We must predict everything, so we cannot drop the NA
    X_test.fillna(X_test.mean(), inplace=True)
    print("Predicting")
    y_test = clf.predict(X_test)
    y_test = pd.DataFrame(y_test, columns=y_train.columns)
    y_test.index = X_test.index
    return y_test


def predict1(DataSet):
    return genericPredict(DataSet, sknb.GaussianNB())


def predict2(DataSet):
    # Because the best algo on the first munging is not the best with better munging
    return genericPredict(DataSet, sken.BaggingClassifier(sken.RandomForestClassifier(n_jobs=-1)))

def predict3(DataSet):
    # ...
    return genericPredict(DataSet, sksvm.Linar())

def predict4(DataSet):
    # ...
    return genericPredict(DataSet, sktr.DecisionTreeClassifier(max_depth=4))


if __name__ == "__main__":
    #DataSetTrial2 is the best up to now
    #We will replay the learner selection with that
    #predicted = predict1(DataSetTrial1)
    #predicted = predict1(DataSetTrial2)     # The best up to now
    #predicted = predict1(DataSetTrial3)
    #predicted = predict2(DataSetTrial1)
    predicted = predict4(DataSetTrial2)
    predicted.index.name = 'ID'
    predicted.reset_index(inplace=True)
    print("Saving predictions to predicted.csv")
    predicted.to_csv("predicted.csv", index=False)
