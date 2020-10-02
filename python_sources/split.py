import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split #GridSearchCV, 
import pandas as pd
import itertools

def splitter(index, s, seed=None, groups=None):
    """Creates Series of Booleans to be used in split. 
       The groups keyword is a series with index index, whose entries
       give the names of groups into which the indices are divided.
       If it is omitted, each index is assigned to a unique group.
       Expected proportion of false values of groups given by s.
       Indices in the same group are assigned the same truth values"""
    if isinstance(seed, int):
        np.random.seed(seed)
    if groups is None:
        groups = pd.Series(list(range(index.size)), index=index)
    group_names = groups.unique()
    train_group, test_group = train_test_split(group_names, test_size=s)
    return groups.isin(train_group)

def split(L, sp):
    """L: iterable of DataFrames, all having the same index
       sp: Series of Booleans, with the same index as the DataFrames in L"""
    return [X[sp] for X in L] + [X[~sp] for X in L]

class WeightedGroupKFold(KFold):
    """Similar to GroupKFold, except that this ensures that the number of groups in each fold is (almost) the same"""
    def __init__(self, n_splits, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.group_kfold = KFold(n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y=None, groups=None):
        try:
            index = X.index
        except:
            print("X must be a DataFrame")
        positions = pd.Series([i for i in range(index.size)], index=index)
        if groups is None:
            groups = pd.Series(np.arange(index.size), index=index) #list(range(index.size)) #
            group_values = groups.values
        else:
            groups = groups[index]     # should we require that this is true? Give a warning if it isn't?
            group_values = groups.unique()
        group_splits = self.group_kfold.split(group_values)
        test_bools = [groups.isin(group_values[test]) for train, test in group_splits]
        return  [(positions[~tb].values, positions[tb].values) for tb in test_bools]
    
class StratifiedWeightedGroupKFold(WeightedGroupKFold):
    """An extension of WeightedGroupKFold designed for binary classifiers that returns the n_repeats splittings out 
    of len(random_states) that have the lowest stratification error. The stratification error of a single split is the 
    absolute difference of proportions of positives in the test and training sets. The stratification error of a k-fold split
    is the maximum stratification error among all splits.
    
    Parameters
    ----------
    n_splits : int
        Number of folds in each split
    n_repeats : int
        Number of splits to use
    shuffle : bool
        Whether to shuffle the data or not
    random_states : iterable of ints
        Seeds for WeightedGroupKFold splits
    weights : None or pd.Series 
        If weights is a series, any X used in splitting must have X.index contained in weights.index
    """
    def __init__(self, n_splits=5, n_repeats=1, shuffle=False, random_states=np.arange(0,50), weights=None):
        self.n_splits = n_splits*n_repeats
        self.n0_splits = n_splits
        self.n_repeats = n_repeats
        self.weights = weights
        self.random_states = random_states
        
    def split(self, X, Y, groups=None):
        """Returns a list of n_splits * n_repeats splits, chosen so that the stratification error of each split is minimised
         
        Parameters
        ----------
        X : pd.DataFrame
            X.index must be contained in self.weights.index
        Y : pd.Series or np.array 
            Length must equal X.shape[0], and values must be either 0 or 1
        groups : None or pd.Series
            If groups is a series, groups.index must equal X.index
        """
        try:
            Y = Y.values
        except:
            pass
        if self.weights is None:
            weights = pd.Series(1, index=X.index)
        else:
            weights = self.weights[X.index]
        best_splits = pd.DataFrame(np.ones(self.n_repeats), dtype='float64', columns=['Discrepancies'])
        best_splits['Splits'] = None
        for seed in self.random_states:
            kfold = WeightedGroupKFold(n_splits=self.n0_splits, shuffle=True, random_state=seed)
            splits = kfold.split(X, Y, groups)
            d = []
            for t, v in splits:
                t_avg = np.average(Y[t], weights=weights.values[t])
                v_avg = np.average(Y[v], weights=weights.values[v])
                d.append(t_avg - v_avg)
            discrepancy = np.max(np.abs(np.array(d)))
            if discrepancy < best_splits['Discrepancies'].values[0]:
                best_splits['Discrepancies'].values[0] = discrepancy
                best_splits['Splits'].values[0] = splits
                best_splits.sort_values(by='Discrepancies', inplace=True, ascending=False)
        #print(best_splits['Discrepancies'].values)
        return list(itertools.chain.from_iterable(best_splits['Splits']))
    
    