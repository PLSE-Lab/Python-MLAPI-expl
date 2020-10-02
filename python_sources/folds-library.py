import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

#Any files you save will be available in the output tab below
train.to_csv('copy_of_the_training_data.csv', index=False)

#defining a generator of StratifiedKFold named StratifiedKFold_generator
def StratifiedKFold_generator(X, n_folds = 10,features=None,label=False,labels=None):
    
    """
    Generate StratifiedKFold for multiclass features or label.
    This method does not require a (multi)classification problem.
    This method returns a tuple(features_chosen,folds)
    
    features_chosen is the list of features chosen to perform Stratified-K-Folds.
    folds is the dictionary of folds created by Stratified-K-Folds. 
    Its keys are the index of the features, its values the fold itself.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.
        
    n_folds : int >=2
    Number of folds to create for each available feature
    Default value is set to 10  
    
    features : list, default 'None'
    List of the features to consider when performing StratifiedKFold.
    When 'None', all features are considered. 
              
    label : Boolean, default value is 'False'
    Indicates if the labels are to be tested, i.e. if the problem is a (multi)classification
    If set to "True", a label array is to be given to the algorithm.
            
    labels : array-like, shape(n_samples,1), default value is 'None'
    Labels of the dataset. For now this method only works for simple array(1D)          
    
    
    
    """
    
    from sklearn.cross_validation import StratifiedKFold
    import numpy as np
    # Getting the number of features and the number of samples
    (n_samples,n_features)=X.shape
    
    # Initializing our dictionary : keys are the indices choosen, values are the StratifiedKFold generator induced 
    folds= {}
    
    # We explore all the features to see which one can be choosen to create Stratified-K-Folds
    if features ==None:
        iterator = range(0,n_features-1)
    else :
        iterator = features
        
    for i in iterator:
        (unique,times)= np.unique(X[:,i],return_counts = True) 
        # unique is a list of the different values in X[;,i]
        # times is a list of the number of times each of the unique values 
        # comes up in X[:,i]
        if len(unique)<= n_folds : # our feature seems multi-class 
            minimun = times[0]
            for j in range(1,len(unique)-1):
                if times[j]<minimun :
                    minimum = times[j]
            if minimum >= n_folds : 
                # our feature will do just fine with StratifiedKFolds
                fold_i = StratifiedKFold(X[:,i],n_folds = n_folds,shuffle=True)    
                folds[i] = fold_i
    
    # We finish by using the labels if told to do so :
    if label==True:
        fold_labels = StratifiedKFold(labels,n_folds = n_folds,shuffle=True)    
        folds["labels"] = fold_labels
    
    # We want to know which features and/or label have been chosen to perform StratifiedKFolds.
    features_chosen=[]
    
    for key in folds.keys():
        features_chosen.append(key)          
    
    return (features_chosen,folds)


# defining a super generator which includes all kinds of stratified ways of splitting a dataset
def Stratified_generator(X, n_folds = 10,n_iter = 10, features=None,label=False,labels=None,method=["Fold","ShuffleSplit"]):
    
    """
    Generate StratifiedKFold and/or StratifiedShuffleSplit for multiclass features or label.
    This method does not require a (multi)classification problem.
    This method returns a tuple(features_chosen,folds)
    
    features_chosen is the list of features chosen to perform the desired splitting.
    folds is the dictionary of folds created by Stratified-K-Folds. 
    Its keys are the index of the features, its values the fold itself.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.
        
    n_folds : int >=2, necesary for StratifiedKFold
    Number of folds to create for each available feature
    Default value is set to 10  
    
    n_iter : int, necesary for StratifiedShuffleSplit
    Number of re-shuffling and splitting iterations.
    Default value is set to 10
    
    features : list, default 'None'
    List of the features to consider when performing StratifiedKFold.
    When 'None', all features are considered. 
              
    label : Boolean, default value is 'False'
    Indicates if the labels are to be tested, i.e. if the problem is a (multi)classification
    If set to "True", a label array is to be given to the algorithm.
            
    labels : array-like, shape(n_samples,1), default value is 'None'
    Labels of the dataset. For now this method only works for simple array(1D)          
    
    method: list of strings. 
    Indicates which methods to use to generate the folds.
    "Fold" and "Split" are the only possible values for now.
    
    """
    
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.cross_validation import StratifiedShuffleSplit
    import numpy as np
    # Getting the number of features and the number of samples
    (n_samples,n_features)=X.shape
    
    # Initializing our dictionaries : keys are the indices choosen, values are the StratifiedKFold and/orStratifiedShuffleSplit generator induced 
    folds = {}
    splits = {}
    
    # We explore all the features to see which one can be choosen to create Stratified-K-Folds
    if features ==None:
        iterator = range(0,n_features-1)
    else :
        iterator = features
        
    for i in iterator:
        (unique,times)= np.unique(X[:,i],return_counts = True) 
        # unique is a list of the different values in X[;,i]
        # times is a list of the number of times each of the unique values 
        # comes up in X[:,i]
        if len(unique)<= n_folds : # our feature seems multi-class 
            minimun = times[0]
            for j in range(1,len(unique)-1):
                if times[j]<minimun :
                    minimum = times[j]
            if minimum >= n_folds : 
                if "Fold" in method :
                    fold_i = StratifiedKFold(X[:,i],n_folds = n_folds,shuffle=True)    
                    folds[i] = fold_i
                if "ShuffleSplit" in method:
                    split_i = StratifiedShuffleSplit(X[:,i],n_iter = n_iter)
                    splits[i]=split_i
    
    # We finish by using the labels if told to do so :
    if label==True:
        if "Fold" in method:
            fold_labels = StratifiedKFold(labels,n_folds = n_folds,shuffle=True)    
            folds["labels"] = fold_labels
        if "ShuffleSplit" in method:
            split_label = StratifiedShuffleSplit(labels,n_iter = n_iter)
            splits["labels"]=split_label
    # We want to know which features and/or label have been chosen to perform StratifiedKFolds.
    features_chosen=[]
        
    for key in folds.keys():
        features_chosen.append(key) 
   
    return (features_chosen,folds,splits)
