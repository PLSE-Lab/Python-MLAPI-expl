#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
base_skin_dir = os.path.join('..', 'input')


# # read prepare further data
# * fill in lesions, imagepath,celltype and cancer type
# * read mnist28 file since this one has prepared a compressed format of all images, lets have a look if this files does the job
# * check if the minst file label is the same as the tile cell type idx
# * encode the tile_df such that the basic fields become all numeric

# In[ ]:


imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'melanoma ',   #this is an error in the other scripts
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


# In[ ]:


tile_df = pd.read_csv(os.path.join(base_skin_dir, '../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv'))
tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get) 
tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes
tile_df.sample(3)
tile_df.describe(exclude=[np.number])


# In[ ]:


images=pd.read_csv('../input/dermatology-mnist-loading-and-processing/hmnist_64_64_RGB.csv')
imagesL=pd.read_csv('../input/dermatology-mnist-loading-and-processing/hmnist_128_128_L.csv')


# In[ ]:


#check  image label equals tiledf celltype
(images.label==tile_df.cell_type_idx).mean()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
Encoder_X = LabelEncoder() 
for col in tile_df.columns:
    if tile_df.dtypes[col]=='object':
        tile_df[col]=col+'_'+tile_df[col].map(str)
        tile_df[col] = Encoder_X.fit_transform(tile_df[col])
Encoder_y=LabelEncoder()
#tile_df


# In[ ]:


tile_df[['dx','dx_type','age','sex','localization','cell_type']].head()


# # define three functions
# * list of classifiers
# * tree function
# * classification function

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC,SVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import PassiveAggressiveClassifier,Perceptron,LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier,MLPRegressor,BernoulliRBM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.svm import LinearSVR,SVC
from sklearn.utils import check_array


class StackingEstimator(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed
    
Classifiers = [
               #Perceptron(n_jobs=-1),
               #SVR(kernel='rbf',C=1.0, epsilon=0.2),
               #CalibratedClassifierCV(LinearDiscriminantAnalysis(), cv=4, method='sigmoid'),    
               #OneVsRestClassifier( SVC(    C=50,kernel='rbf',gamma=1.4, coef0=1,cache_size=3000,)),
               KNeighborsClassifier(10),
               DecisionTreeClassifier(),
               RandomForestClassifier(n_estimators=200),
               ExtraTreesClassifier(n_estimators=250,random_state=0), 
               OneVsRestClassifier(ExtraTreesClassifier(n_estimators=10)) , 
               #MLPClassifier(alpha=0.510,activation='logistic'),
               LinearDiscriminantAnalysis(),
               #OneVsRestClassifier(GaussianNB()),
               AdaBoostClassifier(),
               #GaussianNB(),
               #QuadraticDiscriminantAnalysis(),
               SGDClassifier(average=True,max_iter=100),
               XGBClassifier(max_depth=5, base_score=0.005),
               LogisticRegression(C=1.0,multi_class='multinomial',penalty='l2', solver='saga',n_jobs=-1),
               LabelPropagation(n_jobs=-1),
               #LinearSVC(),
               #MultinomialNB(alpha=.01),    
                   make_pipeline(
                    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
                    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
                    AdaBoostClassifier()
                ),

              ]


# In[ ]:


def treeprint(estimator):
    # Using those arrays, we can parse the tree structure:

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold


    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))
    print()
    return


# In[ ]:


def klasseer(e_,mtrain,mtest,veld,idvld,thres,probtrigger):
    # e_ total matrix without veld, 
    # veld the training field
    #thres  threshold to select features
    label = mtrain[veld]
    # select features find most relevant ifo threshold
    #e_=e_[1:]
    clf = ExtraTreesClassifier(n_estimators=100)
    ncomp=int(e_.shape[1]-3)
    model = SelectFromModel(clf, prefit=True,threshold =(thres)/1000)
       # SVD
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=ncomp, n_iter=7, random_state=42)
    #svd transformation is trying to throw away the noise
    #e_=svd.fit_transform(e_)


       #tsne not used
    from sklearn.manifold import TSNE
    e_=TSNE(n_components=3).fit_transform(e_)
    #from sklearn.metrics.pairwise import cosine_similarity
    
    #robustSVD not used
    #A_,e1_,e_,s_=robustSVD(e_,140)
    clf = clf.fit( e_[:len(mtrain)], label)
    New_features = model.transform( e_[:len(mtrain)])
    Test_features= model.transform(e_[-len(mtest):])
    #New_features =  e_[:len(mtrain)]
    #Test_features= e_[-len(mtest):] 
    pd.DataFrame(New_features).plot.scatter(x=0,y=1,c=mtrain[veld]+1)
    pd.DataFrame(np.concatenate((New_features,Test_features))).plot.scatter(x=0,y=1,c=['r' for x in range(len(mtrain))]+['g' for x in range(len(mtest))])    

    print('Model with threshold',thres/1000,New_features.shape,Test_features.shape,e_.shape)
    print('____________________________________________________')
    
    Model = []
    Accuracy = []
    for clf in Classifiers:
        #train
        fit=clf.fit(New_features,label)
        #if clf.__class__.__name__=='DecisionTreeClassifier':
            #treeprint(clf)
        pred=fit.predict(New_features)
        Model.append(clf.__class__.__name__)
        Accuracy.append(accuracy_score(mtrain[veld],pred))
        #predict
        sub = pd.DataFrame({idvld: mtest[idvld],veld: fit.predict(Test_features)})
        sub.plot(x=idvld,kind='kde',title=clf.__class__.__name__ +str(( mtrain[veld]==pred).mean()) +'prcnt') 
        sub2=pd.DataFrame(pred,columns=[veld])
        #estimate sample if  accuracy
        if veld in mtest.columns:
            print( clf.__class__.__name__ +str(round( accuracy_score(mtrain[veld],pred),2)*100 )+'prcnt accuracy versus unknown',(sub[veld]==mtest[veld]).mean() )
        #write results
        klassnaam=clf.__class__.__name__+".csv"
        sub.to_csv(klassnaam, index=False)
        if probtrigger:
            pred_prob=fit.predict_proba(Test_features)
            sub=pd.DataFrame(pred_prob)
    return sub


# # Split in train and test
# * make a numeric file without the index field since this fields has some 'forecasting value in the tree classifiers methods'
# * dx must be dropped too

# In[ ]:


images=images.reset_index()
images=(images.T.append(tile_df[['dx_type','age','sex','localization']].T)).T
images=(images.T.append(imagesL.drop('label',axis=1).T)).T
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images.drop(['label'],axis=1),images['label'], test_size=0.2, random_state=42)


# # tam tam, here are the results
# * suprisingly i can get a 100% correctforecast 
# * index leaks  information...
# * remove index
# 

# In[ ]:


totaal=(X_train.append(X_test)).fillna(0)
totaal=totaal.drop('index',axis=1).values

subx=klasseer(totaal,(X_train.T.append(y_train.T)).T,(X_test.T.append(y_test.T)).T,'label','index',0.01,False)

