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


tile_df = pd.read_csv( '../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')
tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get) 
tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes
tile_df.sample(3)
tile_df.describe(exclude=[np.number])


# In[ ]:


images=pd.read_csv('../input/dermatology-mnist-loading-and-processing/hsv.csv')
images28=pd.read_csv('../input/fork-of-vectorizing-and-other-techniques/hmnist_28_28_RGB.csv')
imageshu=pd.read_csv('../input/dermatology-mnist-loading-and-processing/mhu.csv')

#del images


# In[ ]:


#check  image label equals tiledf celltype
(images.label==tile_df.cell_type_idx).mean()


# In[ ]:


def hamming2(x,y):
    """Calculate the Hamming distance between two bit strings"""
    assert len(x) == len(y)
    count,z = 0,int(x,2)^int(y,2)
    while z:
        count += 1
        z &= z-1 # magic!
    return count

def hw(a):
    """Calculate the Hamming Weight of an input.
    Keyword arguments:
    a -- the input as an integer
    """
    if not isinstance(a, int):
        raise TypeError("Input parameter must be an integer.")

    count = 0
    while a:
        # Source: http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
        a &= a - 1
        count += 1

    return count


def hd(a, b):
    """Calculate the Hamming Distance of two inputs.
    Keyword arguments:
    a -- the first input as an integer
    b -- the second input as an integer
    """
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Input parameters must be an integer.")

    distance = hw(a ^ b)

    return distance


def ham2(s1,s2):
    return hamming_distance(s1, s2)
def hamming_distance(s1, s2):
    assert len(s1) == len(s2)
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))


# from PIL import Image
# import six
# import imagehash
# def plti(im, h=3, **kwargs):
#     """
#     Helper function to plot an image.
#     """
#     y = im.shape[0]
#     x = im.shape[1]
#     w = (y/x) * h
#     plt.figure(figsize=(w,h))
#     #plt.imshow(im, interpolation="none", **kwargs)
#     has1=imagehash.dhash(Image.fromarray(im,'RGB'))
#     imr=np.rot90(im)
#     plt.imshow(imr, interpolation="none", **kwargs)
#     has2=imagehash.dhash(Image.fromarray(imr,'RGB'))
#     imu=np.rot90(imr)
#     plt.imshow(imu, interpolation="none", **kwargs)
#     has3=imagehash.dhash(Image.fromarray(imu,'RGB'))
#     imn=np.rot90(imu)
#     plt.imshow(imn, interpolation="none", **kwargs)
#     has4=imagehash.dhash(Image.fromarray(imn,'RGB'))
#     print(has1,has2,has3,has4)
#     print(ham2(str(has1),str(has2)),ham2(str(has2),str(has3)),ham2(str(has3),str(has4)),ham2(str(has1),str(has4)),ham2(str(has1),str(has1)))
#     plt.axis('off')
# 
# 
# 
# 
# img = images.iloc[9,:-1].values.reshape(64,64,3) # reshapes 784 => 28x28
# plti(img)
# 

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


imageshu


# In[ ]:


tile_df[['dx','dx_type','age','sex','localization','cell_type']].head(20)


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
              # MLPClassifier(alpha=0.510,activation='logistic'),
               LinearDiscriminantAnalysis(),
               #OneVsRestClassifier(GaussianNB()),
               #AdaBoostClassifier(),
               #GaussianNB(),
               #QuadraticDiscriminantAnalysis(),
               SGDClassifier(average=True,max_iter=100),
               XGBClassifier(max_depth=5, base_score=0.005),
               #LogisticRegression(C=1.0,multi_class='multinomial',penalty='l2', solver='saga',n_jobs=-1),
               LabelPropagation(n_jobs=-1),
               LinearSVC(),
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
    ncomp=int(e_.shape[1]/5)
    model = SelectFromModel(clf, prefit=True,threshold =(thres)/1000)
       # SVD
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=ncomp, n_iter=7, random_state=42)
    #svd transformation is trying to throw away the noise
    #e_=svd.fit_transform(e_)


       #tsne not used
    from sklearn.manifold import TSNE
    #e_=TSNE(n_components=3).fit_transform(e_)
    #from sklearn.metrics.pairwise import cosine_similarity
    
    #robustSVD not used
    #A_,e1_,e_,s_=robustSVD(e_,140)
    clf = clf.fit( e_[:len(mtrain)], label)
    New_features = model.transform( e_[:len(mtrain)])
    Test_features= model.transform(e_[-len(mtest):])
    #New_features =  e_[:len(mtrain)]
    #Test_features= e_[-len(mtest):] 
    pd.DataFrame(New_features).plot(x=0,y=1,c=mtrain[veld]+1,kind='scatter',title='classesplot',colormap ='jet')
    pd.DataFrame(np.concatenate((New_features,Test_features))).plot.scatter(x=0,y=1,c=['r' for x in range(len(mtrain))]+['g' for x in range(len(mtest))])    

    print('Model with threshold',thres/1000,New_features.shape,Test_features.shape,e_.shape)
    print('____________________________________________________')
    
    Model = []
    Accuracy = []
    for clf in Classifiers:
        #train
        fit=clf.fit(New_features,label)
        if clf.__class__.__name__=='DecisionTreeClassifier':
            treeprint(clf)
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


#images=images.reset_index()
#images=(images.T.append(images28.drop('label',axis=1).T)).T
#images=(images.T.append(imageshu.drop('label',axis=1).T)).T
#images=(images.T.append(images8L.drop('label',axis=1).T)).T
#images=(images.T.append(tile_df[['dx_type','age','sex','localization']].T)).T

images=tile_df[['dx_type','age','sex','localization']]
images=images.reset_index()
images=(images.T.append(imageshu.T)).T

from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(images.drop(['label'],axis=1),images['label'], test_size=0.2, random_state=42)
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

