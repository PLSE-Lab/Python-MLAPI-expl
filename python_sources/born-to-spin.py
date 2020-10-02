#!/usr/bin/env python
# coding: utf-8

# # Theoretical Introduction
# __This is my first kernel on kaggle. I would love to receive some feedback or opinions about it !__ 
# > __(If you are only interested only in machine learning you can skip that part and go to Machine Learning Part).__ <br>
# 
# <br>
# The universe is made of many fascinating astronomical objects which we don't fully understand yet. After explosive transformation of massive star an 
# amazing object, Pulsar, is born ( first detection in 1967 by Jocelyn Bell [1], theoretically predicted in 1934 by Baade & Zwicky [2]).
# <br> What are they ? Let's understand them better and why they are unique with study of properties of the first known pulsar ( PSR B1919+21 ). <br> 
# <br>
# They are called pulsars because of their first amazing property ; they spin really fast ! the first observed pulsar has a spin period of 1.3373 second. there are pulsars that spin as fast as 700 times per second ( for example PSR J1748-2446ad ) ! 
# To compare with more everyday values to get sense how much that is : 
# * Blender spins ~ 380 times per second [4]
# * Typical motor used for drones ( they spin really fast if you played a bit with them ) spins ~ 260 times per second (if we assume motor 1400 KV with 11.1 LiPol Battery ). 
# * Car that travels with 72 km / h spins arround 40 times per second ! <br>
# <br>If  you are curious, you can listen the signal converted to audio file here ( __Warning__ : some audio might not be really pleasant and can be really loud ) :
# https://www.parkes.atnf.csiro.au/people/sar049/eternal_life/supernova/0329.wav or here http://www.jb.man.ac.uk/research/pulsar/Education/Sounds/
# <br><br>

# Another amazing property : their size and mass. Imagine a sphere with a 10 km radius ( to get more sense of how small that value is ( for a star) check map in next cell ( you can change location by changing default longitude and latitude parameters)). 
# But that sphere has mass of approximately 1.4 Mass of Sun (!). The Density of that object is $4.75 \cdot 10^{17} \frac{kg}{m^{3}}$ ( Highest known density in universe ). 
# <br>

# In[ ]:


import folium

# Change these to your city
latitude_def  = 51.509865
longitude_def = -0.118092

m = folium.Map(location=[latitude_def, longitude_def])
folium.Circle(radius=10000,
        location=[latitude_def, longitude_def],
        popup='PSR B1919+21',
        color='blue',
        fill=True,).add_to(m)
m


# To better understand how enormously big that density value is,  let's conduct a thought experiment to visualize that better. <br> There are 7 billions humans on the planet. let's approximate that average
# weight of human is 70 kg. So total mass of every human is $ 7 \cdot 10^{9} \cdot 70 = 4.9 \cdot 10 ^{11} kg$. 
# <br>
# 
# Now the density of sugar cube is 4.93 cubic centimeters [3] which is $ 4.93 \cdot 10^{-6} m^{3}$. If we take a sugar cube of stuff 
# from neutron star ( some literature refer to it as neutron-degenerate matter, other as neutronium ; it's extremly weird and interesting material ( just google "nuclear pasta" )) then weight of sugar cube made of neutron star is $ 4.75 \cdot 10^{17} \cdot 4.93 \cdot 10 ^{-6} = 23.4175 \cdot 10^{11} kg $. <br> This is equal to almost 5 earths of 7 billions humans !
# <br><br>
# Yet another amazing property is their magnetic field with values reaching as high as $ 10^{15} G $ ( It is highest known magnetic field in universe, earth has $0.5 G$ ). 
# Projecting such enormous value into real world is much harder than angular velocity or mass but here are some interesing facts that will hopefully help you understand how much that is. 
# * famous diamagnetic frog levitation[5] required magnetic field of $ 1.6 * 10^{5} G $ 
# * If between moon and earth there would be magnetar, it would erase all data from credit cards.
# * On December 27, 2004 Starquake ( like earthquake but for stars ) on magnetar ( type of neutron star with enormous magnetic field (reaching $10^{15} $))  caused just enormous amount of energy ( in one burst it released same energy as our sun in ~ 800 years ).  That amount energy would eliminate all forms of biological life in radius of 10 light years [6].

# Pulsars are currently area of many scientific research. It's fascinating object of theoretical research ( testing general relativity, famous example would be observation of gravitational waves from 2 colliding neutron stars [7]) and might even have practical practical application : because their relatively stable ( very slow down rate reaching $ 10^{-20} s / s $ for milisecond pulsars )
# rotation speed, you can measure time with them ( pulsar clock [8] ). 
# 
# More Information you can find at : <br> http://www.astro.umd.edu/~miller/nstar.html,<br>
# https://www.cv.nrao.edu/course/astr534/Pulsars.html. <br>
# https://en.wikibooks.org/wiki/Pulsars_and_neutron_stars <br><br>
# [1] Publication : Observation of a Rapidly Pulsating Radio Source,  A. Hewish, S. J. Bell, J. D. H. Pilkingon, P. F. Scott, R. A. Collins <br>
# [2] Publication : On Super-novae W. Baade; F. Zwicky <br>
# [3] https://www.quora.com/What-is-the-volume-of-universe-in-sugar-cubes <br>
# [4] https://hackaday.com/2014/06/13/do-you-have-any-idea-how-fast-your-blender-was-going/ <br>
# [5]  Publication : Diamagnetic levitation: Flying frogs and floating magnets <br>
# [6] http://news.bbc.co.uk/2/hi/science/nature/4278005.stm <br>
# [7] Publication : GW170817: Observation of Gravitational Waves from a Binary Neutron Star Inspiral <br>
# [8] https://en.wikipedia.org/wiki/Pulsar_clock

# # Machine learning part
# ## Exploratory data analysis and Data Processing 
# Up to this point all of the feature engineering is done by extracting key features from time series data. To data I will only apply standard scaling ( dimensionality reduction is only used to visualize data because transforming to lower dimensionality and feeding it to machine learning models didn't produce any meaningful improvements ). 

# In[ ]:


# utils 
import os, sys
import pandas as pd
import numpy as np

# machine learning
import sklearn as sk
import statsmodels as sms 

# plotting
import plotly_express
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')

data_folder = '/kaggle/input/predicting-a-pulsar-star/'
os.listdir(data_folder)


# In[ ]:


df = pd.read_csv(os.path.join(data_folder,"pulsar_stars.csv"))
df.head()


# In[ ]:


df.describe()


# |Before                                       |  After 
# |---------------------------------------------|--------
# |Mean of the integrated profile               | meanIP 
# |Standard deviation of the integrated profile | stdIP
# |Excess kurtosis of the integrated profile    | kurtIP
# |Skewness of the integrated profile           | skewIP
# |Mean of the DM-SNR curve                     | meanDS
# |Standard deviation of the DM-SNR curve       | stdDS
# |Excess kurtosis of the DM-SNR curve	      | kurtDS
# |Skewness of the DM-SNR curve                 | skewDS
# |target_class                                 | target

# In[ ]:


names = ['meanIP', 'stdIP', 'kurtIP', 'skewIP', 'meanDS', 'stdDS', 'kurtDS', 'skewDS', 'target']
df.columns = names


# In[ ]:


plt.style.use('ggplot')
plt.figure(figsize=(8,8))

def plotHist(df):
    plt.title("Percent of pulsars {} %".format( sum(df == 1) / len(df) * 100))
    plt.bar(['not pulsar', 'pulsar'], [sum(df == 0), sum(df == 1)])
    
plotHist(df['target'])


# In[ ]:


sns.set(style="ticks", color_codes=True)
sns.pairplot(df, hue='target', markers=['o', 's'], vars =names[:-1])
plt.show()


# Many of the values are highly correlated ( by applying some transformatiations like squaring, transforming to hiperbolic we can get kinda linear dependencies) with each other ( which was expected based on parameters descriptions ).

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_features, labels = df[names[:-1]], df[names[-1]]
X_features_scaled, Y_features_scaled = df[names[:-1]].copy(), df[names[-1]].copy()
print("All        {} : {}".format(X_features.shape, labels.shape))


# In[ ]:


scaler = StandardScaler()

# stratification by class and shuffling are default 
X_train, X_test, Y_train, Y_test = train_test_split(X_features, labels, test_size = 0.1, random_state = 42)
X_train, X_val, Y_train, Y_val  = train_test_split(X_train, Y_train, test_size=2/9, random_state=42)

# X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled = train_test_split(X_features_scaled, Y_features_scaled, test_size = 0.1, random_state = 42)
# X_train_scaled, X_val_scaled, Y_train_scaled, Y_val_scaled  = train_test_split(X_train_scaled, Y_train_scaled, test_size=2/9, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)
X_test_scaled = scaler.fit_transform(X_test)

print("Training   {} : {}".format(X_train.shape, Y_train.shape))
print("Validation {}  : {}".format(X_val.shape, Y_val.shape))
print("Testing    {}  : {}".format(X_test.shape, Y_test.shape))


# we will try few well-performing classifers without thinking much about feature engineering ( most of it is actually done for us, only thing I did is to apply standard scaling ). <br><br>
# 
# ### About Training 
# Because we deal with inbalanced dataset, our accuracy metric will be F1 score ( very common metric for inbalanced datasets, defined as harmonic mean of precision and recall ), confusion matrix, precision-recall and roc curve. We will perform grid search on hyperparameters of models, evaluate model on validation dataset ( In this notebook i don't perform KFold ) and return model and predictions of model that performs the best on validation dataset. Then we evaluate that model on testing data. 

# In[ ]:


from itertools import product

# Here's poor implementation of grid searching through parameters and picking the best 
# Implemented because i couldn't find a way ( maybe it's simple ) to call sklearn GridSearchCV 
# and evaluate classifier from that on validation set
def iterDicts(options):
    values = [options[key] for key in options.keys()]
    return [dict(zip(options.keys(), it)) for it in product(*values)]

def GridSearchAndCrossValidate(Classifier, options, X_train_scaled, Y_train_scaled, 
                               X_val_scaled, Y_val_scaled, accuracy_metric, verbose=False):
    best_acc = 0
    best_classifier = None

    for option in iterDicts(options):
        classifier = Classifier(**option)
        classifier.fit(X_train_scaled, Y_train_scaled)
        predictions = classifier.predict(X_val_scaled)
        accuracy = accuracy_metric(predictions, Y_val_scaled)
    
        if accuracy > best_acc:
            best_acc = accuracy
            best_classifier = classifier
    
        if verbose:
            print("for {} val acc is {}".format(option, accuracy))
    if verbose:
        print("-"*32)
        print("Final best accuracy {}\n for {}".format(best_acc, best_classifier))
        print("-"*32)
    
    return best_classifier, best_acc


# Below is rather ugly solution of searching through hiperparameters ( should be in config file ). 

# In[ ]:


from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve, roc_auc_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from time import time

class model():
    def __init__(self, Model, plotName, options):
        self.model = Model
        self.plotName = plotName
        self.options = options
        
# not really pretty solution.
# this should be in some sort of config file ( json or whatever) 
models = [ model(SVC, "SVC", {"kernel": ['rbf', 'poly'], "C": [0.5, 1, 2, 4, 6, 8, 10], 'random_state':[42], 'probability':[True]}),
           model(DecisionTreeClassifier, "Tree",  {'min_samples_split' : [2, 4, 6, 8, 10, 15, 20, 25] , "criterion" : ["gini", 'entropy'], 'random_state' : [42]}),
           model(RandomForestClassifier, "R. forest", {'criterion' : ['gini', 'entropy'], "max_depth": [5, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100] , 'n_estimators': [10, 50, 100, 250] ,'random_state' : [42], 'n_jobs': [-1]}),
           model(AdaBoostClassifier, "AdaBoost", {'n_estimators' : [10, 100, 500], 'random_state' : [42]}),
           model(BaggingClassifier, "Bagging", {'n_estimators' : [2, 5, 10, 15], 'n_jobs' : [-1], 'random_state' : [42]}),
           model(MLPClassifier, "neural network", {"hidden_layer_sizes" : [(64, 1), (64, 2), (64, 3)], 'learning_rate': ['constant', 'invscaling', 'adaptive'], 'max_iter' : [500], 'random_state' : [42]}),
           model(LGBMClassifier, "lightGBM", {'boosting_type' : ['gbdt', 'dart', 'goss'], "num_leaves" : [16, 32, 64 ], 'max_bin' : [50, 100, 200, 400], 'learning_rate' : [0.005, 0.01, 0.1], 'num_iterations' : [100, 200], 'random_state' : [42]}),
           model(XGBClassifier, "xgboost", {"booster" : ["gbtree", 'gblinear'], 'n_jobs' : [-1],  'max_depth' : np.arange(3, 11,1), 'learning_rate' : [0.01, 0.05, 0.1, 0.2], 'random_state' : [42]})]


def plot_sklearn_models(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):

    accuracies = {}
    plt.style.use('ggplot')
    plt.figure(figsize=(len(models) * 2, len(models) * 5)) 
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    cnt = 1
    
    for model in models:

        t = time()
        # perform grid Search
        classifier, acc = GridSearchAndCrossValidate(model.model, model.options, X_train, Y_train, 
                                             X_valid, Y_valid, f1_score, verbose=False)
        print("Trained {} in {} seconds ".format(model.plotName, time() - t))
    
        # plot confusion matrix 
        plt.subplot(len(models), 3, cnt)
        test_predictions = classifier.predict(X_test)
        
        # our accuracy metric for test data
        acc = f1_score(test_predictions, Y_test)
        
        # calculate confustion matrix from test data
        conf = confusion_matrix(test_predictions,Y_test) 
        plt.title("Conf matrix {0}, F1 {1: .3f}".format(model.plotName, acc))
        sns.heatmap(conf, annot=True, annot_kws={"size": 16})

    
        # plot precision recall curve 
        plt.subplot(len(models), 3, cnt+1)
        
        # get probabilities of classes from test data from clasifier 
        # ( for SVM probability parameter must be set to True)
        probs = classifier.predict_proba(X_test)[:, 1]
        
        # create precision recall curve for testing data
        precision, recall, _ = precision_recall_curve(Y_test, probs)
        plt.title("Precision Recall curve")
        plt.step(precision, recall)
        plt.xlabel("precision")
        plt.ylabel("recall")
        plt.fill_between(precision, recall, alpha=.2, step = 'pre')
        
        plt.subplot(len(models), 3, cnt+2)        
        # calculate area under curve from ROC
        roc_score = roc_auc_score(Y_test, probs)
        plt.title("AUC-ROC {0:.3f}".format(roc_score))
        
        # Roc Curve
        fpr, tpr, threshs = roc_curve(Y_test, probs)
        plt.plot(fpr, tpr)
        plt.plot([[0, 0], [1, 1]], linestyle='dashed')
        cnt += 3
        
        accuracies[model.plotName] = {"F1" : acc, "roc_auc" : roc_score, "classifier" : classifier} 
        
    plt.show()
    return accuracies
    
sample_accuracies = plot_sklearn_models(X_train_scaled, Y_train, X_val_scaled, Y_val, X_test_scaled, Y_test)


# Below is detailed list of best performing models on validation dataset ( with all hiperparameters and accuracies )

# In[ ]:


sample_accuracies


# Models are performing rather well (Even simple tree is performing fairly well ). Interesting thing is that every model make kinda simillar number of mistakes of true pulsars ( 19-28, about 12-17.5 % of all true pulsars ). That samples be can result of noisy dataset ( not enough features to distinguish between not pulsar and pulsar or they just outline cases ).

# ## Latent space transformations 
# We can also try project that dataset to lower dimensional space ( just to have a chance of data visualization). There are many algorithms for dimensionality reducation but i will apply 2 of them ( mainly because applying other didn't produce satisfying result ) : PCA ( most common one, it doesn't produce good result but I'm using it as a reference point ( many other projections looked same way )) and UMAP (recently developed, very good manifold learning algorithm [ __UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction__] ). 

# In[ ]:


s = StandardScaler()
features_scaled = s.fit_transform(X_features)


# In[ ]:


from sklearn.decomposition import PCA, KernelPCA
import plotly_express as pe

string_labels = ((labels.values).copy()).reshape(-1, 1).astype("str")
string_labels[string_labels == '0'] = "not pulsar"
string_labels[string_labels == '1'] = "pulsar"

def get_lower_dim_space(algorithm, options, features, labels):
    transformer = algorithm(**options)
    latent_space = transformer.fit_transform(features)
    all_data     = np.concatenate([latent_space, labels.reshape(-1, 1)], axis=1)
    
    if latent_space.shape[1] == 2:
        return pd.DataFrame(all_data, columns=['x', 'y', 'label']) 
    else:
        return pd.DataFrame(all_data, columns=['x', 'y', 'z', 'label'])
        
df = get_lower_dim_space(KernelPCA, {'kernel':"linear", "n_components" : 2}, features_scaled, string_labels)
pe.scatter(df, "x", "y", "label")


# In[ ]:


df = get_lower_dim_space(KernelPCA, {'kernel':"linear", "n_components" : 3}, features_scaled, string_labels)
pe.scatter_3d(df, "x", "y", "z", "label")


# Unfortunetly, PCA doesn't separate properly clusters ( There are many outliers and we don't have separation between clusters).
# Other great methods such as Isomap also didn't produce good results. 

# Now we can switch to wonderfully performing umap which actually produce nice result.

# In[ ]:


from umap import UMAP
df = get_lower_dim_space(UMAP, { "n_components" : 2, 'n_neighbors':15}, features_scaled, string_labels)
pe.scatter(df, "x", "y", "label")


# In[ ]:


df = get_lower_dim_space(UMAP, { "n_components" : 3, "n_neighbors": 15}, features_scaled, string_labels)
pe.scatter_3d(df, "x", "y", "z", "label")


# Result from umap is much better than PCA ( 2 almost distinctive clusters but many misplaced values ) and It's probably the best we can do. Problem with achieving better result could be caused by that misplaced values ; we are 100% that those values belong to their classes but we are lacking some crucial information/s ( feature/s ) that are necessary for achieving better classification ( for models) and seperation ( for umap ).
