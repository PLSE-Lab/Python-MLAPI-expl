#!/usr/bin/env python
# coding: utf-8

# # Do Not Overfit Challenge

# #### Kaggle recently launched a challenge to run classification models on datasets with a very particular goal - do not overfit!!
# 
# #### Datasets had only 250 rows and features around 320. Since our number of independent features is greater than sample size, we are faced with a very pressing overfitting challenge.
# 
# ##### Below I enumerate the methodology to reduce overfitting :
# 
# ###### 1. Reduce feature size using two specific techniques: pearsons  correlation coefficient and dimensionality reduction techniques
# ###### 2. Create handful of samples with goal on simplicity. eg less number of estimators,less number of iterations etc
# ###### 3. Provide regularization parameters where applicable
# ###### 4. Plot precision-recall curves and roc-auc curves for each model. Check learning curve as well.
# ###### 5. Use StratifiedKFold cross validation to parameter tune
# ###### 6. Deal with imbalance in target classes using under sampling and over sampling techniques
# 
# 
# 
# Note: baseline model creation(using all features + regularization + tsne dimensionality reduction + over/under sampling) technique provided no real use. CV scores for roc auc did not exceed 0.6 in every scenario possible.
# 
# 
# 
# 
# 

# In[ ]:


import numpy as np
import pandas as panda
from matplotlib import pyplot as plot
import seaborn as sns

import pandas as panda
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler,label_binarize

from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,GridSearchCV,RepeatedStratifiedKFold,learning_curve

from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve,         classification_report,confusion_matrix,average_precision_score
from sklearn.linear_model import Perceptron, LogisticRegression,RidgeClassifier,SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plot
from itertools import cycle
import numpy as np 
from scipy import interp
import seaborn as sns
import itertools, time, datetime
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from numpy import bincount, linspace, mean, std, arange, squeeze

import warnings

warnings.simplefilter('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


np.random.seed(143)


# In[ ]:


train_data = panda.read_csv('../input/train.csv')


# In[ ]:


train_data.target.value_counts(), train_data.shape


# In[ ]:


train_data['target'] = train_data.target.astype(np.int64)


# In[ ]:


plot.figure(figsize=(10,4))
train_data.target.value_counts().plot(kind='bar')
plot.show()


# In[ ]:


train_data[[i for i in train_data.columns.tolist() if i not in ['target','id']]].describe(include='all').T


# In[ ]:


data_type = train_data.dtypes.to_frame().reset_index()
data_type.columns  = ['col_name','col_type']
data_type[data_type.col_type==np.object].head()


# In[ ]:


train_data.isnull().any().sum()


# In[ ]:


col_names = [i for i in train_data.columns if i not in ['target','id']]


# In[ ]:


def draw_relevant_plots(table, columns):
    
    
    fig, axes = plot.subplots(1, 5, figsize=(10, 7), sharex=True)
    count = 0
    for col in columns:
        
        ax = axes[count]
        sns.distplot(table[col], ax = axes[count])
        
        count+= 1
    plot.show()  
    
    plot.figure(figsize = (10,4))
    data = table[columns]
    data = panda.concat([data,table['target']], axis = 1)
    correlation_map = np.corrcoef(data.values.T)
    sns.heatmap(correlation_map,
                cbar = True,
                annot=True,
                square = True,
                fmt = '.2f',
                annot_kws = {'size':15},
                yticklabels = data.columns.tolist(),
                xticklabels = data.columns.tolist(),
               )
    plot.show()
        


# In[ ]:


col_chunks = [col_names[i:i+5] for i in range(0,len(col_names),5)]

for item in col_chunks:
    draw_relevant_plots(train_data, item)


# In[ ]:


from scipy import stats
def calculateCorrelationCoefficientsAndpValues(x_data, y_data, xlabel):
    
    pearson_coef, p_value = stats.pearsonr(x_data, y_data)
    print("The Pearson Correlation Coefficient for %s is %s with a P-value of P = %s" %(xlabel,pearson_coef, p_value))
    
    return (pearson_coef,p_value)


# In[ ]:


pearson_coeff = []
p_value = []
col_name = []

for col in [i for i in train_data.columns.tolist() if i not in ['id','target']]:
    
    x,y = calculateCorrelationCoefficientsAndpValues(train_data[col], train_data['target'], col)
    pearson_coeff.append(x)
    p_value.append(y)
    col_name.append(col)
    
pearson_table = panda.DataFrame({'column_name':col_name , 'pearson_coeff':pearson_coeff, 'p_value': p_value})
pearson_table.head()


# In[ ]:


pearson_table[(pearson_table.pearson_coeff>0.1) | (pearson_table.pearson_coeff<-0.1)].sort_values(by=['pearson_coeff'], ascending=False)


# In[ ]:


reqd_columns = pearson_table[(pearson_table.pearson_coeff>0.1) | (pearson_table.pearson_coeff<-0.1)].sort_values(by=['pearson_coeff'], ascending=False).column_name.values.tolist()
reqd_columns[:5], len(reqd_columns)


# There are a couple of observations that we can make out of the above diagrams:
# 
# 1. ALmost all independent variables are distributed within -2.5 and 2.5
# 2. Almost all have near normal distributions
# 3. Almost none have any significant outliers
# 4. Pearsons correlation coeff for each indedepent variable with target variable is pretty low
# 5. There are no categorical data
# 6. pearsons correlation coefficient scores are pretty less and we will take all whose scores are above 0.1 and below -0.1

# #### Based on above data analysis, we are going to attempt 3 things
# 
# 1. create a baseline model out of gridsearch for the data using the usual suspects of classifiers. 
# 
# 
# 2. Plot learning curve and auc/roc scores for each
# 
# 
# 3. try dimensionality reduction techniques such as PCA,LDA and t-SNE and run on baseline models created in step 1.
# 
# 
# 4. try SMOTE and run on the same model created in point 1 and compare scores
# 
# 

# ### STEP 1: Run handful of usual suspect classifiers using selected 50 features giving highest coeff scores
# 
# <br><br>

# In[ ]:


class CodeTimer:
    
    """
        Utility custom contextual class for calculating the time 
        taken for a certain code block to execute
    
    """
    def __init__(self, name=None):
        self.name = " '"  + name + "'" if name else ''

    def __enter__(self):
        self.start = time.clock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (time.clock() - self.start) * 1000.0
        time_taken = datetime.timedelta(milliseconds = self.took)
        print('Code block' + self.name + ' took(HH:MM:SS): ' + str(time_taken))
        
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plot.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    plot.title(title)
    plot.colorbar()
    tick_marks = arange(len(classes))
    plot.xticks(tick_marks, classes, rotation=45)
    plot.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plot.ylabel('True label')
    plot.xlabel('Predicted label')
#     plot.tight_layout()
    plot.show()


# In[ ]:



def plotLearningCurve(_x_train, _y_train, learning_model_pipeline,  model_name, k_fold = 10, training_sample_sizes = linspace(0.1,1.0,10), jobsInParallel = 1):
    
    training_size, training_score, testing_score = learning_curve(estimator = learning_model_pipeline,                                                                 X = _x_train,                                                                 y = _y_train,                                                                 train_sizes = training_sample_sizes,                                                                 cv = k_fold,                                                                 n_jobs = jobsInParallel) 


    training_mean = mean(training_score, axis = 1)
    training_std_deviation = std(training_score, axis = 1)
    testing_std_deviation = std(testing_score, axis = 1)
    testing_mean = mean(testing_score, axis = 1 )

    ## we have got the estimator in this case the perceptron running in 10 fold validation with 
    ## equal division of sizes betwwen .1 and 1. After execution, we get the number of training sizes used, 
    ## the training scores for those sizes and the test scores for those sizes. we will plot a scatter plot 
    ## to see the accuracy results and check for bias vs variance

    # training_size : essentially 10 sets of say a1, a2, a3,,...a10 sizes (this comes from train_size parameter, here we have given linespace for equal distribution betwwen 0.1 and 1 for 10 such values)
    # training_score : training score for the a1 samples, a2 samples...a10 samples, each samples run 10 times since cv value is 10
    # testing_score : testing score for the a1 samples, a2 samples...a10 samples, each samples run 10 times since cv value is 10
    ## the mean and std deviation for each are calculated simply to show ranges in the graph

    plot.plot(training_size, training_mean, label= "Training Data", marker= '+', color = 'blue', markersize = 8)
    plot.fill_between(training_size, training_mean+ training_std_deviation, training_mean-training_std_deviation, color='blue', alpha =0.12 )

    plot.plot(training_size, testing_mean, label= "Testing/Validation Data", marker= '*', color = 'green', markersize = 8)
    plot.fill_between(training_size, testing_mean+ training_std_deviation, testing_mean-training_std_deviation, color='green', alpha =0.14 )

    plot.title("Scoring of our training and testing data vs sample sizes for model:"+model_name)
    plot.xlabel("Number of Samples")
    plot.ylabel("Accuracy")
    plot.legend(loc= 'best')
    plot.show()
    
def plot_roc_auc_curve(false_positive_rate, true_positive_rate, model_name):
        
    plot.figure(figsize=(10,3))
    plot.plot(list(false_positive_rate), list(true_positive_rate),  label = "ROC Curve for model: "+model_name)     
    plot.plot([0, 1], [0, 1], 'k--', label = 'Random Guessing')
    plot.plot([0, 0, 1], [0,1, 1], ':', label = 'Perfect Score')
    auc_score = auc(false_positive_rate, true_positive_rate)
    plot.title('ROC Curve for model: %s with AUC %.2f'%(model_name, auc_score))
    plot.xlabel('False Positive Rate')
    plot.ylabel('True Positive Rate')
    plot.legend(loc='best')
    plot.show()
    
    
def plot_precision_recall_curve(precision, recall, model_name):
    
    plot.figure(figsize=(10,3))
    plot.plot(list(recall), list(precision),  label = "Precision/Recall Curve for model: "+model_name)     
#     plot.plot([0, 1], [0, 1], 'k--', label = 'Random Guessing') #
    plot.title('Precision Recall Curve for model: %s'%model_name)
    plot.xlabel('Recall')
    plot.ylabel('Precision')
    plot.legend(loc='best')
    plot.show()


    


# In[ ]:


def runGridSearchAndPredict(pipeline,model_name, x_train, y_train, x_test, y_test, param_grid, n_jobs = 1, cv = 10, score = 'accuracy'):
#     pass

    response =  {}
    training_timer       = CodeTimer('training')
    testing_timer        = CodeTimer('testing')
    learning_curve_timer = CodeTimer('learning_curve')
    predict_proba_timer  = CodeTimer('predict_proba')
    
    with training_timer:
        
        gridsearch = GridSearchCV(estimator = pipeline, param_grid = param_grid, cv = cv, n_jobs = n_jobs, scoring = score)

        search = gridsearch.fit(x_train,y_train)

        print("Grid Search Best parameters ", search.best_params_)
        print("Grid Search Best score ", search.best_score_)

    with testing_timer:
        y_prediction = gridsearch.predict(x_test)
            
    print("F1 score %s" %f1_score(y_test,y_prediction, average ='weighted'))
    print("Classification report  \n %s" %(classification_report(y_test, y_prediction)))
    
    with learning_curve_timer:
        plotLearningCurve(x_train, y_train, search.best_estimator_, model_name)
#         _matrix = confusion_matrix(y_true = _y_test ,y_pred = y_prediction, labels = list(range(_y_test.shape[1])))
        _matrix = confusion_matrix(y_true = y_test ,y_pred = y_prediction, labels = list(set(y_test)))
        classes = list(set(y_test))
        plot_confusion_matrix(_matrix, classes, title = "Confusion matrix for model:"+model_name)
        
    with predict_proba_timer:

        if hasattr(gridsearch.best_estimator_, 'predict_proba'):
            
            print('inside decision function')
            y_probability = gridsearch.predict_proba(x_test)
            number_of_classes = len(np.unique(y_train))
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_probability[:, 1])
            response['roc_auc_score'] = roc_auc_score(y_test, y_probability[:,1])
            response['roc_curve'] = (false_positive_rate, true_positive_rate)  
            response['roc_curve_false_positive_rate'] = false_positive_rate
            response['roc_curve_true_positive_rate'] = true_positive_rate
            precision, recall, _ = precision_recall_curve(y_test, y_probability[:,1])
            plot_roc_auc_curve(false_positive_rate, true_positive_rate, model_name)
            plot_precision_recall_curve(precision, recall, model_name)
            
        else: ## eg SVM, Perceptron doesnt have predict_proba method
            
            response['roc_auc_score'] = 0
            response['roc_curve'] = 0
            response['roc_curve_false_positive_rate'] = 0
            response['roc_curve_true_positive_rate'] = 0
    
    response['learning_curve_time'] = learning_curve_timer.took
    response['testing_time'] = testing_timer.took
    response['_y_prediction'] = y_prediction
#     response['accuracy_score'] = accuracy_score(y_test,y_prediction)
    response['training_time'] = training_timer.took
    response['f1_score']  = f1_score(y_test, y_prediction, average ='weighted')
    response['f1_score_micro']  = f1_score(y_test, y_prediction, average ='micro')
    response['f1_score_macro']  = f1_score(y_test, y_prediction, average ='macro')
    response['best_estimator'] = search.best_estimator_
    response['confusion_matrix'] = _matrix
    
    return response


def plotROCCurveAcrossModels(positive_rates_sequence, model_name):
    
    plot.figure(figsize=(10,5))
    for plot_values, label_name in zip(positive_rates_sequence, model_name):
        
        plot.plot(list(plot_values[0]), list(plot_values[1]),  label = "ROC Curve for model: "+label_name)
        
    plot.plot([0, 1], [0, 1], 'k--', label = 'Random Guessing') #
    plot.title('ROC Curve across models')
    plot.xlabel('False Positive Rate')
    plot.ylabel('True Positive Rate')
    plot.legend(loc='best')
    plot.show()


# In[ ]:



def execute( _x_train,
             _y_train,
             _x_test,
             _y_test, 
            classifiers, 
            classifier_names, 
            classifier_param_grid,
            cv  = 10 , 
            score = 'accuracy',
            scaler = StandardScaler()
           ):
    
    '''
    This method will run your data sets against the model specified 
    Models will be fed through a pipeline where the first step would be to
    execute a scaling operation.
    
    Method will also call additional lower level methods in order to plot
    precision curve, roc curve, learning curve and will also prepare a confusion matrix
    
    :returns: dict containing execution metrics such as time taken, accuracy scores
    :returntype: dict
    
    '''

    timer = CodeTimer(name='overalltime')
    model_metrics = {}

    with timer:
        for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):

            pipeline_steps = [('scaler', scaler),(model_name, model)] if scaler is not None else [(model_name, model)]
            pipeline = Pipeline(pipeline_steps)

            result = runGridSearchAndPredict(pipeline, 
                                             model_name,
                                             _x_train,
                                             _y_train,
                                             _x_test,
                                             _y_test, 
                                             model_param_grid ,
                                             cv = cv,
                                             score = score)

            _y_prediction = result['_y_prediction']

            model_metrics[model_name] = {}
            model_metrics[model_name]['confusion_matrix'] = result.get('confusion_matrix')
            model_metrics[model_name]['training_time'] = result.get('training_time')
            model_metrics[model_name]['testing_time'] = result.get('testing_time')
            model_metrics[model_name]['learning_curve_time'] = result.get('learning_curve_time')
            model_metrics[model_name]['f1_score'] = result.get('f1_score')
            model_metrics[model_name]['f1_score_macro'] = result.get('f1_score_macro')
            model_metrics[model_name]['f1_score_micro'] = result.get('f1_score_micro')
            model_metrics[model_name]['roc_auc_score'] = result.get('roc_auc_score')
            model_metrics[model_name]['roc_curve_true_positive_rate'] = result.get('roc_curve_true_positive_rate')
            model_metrics[model_name]['roc_curve_false_positive_rate'] = result.get('roc_curve_false_positive_rate')

            model_metrics[model_name]['best_estimator'] = result.get('best_estimator')


    print(timer.took)
    
    return model_metrics


# In[ ]:



classifiers = [
    Perceptron(random_state = 1),
    LogisticRegression(random_state = 1),
    LogisticRegression(random_state = 1, solver='liblinear'),
    LogisticRegression(random_state = 1, solver='newton-cg'),
    LogisticRegression(random_state = 1, solver='sag'),
    DecisionTreeClassifier(),
    RandomForestClassifier(random_state = 1),
    KNeighborsClassifier(metric = 'minkowski'),
    RidgeClassifier(random_state = 123), 
    SVC(kernel="linear"),
    SVC(),
    ExtraTreeClassifier(random_state = 123),
    GaussianProcessClassifier(random_state = 123),
    BernoulliNB(),
    BaggingClassifier(base_estimator = LogisticRegression(random_state = 1)),
    BaggingClassifier(base_estimator = BernoulliNB()),
    GradientBoostingClassifier(random_state= 123),
    LGBMClassifier(objective = 'binary'),
    XGBClassifier(objective = 'binary:logistic')
]


classifier_names = [
            'perceptron',
            'logisticregression',
            'logisticregression_liblinear_l2',
            'logisticregression_newton_cg',
            'logisticregression_sag',
            'decisiontreeclassifier',
            'randomforestclassifier',
            'kneighborsclassifier',
            'ridge',
            'linear_svc',
            'gamma_svc',
            'extra_trees',
            'gaussian_process',
            'bernoulli',
            'bagging_logistic',
            'bagging_bernoulli',
            'gradient_boosting_classifier',
            'lgbm_classifier',
            'xgb'
]

classifier_param_grid = [
            
            {'perceptron__max_iter': [5,10,30], 'perceptron__eta0': [.1]},
            {
             'logisticregression__C':[1.2,0.02,2.2,4, 0.01, 0.05], 
             'logisticregression__penalty':['l1','l2'],
             'logisticregression__solver':['saga','liblinear']
            },
            {
             'logisticregression_liblinear_l2__C':[1.2,0.02,2.2,4, 0.01, 0.05], 
             'logisticregression_liblinear_l2__penalty':['l2'],
             'logisticregression_liblinear_l2__dual':[True]
            },
            {
             'logisticregression_newton_cg__C':[1.2,0.02,2.2,4, 0.01, 0.05], 
             'logisticregression_newton_cg__penalty':['l2'],
            },
            {
             'logisticregression_sag__C':[1.2,0.02,2.2,4, 0.01, 0.05], 
             'logisticregression_sag__penalty':['l2'],
            },
    
            {'decisiontreeclassifier__max_depth':[6,8,10],
             'decisiontreeclassifier__criterion':['gini','entropy'],
             'decisiontreeclassifier__max_features':['auto','sqrt','log2'],
            },
            {'randomforestclassifier__n_estimators':[6,8,12],'randomforestclassifier__criterion': ['gini','entropy']} ,
            {'kneighborsclassifier__n_neighbors':[4,6,10]},
            {'ridge__alpha':[1,1.2,0.9],'ridge__max_iter':[100,300,500]},
            {'linear_svc__C':[0.025]},
            {'gamma_svc__gamma':[2,4],'gamma_svc__C':[1,5]},
            {'extra_trees__max_depth':[6,8,12],'extra_trees__criterion': ['gini','entropy']} ,
            {'gaussian_process__max_iter_predict':[200,400]} ,
            {'bernoulli__alpha':[0.2,0.6,1.2]} ,
            {'bagging_logistic__base_estimator__C':[1.2,0.02,2.2,4], 
             'bagging_logistic__base_estimator__penalty':['l1','l2'],
             'bagging_logistic__n_estimators': [5,8,10]
            },
            {'bagging_bernoulli__base_estimator__alpha':[1.2,0.02,2.2,4], 
             'bagging_bernoulli__n_estimators': [5,8,10]
            },
            {
                'gradient_boosting_classifier__loss':['deviance','exponential'],
                'gradient_boosting_classifier__learning_rate':[0.5,1.2],
                'gradient_boosting_classifier__n_estimators':[100,500,1000],
                'gradient_boosting_classifier__criterion':['friedman_mse','mse','mae'],
                'gradient_boosting_classifier__max_depth':[6,8,16,20],
            },
            {
                 'lgbm_classifier__num_leaves':[25,], \
#                  'lgbm_classifier__min_data_in_leaf':[20],\
                 'lgbm_classifier__max_depth':[20,], \
                 'lgbm_classifier__learning_rate' : [0.01,],\
                 'lgbm_classifier__min_child_samples' :[2,], \
                 'lgbm_classifier__n_estimators' : [5000,], \
                 'lgbm_classifier__num_boost_round' : [100], \
                 'lgbm_classifier__feature_fraction' : [0.9,], \
                 'lgbm_classifier__bagging_freq' : [1,], \
                 'lgbm_classifier__bagging_seed' : [123], \
            },
             {
                'xgb__max_depth':[6,8,10],
                 'xgb__learning_rate':[0.1,0.5,1,2],
                 'xgb__n_estimators':[100,400,1000],             
                 'xgb__booster':['gbtree','dart'],
                 'xgb__subsample':[0.5, 0.2,0.8]
            },
    
]


# In[ ]:


# x = train_data[[i for i in train_data.columns.tolist() if i not in ['target','id']]]
x = train_data[reqd_columns[:51]]
y = train_data['target']

x_train,x_test,y_train,y_test = train_test_split(x,y , stratify = y, test_size = 0.3, random_state = 123)


# In[ ]:


cv = StratifiedKFold(n_splits = 5, shuffle= True, random_state =123)
score= 'roc_auc'


# In[ ]:


response = execute(
        x_train,
        y_train,
        x_test,
        y_test,
        classifiers,
        classifier_names,
        classifier_param_grid,
        cv=cv,
        score=score,
        scaler=StandardScaler())


# In[ ]:


results = panda.DataFrame(response).transpose()
results.head()
results[['f1_score',
         'f1_score_macro',
         'f1_score_micro',
         'learning_curve_time',
         'roc_auc_score',
         'testing_time',
         'training_time',
        ]]\
.sort_values(by=['roc_auc_score',],ascending=False)


# In[ ]:



roc_rates = []
model_name = []
for index, key in enumerate(response):
    
    
    estimator = response.get(key)
    if estimator.get('roc_auc_score')!=0:
        roc_curve_true_positive_rate = estimator.get('roc_curve_true_positive_rate')
        roc_curve_false_positive_rate = estimator.get('roc_curve_false_positive_rate')
        roc_rates.append([roc_curve_false_positive_rate,roc_curve_true_positive_rate])
        model_name.append(key)

plotROCCurveAcrossModels(roc_rates,model_name) 


# In[ ]:


results['learning_curve_time'] = results['learning_curve_time'].astype('float64')
results['testing_time'] = results['testing_time'].astype('float64')
results['training_time'] = results['training_time'].astype('float64')
results['f1_score'] = results['f1_score'].astype('float64')
results['f1_score_micro'] = results['f1_score_micro'].astype('float64')
results['f1_score_macro'] = results['f1_score_macro'].astype('float64')
results['roc_auc_score'] = results['roc_auc_score'].astype('float64')
# results['roc_auc_macro'] = results['roc_auc_macro'].astype('float64')

#scaling time parameters between 0 and 1
results['learning_curve_time'] = (results['learning_curve_time']- results['learning_curve_time'].min())/(results['learning_curve_time'].max()- results['learning_curve_time'].min())
results['testing_time'] = (results['testing_time']- results['testing_time'].min())/(results['testing_time'].max()- results['testing_time'].min())
results['training_time'] = (results['training_time']- results['training_time'].min())/(results['training_time'].max()- results['training_time'].min())

results.plot(kind='barh',figsize=(12, 10))
plot.title("Scaled Estimates across different classifiers used")
plot.show()


# ### Conclusion:
# 
# 1. We got highest scores of ROC/AUC 0.92 in logistic regression with l2 parameter
# 
# 2. From the learning curve, we also see that overfitting tendency is less
# 

# In[ ]:


test_data = panda.read_csv('../input/test.csv')

test_data_x = test_data[reqd_columns]


# ### STEP 2: Run Dimensionality technique, specifically t-SNE
# 
# <br><br>

# In[ ]:



## perplexity parameters were tuned
tsne = TSNE(perplexity=35, learning_rate=15)
scaler = StandardScaler()
x_train_tsne = tsne.fit_transform(scaler.fit_transform(x_train))
x_test_tsne = tsne.fit_transform(scaler.fit_transform(x_test))


# In[ ]:


# plot.scatter(x_train_tsne[:,0]x_train_tsne[:,0], x_train_tsne[:,1])
plot.scatter(x_train_tsne[:,0], y_train,  marker='^', c='blue')
plot.scatter(x_train_tsne[:,1], y_train,  marker='o', c='red')
plot.show()
# x_train_tsne[:,1].shape, y_train.shape


# In[ ]:



##runing without scaling, since scaling was already done prior to tsne
response = execute(
        x_train_tsne,
        y_train,
        x_test_tsne,
        y_test,
        classifiers,
        classifier_names,
        classifier_param_grid,
        cv=cv,
        score=score,
        scaler=None)


# In[ ]:


results = panda.DataFrame(response).transpose()
results.head()
results[['f1_score',
         'f1_score_macro',
         'f1_score_micro',
         'learning_curve_time',
         'roc_auc_score',
         'testing_time',
         'training_time',
        ]]\
.sort_values(by=['roc_auc_score',],ascending=False)


# In[ ]:


results['learning_curve_time'] = results['learning_curve_time'].astype('float64')
results['testing_time'] = results['testing_time'].astype('float64')
results['training_time'] = results['training_time'].astype('float64')
results['f1_score'] = results['f1_score'].astype('float64')
results['f1_score_micro'] = results['f1_score_micro'].astype('float64')
results['f1_score_macro'] = results['f1_score_macro'].astype('float64')
results['roc_auc_score'] = results['roc_auc_score'].astype('float64')
# results['roc_auc_macro'] = results['roc_auc_macro'].astype('float64')

#scaling time parameters between 0 and 1
results['learning_curve_time'] = (results['learning_curve_time']- results['learning_curve_time'].min())/(results['learning_curve_time'].max()- results['learning_curve_time'].min())
results['testing_time'] = (results['testing_time']- results['testing_time'].min())/(results['testing_time'].max()- results['testing_time'].min())
results['training_time'] = (results['training_time']- results['training_time'].min())/(results['training_time'].max()- results['training_time'].min())

results.plot(kind='barh',figsize=(12, 10))
plot.title("Scaled Estimates across different classifiers used")
plot.show()


# ### Conclusion: dimensionality reduction gives worse predictions

# ### STEP 3: Upsampling / Downsampling Techniques
# 
# <br><br>

# In[ ]:



classifiers = [
    Perceptron(random_state = 1),
    LogisticRegression(random_state = 1),
    LogisticRegression(random_state = 1, solver='liblinear'),
    LogisticRegression(random_state = 1, solver='newton-cg'),
    LogisticRegression(random_state = 1, solver='sag'),
    DecisionTreeClassifier(),
    RandomForestClassifier(random_state = 1),
    KNeighborsClassifier(metric = 'minkowski'),
    RidgeClassifier(random_state = 123), 
    SVC(kernel="linear"),
    SVC(),
    ExtraTreeClassifier(random_state = 123),
    GaussianProcessClassifier(random_state = 123),
    BernoulliNB(),
    BaggingClassifier(base_estimator = LogisticRegression(random_state = 1)),
    BaggingClassifier(base_estimator = BernoulliNB()),
 
]


classifier_names = [
            'perceptron',
            'logisticregression',
            'logisticregression_liblinear_l2',
            'logisticregression_newton_cg',
            'logisticregression_sag',
            'decisiontreeclassifier',
            'randomforestclassifier',
            'kneighborsclassifier',
            'ridge',
            'linear_svc',
            'gamma_svc',
            'extra_trees',
            'gaussian_process',
            'bernoulli',
            'bagging_logistic',
            'bagging_bernoulli',
 
]

classifier_param_grid = [
            
            {'perceptron__max_iter': [5,10,30], 'perceptron__eta0': [.1]},
            {
             'logisticregression__C':[1.2,0.02,2.2,4, 0.01, 0.05], 
             'logisticregression__penalty':['l1','l2'],
             'logisticregression__solver':['saga','liblinear']
            },
            {
             'logisticregression_liblinear_l2__C':[1.2,0.02,2.2,4, 0.01, 0.05], 
             'logisticregression_liblinear_l2__penalty':['l2'],
             'logisticregression_liblinear_l2__dual':[True]
            },
            {
             'logisticregression_newton_cg__C':[1.2,0.02,2.2,4, 0.01, 0.05], 
             'logisticregression_newton_cg__penalty':['l2'],
            },
            {
             'logisticregression_sag__C':[1.2,0.02,2.2,4, 0.01, 0.05], 
             'logisticregression_sag__penalty':['l2'],
            },
    
            {'decisiontreeclassifier__max_depth':[6,8,10],
             'decisiontreeclassifier__criterion':['gini','entropy'],
             'decisiontreeclassifier__max_features':['auto','sqrt','log2'],
            },
            {'randomforestclassifier__n_estimators':[6,8,12],'randomforestclassifier__criterion': ['gini','entropy']} ,
            {'kneighborsclassifier__n_neighbors':[4,6,10]},
            {'ridge__alpha':[1,1.2,0.9],'ridge__max_iter':[100,300,500]},
            {'linear_svc__C':[0.025]},
            {'gamma_svc__gamma':[2,4],'gamma_svc__C':[1,5]},
            {'extra_trees__max_depth':[6,8,12],'extra_trees__criterion': ['gini','entropy']} ,
            {'gaussian_process__max_iter_predict':[200,400]} ,
            {'bernoulli__alpha':[0.2,0.6,1.2]} ,
            {'bagging_logistic__base_estimator__C':[1.2,0.02,2.2,4], 
             'bagging_logistic__base_estimator__penalty':['l1','l2'],
             'bagging_logistic__n_estimators': [5,8,10]
            },
            {'bagging_bernoulli__base_estimator__alpha':[1.2,0.02,2.2,4], 
             'bagging_bernoulli__n_estimators': [5,8,10]
            },
    
]


# In[ ]:





# In[ ]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

from imblearn.combine import SMOTETomek


# In[ ]:



tl = TomekLinks(return_indices=True, ratio='majority')
_x_train_tomek, _y_train_tomek, id_tl = tl.fit_sample(x_train, y_train)

smt = SMOTETomek(ratio='auto')
_x_train_smt, _y_train_smt = smt.fit_sample(x_train, y_train)

smote = SMOTE(ratio='minority')
x_train_smote, y_train_smote = smote.fit_sample(x_train,y_train)




# In[ ]:


_x_train_tomek.shape, _y_train_tomek.shape,x_test.shape,y_test.shape


# In[ ]:


response1 = execute(
        _x_train_tomek,
        _y_train_tomek,
        x_test,
        y_test,
        classifiers,
        classifier_names,
        classifier_param_grid,
        cv=cv,
        score=score,
        scaler=None)


# In[ ]:


results = panda.DataFrame(response1).transpose()
results.head()
results[['f1_score',
         'f1_score_macro',
         'f1_score_micro',
         'learning_curve_time',
         'roc_auc_score',
         'testing_time',
         'training_time',
        ]]\
.sort_values(by=['roc_auc_score',],ascending=False)


# In[ ]:


response = execute(
        _x_train_smt,
        _y_train_smt ,
        x_test,
        y_test,
        classifiers,
        classifier_names,
        classifier_param_grid,
        cv=cv,
        score=score,
        scaler=None)


# In[ ]:


results = panda.DataFrame(response).transpose()
results.head()
results[['f1_score',
         'f1_score_macro',
         'f1_score_micro',
         'learning_curve_time',
         'roc_auc_score',
         'testing_time',
         'training_time',
        ]]\
.sort_values(by=['roc_auc_score',],ascending=False)


# In[ ]:





# In[ ]:


response2 = execute(
        x_train_smote,
        y_train_smote ,
        x_test,
        y_test,
        classifiers,
        classifier_names,
        classifier_param_grid,
        cv=cv,
        score=score,
        scaler=None)


# In[ ]:


results = panda.DataFrame(response2).transpose()
results.head()
results[['f1_score',
         'f1_score_macro',
         'f1_score_micro',
         'learning_curve_time',
         'roc_auc_score',
         'testing_time',
         'training_time',
        ]]\
.sort_values(by=['roc_auc_score',],ascending=False)


# ### Conclusion : We got the best performing model as Liblinear L2 Logistic Regression with a ROC-AUC score of 0.93 and f1 score of 0.82 using TomekLink downsampling

# In[ ]:


logistic_regression_liblinear_2 = response1.get('logisticregression_liblinear_l2',{}).get('best_estimator')


# In[ ]:


# _x_tomek, _y_tomek, id_tl = tl.fit_sample(x, y)


# In[ ]:


test_data_x.shape


# In[ ]:


# logistic_regression_liblinear_2.fit(_x_tomek, _y_tomek)


# In[ ]:


test_target = logistic_regression_liblinear_2.predict(test_data_x)


# In[ ]:


np.bincount(test_target)


# In[ ]:


final_submission = panda.DataFrame({'target':test_target})
final_submission['id'] = test_data['id']
final_submission[['id','target']].head() 

# np.bincount(test_target)

final_submission[['id','target']].to_csv('sample_submission_8.csv', index = False)

