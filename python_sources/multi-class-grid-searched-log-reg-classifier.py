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

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random
import sys
import seaborn as sns

""" ######################### I/O #################################### """
folder=r"C:/Users/admin/Desktop/LEARNING/KAGGLE/Spooky Author comp/"
inname=r'XY.csv'
trainset = pd.read_csv(folder+inname,encoding="utf-8")
XY = pd.read_csv(folder+inname, skipinitialspace=True, encoding="utf-8")
feature_names=XY.drop("Y", 1).columns
y=np.array(XY.Y)
class_labels=np.unique(y)
class_proxies={"EAP":1,"MWS":2,"HPL":3,"unlabeled":-1}
inverse_class_proxies={y:x for x,y in class_proxies.items()}
actual_class_labels=[inverse_class_proxies[i] for i in class_labels]
n_classes = len(class_labels)
X=np.array(XY.drop("Y", 1))
"""########## Is a separate "REAL WORLD" X-set provided? ##############"""
inname=r'XYtest.csv'
XYtest = pd.read_csv(folder+inname, skipinitialspace=True, encoding="utf-8")
X_real=np.array(XYtest)

"""###################################################### """
"""#############  Independent from model  ############### """
"""###################################################### """

preview_n_rows=3
for j in range(preview_n_rows):
    preview_dict={}

    for i in range(len(feature_names)):
        preview_dict[feature_names[i]]=X[j][i]
    print("\nPreview of data of row %d:" % j)
    print(preview_dict)
    print("Preview of label: ", y[j])

print("Shape of data (X): ", X.shape)
print("Class labels: ", class_labels)
print("Shape of dataset labels (y): ", y.shape)
print("Examples count per class: \n {}".format(np.bincount(y)))

scaling=True
########################## SCALING dataset #####################

if scaling:
    from sklearn.preprocessing import StandardScaler
    X_scaled=np.array([])
    X_real_scaled=np.array([])
    scaler = StandardScaler()
    print ("\nData scaling: Standard", end="")
    X_scaled=scaler.fit(X).transform(X)
    X_real_scaled=scaler.fit(X).transform(X_real)  ### Test data must also be scaled by the SAME scaler!
    if X_scaled.any(): print (" --> Data scaled!")

    """ Previewing after scaling! """
    print("Previewing data after scaling!\n")
    for j in range(preview_n_rows):
        preview_dict={}
        for i in range(len(feature_names)):
            preview_dict[feature_names[i]]=X_scaled[j][i]
        print("\nPreview of data of row %d:" % j)
        print(preview_dict)
    X_trans=X_scaled
    X_real=X_real_scaled
else:
    X_trans=X

print("\n------SPLITTING the training set in train and test samples--------\n")
from sklearn.model_selection import train_test_split
split=0.1
X_train, X_test, y_train, y_test = train_test_split(X_trans,y,stratify=y, test_size=split,random_state=201)

print("shape of training data: \n", X_train.shape)
print("shape of testing data: \n", X_test.shape)
print("Class breakdown in train set targets:",np.bincount(y_train))
print("...and in percentages: {}".format(np.bincount(y_train)*100/y_train.shape[0]))
print("Class breakdown in test set targets:",np.bincount(y_test))
print("...and in percentages: {}".format(np.bincount(y_test)*100/y_test.shape[0]))

"""###################################################### """
"""#######  From here on below: MODEL SPECIFIC  ######### """
"""###################################################### """

print("\n------ Running cross-validated grid search on LogisticRegression --------\n")
from sklearn.model_selection import GridSearchCV

X_trainval, X_test, y_trainval, y_test= X_train, X_test, y_train, y_test

from sklearn.linear_model import LogisticRegression
njobs=-1
model=LogisticRegression()
####### F1 score averaging technique: macro or micro?
####### Do we have a preference for Recall *OR* Precision            ---> Micro
####### ... or care of accuracy of prediction for each class equally ---> Macro
score_metric="f1_macro"
# score_metric="f1_micro"
averaging_heuristic="macro"
# averaging_heuristic="micro"
#####################################################
params=['C']
param_grid = {  params[0]: [0.001,0.01,0.1,1.,10,100]}
cv=5
grid_classifier=GridSearchCV(model,param_grid,cv=cv, scoring=score_metric) #,n_jobs=njobs, scoring="roc_auc"
grid_classifier.fit(X_trainval,y_trainval)
results = pd.DataFrame(grid_classifier.cv_results_) #######?
best = np.argmax(results.mean_test_score.values)

################################################################################

print("\n------Plotting Cross-Validated Grid Search Results--------\n")
def plot_Xvalidated_grid(results,grid_classifier):
    plt.figure(figsize=(10, 3))
    plt.xlim(-1, len(results))

    for i, (_, row) in enumerate(results.iterrows()):
        scores = row[['split%d_test_score' % i for i in range(cv)]]
        marker_cv, = plt.plot([i] * cv, scores, '^', c='gray', markersize=5,
                                  alpha=.5)
        marker_mean, = plt.plot(i, row.mean_test_score, 'v', c='none', alpha=1,
                                    markersize=10, markeredgecolor='k')
        if i == best:
            marker_best, = plt.plot(i, row.mean_test_score, 'o', c='red',
                                        fillstyle="none", alpha=1, markersize=20,
                                        markeredgewidth=3)

    plt.xticks(range(len(results)), [str(x).strip("{}").replace("'", "") for x
                            in grid_classifier.cv_results_['params']],rotation=90)
    plt.ylabel("Validation score")
    plt.xlabel("Parameter settings")
    plt.legend([marker_cv, marker_mean, marker_best],
                   ["cv "+score_metric, "mean "+score_metric, "best parameter setting"],
                   loc="best")
    plt.show()

plot_Xvalidated_grid(results,grid_classifier)

################################################################################

print("Best parameters: {}".format(grid_classifier.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_classifier.best_score_))
print("Best estimator:\n{}".format(grid_classifier.best_estimator_))

print("\n---- Trained a Logistic Regression classifier with optimal parameters ----\n")
classifier=grid_classifier.best_estimator_
print("\nAccuracy on the train set: %.3f "% classifier.score(X_train,y_train))
print("\nAccuracy on the TEST set: %.3f "% classifier.score(X_test,y_test))

from sklearn.metrics import confusion_matrix,f1_score, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
predictions=classifier.predict(X_test)
confidence=classifier.decision_function(X_test)


################################################################################

print("\n------ Plotting MULTI-CLASS ROC Curve --------\n")
def plot_MULTI_CLASS_roc(X_train,X_test,y_train,y_test, classifier,
                            n_classes, class_labels,averaging_heuristic):

    # Binarize (oneVSall) the multiclass predictions, for ROC curve plotting purposes
    y_train_binarized = label_binarize(y_train, classes=class_labels)
    y_test_binarized = label_binarize(y_test, classes=class_labels)
    # # Learn to predict each class against the other
    binarized_predictor = OneVsRestClassifier(classifier) # n_jobs=njobs
    y_bin_score = binarized_predictor.fit(X_train, y_train_binarized).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()
    for cl in range(n_classes):
        fpr[cl], tpr[cl], thresholds[cl] = roc_curve(y_test_binarized[:, cl], y_bin_score[:, cl])
        roc_auc[cl] = auc(fpr[cl], tpr[cl])
        plt.plot(fpr[cl], tpr[cl], label=("ROC Curve of class "+str(class_labels[cl])+" VS. all"))
        op_default = np.argmin(np.abs(thresholds[cl]))
        # plt.plot    (fpr[cl][op_default],tpr[cl][op_default],
        #             "^",label="default operation point\n(50% probability threshold)")


    # # Compute average ROC curve and ROC area
    # fpr[averaging_heuristic], tpr[averaging_heuristic], _ = roc_curve(y_test.ravel(), y_bin_score.ravel())
    # roc_auc[averaging_heuristic] = auc(fpr[averaging_heuristic], tpr[averaging_heuristic])

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive\nRate (Recall)")

    # auc=roc_auc_score(y_test,classifier.decision_function(X_test))
    # plt.suptitle('AUC (area-under-curve) = {:.2f} %'.format(100.*auc), color="g", x=0.45,y=0.3, fontsize=14, va='bottom',ha='left')
    plt.legend(loc="best")
    plt.show()

plot_MULTI_CLASS_roc(X_train,X_test,y_train,y_test, classifier, n_classes, class_labels,averaging_heuristic)

################################################################################

print("\nf1 score: {:.2f}\n".format(f1_score(y_test,predictions, average=averaging_heuristic)))
# print("\nArea-under-ROC-Curve score: {:.2f}\n".format(roc_auc_score(y_test,confidence)))
print("Confusion matrix:\n{}\n".format(confusion_matrix(y_test,predictions)))
print("------ Classifier Evaluation report ------\n")
print(classification_report(y_test,predictions,target_names=class_labels.astype("str")))

################################################################################

print("\n------ Plotting Feature coefficients --------\n")
def plot_feature_coeffs(classifier,feature_names,features_coeff_to_plot):
    coef = classifier.coef_[0] #,classifier.coef_!= 0]

    n_top_features=int(0.5*features_coeff_to_plot*len(feature_names))
    pos_coeffs = np.argsort(coef)[-n_top_features:]
    neg_coeffs = np.argsort(coef)[:n_top_features]
    relevant_coeffs = np.hstack([neg_coeffs,pos_coeffs])
    # print("Relevant coeffs: {}".format(coef[relevant_coeffs]))
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[relevant_coeffs]]
    plt.bar(np.arange(2*n_top_features), coef[relevant_coeffs], color=colors)
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(np.arange(1, 2 * n_top_features),
               feature_names[relevant_coeffs], rotation=60, ha="right")
    plt.xlabel("Feature name")
    plt.ylabel("Feature magnitude\n(Regression coeff)")
    plt.show()

features_coeff_to_plot=0.8 ### what % of feature coeffs to plot, max?
plot_feature_coeffs(classifier,feature_names,features_coeff_to_plot)

################################################################################
    ########################## Competition output ##########################
################################################################################

testname=r'\test.csv'
output=pd.DataFrame()
samples = pd.read_csv(folder+testname,encoding="utf-8")
samples["pred_class"]=classifier.predict(X_real)
samples.replace({"pred_class":inverse_class_proxies}, inplace=True)
output=samples[["id","pred_class"]]
# print(output.shape)
# print(classifier.predict(X_real).shape)
# print(classifier.predict(X_real)[:5])
# print(output.head())
outname=r"\KAGGLE.csv"
print("\nSaving the competition formatted output to file (%s)- ready for KAGGLE scoring!!! \n" % outname)
output.to_csv(folder+outname, index=False,encoding="utf-8")

################################################################################
print("Real world set: Predicted Class histogram!!! ")
sns.countplot(output.pred_class)
plt.show()

################################################################################
print("Real world set: Feature histogram per predicted Class !!! ")
def feature_histogram_by_predclass(feature_names,n_classes,actual_class_labels,output,X_real):
    hist_features=feature_names
    y=np.array(output.pred_class)
    X_hist=X_real
    fig, axes = plt.subplots(int((len(hist_features)+1)*0.5),2, figsize=(5, 10))
    plt.title("Per predicted Class: feature histogram")
    classes=[]
    ax = axes.ravel()
    for cl in range(n_classes):
        classes.append(X_hist[y==actual_class_labels[cl]])
    for i in range(X_hist.shape[1]):
        _, bins = np.histogram(X_hist[:, i], bins=20)
        for cl in range(n_classes):
            ax[i].hist(classes[cl][:,i], bins=bins, alpha=.5)
        ax[i].set_title(hist_features[i], loc="right")
        ax[i].set_yticks(())
        ax[i].set_xlabel("Feature magnitude")
        ax[i].set_ylabel("Frequency")
        ax[i].legend(class_labels, loc="best")
    fig.tight_layout()
    plt.show()

feature_histogram_by_predclass(feature_names,n_classes,actual_class_labels,output,X_real)
################################################################################
################################################################################