

from __future__ import print_function, division

import pandas as pd
import matplotlib.pyplot as plt
import h5py
import csv
import numpy as np
from numpy import array 
import xlrd

import time
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from pandas import set_option
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

file = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
array = np.array(file)
dataframe = pd.DataFrame(file)

label = dataframe['SARS-Cov-2 exam result']
dados = dataframe.drop(['Patient ID',
						'SARS-Cov-2 exam result',
						'Patient addmited to regular ward (1=yes, 0=no)',
						'Patient addmited to semi-intensive unit (1=yes, 0=no)',
						'Patient addmited to intensive care unit (1=yes, 0=no)',
						'Respiratory Syncytial Virus',
						'Influenza A',
						'Influenza B',
						'Parainfluenza 1',
						'CoronavirusNL63',
						'Rhinovirus/Enterovirus',
						'Mycoplasma pneumoniae',
						'Coronavirus HKU1',
						'Parainfluenza 3',
						'Chlamydophila pneumoniae',
						'Adenovirus',
						'Parainfluenza 4',
						'Coronavirus229E',
						'CoronavirusOC43',
						'Inf A H1N1 2009',
						'Bordetella pertussis',
						'Metapneumovirus',
						'Parainfluenza 2',
						'Sodium',
						'Influenza B, rapid test',
						'Influenza A, rapid test',
						'Strepto A',
						'Urine - Esterase',
                		 'Urine - Aspect',
              	   	'Urine - pH',
             	    	'Urine - Hemoglobin',
                 	'Urine - Bile pigments',
                	 'Urine - Ketone Bodies',
             	    'Urine - Nitrite',
                 'Urine - Density',
                 'Urine - Urobilinogen',
                 'Urine - Protein',
                 'Urine - Sugar',
                 'Urine - Leukocytes',
                 'Urine - Crystals',
                 'Urine - Red blood cells',
                 'Urine - Hyaline cylinders',
                 'Urine - Granular cylinders',
                 'Urine - Yeasts',
                 'Urine - Color',
						], axis=1)

dados = dados.fillna(0)


feature_names = ['Patient age quantile',
                 'Hematocrit',
                 'Hemoglobin',
                 'Platelets',
                 'Mean platelet volume ',
                 'Red blood Cells',
                 'Lymphocytes',
                 'Mean corpuscular hemoglobin concentration (MCHC)',
                 'Lymphocytes',
                 'Basophils',
                 'Mean corpuscular hemoglobin (MCH)',
                 'Eosinophils',
                 'Mean corpuscular volume (MCV)',
                 'Monocytes',
                 'Red blood cell distribution width (RDW)',
                 'Serum Glucose',
                 'Neutrophils',
                 'Urea',
                 'Proteina C reativa mg/dL',
                 'Creatinine',
                 'Potassium',
                 'Sodium',
                 'Aspartate transaminase',
                 'Alanine transaminase',
                 'Total Bilirubin',
                 'Total Bilirubin',
                 'Indirect Bilirubin',
                 'Alkaline phosphatase',
                 'Ionized calcium',
                 'Magnesium',
                 'pCO2 (venous blood gas analysis)',
                 'Hb saturation (venous blood gas analysis)',
                 'Base excess (venous blood gas analysis)',
                 'pO2 (venous blood gas analysis)',
                 'Fio2 (venous blood gas analysis)',
                 'Total CO2 (venous blood gas analysis)',
                 'pH (venous blood gas analysis)',
                 'HCO3 (venous blood gas analysis)',
                 'Rods #',
                 'Segmented',
                 'Promyelocytes',
                 'Metamyelocytes',
                 'Myelocytes',
                 'Myeloblasts',
                 'Partial thromboplastin time (PTT) ',
                 'Relationship (Patient/Normal)',
                 'International normalized ratio (INR)',
                 'Lactic Dehydrogenase',
                 'Prothrombin time (PT), Activity',
                 'Vitamin B12',
                 'Creatine phosphokinase (CPK)',
                 'Ferritin',
                 'Arterial Lactic Acid',
                 'Lipase dosage',
                 'D-Dimer',
                 'Albumin',
                 'Hb saturation (arterial blood gases)',
                 'pCO2 (arterial blood gas analysis)',
                 'Base excess (arterial blood gas analysis)',
                 'pH (arterial blood gas analysis)',
                 'Total CO2 (arterial blood gas analysis)',
                 'HCO3 (arterial blood gas analysis)',
                 'pO2 (arterial blood gas analysis)',
                 'Arteiral Fio2',
                 'Phosphor',
                 'ctO2 (arterial blood gas analysis)']
                 

def trocar(num):
    if num == 'negative':
        return 0
    elif num == 'positive':
        return 1

label = label.map(trocar)            

                 
X_train, X_test, y_train, y_test = train_test_split(dados, label, test_size=0.3,random_state=200)

train = xgb.DMatrix(X_train,label=y_train)
                    
test = xgb.DMatrix(data=X_test,label=y_test)
                   
print('Number of training samples: {}'.format(train.num_row()))
print('Number of testing samples: {}'.format(test.num_row()))


param = []

# Boost5er parameters
param += [('eta',        0.07)]              # learning rate
param += [('max_depth',    5)]               # maximum depth of a tree
#param += [('gamma',    8)]  
param += [('subsample',  0.3)]               # fraction of events to train tree on
#param += [('max_depth',  10)]              

param += [('colsample_bytree', 0.3)]          # fraction of features to train tree on
# Learning task parameters
param += [('objective', 'binary:logistic')]   # objective function
param += [('eval_metric', 'error')]           # evaluation metric for cross validation
param += [('eval_metric', 'logloss')] + [('eval_metric', 'rmse')]
print(param)

num_trees = 100  # number of trees to make

booster = xgb.train(param,train,num_boost_round=num_trees)

print(booster.eval(test))

predictions = booster.predict(test)


from sklearn.metrics import roc_curve,precision_recall_curve
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test,predictions)
prec_xgb, rec_xgb, threshs_xgb = precision_recall_curve(y_test, predictions)


plt.plot(rec_xgb,prec_xgb,"b:")
plt.xlabel("efficiency")
plt.ylabel("purity")
plt.title("Train for xgbgoost")
#plt.savefig("PRC_train_xgb.png") # +suffix+
plt.show()

plt.plot(rec_xgb,prec_xgb*rec_xgb,"g:")
plt.xlabel("efficiency")
plt.ylabel("efficiency*purity")
plt.title("Train for xgbgoost")
#plt.savefig("RecPurity_train_xgb.png") # +suffix+
plt.show()

#abaixo está o melhor corte para o xgb para separar o signal de Background
bidxg_xgb = np.argmax(prec_xgb*rec_xgb)
best_cut_xgb = threshs_xgb[bidxg_xgb]


y_test_pred_best_xgb = predictions >= best_cut_xgb

from sklearn.metrics import precision_score,recall_score,accuracy_score
print("purity in test sample for xgbgoost     : {:2.2f}%".format(100*precision_score(y_test,y_test_pred_best_xgb)))
print("efficiency in test sample for xgbgoost : {:2.2f}%".format(100*recall_score(y_test,y_test_pred_best_xgb)))
print("accuracy in test sample for xgbgoost   : {:2.2f}%".format(100*accuracy_score(y_test,y_test_pred_best_xgb)))


hbgt_xgb =  plt.hist(predictions[y_test==0],bins=100,range=(0,1),histtype='step',label='negative')
hsigt_xgb = plt.hist(predictions[y_test==1],bins=100,range=(0,1),histtype='step',label='positive')
uppery_xgb=np.max(hbgt_xgb[0])*1.1
plt.plot([best_cut_xgb,best_cut_xgb],[0,uppery_xgb],"r:",label='best_cut_grid')
plt.axis([-0.01,1.01,0,uppery_xgb])
plt.xlabel("probability")
plt.ylabel("Number of events/bins of 0.01 width")
plt.title("Probability of signal for test sample validation for xgbgoost")
plt.legend(loc="upper right")
#plt.savefig("ProbTest_xgb.png") # +suffix+
plt.show()


from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score,precision_score,recall_score


def plotROC(predictions,y_test):
    # choose score cuts:
    cuts = np.linspace(0,1,500);
    n_truePos = np.zeros(len(cuts));
    n_falsePos = np.zeros(len(cuts));
    n_TotPos = len(np.where(y_test==1)[0])
    for i,cut in enumerate(cuts):
       y_pred = np.array([i>cut for i in predictions ])
       n_truePos[i] = len(np.where(predictions[y_test==1] > cut)[0]);
       n_falsePos[i] = len(np.where(predictions[y_test==0] > cut)[0]);
       if i%50 ==0:
         ascore = accuracy_score(y_test,y_pred)
         pscore = precision_score(y_test,y_pred)
         rscore = recall_score(y_test,y_pred)
         print("corte em {:2.1f} --> eficiência  {:2.1f} % e  pureza {:2.1f} %".format(cut,n_truePos[i]/n_TotPos *100,n_truePos[i]/(n_truePos[i]+n_falsePos[i])*100))
         print("                                                             accuracy_score = {:2.4f}     precision_score = {:2.4f}     recall_score = {:2.4f}\n".format(ascore,pscore,rscore))
    # plot efficiency vs. purity (ROC curve)
    plt.figure();

    custom_cmap3 = ListedColormap(['orange','yellow','lightgreen',"lightblue","violet"])
    plt.scatter((n_truePos/n_TotPos),n_truePos/(n_truePos + n_falsePos),c=cuts,cmap=custom_cmap3,label="ROC");
    # make the plot readable
    plt.xlabel('Efficiency',fontsize=12);
    plt.ylabel('Purity',fontsize=12);
    plt.colorbar()
    #plt.savefig("efficiency_x_purity_xgb.png")

    #plt.show()

plotROC(predictions,y_test)

xgb.plot_importance(booster,grid=False) # A "pontuação F" é o número de vezes que cada recurso é usado para dividir os dados em todas as árvores (vezes o peso dessa árvore).
#plt.savefig("booster_importance_xgb.png")
plt.show()