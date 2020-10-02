#!/usr/bin/env python
# coding: utf-8

# ### Author: Guilherme Resende
# This practical work was developed over the [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset and is meant to analyze the data and build a predictive model capable of distinguish whether a given transaction is or is not a fraud. [Anomaly Detection](https://pt.wikipedia.org/wiki/Detec%C3%A7%C3%A3o_de_anomalias) problems are interesting because they demand a different approach given the regular methods usually do not perform well in such scenarios.
# 
# Here I decided to test a stacked approach based on Wolpert's work [[DH Wolpert - 1992](https://www.sciencedirect.com/science/article/pii/S0893608005800231)]. This approach presents the input data $X_1$ to a given layer($L_1$) of learning models, combine their outputs with the actual input $X_2 = [X_1, L_1(X_1)]$ and present the new data to the next level of learning models. This procedure is performed until the structure gets to the final stack layer, that is, the final output. The outputs from inner layers are called meta-features.
# 
# ![stacked_generalization.png](attachment:stacked_generalization.png)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as pyo, plotly.graph_objs as go
import missingno as msno
import time

from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import *
from scikitplot.plotters import plot_ks_statistic
from scikitplot.metrics import plot_ks_statistic, plot_roc
from matplotlib import rcParams

from xgboost.sklearn import XGBClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest


# In[ ]:


class Stacking(object):
    def __init__(self, data, important_cols=None, fl_models=None, fl_models_names=None):
        self.data = data
        self.important_cols = important_cols
        self.fl_models = fl_models
        self.fl_models_names = fl_models_names
        for name in self.fl_models_names:
            self.data[name] = 0
        
        self.fl_data, self.sl_data = self._stacking_split()
        self.sl_model = XGBClassifier()
    
    
    def _fl_fit(self, X_train, Y_train):        
        for model in self.fl_models:
            model.fit(X_train, Y_train)
    
    
    def _fl_predict(self, X_test):
        preds = []
        for i, model in enumerate(self.fl_models):
             preds.append(model.predict(X_test).tolist())
        return np.array(preds).T.tolist()
            
    
    def _build_fl_models(self):
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        all_indices = []
        preds = []
        for index, (train_indices, test_indices) in enumerate(kf.split(self.fl_data[important_cols], self.fl_data['Fraud'])):
            train = self.fl_data.iloc[train_indices]
            test = self.fl_data.iloc[test_indices]
            all_indices += test_indices.tolist()
            
            self._fl_fit(train[self.important_cols].values, train['Fraud'].values)
            preds += self._fl_predict(test[important_cols].values)
            print("{}/5 of data ended training".format(index+1))
        preds = pd.DataFrame(preds, columns=self.fl_models_names)
        self.fl_data = self.fl_data.iloc[all_indices]
        for name in self.fl_models_names:
            self.fl_data[name] = preds[name].values

            
    def _sl_fit(self, X_train, Y_train):
        self.sl_model.fit(X_train, Y_train)
        
        
    def _sl_predict(self, X_test):
        return self.sl_model.predict(X_test)
    
    
    def _sl_predict_probs(self, X_test):
        return self.sl_model.predict_proba(X_test)
    
    
    def run_stacked(self):
        tik = time.time()
        self._build_fl_models()
        self._sl_fit(self.fl_data[important_cols+self.fl_models_names], self.fl_data['Fraud'])
        tok = time.time()
        
        minutes = int((tok-tik) / 60)
        hours = int(minutes / 60)
        minutes = minutes % 60

        print("The Training Process Took {}h {} min".format(hours, minutes))
        
        preds = pd.DataFrame(self._fl_predict(self.sl_data[important_cols].values), columns=self.fl_models_names)
        for name in self.fl_models_names:
            self.sl_data[name] = preds[name].values
            
        self.preds = self._sl_predict(self.sl_data[important_cols+self.fl_models_names])
        self.probs = self._sl_predict_probs(self.sl_data[important_cols+self.fl_models_names])

        
    def _stacking_split(self, split_at=0.7):
        frauds = self.data[self.data['Fraud'] == 1]
        non_frauds = self.data[self.data['Fraud'] == -1]
        fraud_pivot, non_frauds_pivot = int(len(frauds)*split_at), int(len(non_frauds)*split_at)

        fl_data = pd.concat([frauds.iloc[:fraud_pivot], non_frauds.iloc[:non_frauds_pivot]])
        sl_data = pd.concat([frauds.iloc[fraud_pivot:], non_frauds.iloc[non_frauds_pivot:]])
        
        return fl_data, sl_data


# In[ ]:


# Sets the figure size to Seaborn and Matplotlib plots
rcParams['figure.figsize'] = [15, 15]
# Allows Plotly to plot offline
pyo.init_notebook_mode(connected=True)


# In[ ]:


col_names = ['V'+str(index) for index in range(29)]


# First thing we need to do is to read the data from the repository and check if everything was read as expected

# In[ ]:


data = pd.read_csv("../input/creditcard.csv")


# In[ ]:


data.tail(3)


# Once the data is read correctly, it is needed to check if the dataset is not corrupted and if there isno null values. As the dataset is originated from a dimensionality reduction, it is expected to not have missing values, however, it is good to make sure.

# In[ ]:


msno.matrix(data)


# As we can see above there was no missing values in the dataset. Next thing needed is to create a naming pattern to the columns and then make sure all data is normalized. In order to standardize all column names,  I've changed 'Time' and 'Amount' to $V_0$ and $V_{29}$, respectively. Posteriorly, I do a standard scaling setting unit variance to the two columns here mentioned. As the machine learning models used below outputs -1 to anomalies/outliers and 1 to normal records/inliers and the dataset uses 1 to anomalies/outliers and 0 to normal records/inlies, I here do a mapping from $1 \rightarrow -1$ and $0 \rightarrow 1$.

# In[ ]:


data = data.rename(index=str, columns={"Class": "Fraud", "Time": "V0", "Amount": "V29"})
for col in ["V0", "V29"]:
    data[col] = scale(data[col].values)
    
data["Fraud"] = data["Fraud"].apply(lambda x: -1 if x else 1)


# The ideal case to treat a problem as anomaly detection problem, is when one of the classes is pretty narrow and the other one is abundant. Thus, to make sure I am, indeed, dealing with an anomaly detection problem, I present the percentage of the data that corresponds to frauds:

# In[ ]:


percentage = round(len(data[data.Fraud == -1]) / len(data) * 100, 4)
print("Percentage of Frauds is {}%".format(percentage))


# As we can see above, the frauds corresponds to only a small amount of the whole data, that is, 0.1727%, way less than 1 percent. Hence, I am indeed dealing with an anomaly detection problem!
# <p> To start going down the road to build the machine learning model to detect anomalies, I first need to check if the 'col_names' variable is as I expect it to be, that is, do not contains any undesirable column such as the fraud labels.

# In[ ]:


col_names


# Since everything is OK with 'col_names', I will now remove some of the features that do not aggregates lots of information to the analysis - present redundant information. To do that, I am going to fit a XGBoost model and then get the features importance learned by XGBoost. In order to filter the "unecessary" features, I will stablish a minimum importance level (previously tested, but here omitted for simplicity's sake) of 0.015, that is, every feature that has importance of at least 0.015 will remain for further analysis.

# In[ ]:


X, Y = data[col_names].values, data["Fraud"].values

model = XGBClassifier()
model.fit(X, Y)
feat_importances = dict(zip(col_names, model.feature_importances_))


# In[ ]:


important_cols = []
importance = 0
for key, value in feat_importances.items():
    if(value > 0.015):
        important_cols.append(key)
        importance += value


# In[ ]:


print("Num. Features Initially: {}".format(len(feat_importances)))
print("Num. Features After: {}".format(len(important_cols)))
print("Preserved Importance: {}".format(round(importance, 4)))


# As we can see above, the feature importance analysis was able to reduce the number of features from 29 to 17 and yet, preserve $\approx 89%$ of the total importance described by the XGBoost model. However, it is just an estimate, since I do not expect the model to correctly learn all data characteristics. 
# 
# <p> Next step I will take in this analysis is to plot all features histograms in order to check their display. I do not expect them all to look Gaussian like, however, it would be good if they do.

# In[ ]:


figure, axes = plt.subplots(nrows=4, ncols=4, sharey=True)

for i in range(4):
    for j in range(4):
        if(i == 3 and j > 0):
            break
            
        axes[i, j].hist(data[important_cols[i*4 + j]].values, bins=75)
        axes[i, j].set_title(important_cols[i*4 + j])
        
plt.show()


# As it was evidenced above, almost all features already follows a Gaussian distribution, except for $V_{6}$ and $V_4$. However, even though these columns have some deviational behaviour, they do resambles a bell shaped curve and hence will be used without problem on the following approachs.
# 
# <p> For the following step, I am going to plot the features correlation in order to avoid inputting redundant information on the models I am about to build

# In[ ]:


sns.heatmap(data[important_cols].corr().values, 
            linewidth=0.5, 
            xticklabels=important_cols, 
            yticklabels=important_cols)

plt.title("Important Features Correlation Matrix")
plt.show()


# As the plot above shows, the features are almost nothing correlated, so, there will be no problem inputting all of them together. Once this things are known, the models are ready to be defined and created. For this dataset I will develop a two level stacking approach, in which the first layer is composed by two supervised learning models (Elliptic Envelope and Isolation Forest) and one unsupervised (Local Outlier Factor), and the second layer is composed of a XGBoost classifier.

# In[ ]:


models_names = ["EllipticEnvelope", 
                    "IsolationForest",
                    "LocalOutlierFactor"]

models = [EllipticEnvelope(support_fraction=0.7), 
              IsolationForest(behaviour="new", contamination="auto"), 
              LocalOutlierFactor(novelty=True, contamination="auto")]


# When dealing with supervised learning models, we need to try to reduce the chance of overfitting, that is, do not let it spend too much time trying to learn the data specificities/noise and lose the ability to generalize the results. When a model overfits we can expect it to have a great performance on the training data and have a poor performance on test data. One way to avoid overfitting is to perform K-Fold-1-Out Cross-Validation on the data. That means divide the whole dataset into K folds, train with K-1 folds and then test it on the one left out. To this work I decided to set $K=5$. Hence, as I go through the 5 left out folds, all data will have been validated with a test set.
# 
# As I am dealing with a very narrow class, it would be good to make sure the number of anomalies is approximately the same over all splits. To do so, I will use a Stratified K-Fold-1-Out Cross-Validation process.For each model defined above, there will be created a new column representing the model's outputs to be used as meta-features on the second learning level.

# In[ ]:


get_ipython().run_cell_magic('capture', '', "stack = Stacking(data=data[important_cols + ['Fraud']], \n                 important_cols=important_cols, \n                 fl_models=models, \n                 fl_models_names=models_names)")


# In[ ]:


stack.run_stacked()


# As aforementioned, once the K-Fold-1-Out Cross-Validation process ends, every new column will have been built using only the test folds. So, for each new column created, I calculate Precision, Recall and F1-score all at once.

# In[ ]:


recs = []
precs = []
f1s = []

for name in models_names:
    prediction = stack.fl_data[name].values
    Y_fl = stack.fl_data['Fraud'].values
    precs.append(precision_score(Y_fl, prediction))
    recs.append(recall_score(Y_fl, prediction))
    f1s.append(f1_score(Y_fl, prediction))


# In[ ]:


trace1 = go.Bar(x=models_names, 
                y=precs, 
                name="Precision")
trace2 = go.Bar(x=models_names, 
                y=recs, 
                name="Recall")
trace3 = go.Bar(x=models_names, 
                y=f1s, 
                name="F1")

traces_list = [trace1, trace2, trace3]
layout = go.Layout(title="First Layer Models Performance")
figure = go.Figure(traces_list, layout)

pyo.iplot(figure)


# As a way to check if a stacking approach is suitable for this dataset, we need to know if the first level models outputs are usually different from each other. If so, they probably will aggregate value to the learning process as a whole and improve the performance, that is, the final model only need to know who to trust in each case

# In[ ]:


get_ipython().run_cell_magic('capture', '', '\n# Test the discordance rate over first level outputs\ndf_tmp = stack.fl_data[["Fraud"]]\ndf_tmp["outputs_sum"] = stack.fl_data[models_names].sum(axis=1).values\n\nall_anomalies = len(df_tmp[df_tmp["Fraud"] == -1])\noutputs_sum = df_tmp[df_tmp["Fraud"] == -1]["outputs_sum"].values\n\n# If it is equals 3, all first level models agreed that it was \n# not an anomaly and consequently, if it is equals -3, all first \n# level models considered a record as an anomaly\noutputs_sum = [0 if value == 3 or value == -3 else 1 for value in outputs_sum]')


# In[ ]:


print(outputs_sum[:30])


# In[ ]:


print("The discordance among first level models is: {} (regarding anomalous records)"                                                  .format(sum(outputs_sum) / all_anomalies))


# The whole process performance can be measured contrasting the last layer outputs with the actual labels. In order to measure how well the developed approach performed I first plot the values regarding Precision, Recall and F1-score.

# In[ ]:


X = ['Precision','Recall','F1-Score']

Y_sl = stack.sl_data['Fraud']
Y = [np.round(precision_score(Y_sl, stack.preds)*100, 2), 
     np.round(recall_score(Y_sl, stack.preds)*100, 2),
     np.round(f1_score(Y_sl, stack.preds)*100, 2)]


# In[ ]:


trace = go.Bar(x=X, y=Y)
traces = [trace]

layout = go.Layout(title='Stack Performance Over Metrics')
figure = go.Figure(traces, layout)

pyo.iplot(figure)


# As we can see, the stacking process seems to be pretty performatic, reaching almost 100% score in Precision, Recall and F1. Hence, in order to make sure the model is capable of well separate the classes, I present the ROC curve visualization together with the respective AUC metric regarding the anomalous records

# In[ ]:


# Sets the figure size to Seaborn and Matplotlib plots
rcParams['figure.figsize'] = [9, 9]


# In[ ]:


plot_roc(Y_sl, stack.probs, plot_micro=False, plot_macro=False)
plt.show()


# As presented above, the model seems to be, indeed, pretty good in classifying whether a record is an anomalous transation or not. Then, as a way to certify the model is not only getting the answers right by guess, I present the visualization and values of [Kolmogorov-Smirnov Test (KS Test)](https://pt.wikipedia.org/wiki/Teste_Kolmogorov-Smirnov). The KS Test points out how much a sampled distribution is close to the original distribution, in other words, considering the true labels as the original distribution and the model's output as the sampled distribution, how close is the sample probability distribution to the actual outputs?

# In[ ]:


plot_ks_statistic(Y_sl, stack.probs)
plt.show()


# Considering that, all values are being predicted with a high confidence level, we are able to say that the approach here developed is pretty satisfactory. The approach performed way better than expected, almost mimicking all actual outputs. However, considering that, we do not have the actual data distribution the model probably got biased by the sampled distribution presented to it. In order to further validate the model, I intend to run the current approach with a new database taken from a different distribution.

# In[ ]:




