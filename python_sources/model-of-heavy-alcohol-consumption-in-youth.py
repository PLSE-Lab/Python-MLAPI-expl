#!/usr/bin/env python
# coding: utf-8

# [back to executive summary](https://www.kaggle.com/joeycorea/academicsuccessalcoholconsumptionexecutivesummary)
# # Model of heavy alcohol consumption among high school students
# 
# This section of the analysis will attempt to construct a model to predict which students are at risk of heavy drinking behaviour.
# You may recall that a student is labelled a heavy drinker when they have
# 
# * a high level (Walc = 4-5) of weekend alcohol consumption accompanied by 
# * a high level (Dalc = 4-5) of weekday alcohol consumption.
# 
# <a id="executive_summary"></a>
# # Executive Summary
# The target variable was highly unbalanced with the positive class (heavy drinkers) making up less than 5% of the population. For this reason, recall was chosen as the performance metric rather than accuracy. That is, performance should be weighted towards identifying heavy drinkers even if it means incorrectly classifying some normal drinkers as heavy drinkers.
# 
# The most effective model proved to be a Principle Component Analysis (PCA) followed by logistic regression. The PCA reduced the 50 feature space to three features and achieved the most impressive recall in finding the heavy drinkers but also incorrectly classifying many non-drinkers as heavy drinkers.
# 
# In an attempt to detect any subtle signals, both the maths and Portugese classes were combined in the same analysis. If more data could be obtained, these should be considered separately. The shape of the learning curves on both simple and complex models suggest that more data would help amplify the subtle signals and perhaps smoothen out the noise found in this analysis.
# 
# 
# # Uninteresting Helper Code
# [Fast forward to the meat](#reading_in_file)

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.ticker as ticker

from IPython.core.debugger import set_trace #https://docs.python.org/3/library/pdb.html

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, PowerTransformer

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.metrics import roc_curve, auc, confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
#to allow plots to render in the notebook

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 18})


# In[ ]:


class AlcoholStudyDataPreparer:
    
    def __init__(self):
        
        pass
    
    def read_in_data_from(self, path):
        self.data= pd.read_csv(path)
        
    def data_length_checker(self):
        print("There are ", str(self.data.shape[1]), " columns in the data")
        
    def move_alcohol_related_variables_to_end_of_column_list(self, data):
        self.data = data
        self.data_length_checker()
        print("Moving alcohol related variables to the end of the list")
        alcohol_related_variables = ["binge_drinker", "heavy_drinker", "Dalc", "Walc"]
        cols = self.data.columns.tolist()
        print("Removed extra index column resulting from csv import")
        cols = set(cols) - set(alcohol_related_variables) - set(["Unnamed: 0"])
        cols = list(cols) + alcohol_related_variables
        self.data = data[cols]
        self.data_length_checker()
        
    def ready_data_for_experiments(self, path):
        self.read_in_data_from(path)
        self.move_alcohol_related_variables_to_end_of_column_list(self.data)
        print(self.data.shape)
        return self
    
    def plot_dependent_variable(self):
        data = self.data
        f, ax = plt.subplots(1,1)
        sns.countplot("heavy_drinker", data=data, ax=ax)
        f.suptitle("Distribution of heavy drinkers")
        print(data.groupby(data["heavy_drinker"])["heavy_drinker"].count())
        
class Experiment:
    
    def __init__(self):
        
        pass
    
    def run_experiment(self, data, y_col_name, estimator, evaluation):
        self.data = data
        model = ModelFitter(data)
        self.X_train, self.X_test, self.y_train, self.y_test = model.get_test_train_splits(y_col_name)
        fit_model = model.fit_model_and_evaluate(estimator, evaluation)
        Evaluation().plot_learning_curve(self.data, estimator)
        return fit_model
        
    
class ModelFitter:
    
    def __init__(self, data):
        self.data = data
        pass
    
    def separate_X_from_y(self, y_col_name):
        self.X = self.data.iloc[:, :-4]
        self.y = self.data.loc[:, y_col_name].values.reshape(-1, )
        return self.X, self.y
    
    def divide_into_training_and_test(self, train_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, random_state=0, train_size=train_size, test_size=(1-train_size)
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_test_train_splits(self, y_col_name, train_size=0.7):
        self.separate_X_from_y(y_col_name)
        return self.divide_into_training_and_test(train_size)
    
    def fit_model(self, estimator, X_train, y_train):
        self.model = estimator(X_train, y_train)
        fit_model = self.model.fit(X_train, y_train)
        return fit_model
    
    def fit_model_and_evaluate(self, estimator, evaluation):
        fitted_model = self.fit_model(estimator, self.X_train, self.y_train)
        evaluation(fitted_model, self)
        return fitted_model
    
class Evaluation:
    
    def print_accuracy_score(self, model, data):
        print("Accuracy is: " + str(model.score(data.X_train, data.y_train)))
        
    def print_confusion_matrix(self, model, data, figsize=(20,8), fontsize=14, model_name=""):
        #tweaked from shyapal5 https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
        class_names = ["normal_drinker", "heavy_drinker"]
        
        f, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)
        
        def __create_single_heatmap(X, y, name, ax):
            y_pred = model.predict(X)
            conf_matrix = confusion_matrix(y, y_pred)
            df_cm = pd.DataFrame(
                conf_matrix, index=class_names, columns=class_names
            )
            try:
                heatmap = sns.heatmap(df_cm, annot=True, fmt="d", ax=ax)
            except ValueError:
                raise ValueError("Confusion matrix values must be integers")
            heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
            heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
            heatmap.set_title(("Confusion matrix for " + name + " set"))
            #plt.ylabel('True label')
            heatmap.set_xlabel('Predicted label')
            
        __create_single_heatmap(data.X_train, data.y_train, "train", ax1)
        __create_single_heatmap(data.X_test, data.y_test, "test", ax2)
        ax1.set_ylabel("True Label")
        ax2.set_ylabel("")
    
    def __compute_auc(self, y, y_pred):
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        return auc(fpr, tpr)
    
    def __compute_confusion_matrix(self, y, y_pred):
        return confusion_matrix(y, y_pred)
    
    def __calculate_metric(self, fitted_model, X, y):
        y_pred = fitted_model.predict(X)
        return self.metric(self, y, y_pred) #if I were passing in a custom metric, would it be tricky to pass in the Evaluation in stance from the caller?
        
    def __get_train_and_test_metric_for_train_size(self, train_size):
        model = ModelFitter(self.data)
        #TODO: find a way to get rid of this magic string
        self.X_train, self.X_test, self.y_train, self.y_test = model.get_test_train_splits("heavy_drinker", train_size)
        fitted_model = model.fit_model(self.estimator, self.X_train, self.y_train)
        train_metric = self.__calculate_metric(fitted_model, self.X_train, self.y_train)
        test_metric = self.__calculate_metric(fitted_model, self.X_test, self.y_test)
        return (train_size, train_metric, test_metric)
    
    def __get_learning_curve_data(self):
        metrics = [self.__get_train_and_test_metric_for_train_size(train_size) for train_size in self.train_sizes]
        train_df = pd.DataFrame(data=[(metric[0], metric[1], "train") for metric in metrics])
        test_df = pd.DataFrame(data=[(metric[0], metric[2], "test") for metric in metrics])        
        metrics = pd.concat([train_df, test_df])
        metrics.columns = ["train_size", "auc", "split"]

        return metrics
    
    def plot_learning_curve(self, data, estimator, train_sizes=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9], metric=__compute_auc):
        
        #plots the learning_curves
        self.metric = metric
        self.data = data
        self.train_sizes = train_sizes
        self.estimator = estimator
        lc_data = self.__get_learning_curve_data()
        fig, ax = plt.subplots(1,1)
        fig.suptitle("Learning Curve")
        sns.pointplot("train_size", "auc", hue="split", data=lc_data, ax=ax)
        return lc_data

class PCAInvestigator:
    
    def __init__(self, pca, data):
        self.pca = pca
        self.data = data
        self.create_pca_df()
        self.transformed = self.transform_to_pca_components()
        pass

    def create_pca_df(self):
        components = self.pca.components_
        self.component_labels = [("PCA" + str(i+1)) for i, val in enumerate(components)]
        df = pd.DataFrame(data=components, columns=self.data.X_train.columns).transpose()
        df.columns = self.component_labels
        self.df = df
        return df
    
    def get_n_largest_features_for_component(self, component_name, n):
        
        abs_col = str(component_name + "_abs")
        self.df[abs_col] = self.df[component_name].abs() #for ranking purposes, we need the absolute value of the coeff
        components = self.df.nlargest(n, abs_col)[[component_name, abs_col]]
        df = pd.DataFrame(data=components).reset_index() #unhooks the feature names from the index
        df["rank"] = df[abs_col].rank(ascending=False)
        df = df.loc[:, ["index", component_name, "rank"]] #after we have rank, we no longer need abs_col
        df.columns=["feature_name_" + component_name, "variance_explained_" + component_name, "rank"]
        return df
    
    def get_n_largest_features_for_pca(self, n):
        
        n_largest_features = [self.get_n_largest_features_for_component(label, n) for label in self.component_labels]
        
        if len(n_largest_features) > 1:
            length = len(self.component_labels[:-1]); 
            i = 0;
            
            def merge_feature_dfs(df, dfs, labels, length, i):
                i = i + 1
                combined_df = pd.merge(df, dfs[i], on="rank")
                if (i >= length):
                    return combined_df
                else:
                    return merge_feature_dfs(combined_df, dfs, labels, length, i)
            
            self.largest_features_per_factor = merge_feature_dfs(n_largest_features[0], n_largest_features, self.component_labels, length, i).set_index("rank")
        else:
            self.largest_features_per_factor = n_largest_features[0]
        return self.largest_features_per_factor
    
    
    def transform_to_pca_components(self):
        
        pca_results = self.pca.transform(self.data.X_train)
        pca_results = pd.DataFrame(data=pca_results)
        pca_results.columns = self.component_labels
        pca_results["heavy_drinker"] = self.data.y_train
        return pca_results
    
    def plot_pca_components(self):
        
        f, ax = plt.subplots(1,1, figsize = (10,7))
        f.suptitle("Heavy Drinker instances across PCA1 vs PCA2")
        sns.scatterplot("PCA1", "PCA2", hue="heavy_drinker", data=self.transformed);
        
    def plot_bar_chart_of_component_features(self, component_name):
        #these lines are scoping the columns to be selected
        feature_name = "feature_name_" + component_name
        variance_explained = "variance_explained_" + component_name
        columns = [feature_name, variance_explained]
        
        f, ax = plt.subplots(1,1, figsize = (10,7))
        f.suptitle("Variance explained for features in component: " + component_name)
        barplot = sns.barplot(feature_name, variance_explained, data=self.largest_features_per_factor[columns], ax=ax)
        barplot.xaxis.set_ticklabels(barplot.xaxis.get_ticklabels(), rotation=45, ha='right')
        
    def plot_distplot_of_component_features(self, component_name):
        
        f, ax = plt.subplots(1,1, figsize = (10,7))
        f.suptitle("Distribution of " + component_name)
        sns.distplot(a=self.transformed[component_name])
        
    def plot_pca_component(self, y):
        
        f, ax = plt.subplots(1,1, figsize = (10,7))
        f.suptitle("Distribution of heavy drinkers across " + y)
        sns.swarmplot(x="heavy_drinker", y=y, data=self.transformed);
        


# <a id="reading_in_file"></a>
# # Reading in file and checking columns

# In[ ]:


asdp = AlcoholStudyDataPreparer().ready_data_for_experiments('../input/mathportugeseyouthalcoholstudy/student_math_por_formatted.csv')


# # Metric Selection

# In[ ]:


asdp.plot_dependent_variable()


# The figure above shows that the classes are imbalanced; thus accuracy may not be a suitable measure.
# 
# Take, for example, an accuracy of 95%. Since the heavy drinkers make up less than 5% of the sample, a classifier could fail to detect all of the heavy drinkers in the sample and still achieve a 95% accuracy.
# 
# The baseline model below will make this more concrete.
# 
# Effectiveness of this model would be 
# * finding all the heavy drinkers so that they can be targeted for any interventions.
# * the risk of classifying normal drinkers as heavy_drinkers is acceptable but should be minimised.
# * the risk of classigying heavy_drinkers as normal_drinkers is not acceptable.
# 
# # Modelling
# ## Baseline performance: Stratified Dummy Classifier

# In[ ]:


#exp1_classifier = DummyClassifier(strategy="stratified", random_state=0)
def exp1_classifier(X_train, y_train):
    return DummyClassifier(strategy="stratified", random_state=0)

exp1_eval = Evaluation().print_confusion_matrix
exp1 = Experiment()
dummy = exp1.run_experiment(asdp.data, "heavy_drinker", exp1_classifier,exp1_eval)
print("accuracy score: " + str(dummy.score(exp1.X_test, exp1.y_test)))


# ### Lots of room for improvement
# The results above illustrate the inappropriateness of using accuracy as a performance metric: The accuracy score is 93%. However, as can be seen from the confusion matrix, only one heavy_drinker was predicted correctly. What's worse, nine normal drinkers were incorrectly classified as heavy drinkers (somewhat acceptable) and nine heavy drinkers were incorrectly classified as normal drinkers (not acceptable).
# 
# A dummy classifier is meant to give us only a baseline of recall performance. Hopefully, things will only improve from here!
# 
# #### Learning Curves
# 
# As noted in the [previous exploration](https://www.kaggle.com/joeycorea/academicsuccessalcoholconsumptioneda), the small sample size is a concern. Thus, learning curves will be plotted and interpreted to see if there is a case for requesting more data. The learning curves will be plotted using the area under the ROC curve metric.
# <a id="logistic-regression"></a>
# ## Logistic Regression Approach
# ### Preprocessing
# 1. Remove collinearity by applying PCA.
# 2. Investigate factors for linear relationship with heavy_drinker variable.
# 3. Apply transformations as required to remove outliers

# In[ ]:


def exp2_classifier(X_train, y_train):
    
    estimators = [('robust_scaler', RobustScaler()), 
                  ('reduce_dim', PCA(random_state=0)), 
                  ('normalise_pca', PowerTransformer()), #I applied this as the distribution of the PCA factors were skew
                  ('clf', LogisticRegression(random_state=0, solver="liblinear"))] 
                #solver specified here to suppress warnings, it doesn't seem to effect gridSearch
    pipe = Pipeline(estimators)
    pipe.fit(X_train, y_train)
    
    param_grid = dict(reduce_dim__n_components=[1, 2, 3, 4, 5, 10],
                    clf__solver=["liblinear", "lbfgs"],
                     clf__class_weight=["balanced", None])
    return GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="recall", iid=False)

exp2_eval = Evaluation().print_confusion_matrix
exp2 = Experiment()
logit_grid = exp2.run_experiment(asdp.data, "heavy_drinker", exp2_classifier, exp2_eval);

#TODO: link this kernel here: https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search#354172


# ### Performance: Promising
# 
# The logistic regression appears promising. Note, however, that recall is still not perfect as there are still 3 cases of heavy_drinkers getting classified as normal drinkers. 
# 
# ### Learning Curve
# 
# The test performance is climbing with increased training size which suggests that more training data might be useful.
# 
# Since there are relatively few instances in the target class, Linear Discriminant Analysis might be useful to apply
# 
# ## Linear Discriminant Analysis

# In[ ]:


def exp3_classifier(X_train, y_train):
    
    estimators = [('robust_scaler', RobustScaler()), 
                  ('reduce_dim', PCA(random_state=0)), 
                  ('normalise_pca', PowerTransformer()), #I applied this as the distribution of the PCA factors were skew
                  ('clf', LinearDiscriminantAnalysis())] 
                #solver specified here to suppress warnings, it doesn't seem to effect gridSearch
    pipe = Pipeline(estimators)
    pipe.fit(X_train, y_train)
    
    param_grid = dict(reduce_dim__n_components=[1, 2, 3, 4, 5, 10])
    return GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="recall", iid=False)

exp3_eval = Evaluation().print_confusion_matrix
exp3 = Experiment()
gbdt_grid = exp3.run_experiment(asdp.data, "heavy_drinker", exp3_classifier, exp3_eval);


# ### Performance: Surprisingly Poor
# 
# Running with the same pipeline as the logistic regression resulted in anomalously poor peformance. Interestingly, removing the PCA step to the pipeline resulted in better performance, even in the test set. However, doing so violates the assumption of minimum collinearity thus this approach should probably not be considered.
# 
# ### Learning Curve
# 
# The learning curve is difficult to inpterpet here and perhaps should be discarded along with the performance of this model.
# 
# Given that *any* improvement seen in the LDA model was trivial compared to logistic regression, the anomalies seen in this learning curve won't be investigated further (but is included in this report to show the reader the diversity of methods employed).
# 
# ## Gradient Boosted Decision Trees
# 
# 3 models will be investigated here and [discussed collectively at the end of the section](#GBDT_Conclusion).
# 
# ### Using the same pipe / GridSearch params (where applicable)

# In[ ]:


def exp3_classifier(X_train, y_train):
    
    estimators = [('robust_scaler', RobustScaler()), 
                  ('reduce_dim', PCA(random_state=0)), 
                  ('normalise_pca', PowerTransformer()), #I applied this as the distribution of the PCA factors were skew
                  ('clf', GradientBoostingClassifier(random_state=0))] 
                #solver specified here to suppress warnings, it doesn't seem to effect gridSearch
    pipe = Pipeline(estimators)
    pipe.fit(X_train, y_train)
    
    param_grid = dict(reduce_dim__n_components=[1, 2, 3, 4, 5, 10])
    return GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="recall", iid=False)

exp3_eval = Evaluation().print_confusion_matrix
exp3 = Experiment()
gbdt_grid = exp3.run_experiment(asdp.data, "heavy_drinker", exp3_classifier, exp3_eval);


# The improvement over baseline is trivial and outclassed by logistic regression. However, it could be that the dimensionality reduction is oversimplifying the space for the model. In the next step, the model will have the opportunity to search the entire feature space by removing the preprocessing pipeline.

# ### Using no transformation pipeline nor GridSearch

# In[ ]:


def exp4_classifier(X_train, y_train):
    
    return GradientBoostingClassifier(random_state=0)

exp4_eval = Evaluation().print_confusion_matrix
exp4 = Experiment()
gbdt_grid = exp4.run_experiment(asdp.data, "heavy_drinker", exp4_classifier, exp4_eval);


# ### Using GridSearchCV
# See the param_grid variable below for the hyperparameters searched over.

# In[ ]:


def exp5_classifier(X_train, y_train):
    
    clf = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
    
    param_grid = dict(loss=["deviance", "exponential"],
                     learning_rate=[0.1,0.2,0.5,0.8],
                     n_estimators=[50, 100, 150],
                    max_depth=[1,3,5],
                      max_features=["sqrt", "log2", 3] #using the rule of 3s
                     )
    
    return GridSearchCV(clf, param_grid=param_grid, cv=3, scoring="recall", iid=False) 

exp5_eval = Evaluation().print_confusion_matrix
exp5 = Experiment()
gbdt_grid = exp5.run_experiment(asdp.data, "heavy_drinker", exp5_classifier, exp5_eval);


# <a id="GBDT_Conclusion"></a>
# ### Gradient Boosted Decision Tree Suite Conclusion
# 
# #### Performance: Relatively Poor
# 
# Overall, the suite of GBDT models a far inferior to logistic regression in recall performance. Indeed, there was very little difference between the different GBDT iterations applied above.
# 
# #### Learning Curves
# 
# The following interpretations of the curves suggest that more training data could allow the model to better classify normal from heavy drinkers:
# 
# * the upward trend of test performance with increasing training set size
# * in all cases, it appeared like the models were overfitting the training data

# ## Random Forest Classifier
# 
# 2 models will be investigated and will be [discussed collectively at the end of the section](#RF_Conclusion)
# ### Without GridSearchCV

# In[ ]:


def exp6_classifier(X_train, y_train):
    
    return RandomForestClassifier(random_state=0, n_estimators=100)

exp6_eval = Evaluation().print_confusion_matrix
exp6 = Experiment()
gbdt_grid = exp6.run_experiment(asdp.data, "heavy_drinker", exp6_classifier, exp6_eval);


# ### With GridSearchCV

# In[ ]:


def exp7_classifier(X_train, y_train):
    
    estimators = [('robust_scaler', RobustScaler()), 
                  ('reduce_dim', PCA(random_state=0)), 
                  ('normalise_pca', PowerTransformer()), #I applied this as the distribution of the PCA factors were skew
                  ('clf', RandomForestClassifier(random_state=0, n_estimators=100))] 
                #solver specified here to suppress warnings, it doesn't seem to effect gridSearch
    pipe = Pipeline(estimators)
    pipe.fit(X_train, y_train)
    
    param_grid = dict(reduce_dim__n_components=[1, 2, 3, 4, 5, 10])
    return GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="recall", iid=False)

exp7_eval = Evaluation().print_confusion_matrix
exp7 = Experiment()
exp7_grid = exp7.run_experiment(asdp.data, "heavy_drinker", exp7_classifier, exp7_eval);


# <a id="RF_Conclusion"></a>
# ### Random Forest Suite Conclusion
# 
# #### Performance
# 
# Overall, the suite of Random Forest models a far inferior to logistic regression in recall performance. Like the GBDT suite, there was very little difference between the different Random Forest iterations applied above.
# 
# #### Learning Curves
# 
# The suggestions from the learning curves were mixed:
# * random forest alone showed an upward trend of test performance with training set size.
# * random forest with dimensionality reduction and GridSearchCV showed a relationship between test performance and training size that was too noisey to interpret.
# * in both cases, it appeared like the models were overfitting the training data
# 
# # Deeper Dive into Dimensionality Reduction Analysis: 3 Components
# 
# Logistic Regression combined with dimensionality reduction to 3 components provided the best result out of all the models studied above. The following section will take a closer look at these components.

# In[ ]:


pca_investigation = PCAInvestigator(logit_grid.best_estimator_.steps[1][1], exp2)
pca_investigation.get_n_largest_features_for_pca(10)


# Using the grid search CV, it was found that the pipeline worked best when the feature space was reduced to 3 features.
# ##### PCA 1 Analysis

# The barplot below suggests PCA1 emphasises academic performance and possibly conscientiousness

# In[ ]:


pca_investigation.plot_bar_chart_of_component_features("PCA1")


# In[ ]:


pca_investigation.plot_distplot_of_component_features("PCA1")


# In[ ]:


pca_investigation.plot_pca_component(y="PCA1")


# <a id="a_telling_swarm"></a>
# From the plot below, it is a surprise that the model was able to pick the heavy drinkers from the crowd: For PCA 1, the heavy drinkers somewhat evenly distributed among the normal drinkers. It should be noted, however, in the test set, 113 normal drinkers were classified as heavy drinkers. Depending on the investment in any interventions on suspected heavy drinkers, this may be unacceptable.
# 
# [back to executive summary](#executive_summary)

# ##### PCA2 Analysis
# 
# This component is difficult to interpret. It might point to health (high absences) or family issues (low famrel).

# In[ ]:


pca_investigation.plot_bar_chart_of_component_features("PCA2")


# In[ ]:


pca_investigation.plot_distplot_of_component_features("PCA2")


# In[ ]:


pca_investigation.plot_pca_component(y="PCA2")


# ##### PCA3 Analysis
# 
# This factor is difficult to interpret in ioslation, let alone in the context of PCA1 and PCA2. The top five components relate to an emphasis on extracurricular activities.

# In[ ]:


pca_investigation.plot_bar_chart_of_component_features("PCA3")


# In[ ]:


pca_investigation.plot_distplot_of_component_features("PCA3")


# In[ ]:


pca_investigation.plot_pca_component(y="PCA3")


# ##### PCA Analysis Conclusion
# 
# The feature space appears to be reduced to 
# * PCA1: conscientiousness and academic achievement
# * PCA2: health and family relationship quality
# * PCA3: extracurricular activities and responsibilities
# 
# The second and third features mentioned appear very similar in terms of their distribution and are particularly noisey when compared with PCA1.

# # Conclusion
# 
# Despite the poor performance of most of the models implemented above, there is reason to believe that the investigation can be improved with more training data. This is suggested by:
# 
# * the upward trend between test performance and training set size on most learning curves.
# * the tendency for complex models like GBDT and Random Forest to overfit the training data.
# 
# The finding that the relatively simplistic logistic regression could recall many of the heavy drinkers in the test set was promising. Part of this procedure was reducing the feature space from 50 features to 3. These features seemed to correspond to:
# 
# * conscientiousness and academic achievement
# * health and family relationship quality
# * extracurricular activities and responsibilities
# 
# However, the interpretations of these are fairly loose as the mapping of the features to the components was noisey (especially for the second and third features).
# [back to executive summary](https://www.kaggle.com/joeycorea/academicsuccessalcoholconsumptionexecutivesummary)
