#!/usr/bin/env python
# coding: utf-8

# # 1. Import relevant packges / functions:

# In[ ]:


# Data manipulation:
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# ML packages:
from sklearn.model_selection import cross_validate, KFold
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, make_scorer
from sklearn.model_selection import train_test_split

# Auxilliary:
import os
from multiprocessing import cpu_count


# # 2. Load raw data:
# <font color=red>Note:</font>make sure the data is in the current working directory, or change `dataset_path` accordingly.

# In[ ]:


# Deinfe integer encoding for the 6 classes:
activity_to_code = {'dws': 1, 'ups': 2, 'sit': 3, 'std': 4, 'wlk': 5, 'jog': 6}
code_to_activity = {v:k for k,v in activity_to_code.items()}


# ### Load all data from all subjects / experiments, as well as their demographic data:

# In[ ]:


def load_raw_data(data_directory_path):
    """
    Given a path to the motionsense-dataset directory, loop through the different CSVs and concatenate them 
    to one pandas DataFrame. Join with demographic data, and return the DataFrame.
    """
    subjects_data_directory_path = os.path.join(dataset_path, "data_subjects_info.csv")
    
    # Load demographic data of subjects:
    subject_data = pd.read_csv(subjects_data_directory_path).rename(columns={'code':'subject'}) # rename for clarity
    subject_data['subject'] = subject_data.subject.astype(str)

    # Load data from sensor:
    motion_data_directory_path = os.path.join(dataset_path, r"A_DeviceMotion_data/A_DeviceMotion_data")
    dirs = os.listdir(motion_data_directory_path)
    dfs = []
    for d in dirs:
        activity_name, experiment_id = d.split("_")
        for subject in os.listdir(os.path.join(motion_data_directory_path, d)):
            filepath = os.path.join(os.path.join(motion_data_directory_path, d), subject)
            df = pd.read_csv(filepath, index_col=0)
            df['subject'] = subject.split(".")[0].split("_")[1] # keep only the subject's numerical i.d.
            df['activity'] = activity_to_code[activity_name]
            df['experiment_id'] = int(experiment_id)
            df['experiment_step'] =  np.arange(0, len(df)) # assign a numerical step number for every measurement in the experiment
            dfs.append(df)

    motion_data = pd.concat(dfs)
    
    # Join demographic data to final dataframe:
    final_df = motion_data.merge(subject_data, on=['subject'])
    
    return final_df, subject_data


# In[ ]:


dataset_path = r"../input/motionsense-dataset" # Assuming here dataset is in cwd.
raw_df, subject_data = load_raw_data(dataset_path)


# In[ ]:


print("raw data shape: ", raw_df.shape)
print("\nraw data columns:\n", raw_df.columns)


# # 3. EDA:

# ### Check for missing values:

# In[ ]:


raw_df.isnull().sum(axis=0)
# There are no missing values


# <font color=blue>There are no missing values in the data.</font>

# ### Inspect the distribution of the raw data:

# In[ ]:


MOTION_SENSOR_COLUMNS = ['attitude.roll', 'attitude.pitch', 'attitude.yaw', 'gravity.x', 'gravity.y', 'gravity.z', 
                         'rotationRate.x', 'rotationRate.y', 'rotationRate.z', 'userAcceleration.x', 'userAcceleration.y',
                        'userAcceleration.z']
DEMOGRAPHIC_FEATURES = ['weight', 'height', 'age', 'gender']


# In[ ]:


raw_df[MOTION_SENSOR_COLUMNS].describe(percentiles=[0.001,0.01,0.25,0.5,0.75,0.95,0.99, 0.999]).round(3)


# <font color=blue>
# We can see that some of the feature exhibit some extremely large values (relatively to their distribution) - for 
#     example rotationRate.y. This appears to happen in both sides of the distribution (extremely small values & extremely large ones), and thus less likely to be measurements errors.
# </font>

# In[ ]:


# Plot histograms of the motion features:

samp = raw_df.sample(10**5) # plotting a sample, for run-time considerations
fig, ax = plt.subplots(3, 4, sharex='col', sharey='row', figsize=(10, 8))

m=0
for i in range(3):
    for j in range(4):
        colname = MOTION_SENSOR_COLUMNS[m]
        samp[colname].plot(kind='hist', ax=ax[i,j], bins=20, title=colname, density=True, xlim=(-4,4))
        m += 1
plt.show()


# <font color=blue>
# We observe that most raw features exhibit bell-shaped distributions around 0, while others (like attitute.pitch) are highly skewed</font>

# ### Examine "demographics" distibution:

# In[ ]:


raw_df[['weight','height','age']].describe().round(2)


# <font color=blue>No suspicious values such as negative height or impossible age.</font>

# ### Inspect target-class distribution:

# In[ ]:


# Plot class distribution:
activiry_counts = raw_df.activity.apply(lambda x: code_to_activity[x] ).value_counts()
activiry_counts.plot(kind='bar', title='Activity Class Distibution')
plt.show()


# <font color=blue>classes are not balances, with (walk, sit, stand) ~2X larger than (upstairs, jog, downstairs)</font>

# # Calculate features as summary statistics over fixed window:
# 
# We'll split consecutive measurements (within subject, within experiment) to windows of a fixed size (for example, a window size of 50 corresponds to 1 second of measurements as the measure frequency is 50htz), and calculate a set of summary statistics for the measurements in the window.
# Additionaly, we'll add the demographic features (age, weight, etc.) per subject.

# In[ ]:


def get_processed_df(window_size, summary_statistics):
    """
    Group measurements per subject per experiment to windows of requested size, and calculate a set of summary_statistics 
    withing that window 
    window_size: int, size of window.
    summary_statistics: list, either of string representations of aggregation functions or aggregation function objects
    """
    grouped_df = raw_df.groupby(['subject', 'experiment_id'])
    processed_dfs = []
    for name, group in grouped_df: # iterate over data from consecutive meaurements, per subject per experiment
        subject, experiment = name
        nbins = int(len(group) / window_size) # num of bins depends on length of epxeriment
        bin_edges = pd.cut(group.experiment_step, bins=nbins) 
        activity = group.activity.values[0] # all activites in this df are the same
        agg_per_bin = group[MOTION_SENSOR_COLUMNS].groupby(bin_edges).agg(summary_statistics)

        # fix colum names:
        agg_per_bin = agg_per_bin.reset_index()
        cols = list(agg_per_bin.columns)
        fixed_cols = [str(c[0])+"_"+str(c[1]) for c in cols]
        agg_per_bin.columns = fixed_cols

        # Add the constant features (constant per sbject and experiment):
        agg_per_bin['experiment_id'] = experiment
        agg_per_bin['activity'] = activity # this will be the label
        agg_per_bin['subject'] = subject
        agg_per_bin = agg_per_bin.merge(subject_data, on='subject')
        
        processed_dfs.append(agg_per_bin)
    
    processed_data = pd.concat(processed_dfs)
    return processed_data


# We'll choose the following summary statistics, as they seem like a reasonable approximation for the disribution of a feature within the window. 
# I'd expect these values to exibit different mean, std, etc. when a person switches between activities.
# Additionaly, such measures could vary between different gender/height/weight/age groups, so we keep the demographoc features as well.
# <font color=red>Following cell takes ~1m to run (on a local i7 processor):</font>
# 

# In[ ]:


summary_statistics = ['mean', 'median', 'max', 'min', 'std', 'skew']
processed_data = get_processed_df(window_size=100, summary_statistics=summary_statistics)


# In[ ]:


print(processed_data.columns)


# ### Examine feature behaviour accross classes:
# 
# We'll look at the difference in distribution of some of the features accross classes"

# In[ ]:


features_to_plot = ["attitude.roll_mean", "userAcceleration.y_skew", "gravity.y_median"] # arbitrary choice
for feature in features_to_plot:
    processed_data.groupby('activity')[feature].plot(kind='kde',legend=True,
                                                    title="'%s' density per class" % feature)
    plt.show()


# <font color=blue>It's evident that the features are distributed differently between classes.</font>

# # Extract feature matrix & labels for model training:

# create a list of all the features to train on (sensor summary statistics & demographic features):

# In[ ]:


motion_sensor_features =  [f for f in processed_data.columns if any([x in f for x in MOTION_SENSOR_COLUMNS])]
feature_names = motion_sensor_features + DEMOGRAPHIC_FEATURES


# Create feature matrix **X** and labels vecotr **y**:

# In[ ]:


X = processed_data[feature_names]
y = processed_data['activity']

print("feature matrix shape: ",X.shape)
print("target vector matrix shape: ",y.shape)


# ### Train and evaluate using k-ofld CV, to estimate MCC (and is standard deviation):

# We'll choose the xgboost implementation of **gradient boosted trees clasifier**, as a good off-the-shelf classifier with relative robustness to redundant features and fast training time, as well as explainable results. We will also compare it's perofrmance to multiclass **logistic-regression**.

# In[ ]:


xgb_clf = XGBClassifier(n_jobs=-1)
lr = LogisticRegression(n_jobs=-1)


# Define a scorrer and a CV splitter:
# The CV splitter has to explicitly be defined so that we can demand that data be shuffeld before splitting. This is required because (X,y) are sorted by subject/experiment, so unshuffled splits will produce biased estimators.

# In[ ]:


mcc_scorer = make_scorer(matthews_corrcoef) # defining a custom scorer (requried for computing MCC + accuracy in one go)
kfold_splitter = KFold(n_splits=5, shuffle=True)
named_classifiers = [(xgb_clf, "xgb_classifier"), (lr, "logistic_regression")]


# Train & evaluate both classifeirs <font color=red>(will take a few minutes on the full dataset):</font>
# 

# In[ ]:


cv_results = {}
for model, name in named_classifiers:
    print("fitting %s..." % name)
    cv_results[name] =         cross_validate(model, X, y, 
                        cv=kfold_splitter, scoring={"mcc":mcc_scorer, "accuracy":"accuracy"}, 
                       n_jobs=cpu_count()-1, verbose=3, return_estimator=True) # use all cores but one
    print("Done")


# ### Parse and report cv results - keep only the evaluation metrics (cross_validate also reports timing metrics):

# In[ ]:


test_metrics = {}
for model, results in cv_results.items():
    test_metrics[model] = {}
    for metric, values in results.items():
        if "test" in metric: # skip timing metrics
            test_metrics[model].update({
                metric: np.mean(values),
                metric + "_std": np.std(values)
            })
test_metrics = pd.DataFrame(test_metrics)

print(test_metrics)


# <font color=blue>We can see that the gradient boosted trees model achieves high MCC and accuracy,
# with very small standard deviations. logistic_regression doesn't perform as well.</font>
# 
# 
# 

# # Inspect feature importances

# ### Extract feature importance from one of the trained xgb_classifiers:

# In[ ]:


trained_xgb_clf = cv_results["xgb_classifier"]["estimator"][0] # 0 is abritrary, the estimator from the 1st fold

# get feature importances:
fi = pd.DataFrame(zip(X.columns, trained_xgb_clf.feature_importances_), columns=['feature','importance'])
fi.sort_values(by='importance', ascending=False, inplace=True)
fi.set_index('feature', inplace=True)


# Plot the 20 most important features:

# In[ ]:


fi.head(20).plot(kind='barh', figsize=(10,10), title="Most important features")
plt.show()


# <font color=blue>We can see that all 6 statistics (mean, median, min, max, std, skew) are represented in the top-20 features.</font>

# ### Check how many features (and which ones) were excluded form model:

# In[ ]:


excluded_features = fi[fi.importance == 0]
print(excluded_features)


# <font color=blue>Very few features were not used for splits by the model (in my run they were interestingly all "gravity" features, but this may slightly change accross runs).</font>

# # Additional directions:

# * Experiment with different window sizes. current windows requires 2 seconds of data, which may be a lot for some applications
# * Consider other metrics such as multi-class AUC (I didn't add it here because cross_validate doesn't support it, and I chose to go along woth the parallelized implementation othar than iterating over CV folds)
# * Investigate perfomrance metrics per class and per person in the test set - could be interesting.
# * Deal with anbormal values in the raw data - this requires some more info about their scales (consider some transformation such as log-scale for the features with abnormal values)
# * Conduct meta-parameter optimization, and switch from using the vanilla model.
# * Consider additional models, including RNN architectures (could work on the raw data, or on smaller feature-windows)
# * Use bootstrapping to better estimate the variability of the test metrics
# * Add more summary statistics as features, for example I would try adding more percentiles of each raw feature in the window in order to better convey it's distribution to the model (currently we use only the 0.5 percentile - the median).
# 
