# Grab AI SEA Challenge Submission
# By Ng Jia Sheng Jason, a student from Singapore University of Technology and Design
# Summary of project: I decided to use supervised learning as the method for model prediction. As the data given is a time-series data,
# ts_fresh package is used to extract meaningful features such as mean and median out of the time-series data. Afterwards, with further
# data processing and data merging, Extreme Learning Machine is chosen and used as the desired prediction algorithm due to its incredible
# learning speeds and high accuracy scores. With that, my model is able to predict dangerous driving with an accuracy of 76.81%.

# A table of results showing the algorithms and hyperparameters I tested are available in the draft environment.

# importing of all packages that I need
# Kaggle does not have the package sklearn_extensions and it does not seem to install it properly. It would be great if you can kindly copy the whole script to a local kernel to run it.
# I tried to run the code part by part using the kernel on Kaggle using shift enter, but somehow information get lost along the way without the kernel refreshing.
import pandas as pd
import tsfresh
from tsfresh.feature_extraction import MinimalFCParameters
import time
from sklearn.preprocessing import StandardScaler
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# reading of the first data in the feature folder + minor pre-processing
df_1 = pd.read_csv(r"../input/grabai/part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df_2 = pd.read_csv(r"../input/grabai/part-00001-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df_3 = pd.read_csv(r"../input/grabai/part-00002-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df_4 = pd.read_csv(r"../input/grabai/part-00003-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df_5 = pd.read_csv(r"../input/grabai/part-00004-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df_6 = pd.read_csv(r"../input/grabai/part-00005-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df_7 = pd.read_csv(r"../input/grabai/part-00006-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df_8 = pd.read_csv(r"../input/grabai/part-00007-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df_9 = pd.read_csv(r"../input/grabai/part-00008-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df_10 = pd.read_csv(r"../input/grabai/part-00009-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv")
df = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10])
df = df.sort_values(["bookingID", "second"], ascending = [True, True])
df = df.reset_index(drop = True)

# some checks on the data to see if it requires further pre-processing
df.dtypes # returns int64 and float64
df.isnull().sum() # returns 0 for all rows, there is no missing values in any row.

# as the data given is a time series data, ts fresh package is used to extract relevant features out of the data so that the features produced are more meaningful.
extracted_df = tsfresh.extract_features(df, column_id = "bookingID", column_sort = "second", default_fc_parameters = MinimalFCParameters())
extracted_df = extracted_df.reset_index()
extracted_df.rename(columns = {"id": "bookingID"}, inplace = True)

# reading of the labels data and making an inner join with the target labels
labels = pd.read_csv(r"../input/data-labels/Data labels.csv")
labels = labels.sort_values(["bookingID"], ascending = [True])
labels = labels.reset_index(drop = True)
result = pd.merge(extracted_df, labels, on = "bookingID", how = "inner")
result = result.sort_values(["bookingID"], ascending = [True])
result = result.reset_index(drop = True)

# seperation of target and feature variables
target = result["label"]
features = result.drop("label", axis = 1)

# standardizing the input feature
sc = StandardScaler()
features = sc.fit_transform(features)

# spliting of training and test set
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.25, random_state = 42)

# application of hyperparamter tuning, ELM model and scoring
tune = [{"hidden_layer__n_hidden": [20, 100, 150, 200, 205],
         "hidden_layer__activation_func": ["tanh", "sigmoid", "gaussian"]}]

elm = GridSearchCV(GenELMClassifier(), tune, cv = 10)

start = time.time()
elm.fit(X_train, y_train)
end = time.time()
print("model training time: " + str(end - start) + " seconds")

y_pred = elm.predict(X_test)
print(classification_report(y_test, y_pred))
print("the best parameters are: " + str(elm.best_params_))
print("model accuracy: " + str(elm.best_score_))

# after the application of hyperparameter tuning, I have discovered that the right number of nodes is 205 and the best activation function is sigmoid function.
# this gives an accuracy of 0.7681
# due to the lack of proper GPU and time, I had to use MinimalFCParameters for tsfresh extracting. With a proper GPU and a god amount of time, the full features from each dataset can be extracted. There might be more meaningful features that can help in increasing the accuracy of the model.
# ELM model is used due to its fast training speeds (6.763s on average) and relatively high prediction accuracies.
# the current ELM package from sklearn_extensions does not seem to have a method to extract out the output weights, or we can identify the features that are contributing most to the prediciton model.
# thank you for taking your time to view my work. If you do have any queries, please feel free to email me @ jason_ng@mymail.sutd.edu.sg.
