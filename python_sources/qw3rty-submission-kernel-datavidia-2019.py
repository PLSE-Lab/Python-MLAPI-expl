#!/usr/bin/env python
# coding: utf-8

# <h1> <b>Customer Hotel Cross Seling Prediction on Flights Booking </b></h1>
# <p>By <b>qw3rty</b> </p>
# <ul>
# <li>DP. Nala Krisnanda</li>
# <li>M. Alwi Sukra</li>
# <li>Bimo Arief Wicaksana</li>
# </ul>
# 

# <b> Kernel Outline </b>
#  0. Library and data import 
#  1. Data preprocessing
#  2. Exploratory Data Analysis
#  3. Feature Engineering 
#  4. Modelling 
#  5. Validation

# <br>
# <h4> <b>
# 0. Library and Data Import
# </b> </h4>

# <b> 0.1. Library import </b>
# <p> 
# The cell below is for importing useful library for <i>exploratory data analysis</i> (EDA), data visualization, and of course data modeling
# </p>

# In[ ]:


#linear algebra
import numpy as np

#dataframe
import pandas as pd

#data visualization
import matplotlib.pyplot as plt
import seaborn as sns

#regex
import re

#machine learning
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report,f1_score,accuracy_score
from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from vecstack import stacking

#neural network
import tensorflow as tf
from tensorflow import keras


# <b> 0.2. Data import </b>
# <p> 
# In the cell below we import all the data given by the competition for performing the prediction
# </p>

# In[ ]:


df_flight = pd.read_csv("/kaggle/input/datavidia2019/flight.csv")
df_hotel = pd.read_csv("/kaggle/input/datavidia2019/hotel.csv")
df_test = pd.read_csv("/kaggle/input/datavidia2019/test.csv")


# <br>
# <h4> <b>
# 1. Data Preprocessing
# </b> </h4>

# <b> 1.1. Data Quick Look </b>
# <p> 
# In this section we will take a look at the basic information of the data we've been import in the previous section
# </p>

# In[ ]:


df_flight.sample(3)


# In[ ]:


df_flight.info()


# In[ ]:


df_hotel.head()


# In[ ]:


df_hotel.info()


# In[ ]:


df_test.head(3)


# In[ ]:


df_test.info()


# <p>Luckily, on every datasets we don't face any NaN value and all the features seems pretty consistent on their data type.</p>
# 
# <p>From the quick look above here some informations that we got:
# <ul>
# <li> <mark>flight.csv</mark> contain the customer flight booking data </li>
# <li> <mark>hotel.csv</mark> contain the hotel detail information </li>
# <li> <mark>test.csv</mark> is data for testing purpose which has same features with <mark>flight.csv</mark> but without the <i>hotel_id</i> feature</li>
# </ul>
# </p>
# 
#  <p>We jump to the next section!</p>

# <b> 1.2. Data Wrangling</b>
# <p> 
# In this section we will modify data into another format with the intent of making it more appropriate and valuable
# </p>

# <p>Firstly on the cell below, <i>order_id</i> features will be removed as those are computer generated value for customer data encryption. So it is not really going to help us reach our goal</p>

# In[ ]:


df_flight.drop(["order_id"],axis=1,inplace=True)
df_test.drop(["order_id"],axis=1,inplace=True)


# <p>Next, we're going to fix the visited_city and log_transaction feature as it is seems to be a list but still formatted as string so we need to change it a little bit so we can take advantage of it</p>

# In[ ]:


# function for fix list in form of string into a python accessable list
def stringoflist_to_list(text,as_float=False):
    result = re.sub("[\'\,\[\]]","",text)
    result = re.split("\s",result)
    if(as_float):  
        result = [float(x) for x in result]  
    return result


# In[ ]:


#fix the df_flight dataframe
visited_city_idx = df_flight.columns.get_loc("visited_city")
log_transaction_idx = df_flight.columns.get_loc("log_transaction")
for index, row in df_flight.iterrows():
    df_flight.iat[index, visited_city_idx] = stringoflist_to_list(row["visited_city"],False)
    df_flight.iat[index, log_transaction_idx] = stringoflist_to_list(row["log_transaction"],True)


# In[ ]:


#fix the df_test dataframe
visited_city_idx = df_test.columns.get_loc("visited_city")
log_transaction_idx = df_test.columns.get_loc("log_transaction")
for index, row in df_test.iterrows():
    df_test.iat[index, visited_city_idx] = stringoflist_to_list(row["visited_city"],False)
    df_test.iat[index, log_transaction_idx] = stringoflist_to_list(row["log_transaction"],True)


# In[ ]:


df_flight.sample(3)


# In[ ]:


df_test.sample(3)


# In the previous section, spotted that there are slight difference in <i>member_duration_days</i> and <i>no_of_seats</i> features data type. In cell below we fix the consistency of the feature

# In[ ]:


df_test["member_duration_days"] = df_test["member_duration_days"].astype("float64")
df_test["no_of_seats"] = df_test["no_of_seats"].astype("float64")


# Now we look inside every features, checking is it contain any anomaly that might make our predictor model biased

# In[ ]:


df_flight["member_duration_days"].describe()


# In[ ]:


df_test["member_duration_days"].describe()


# Seems there is no problem for the <i>member_duration_days</i> feature
# <br>

# In[ ]:


df_flight["account_id"].value_counts(sort=True)


# We can see for the <i>account_id</i> feature, several customer is booking multiple times in a year. This is very <b>important</b> fact that we may check further more in the feature engineering section

# In[ ]:


df_flight["gender"].value_counts()


# In[ ]:


df_test["gender"].value_counts()


# As we can see there is an extra gender <b>(None)</b> in the flight dataset which mean some action need to be taken to fix this. This data oddity can be handled in many ways. In this case , we will leave it alone as the third gender.

# In[ ]:


df_flight["trip"].value_counts()


# In[ ]:


df_test["trip"].value_counts()


# From the data_dictionary.csv, told that there are only 2 kind of trip available which is <i>trip</i> and <i>roundtrip</i>. By this fact,<b> we assume that <i>round</i> and <i>roundtrip</i> are the same type of trip </b>. Than we fix the dataset in cell below

# In[ ]:


df_flight["trip"] = np.where(df_flight["trip"]=="trip","trip","roundtrip")
df_test["trip"] = np.where(df_test["trip"]=="trip","trip","roundtrip")


# In[ ]:


df_flight["trip"].value_counts()


# In[ ]:


df_test["trip"].value_counts()


# Now the <i> trip</i> feature is good to go
# <br>

# In[ ]:


df_flight["service_class"].value_counts()


# In[ ]:


df_test["service_class"].value_counts()


# No problem for <i>service_class</i> feature
# <br>

# In[ ]:


df_flight["price"].describe()


# In[ ]:


df_test["price"].describe()


# No problem for <i>price</i> feature

# In[ ]:


df_flight["is_tx_promo"].value_counts()


# In[ ]:


df_test["is_tx_promo"].value_counts()


# No problem for <i>is_tx_promo</i> feature

# In[ ]:


df_flight["no_of_seats"].describe()


# In[ ]:


df_test["no_of_seats"].describe()


# No Problem for <i>no_of_seats</i> feature

# In[ ]:


df_flight["airlines_name"].value_counts()


# In[ ]:


df_test["airlines_name"].value_counts()


# No Problem for <i>airlines_name</i> feature

# In[ ]:


df_flight["route"].value_counts()


# In[ ]:


df_test["route"].value_counts()


# only CGK-DPS route available. So there's no point of using this feature. We remove the <i>route</i> feature in cell below

# In[ ]:


df_flight.drop(["route"],axis=1,inplace=True)
df_test.drop(["route"],axis=1,inplace=True)


# In[ ]:


df_flight["hotel_id"].value_counts()


# we creating a new feature in df_flight named <i>is_cross_sell</i> to check wheter the <i>hotel_id</i> is None or not

# In[ ]:


df_flight["is_cross_sell"] = np.where(df_flight["hotel_id"]=="None",0,1)


# In[ ]:


df_flight["is_cross_sell"].value_counts()


# Now we can drop the <i>hotel_id</i> feature 

# In[ ]:


df_flight.drop(["hotel_id"],axis=1,inplace=True)


# In[ ]:


df_flight.sample(3)


# <br>
# <h4> <b>
# 2. Exploratory Data Analysis (EDA)
# </b> </h4>
# 
# <p> 
# Taking insights from the data that may helping to build accurate model
# </p>

# In[ ]:


Var_Corr = df_flight.corr()
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
plt.title("Correlation heatmap")
plt.show()


# <b>2.1. is_cross_selling</b>
# <p> Firstly, we want to see how is the target variable doing in the dataset</p>

# In[ ]:


sns.countplot(x="is_cross_sell", data=df_flight)
plt.title("is_cross_sell class count")
plt.show()


# we can see from the plot above, this classification problem is purely imbalance

# <b>2.2. member duration days</b>

# In[ ]:


sns.distplot(df_flight["member_duration_days"])
plt.title("member_duration_days data distribution")
plt.show()


# In[ ]:


sns.boxplot(x="is_cross_sell", y="member_duration_days", data=df_flight)
plt.title("is_cross_sell vs member_duration_days")
plt.show()


# We can see that there are slight difference between customer who did cross selling and not. Customer who did cross selling tend to be joined to the platform for longer period of time
# <br>

# <b>2.3. Gender</b>

# In[ ]:


sns.countplot(x="gender", hue="is_cross_sell", data=df_flight)
plt.title("gender count")
plt.show()


# We can see from the chart, where Gender 2 which is female is showing greater is_cross_sell ratio than male or None

# <b>2.4. Trip</b>

# In[ ]:


sns.countplot(x="trip", hue="is_cross_sell", data=df_flight)
plt.title("trip count")
plt.show()


# The number of trip is far more greater than the roundtrip. the cross selling activities seems to be linearly correlated to the number of the type of trip

# <b>2.5. Price</b>

# In[ ]:


sns.distplot(df_flight["price"])
plt.title("price data distribution")
plt.show()


# In[ ]:


sns.boxplot(x="is_cross_sell", y="price", data=df_flight)
plt.title("is_cross_sell vs price")
plt.show()


# <b>2.6. Transaction with promo</b>

# In[ ]:


sns.countplot(x="is_tx_promo", hue="is_cross_sell", data=df_flight)
plt.title("tx_with_promo count")
plt.show()


# <b>2.7. Number of Seats</b>

# In[ ]:


sns.boxplot(x="is_cross_sell", y="no_of_seats", data=df_flight)
plt.title("is_cross_sell vs num_of_seats")
plt.show()


# <b>2.8. Airlines Name</b>

# In[ ]:


sns.countplot(x="airlines_name", hue="is_cross_sell", data=df_flight)
plt.title("airlines_name count")
plt.show()


# <br>
# <b>Takeaways from this EDA section</b>:
# 
# 1. This is an imbalance classification problem as the number of actual positive cross selling is far more smaller than the actual negative cross selling
# 2. From the correlation heatmap we can see that our available feature is not correlated with each other. This is a good sign showing that there are no redundant data
# 3. We haven't dig more information for visited_city and log_transaction feature as both of it still in form of python list. These feature need to be fixed so we can get some information from it

# <br>
# <h4> <b>
# 3. Feature Engineering
# </b> </h4>
# 
# <p> 
# 
# </p>

# <p>First, as we already know from the EDA section, <i>visited_city</i> feature contain list(array) type data which not really helpful. We have to extract the data so the feature may be more useful. We're extracting how many city does the customer already visit. We see later if this approach can help us to predict to cross selling </p>

# In[ ]:


## df_flight

#initial value for the column
df_flight["number_city_visited"] = 0.0

#get the index of the column number_city_visited
the_idx = df_flight.columns.get_loc("number_city_visited")


#assign the value for every row
for index, row in df_flight.iterrows():
    df_flight.iat[index, the_idx] = len(row["visited_city"])
    
#drop the original column as we already got what we need
df_flight.drop(["visited_city"],axis=1,inplace=True)


# In[ ]:


##same process as the above cell but for df_test
df_test["number_city_visited"] = 0.0
the_idx = df_test.columns.get_loc("number_city_visited")
for index, row in df_test.iterrows():
    df_test.iat[index, the_idx] = len(row["visited_city"])
df_test.drop(["visited_city"],axis=1,inplace=True)


# <p>Next, <i>log_transaction</i> is same as <i>visited_city</i> feature which contain list(array) type data. The difference is the data inside the list is in form of continous data. So in the cell below we will extract the descriptive statistics which is number of transaction, min, max, average, and total as a new dataframe column</p>

# In[ ]:


##df_flight
#initial column value
df_flight["amount_spent"] = 0.0
df_flight["min_spend"] = 0.0
df_flight["max_spend"] = 0.0
df_flight["average_spend"] = 0.0
df_flight["number_of_transaction"] = 0.0

#assign value for each row
for index, row in df_flight.iterrows():
    df_flight.iat[index, -5] = sum(row["log_transaction"])
    df_flight.iat[index, -4] = min(row["log_transaction"])
    df_flight.iat[index, -3] = max(row["log_transaction"])
    df_flight.iat[index, -2] = np.average(row["log_transaction"])
    df_flight.iat[index, -1] = len(row["log_transaction"])
    
# drop the original column as we already got what we need
df_flight.drop(["log_transaction"],axis=1,inplace=True)


# In[ ]:


#same process for the above cell but fr df_test
df_test["amount_spent"] = 0.0
df_test["min_spend"] = 0.0
df_test["max_spend"] = 0.0
df_test["average_spend"] = 0.0
df_test["number_of_transaction"] = 0.0
for index, row in df_test.iterrows():
    df_test.iat[index, -5] = sum(row["log_transaction"])
    df_test.iat[index, -4] = min(row["log_transaction"])
    df_test.iat[index, -3] = max(row["log_transaction"])
    df_test.iat[index, -2] = np.average(row["log_transaction"])
    df_test.iat[index, -1] = len(row["log_transaction"])
df_test.drop(["log_transaction"],axis=1,inplace=True)


# In[ ]:


df_flight.sample(3)


# In[ ]:


df_test.sample(3)


# <p>As we already know from the data wrangling section, some customers are doing bookings more than once in a year. We want to see if previous cross selling will trigger another cross selling on their next flight booking. In the cells below, we try to make use of feature account_id to find the is_cross_sell probability of each account </p>

# In[ ]:


ics_history = df_flight.groupby(["account_id"]).mean()["is_cross_sell"]


# In[ ]:


df_flight = pd.merge(df_flight, ics_history, on='account_id', how="left")
old_column_name = df_flight.columns[-1]
df_flight = df_flight.rename(columns = {"is_cross_sell_x" : "is_cross_sell", old_column_name : "ics_probability"})


# In[ ]:


df_test = pd.merge(df_test, ics_history, on='account_id', how="left")
old_column_name = df_test.columns[-1]
df_test[old_column_name] = df_test[old_column_name].fillna(0)
df_test = df_test.rename(columns = {"is_cross_sell_x" : "is_cross_sell", old_column_name : "ics_probability"})


# In[ ]:


df_flight.drop(["account_id"],axis=1,inplace=True)
df_test.drop(["account_id"],axis=1,inplace=True)


# In[ ]:


df_flight.sample(3)


# In[ ]:


df_test.sample(3)


# <b>3.1. Data labelling</b>
# <p>Next, we're going to transform any attribute which the type is text into a number</p>
# 
# 1. binary categorical data will be transform to 0/1 with scikit-learn LabelEncoder class.
# 2. multinomial categorical data which is in this data is gender and airlines name will be transform into one hot encoding with pandas get_dummies method.

# In[ ]:


#trip
le = LabelEncoder()
df_flight['trip'] = le.fit_transform(df_flight['trip'])

#service_class
le = LabelEncoder()
df_flight['service_class'] = le.fit_transform(df_flight['service_class'])

#is_tx_promo
le = LabelEncoder()
df_flight['is_tx_promo'] = le.fit_transform(df_flight['is_tx_promo'])


# In[ ]:


#trip
le = LabelEncoder()
df_test['trip'] = le.fit_transform(df_test['trip'])

#service_class
le = LabelEncoder()
df_test['service_class'] = le.fit_transform(df_test['service_class'])

#is_tx_promo
le = LabelEncoder()
df_test['is_tx_promo'] = le.fit_transform(df_test['is_tx_promo'])


# In[ ]:


# make sure to concat the training and testing so both of the data get equal data dummies
df = pd.concat([df_flight[df_flight.columns[:-1]], df_test],sort = False)
airlines_bin_df = pd.get_dummies(df["airlines_name"], prefix="an")
gender_bin_df = pd.get_dummies(df["gender"], prefix="g")


# In[ ]:


df_flight = pd.concat([df_flight, gender_bin_df.iloc[:len(df_flight)], airlines_bin_df.iloc[:len(df_flight)]], axis = 1)
df_flight.drop(["gender"],axis=1,inplace=True)
df_test.drop(["gender"],axis=1,inplace=True)

df_test = pd.concat([df_test, gender_bin_df.iloc[len(df_flight):], airlines_bin_df.iloc[len(df_flight):]], axis = 1)
df_flight.drop(["airlines_name"],axis=1,inplace=True)
df_test.drop(["airlines_name"],axis=1,inplace=True)


# In[ ]:


df_flight.sample(3)


# In[ ]:


df_test.sample(3)


# <b>3.2 Data Scaling</b>
# <p>The scaling method used here is the minmax scaling. We scale only the continous data not the categorical data</p>

# In[ ]:


feature_to_scale = ["member_duration_days","price","no_of_seats","number_city_visited","amount_spent","min_spend","max_spend","average_spend","number_of_transaction"]


# In[ ]:


minmax_scaler = MinMaxScaler()
df_temp = pd.concat([df_flight.drop(["is_cross_sell"],axis=1), df_test],sort=False)
minmax_scaler.fit(df_temp[feature_to_scale],)


# In[ ]:


df_flight[feature_to_scale] = minmax_scaler.transform(df_flight[feature_to_scale])
df_test[feature_to_scale] = minmax_scaler.transform(df_test[feature_to_scale])


# In[ ]:


df_flight.sample(3)


# In[ ]:


df_test.sample(3)


# So here is the final dataframe after this section finished

# In[ ]:


df_flight.sample(3)


# In[ ]:


df_test.sample(3)


# <br>
# <h4> <b>
# 4. Modeling
# </b> </h4>
# 
# <p> 
# Build ML Model
# </p>

# In[ ]:


X = df_flight.drop(["is_cross_sell"],axis=1)
y = df_flight["is_cross_sell"]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# <b>4.1. Single Model</b>
# <p>We try to see the performance of difference ML algorithms in this data</p>

# <b>Random Forest Classifier</b>

# In[ ]:


# the hyperparameter is tuned on other kernel
rf_clf = RandomForestClassifier(
    bootstrap=False,
    max_depth=50,
    max_features = "auto",
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=1000,
    n_jobs=-1
)


# In[ ]:


rf_clf.fit(X_train,y_train)


# In[ ]:


rf_clf.score(X_test,y_test)


# In[ ]:


f1_score(rf_clf.predict(X_test),y_test)


# <b>K-Nearest Neighbor</b>

# In[ ]:


knn_clf = KNeighborsClassifier(n_neighbors=5)


# In[ ]:


knn_clf.fit(X_train,y_train)


# In[ ]:


knn_clf.score(X_test,y_test)


# In[ ]:


f1_score(knn_clf.predict(X_test),y_test)


# <b>Logistic Regression</b>

# In[ ]:


lr_clf = LogisticRegression(max_iter=1000)


# In[ ]:


lr_clf.fit(X_train,y_train)


# In[ ]:


lr_clf.score(X_test,y_test)


# In[ ]:


f1_score(lr_clf.predict(X_test),y_test)


# <b>Neural Network</b>

# In[ ]:


n_cols = X_train.shape[1] 
nn_clf = keras.Sequential()
nn_clf.add(keras.layers.Dense(20, activation='relu', input_shape=(n_cols,)))
nn_clf.add(keras.layers.Dropout(rate=0.2))
nn_clf.add(keras.layers.Dense(20, activation='relu'))
nn_clf.add(keras.layers.Dropout(rate=0.2))
nn_clf.add(keras.layers.Dense(1, activation='sigmoid'))


# In[ ]:


nn_clf.compile( 
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# In[ ]:


nn_clf.fit(X_train, y_train, epochs=10, verbose=2)


# In[ ]:


nn_clf.evaluate(X_test,y_test)


# In[ ]:


f1_score(np.where(nn_clf.predict(X_test)> 0.5, 1, 0),y_test)


# <b>XGBoost</b>

# In[ ]:


xgb_clf = XGBClassifier(random_state=0)


# In[ ]:


xgb_clf.fit(X_train,y_train)


# In[ ]:


xgb_clf.score(X_test,y_test)


# In[ ]:


f1_score(xgb_clf.predict(X_test),y_test)


# From the attempt did above, we can see the logistic regression somehow outperform the other algorithm. It reach 0.86 F1 score. The score followed by the XGBoost and the random forest classifier
# <p>But the difference between all algorithm is not significance</p>

# <b>4.2. Stacking Model</b>
# <p>The idea of this model is to use the output of ML algorithm (layer 0) as a new feature for another ML algorithm (layer 1) </p>

# <b>4.2.1. Base Classifier</b>
# <p>layer 0 also called as base classifier usually using different kind of algorithm. In this case, for the base classifier we use:</p>
# 
# 1. Distance (neighbor) base algorithm (KNN)
# 2. Ensamble algorithm (RF)
# 3. Linear algorithm (LR)

# <b>Random Forest Classifier</b>

# In[ ]:


rf_clf = RandomForestClassifier(
    bootstrap=False,
    max_depth=50,
    max_features = "auto",
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=1000,
    n_jobs=-1
)


# <b>K-Nearest Neighbor</b>

# In[ ]:


knn_clf = KNeighborsClassifier(n_neighbors=5)


# <b>Logistic Regression</b>

# In[ ]:


lr_clf = LogisticRegression(max_iter=1000)


# In[ ]:


models = [
    rf_clf,
    knn_clf,
    lr_clf,
]


# In[ ]:


S_train, S_test = stacking(
    models,                   
    X_train, y_train, X_test,   
    regression=False, 
    mode='oof_pred_bag', 
    needs_proba=False,
    save_dir=None, 
    metric=accuracy_score, 
    n_folds=4, 
    stratified=True,
    shuffle=True,  
    random_state=0,    
    verbose=2
)


# In[ ]:


S_train = pd.DataFrame(S_train, columns = ["rf","knn","lr"])
S_test = pd.DataFrame(S_test, columns = ["rf","knn","lr"])


# In[ ]:


X_train.reset_index(inplace=True)
X_test.reset_index(inplace=True)


# In[ ]:


df_stack_train = pd.concat([X_train,S_train],axis=1)
df_stack_test = pd.concat([X_test,S_test],axis=1)


# In[ ]:


df_stack_train.drop(["index"],axis=1,inplace=True)
df_stack_train.head()


# In[ ]:


df_stack_test.drop(["index"],axis=1,inplace=True)
df_stack_test.head()


# <b>4.2.2. Meta Classifier</b>
# <p>We will build the meta classifier. In this chance, we try to use neural network and XGBoost Classifier</p>

# In[ ]:


n_cols = df_stack_train.shape[1]


# In[ ]:


nn_clf = keras.Sequential()
nn_clf.add(keras.layers.Dense(16, activation='relu', input_shape=(n_cols,)))
nn_clf.add(keras.layers.Dropout(rate=0.2))
nn_clf.add(keras.layers.Dense(1, activation='sigmoid'))


# In[ ]:


nn_clf.compile( 
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# In[ ]:


xgb_clf = XGBClassifier(random_state=0)


# In[ ]:


nn_clf.fit(df_stack_train, y_train, epochs=10, verbose=2)


# In[ ]:


y_pred_nn = np.where(nn_clf.predict(df_stack_test)> 0.5, 1, 0)


# In[ ]:


f1_score(y_pred_nn,y_test)


# In[ ]:


xgb_clf.fit(df_stack_train,y_train)


# In[ ]:


y_pred_xgb = xgb_clf.predict(df_stack_test)


# In[ ]:


xgb_clf.score(df_stack_test,y_test)


# In[ ]:


f1_score(y_pred_xgb,y_test)


# We can see that using neural network as meta classifier not giving any improvement while using XGBoost giving a slight improvement which is worth to try. So for now we are using this method which is our best accuracy obtained

# <br>
# <h4> <b>
# 5. Validation
# </b> </h4>
# 
# <p> 
# Preparing model for validation in Kaggle platform
# </p>

# <b> 5.1 Base Classifier</b>

# <b>Random Forest Classifier</b>

# In[ ]:


rf_clf = RandomForestClassifier(
    bootstrap=False,
    max_depth=50,
    max_features = "auto",
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=1000,
    n_jobs=-1
)


# <b>K-Nearest Neighbor</b>

# In[ ]:


knn_clf = KNeighborsClassifier(n_neighbors=5)


# <b>Logistic Regression</b>

# In[ ]:


lr_clf = LogisticRegression(max_iter=1000)


# In[ ]:


models = [
    rf_clf,
    knn_clf,
    lr_clf,
]


# In[ ]:


S_train, S_test = stacking(
    models,                   
    X, y, df_test,   
    regression=False, 
    mode='oof_pred_bag', 
    needs_proba=False,
    save_dir=None, 
    metric=accuracy_score, 
    n_folds=4, 
    stratified=True,
    shuffle=True,  
    random_state=0,    
    verbose=2
)


# In[ ]:


S_train = pd.DataFrame(S_train, columns = ["rf","knn","lr"])
S_test = pd.DataFrame(S_test, columns = ["rf","knn","lr"])


# In[ ]:


df_stack_train = pd.concat([X,S_train],axis=1)
df_stack_test = pd.concat([df_test,S_test],axis=1)


# <b>5.2. Meta Classifier</b>
# <p>We will build the meta classifier. In this chance, we try to use neural network and XGBoost Classifier</p>

# In[ ]:


xgb_clf = XGBClassifier(random_state=0)


# In[ ]:


xgb_clf.fit(df_stack_train,y)


# In[ ]:


y_pred_xgb = xgb_clf.predict(df_stack_test)


# In[ ]:


temp = pd.read_csv("/kaggle/input/datavidia2019/test.csv")
temp = temp["order_id"]

d = {'order_id': temp, 'is_cross_sell': y_pred_xgb}
output = pd.DataFrame(data = d)

output["is_cross_sell"] = np.where(output["is_cross_sell"]==0,"no","yes")
output.to_csv("stacking-meta-xgb.csv",index=False)

